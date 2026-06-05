import numpy as np
import torch
import torch.nn as nn

from race.rewards.module import RNN_Reward_Model, Reward_Model


class RARRResidualBase(nn.Module):
    """Base class for residual reward redistribution on top of RARR rewards."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.soc_target = float(getattr(args, "soc_target", 0.3))
        self.soc_deadband = float(getattr(args, "soc_deadband", 0.02))
        self.lambda_soc = float(getattr(args, "lambda_soc", getattr(args, "rarr_lambda_soc", 100.0)))
        self.soc_deadband_mode = str(getattr(args, "soc_deadband_mode", "linear")).lower()
        self.loss_fn = nn.MSELoss(reduction="mean")

    def target_residual_return(self, next_states, episode_length):
        lengths = episode_length.long().reshape(-1)
        batch_indices = torch.arange(next_states.shape[0], device=next_states.device)
        final_indices = torch.clamp(lengths - 1, min=0, max=next_states.shape[1] - 1)
        final_soc = next_states[batch_indices, final_indices, 1]
        raw_error = torch.abs(final_soc - self.soc_target)
        deadband_violation = torch.clamp(raw_error - self.soc_deadband, min=0.0)
        if self.soc_deadband_mode == "linear":
            target_cost = self.lambda_soc * deadband_violation
        elif self.soc_deadband_mode == "squared":
            target_cost = self.lambda_soc * deadband_violation.pow(2)
        else:
            raise ValueError(f"Unknown soc deadband mode: {self.soc_deadband_mode}")
        eta = float(getattr(self.args, "rarr_residual_eta", getattr(self.args, "rarr_vib_eta", 1.0)))
        target_reward = -target_cost
        if abs(eta) > 1e-12:
            target_reward = target_reward / eta
        return target_reward.reshape(-1, 1)

    @staticmethod
    def build_transition_features(states, actions, next_states):
        return torch.cat([states, actions, states - next_states], dim=-1)


class RARRRDRewardDecomposer(RARRResidualBase):
    """RARR + RD: learns a residual terminal-SOC reward with an RD objective."""

    def __init__(self, args):
        super().__init__(args)
        input_dim = int(args.obs_dim * 2 + args.action_dim)
        self.reward_model = Reward_Model(input_dim=input_dim, device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.reward_model.parameters(), lr=3e-4)

    def forward(self, states, actions, next_states):
        features = self.build_transition_features(states, actions, next_states)
        return self.reward_model(features)

    def update(self, states, actions, next_states, episode_return, episode_length, base_rewards=None, dense_rewards=None):
        target_return = self.target_residual_return(next_states, episode_length).detach()
        rewards = self.forward(states, actions, next_states)
        lengths = episode_length.reshape(-1)
        for i in range(rewards.shape[0]):
            rewards[i, int(lengths[i].item()):] = 0
        pred_returns = rewards.sum(dim=1).reshape(-1) / lengths
        target_returns = target_return.reshape(-1) / lengths
        loss = self.loss_fn(pred_returns, target_returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())


class RARRRRDRewardDecomposer(RARRRDRewardDecomposer):
    """RARR + RRD: residual terminal-SOC redistribution with random time-step sampling."""

    def __init__(self, args):
        super().__init__(args)
        self.K = int(getattr(args, "rrd_k", 64))

    def update(self, states, actions, next_states, episode_return, episode_length, base_rewards=None, dense_rewards=None):
        target_return = self.target_residual_return(next_states, episode_length).detach()
        rewards = self.forward(states, actions, next_states)
        sampled_rewards = []
        var_coef = []
        for i in range(rewards.shape[0]):
            local_length = int(episode_length[i].item())
            sampled_steps = np.random.choice(local_length, self.K, replace=self.K > local_length)
            sampled_rewards.append(rewards[i, sampled_steps])
            var_coef.append(1.0 - self.K / local_length)

        sampled_rewards = torch.stack(sampled_rewards, dim=0)
        sampled_rewards_var = torch.sum(
            torch.square(sampled_rewards - torch.mean(sampled_rewards, dim=1, keepdim=True)),
            dim=1,
        ) / max(self.K - 1, 1)
        sampled_rewards_var = torch.mean(
            sampled_rewards_var.squeeze()
            * torch.tensor(var_coef, device=sampled_rewards_var.device, dtype=sampled_rewards_var.dtype)
            / max(self.K, 1)
        )
        pred_returns = sampled_rewards.mean(dim=1).reshape(-1)
        target_returns = target_return.reshape(-1) / episode_length.reshape(-1)
        loss = self.loss_fn(pred_returns, target_returns)
        if bool(getattr(self.args, "rrd_unbiased", False)):
            loss = loss - sampled_rewards_var
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())


class RARRDiasterRewardDecomposer(RARRResidualBase):
    """RARR + Diaster-style residual redistribution."""

    def __init__(self, args):
        super().__init__(args)
        input_dim = int(args.obs_dim + args.action_dim)
        self.sub_reward_model = RNN_Reward_Model(input_dim=input_dim, device=self.device).to(self.device)
        self.reward_model = Reward_Model(input_dim=input_dim, device=self.device).to(self.device)
        self.return_scale = float(getattr(args, "rarr_diaster_return_scale", 10.0))
        self.update_t = 0
        self.sub_reward_optim = torch.optim.Adam(self.sub_reward_model.parameters(), lr=3e-4)
        self.reward_model_optim = torch.optim.Adam(self.reward_model.parameters(), lr=3e-4)

    def forward(self, states, actions, next_states):
        return self.reward_model(torch.cat([states, actions], dim=-1))

    def learn_sub_reward_from(self, states, actions, target_return, mask):
        bs, steps = states.shape[:2]
        mask = mask.view(bs, steps)
        mask_len = mask.sum(dim=-1).long()
        state_action = torch.cat([states, actions], dim=-1).view(bs, steps, -1)
        target = target_return.reshape(-1, 1) / self.return_scale

        total_loss = 0.0
        for it in range(steps):
            break_point = np.random.randint(steps)
            self.sub_reward_model.init_hidden()
            sub_r = self.sub_reward_model(state_action[:, : break_point + 1]) * mask[:, : break_point + 1]
            if break_point < steps - 1:
                self.sub_reward_model.init_hidden()
                sub_r2 = self.sub_reward_model(state_action[:, break_point + 1 :]) * mask[:, break_point + 1 :]
                sub_r = torch.cat((sub_r, sub_r2), dim=-1)
            sub_r = sub_r[torch.arange(bs, device=states.device), mask_len - 1] + (
                mask_len > break_point + 1
            ).float() * sub_r[:, break_point]
            loss = ((sub_r.flatten() - target.flatten()).pow(2)).mean()
            self.sub_reward_optim.zero_grad()
            loss.backward()
            self.sub_reward_optim.step()
            total_loss += float(loss.detach().cpu())
        return total_loss / max(steps, 1)

    def learn_step_reward_from(self, states, actions, target_return, mask):
        bs, steps = states.shape[:2]
        mask = mask.view(bs, steps)
        mask_len = mask.sum(dim=-1).long()
        state_action = torch.cat([states, actions], dim=-1).view(bs, steps, -1)
        target = target_return.reshape(-1, 1) / self.return_scale
        with torch.no_grad():
            self.sub_reward_model.init_hidden()
            sub_r = self.sub_reward_model(state_action) * mask
            sub_r[torch.arange(bs, device=states.device), mask_len - 1] = target.flatten()
            diff_r = sub_r - torch.cat((torch.zeros((bs, 1), device=self.device), sub_r[:, :-1]), dim=-1)

        flat_states = states.reshape(bs * steps, -1)
        flat_actions = actions.reshape(bs * steps, -1)
        probabilities = (mask.flatten() / mask.sum()).detach().cpu().numpy()
        total_loss = 0.0
        for _ in range(steps):
            indices = np.random.choice(np.arange(bs * steps), size=bs, p=probabilities, replace=False)
            index_tensor = torch.tensor(indices, device=states.device, dtype=torch.long)
            step_target = diff_r.flatten()[index_tensor]
            step_reward = self.reward_model(torch.cat([flat_states[index_tensor], flat_actions[index_tensor]], dim=-1)).flatten()
            loss = (step_reward - step_target).pow(2).mean()
            self.reward_model_optim.zero_grad()
            loss.backward()
            self.reward_model_optim.step()
            total_loss += float(loss.detach().cpu())
        return total_loss / max(steps, 1)

    def update(self, states, actions, next_states, episode_return, episode_length, base_rewards=None, dense_rewards=None):
        target_return = self.target_residual_return(next_states, episode_length).detach()
        max_length = int(max(episode_length).item())
        states = states[:, :max_length].contiguous()
        actions = actions[:, :max_length].contiguous()
        next_states = next_states[:, :max_length].contiguous()
        mask = (
            torch.arange(states.shape[1], device=episode_length.device)
            .reshape(1, -1)
            .repeat(states.shape[0], 1)
            < episode_length.reshape(-1, 1)
        )
        if self.update_t % 1000 == 0:
            sub_loss = self.learn_sub_reward_from(states, actions, target_return, mask)
            step_loss = self.learn_step_reward_from(states, actions, target_return, mask)
            loss = 0.5 * (sub_loss + step_loss)
        else:
            loss = 0.0
        self.update_t += 1
        return float(loss)
