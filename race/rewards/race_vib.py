import torch.nn as nn
import torch

from .vib_reward import VIBRewardConfig, VIBRewardModel


class RARRVIBRewardDecomposer(nn.Module):
    """Learns residual return not explained by the RARR base reward."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.eta_vib = float(getattr(args, "rarr_vib_eta", 1.0))
        self.soc_target = float(getattr(args, "soc_target", 0.3))
        self.soc_deadband = float(getattr(args, "soc_deadband", 0.02))
        self.lambda_soc = float(getattr(args, "lambda_soc", getattr(args, "rarr_lambda_soc", 100.0)))
        self.soc_deadband_mode = str(getattr(args, "soc_deadband_mode", "linear"))
        input_dim = int(args.obs_dim * 2 + args.action_dim)
        config = VIBRewardConfig(
            input_dim=input_dim,
            latent_dim=int(getattr(args, "rarr_vib_latent_dim", getattr(args, "IB_latent_dim", 8))),
            hidden_dim=int(getattr(args, "rarr_vib_hidden_dim", 128)),
            beta_vib=float(getattr(args, "rarr_vib_beta", 1e-3)),
            eta_vib=self.eta_vib,
            lr=float(getattr(args, "rarr_vib_lr", 3e-4)),
            reward_clip=getattr(args, "rarr_vib_reward_clip", None),
        )
        self.model = VIBRewardModel(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()

    def build_features(self, states, actions, next_states):
        return torch.cat([states, actions, next_states - states], dim=-1)

    def forward(self, states, actions, next_states):
        x = self.build_features(states, actions, next_states)
        raw_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        rewards, _, _ = self.model(x, deterministic=True)
        return rewards.reshape(*raw_shape, 1)

    def update(
        self,
        states,
        actions,
        next_states,
        episode_return,
        episode_length,
        base_rewards=None,
        dense_rewards=None,
    ):
        if base_rewards is None or dense_rewards is None:
            return 0.0

        x = self.build_features(states, actions, next_states)
        rewards, mu, logvar = self.model(x.reshape(-1, x.shape[-1]), deterministic=False)
        rewards = rewards.reshape(x.shape[0], x.shape[1])

        lengths = episode_length.long().reshape(-1)
        masks = torch.arange(x.shape[1], device=x.device).reshape(1, -1) < lengths.reshape(-1, 1)
        pred_residual = (rewards * masks.float()).sum(dim=1)
        batch_indices = torch.arange(x.shape[0], device=x.device)
        final_indices = torch.clamp(lengths - 1, min=0, max=x.shape[1] - 1)
        final_soc = next_states[batch_indices, final_indices, 1]
        raw_error = torch.abs(final_soc - self.soc_target)
        deadband_soc_violation = torch.clamp(raw_error - self.soc_deadband, min=0.0)
        if self.soc_deadband_mode.lower() == "linear":
            target_soc_cost = self.lambda_soc * deadband_soc_violation
        elif self.soc_deadband_mode.lower() == "squared":
            target_soc_cost = self.lambda_soc * deadband_soc_violation.pow(2)
        else:
            raise ValueError(f"Unknown soc deadband mode: {self.soc_deadband_mode}")
        target_soc_reward = -target_soc_cost
        if abs(self.eta_vib) > 1e-12:
            target_soc_reward = target_soc_reward / self.eta_vib

        pred_loss = self.loss_fn(pred_residual, target_soc_reward.detach())
        kl = self.model.kl_loss(mu, logvar)
        loss = pred_loss + self.model.config.beta_vib * kl

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "rarr_vib_loss": float(loss.detach().cpu()),
            "rarr_vib_pred_loss": float(pred_loss.detach().cpu()),
            "rarr_vib_kl": float(kl.detach().cpu()),
            "rarr_vib_target_soc_cost": float(target_soc_cost.mean().detach().cpu()),
            "rarr_vib_deadband_soc_violation": float(deadband_soc_violation.mean().detach().cpu()),
            "rarr_vib_target_return": float(target_soc_reward.mean().detach().cpu()),
            "rarr_vib_pred_return": float(pred_residual.mean().detach().cpu()),
        }
