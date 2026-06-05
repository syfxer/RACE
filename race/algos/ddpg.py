import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_rarr_residual_method(rd_method):
	return rd_method in {"RARR_VIB", "RARR_RD", "RARR_RRD", "RARR_Diaster", "VIB_StepSOC"}


def is_rrd_like_method(rd_method):
	return "RRD" in rd_method or rd_method == "VIB"


def is_diaster_like_method(rd_method):
	return rd_method in {"Diaster", "RARR_Diaster"}


# Re-tuned version of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action

	
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		return self.l3(q)


class DDPG(object):
	def __init__(self, args, state_dim, action_dim, max_action, discount=0.99, tau=0.005):
		self.args = args
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.discount = discount
		self.tau = tau
		self.total_it = 0

		from race.rewards import import_reward_model
		self.reward_model = import_reward_model(self.args)
		self.total_it = 0


	def select_action(self, state, deterministic=False):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1
		# Sample replay buffer 
		train_info = {}
		state, action, next_state, reward, dense_reward, not_done = replay_buffer.sample(batch_size) #bs,d

		if self.args.dense_r:
			reward = dense_reward
		if self.reward_model is not None and is_rarr_residual_method(self.args.rd_method):
			residual_reward = self.reward_model.forward(state, action, next_state)
			residual_eta = float(getattr(self.args, "rarr_residual_eta", getattr(self.args, "rarr_vib_eta", 1.0)))
			reward = reward + residual_eta * residual_reward.detach()
		elif self.reward_model is not None:
			reward = self.reward_model.forward(state, action, next_state).detach()
		train_info['reward_pred_err'] = F.mse_loss(reward, dense_reward).item()

		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)
		train_info['critic_loss'] = critic_loss.item()

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()
		train_info['actor_loss'] = actor_loss.item()
		
		# Optimize the actor 
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		if self.reward_model is not None:
			if is_rarr_residual_method(self.args.rd_method):
				if self.args.rd_method == "RARR_RRD":
					traj_num = max(int(self.args.batch_size//self.args.rrd_k), 1)
					traj_state, traj_action, traj_next_state, traj_reward, traj_dense_reward, traj_not_done, traj_episode_return, traj_episode_length = replay_buffer.sample_traj(traj_num)
				else:
					traj_state, traj_action, traj_next_state, traj_reward, traj_dense_reward, traj_not_done, traj_episode_return, traj_episode_length = replay_buffer.sample_traj(max(int(batch_size//np.mean(replay_buffer.episode_length)), 1), length_priority=is_diaster_like_method(self.args.rd_method))
			elif is_rrd_like_method(self.args.rd_method):
				traj_state, traj_action, traj_next_state, traj_reward, traj_dense_reward, traj_not_done, traj_episode_return, traj_episode_length= replay_buffer.sample_traj(int(self.args.batch_size//self.args.rrd_k)) #bs,t,d
			elif is_diaster_like_method(self.args.rd_method):
				traj_state, traj_action, traj_next_state, traj_reward, traj_dense_reward, traj_not_done, traj_episode_return, traj_episode_length = replay_buffer.sample_traj(batch_size//4, length_priority=True) #bs,t,d
			else:
				traj_state, traj_action, traj_next_state, traj_reward, traj_dense_reward, traj_not_done, traj_episode_return, traj_episode_length = replay_buffer.sample_traj(max(int(batch_size//np.mean(replay_buffer.episode_length)), 1)) #bs,t,d
			if is_rarr_residual_method(self.args.rd_method):
				reward_model_loss = self.reward_model.update(
					traj_state,
					traj_action,
					traj_next_state,
					traj_episode_return,
					traj_episode_length,
					base_rewards=traj_reward,
					dense_rewards=traj_dense_reward,
				)
			else:
				reward_model_loss = self.reward_model.update(traj_state, traj_action, traj_next_state, traj_episode_return, traj_episode_length)
			if isinstance(reward_model_loss, dict):
				train_info.update(reward_model_loss)
				train_info['reward_model_loss'] = next(iter(reward_model_loss.values()), 0.0)
			else:
				train_info['reward_model_loss'] = reward_model_loss

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		return train_info


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
		
