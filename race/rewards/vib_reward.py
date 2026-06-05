from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class VIBRewardConfig:
    input_dim: int
    latent_dim: int = 8
    hidden_dim: int = 128
    beta_vib: float = 1e-3
    eta_vib: float = 1.0
    lr: float = 3e-4
    reward_clip: Optional[float] = None


class VIBRewardModel(nn.Module):
    def __init__(self, config: VIBRewardConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar_head = nn.Linear(config.hidden_dim, config.latent_dim)
        self.reward_decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = torch.clamp(self.logvar_head(h), min=-10.0, max=5.0)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        reward = self.reward_decoder(z).squeeze(-1)
        if self.config.reward_clip is not None:
            reward = torch.clamp(reward, -self.config.reward_clip, self.config.reward_clip)
        return reward, mu, logvar

    @staticmethod
    def kl_loss(mu, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()
