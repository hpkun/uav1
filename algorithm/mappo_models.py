from __future__ import annotations

import torch
import numpy as np
from torch import nn
from torch.distributions import Normal

ACTION_PROBABILITY_VERSION = "mappo.raw_normal.v1"
TANH_ACTION = 'tanh'
DISK_ACTION = 'world_acceleration_disk'


def transform_raw_actions(raw, transform=TANH_ACTION, disk_radius=8.):
    """Map unconstrained policy outputs to a bounded acceleration disk."""
    if transform == TANH_ACTION:
        return torch.tanh(raw)
    if transform != DISK_ACTION or raw.shape[-1] != 2:
        raise ValueError('Unsupported MAPPO action transform/dimension')
    # Float64 prevents squaring finite float32 latents from overflowing; the
    # derivative at zero is well-defined (unlike a nested norm/hypot at zero).
    value = raw.to(torch.float64)
    return (disk_radius * value / torch.sqrt(1 + value.square().sum(-1, keepdim=True))).to(raw.dtype)


def transform_raw_numpy(raw, transform=TANH_ACTION, disk_radius=8.):
    if transform == TANH_ACTION:
        return np.tanh(raw)
    if transform != DISK_ACTION or raw.shape[-1] != 2:
        raise ValueError('Unsupported MAPPO action transform/dimension')
    value = np.asarray(raw, dtype=np.float64)
    return (disk_radius * value / np.sqrt(1 + np.square(value).sum(-1, keepdims=True))).astype(raw.dtype)


def _init_layer(layer: nn.Linear, gain: float = 1.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ContinuousGaussianActor(nn.Module):
    def __init__(self, observation_dim: int, hidden_size: int = 128, action_dim: int = 2, *, action_transform: str = TANH_ACTION, disk_radius: float = 8.) -> None:
        super().__init__()
        if action_transform not in (TANH_ACTION,DISK_ACTION) or action_transform==DISK_ACTION and action_dim!=2:
            raise ValueError('Unsupported action transform/dimension')
        self.action_transform=action_transform
        if not np.isfinite(disk_radius) or disk_radius<=0:raise ValueError('Positive finite disk radius required')
        self.disk_radius=float(disk_radius)
        self.network = nn.Sequential(
            _init_layer(nn.Linear(observation_dim, hidden_size), 2**0.5),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, hidden_size), 2**0.5),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, action_dim), 0.01),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def distribution(self, observations: torch.Tensor) -> Normal:
        mean = self.network(observations)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _log_probability(distribution, raw_actions, actions):
        return (distribution.log_prob(raw_actions)
                - torch.log(1.0 - actions.square() + 1e-6)).sum(dim=-1)

    def sample(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        raw = distribution.mean if deterministic else distribution.rsample()
        actions = transform_raw_actions(raw,self.action_transform,self.disk_radius)
        if self.action_transform==DISK_ACTION:
            # det J = radius^2/(1+||z||^2)^2 (historical radius=8). PPO uses the
            # saved latent Normal density, without evaluating an inverse map.
            logdet=2*np.log(self.disk_radius)-2*torch.log1p(raw.double().square().sum(-1))
            return actions, distribution.log_prob(raw).sum(-1)-logdet.to(raw.dtype)
        return actions, self._log_probability(distribution, raw, actions)

    def sample_rollout(
        self, observations: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return executed action, original latent and joint Normal log density.

        PPO stores the latent, including when float32 tanh rounds to +/-1.
        The fixed tanh Jacobian cancels in ratios; never invert stored actions.
        ``sample`` remains the historical inference interface.
        """
        distribution = self.distribution(observations)
        raw = distribution.mean if deterministic else distribution.rsample()
        return transform_raw_actions(raw,self.action_transform,self.disk_radius), raw, distribution.log_prob(raw).sum(dim=-1)

    def evaluate_raw_actions(
        self, observations: torch.Tensor, raw_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """PPO likelihood in latent space; retain the original Gaussian entropy."""
        distribution = self.distribution(observations)
        return distribution.log_prob(raw_actions).sum(dim=-1), distribution.entropy().sum(dim=-1)

    def evaluate_actions(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Legacy action-density query, not suitable for saturated PPO rollouts.

        Retained for existing callers; training must use evaluate_raw_actions.
        A squashed action alone cannot identify its float32 saturated latent.
        """
        if self.action_transform!=TANH_ACTION:
            raise ValueError('Disk probabilities require the original latent; use evaluate_raw_actions')
        bounded = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        distribution = self.distribution(observations)
        raw = torch.atanh(bounded)
        # Pre-squash Normal entropy is the PPO entropy approximation, not exact tanh entropy.
        return self._log_probability(distribution, raw, bounded), distribution.entropy().sum(dim=-1)


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _init_layer(nn.Linear(state_dim, hidden_size), 2**0.5),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, hidden_size), 2**0.5),
            nn.Tanh(),
            _init_layer(nn.Linear(hidden_size, 1), 1.0),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)
