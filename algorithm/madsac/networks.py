"""Shared local actor and centralized attention Q network for MADSAC."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def _activation(name: str):
    result = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}.get(name)
    if result is None:
        raise ValueError(f"unsupported activation: {name}")
    return result


class SharedMADSACActor(nn.Module):
    """Homogeneous local-observation tanh-Gaussian policy."""

    def __init__(self, observation_dim=52, action_dim=3, hidden_dim=256,
                 log_std_min=-5.0, log_std_max=2.0, activation="relu"):
        super().__init__()
        act = _activation(activation)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.backbone = nn.Sequential(
            nn.Linear(self.observation_dim, hidden_dim), act(),
            nn.Linear(hidden_dim, hidden_dim), act(),
        )
        self.mean = nn.Linear(hidden_dim, self.action_dim)
        self.log_std = nn.Linear(hidden_dim, self.action_dim)

    def distribution(self, observations: torch.Tensor) -> Normal:
        hidden = self.backbone(observations)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp())

    @staticmethod
    def squashed_log_prob(distribution: Normal, raw_actions: torch.Tensor) -> torch.Tensor:
        # Saturation-safe exact log|d tanh(z)/dz|, shared with validated MAPPO.
        log_jacobian = 2.0 * (
            math.log(2.0) - raw_actions - F.softplus(-2.0 * raw_actions)
        )
        return (distribution.log_prob(raw_actions) - log_jacobian).sum(-1)

    def sample(self, observations: torch.Tensor, generator: torch.Generator | None = None):
        distribution = self.distribution(observations)
        if generator is None:
            raw_actions = distribution.rsample()
        else:
            epsilon = torch.randn(
                distribution.mean.shape, dtype=distribution.mean.dtype,
                device=distribution.mean.device, generator=generator,
            )
            raw_actions = distribution.mean + distribution.stddev * epsilon
        actions = torch.tanh(raw_actions)
        return actions, raw_actions, self.squashed_log_prob(distribution, raw_actions)

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.distribution(observations).mean)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.deterministic(observations)


class CentralizedAttentionQCritic(nn.Module):
    """Per-agent centralized Q(o_1:4,a_1:4) with two-head peer attention."""

    def __init__(self, observation_dim=52, action_dim=3, hidden_dim=256,
                 attention_heads=2, activation="relu"):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        act = _activation(activation)
        self.hidden_dim = int(hidden_dim)
        self.attention_heads = int(attention_heads)
        self.head_dim = self.hidden_dim // self.attention_heads
        self.embedding = nn.Sequential(
            nn.Linear(int(observation_dim) + int(action_dim), hidden_dim), act(),
            nn.Linear(hidden_dim, hidden_dim), act(),
        )
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        self.q_network = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), act(),
            nn.Linear(hidden_dim, hidden_dim), act(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor,
                alive_mask: torch.Tensor, return_attention: bool = False):
        if observations.ndim != 3 or actions.ndim != 3:
            raise ValueError("critic inputs must be [batch, agents, features]")
        if observations.shape[:2] != actions.shape[:2] or alive_mask.shape != observations.shape[:2]:
            raise ValueError("critic observation/action/alive shapes disagree")
        embedding = self.embedding(torch.cat((observations, actions), dim=-1))
        batch, agents, _ = embedding.shape

        def heads(value):
            return value.view(batch, agents, self.attention_heads, self.head_dim).transpose(1, 2)

        query, key, value = heads(self.wq(embedding)), heads(self.wk(embedding)), heads(self.wv(embedding))
        logits = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        alive = alive_mask > 0.5
        not_self = ~torch.eye(agents, dtype=torch.bool, device=logits.device).view(1, 1, agents, agents)
        valid = alive[:, None, None, :].expand(batch, self.attention_heads, agents, agents) & not_self
        weights = torch.softmax(logits.masked_fill(~valid, -1e9), dim=-1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        weights = weights * alive[:, None, :, None].to(weights.dtype)
        context = (weights @ value).transpose(1, 2).contiguous().view(batch, agents, self.hidden_dim)
        q = self.q_network(torch.cat((embedding, context), dim=-1)).squeeze(-1)
        q = q * alive_mask
        return (q, weights) if return_attention else q


__all__ = ["CentralizedAttentionQCritic", "SharedMADSACActor"]

