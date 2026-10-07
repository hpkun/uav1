"""Shared actor and centralized attention value critic for MAPPO."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class SharedMAPPOActor(nn.Module):
    """Two-layer shared tanh-Gaussian policy for homogeneous Red agents."""

    def __init__(
        self,
        observation_dim: int = 52,
        action_dim: int = 3,
        hidden_dim: int = 256,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        activation_cls = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}.get(activation)
        if activation_cls is None:
            raise ValueError(f"unsupported actor activation: {activation}")
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), activation_cls(),
            nn.Linear(hidden_dim, hidden_dim), activation_cls(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def distribution(self, observations: torch.Tensor) -> Normal:
        hidden = self.backbone(observations)
        mean = self.mean(hidden)
        std = self.log_std(hidden).clamp(
            self.log_std_min, self.log_std_max
        ).exp()
        return Normal(mean, std)

    @staticmethod
    def _squashed_log_prob(
        distribution: Normal, raw_actions: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        del actions
        # Exact, saturation-safe log|d tanh(x)/dx|.  Computing this from the
        # latent action avoids the precision loss of log(1-tanh(x)^2).
        log_jacobian = 2.0 * (
            math.log(2.0) - raw_actions - F.softplus(-2.0 * raw_actions)
        )
        return (distribution.log_prob(raw_actions) - log_jacobian).sum(dim=-1)

    def sample(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        raw_actions = distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_prob = self._squashed_log_prob(distribution, raw_actions, actions)
        # Monte-Carlo entropy of the bounded policy, including the tanh
        # Jacobian.  This is deliberately not Normal.entropy().
        entropy = -log_prob
        return actions, raw_actions, log_prob, entropy

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        raw_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bounded = actions
        if raw_actions is None:
            # Diagnostic compatibility only.  Formal PPO updates always pass
            # the latent action saved in the rollout.
            bounded = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            raw_actions = torch.atanh(bounded)
        distribution = self.distribution(observations)
        log_prob = self._squashed_log_prob(distribution, raw_actions, bounded)
        sampled_actions, sampled_raw, sampled_log_prob, entropy = self.sample(
            observations
        )
        del sampled_actions, sampled_raw, sampled_log_prob
        return log_prob, entropy

    def deterministic(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.distribution(observations).mean)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.deterministic(observations)


class CentralizedValueCritic(nn.Module):
    """Attention critic that produces one centralized state value per agent."""

    def __init__(
        self,
        observation_dim: int = 52,
        hidden_dim: int = 256,
        attention_heads: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        activation_cls = {"relu": nn.ReLU, "leaky_relu": nn.LeakyReLU}.get(activation)
        if activation_cls is None:
            raise ValueError(f"unsupported critic activation: {activation}")
        self.hidden_dim = int(hidden_dim)
        self.attention_heads = int(attention_heads)
        self.head_dim = hidden_dim // attention_heads
        self.embedding = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim), activation_cls(),
            nn.Linear(hidden_dim, hidden_dim), activation_cls(),
        )
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        self.value_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), activation_cls(),
            nn.Linear(hidden_dim, hidden_dim), activation_cls(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        observations: torch.Tensor,
        alive_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        if observations.ndim != 3:
            raise ValueError("critic observations must be [batch, agents, features]")
        embedding = self.embedding(observations)
        batch, agents, _ = embedding.shape

        def heads(values: torch.Tensor) -> torch.Tensor:
            return values.view(
                batch, agents, self.attention_heads, self.head_dim
            ).transpose(1, 2)

        query = heads(self.wq(embedding))
        key = heads(self.wk(embedding))
        value = heads(self.wv(embedding))
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if alive_mask is None:
            alive_mask = torch.ones(
                (batch, agents), dtype=logits.dtype, device=logits.device
            )
        alive = alive_mask > 0.5
        not_self = ~torch.eye(
            agents, dtype=torch.bool, device=logits.device
        ).view(1, 1, agents, agents)
        valid = alive[:, None, None, :].expand(
            batch, self.attention_heads, agents, agents
        ) & not_self
        weights = torch.softmax(logits.masked_fill(~valid, -1e9), dim=-1)
        weights = weights * valid.to(logits.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = weights * alive[:, None, :, None].to(logits.dtype)
        context = torch.matmul(weights, value).transpose(1, 2).contiguous().view(
            batch, agents, self.hidden_dim
        )
        values = self.value_network(torch.cat([embedding, context], dim=-1)).squeeze(-1)
        values = values * alive_mask
        return (values, weights) if return_attention else values


class CentralizedMLPCritic(nn.Module):
    """Shared V_i=V_phi(masked global state, masked focal observation)."""

    def __init__(self, observation_dim: int = 52, num_agents: int = 4,
                 hidden_dim: int = 256, activation: str = "relu") -> None:
        super().__init__()
        activation_cls={"relu":nn.ReLU,"leaky_relu":nn.LeakyReLU}.get(activation)
        if activation_cls is None:raise ValueError(f"unsupported critic activation: {activation}")
        self.observation_dim=int(observation_dim);self.num_agents=int(num_agents)
        input_dim=(self.num_agents+1)*self.observation_dim
        self.value_network=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),activation_cls(),
            nn.Linear(hidden_dim,hidden_dim),activation_cls(),nn.Linear(hidden_dim,1))

    def forward(self,observations:torch.Tensor,alive_mask:torch.Tensor|None=None)->torch.Tensor:
        if observations.ndim!=3:raise ValueError("critic observations must be [batch, agents, features]")
        batch,agents,features=observations.shape
        if (agents,features)!=(self.num_agents,self.observation_dim):raise ValueError("MLP critic observation shape mismatch")
        if alive_mask is None:alive_mask=torch.ones((batch,agents),dtype=observations.dtype,device=observations.device)
        if alive_mask.shape!=(batch,agents):raise ValueError("MLP critic alive mask shape mismatch")
        masked=observations*alive_mask.unsqueeze(-1);global_state=masked.reshape(batch,-1)
        global_per_agent=global_state.unsqueeze(1).expand(-1,agents,-1)
        critic_input=torch.cat((global_per_agent,masked),dim=-1)
        values=self.value_network(critic_input).squeeze(-1)
        return values*alive_mask


__all__ = ["CentralizedMLPCritic", "CentralizedValueCritic", "SharedMAPPOActor"]
