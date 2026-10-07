"""Self-query entity attention followed by explicit per-agent temporal memory."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.distributions import Normal
from algorithm.mappo.networks import SharedMAPPOActor


def decompose_observations(observations: torch.Tensor):
    if observations.shape[-1] != 52:
        raise ValueError('STEA observations must have exactly 52 features')
    prefix = observations.shape[:-1]
    return (observations[..., :7], observations[..., 7:28].reshape(*prefix, 3, 7),
            observations[..., 28:52].reshape(*prefix, 4, 6))


class MaskedEntityAttention(nn.Module):
    """Independent multi-head query/key/value projections for one entity type."""
    def __init__(self, entity_dim: int, heads: int):
        super().__init__()
        if heads <= 0 or entity_dim <= 0 or entity_dim % heads:
            raise ValueError('entity_dim must be positive and divisible by attention heads')
        self.entity_dim, self.heads = entity_dim, heads
        self.head_dim = entity_dim // heads
        self.wq = nn.Linear(entity_dim, entity_dim)
        self.wk = nn.Linear(entity_dim, entity_dim)
        self.wv = nn.Linear(entity_dim, entity_dim)

    def forward(self, query: torch.Tensor, entities: torch.Tensor, valid: torch.Tensor):
        prefix, slots = query.shape[:-1], entities.shape[-2]
        q = self.wq(query).reshape(*prefix, self.heads, self.head_dim).unsqueeze(-2)
        k = self.wk(entities).reshape(*prefix, slots, self.heads, self.head_dim).transpose(-3, -2)
        v = self.wv(entities).reshape(*prefix, slots, self.heads, self.head_dim).transpose(-3, -2)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = valid[..., None, None, :]
        # A finite minimum avoids softmax(all -inf); re-mask and normalize make
        # every invalid weight and every empty-set context exactly zero.
        weights = torch.softmax(logits.masked_fill(~mask, torch.finfo(logits.dtype).min), dim=-1)
        weights = weights * mask.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        context = (weights @ v).squeeze(-2).reshape(*prefix, self.entity_dim)
        return context, weights.squeeze(-2)


class SpatioTemporalEntityAttentionActor(nn.Module):
    def __init__(self, entity_dim: int = 64, entity_attention_heads: int = 2,
                 spatial_hidden_dim: int = 128, gru_hidden_dim: int = 128,
                 gru_layers: int = 1, action_dim: int = 3,
                 log_std_min: float = -5., log_std_max: float = 2.):
        super().__init__()
        if gru_layers != 1:
            raise ValueError('STEA v1 requires one unidirectional GRU layer')
        if min(entity_dim, spatial_hidden_dim, gru_hidden_dim) <= 0 or action_dim != 3:
            raise ValueError('invalid STEA actor dimensions')
        if log_std_min > log_std_max:
            raise ValueError('log_std_min must not exceed log_std_max')
        self.gru_hidden_dim = gru_hidden_dim
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.self_encoder = nn.Sequential(nn.Linear(7, entity_dim), nn.ReLU())
        self.ally_encoder = nn.Sequential(nn.Linear(7, entity_dim), nn.ReLU())
        self.enemy_encoder = nn.Sequential(nn.Linear(6, entity_dim), nn.ReLU())
        self.ally_attention = MaskedEntityAttention(entity_dim, entity_attention_heads)
        self.enemy_attention = MaskedEntityAttention(entity_dim, entity_attention_heads)
        self.spatial_fusion = nn.Sequential(nn.Linear(3*entity_dim, spatial_hidden_dim), nn.ReLU())
        self.gru = nn.GRU(spatial_hidden_dim, gru_hidden_dim, num_layers=1, batch_first=True)
        self.actor_head = nn.Sequential(nn.Linear(gru_hidden_dim, gru_hidden_dim), nn.ReLU())
        self.mean = nn.Linear(gru_hidden_dim, action_dim)
        self.log_std = nn.Linear(gru_hidden_dim, action_dim)

    _squashed_log_prob = staticmethod(SharedMAPPOActor._squashed_log_prob)

    def encode_entities(self, observations: torch.Tensor):
        own, allies, enemies = decompose_observations(observations)
        return self.self_encoder(own), self.ally_encoder(allies), self.enemy_encoder(enemies)

    def spatial(self, observations: torch.Tensor, alive: torch.Tensor):
        _, allies, enemies = decompose_observations(observations)
        own_embedding, ally_embeddings, enemy_embeddings = self.encode_entities(observations)
        ally_valid = (allies[..., -1] > .5) & (alive[..., None] > .5)
        enemy_valid = (enemies[..., -1] > .5) & (alive[..., None] > .5)
        ally_context, ally_weights = self.ally_attention(own_embedding, ally_embeddings, ally_valid)
        enemy_context, enemy_weights = self.enemy_attention(own_embedding, enemy_embeddings, enemy_valid)
        fused = self.spatial_fusion(torch.cat((own_embedding, ally_context, enemy_context), -1))
        return fused, {'ally_attention_weights': ally_weights, 'enemy_attention_weights': enemy_weights,
                       'ally_context': ally_context, 'enemy_context': enemy_context}

    def distribution_step(self, observations: torch.Tensor, hidden: torch.Tensor,
                          alive: torch.Tensor, episode_start: torch.Tensor):
        if hidden.shape != (*observations.shape[:-1], self.gru_hidden_dim) or alive.shape != observations.shape[:-1]:
            raise ValueError('actor hidden/alive shapes must match observation batch and agents')
        start = episode_start
        while start.ndim < alive.ndim:
            start = start.unsqueeze(-1)
        hidden = hidden * alive[..., None] * (1 - start.to(hidden.dtype))[..., None]
        spatial, attention = self.spatial(observations, alive)
        _, next_hidden = self.gru(spatial.reshape(-1, 1, spatial.shape[-1]), hidden.reshape(1, -1, self.gru_hidden_dim))
        next_hidden = next_hidden.reshape_as(hidden) * alive[..., None]
        features = self.actor_head(next_hidden)
        distribution = Normal(self.mean(features), self.log_std(features).clamp(self.log_std_min, self.log_std_max).exp())
        return distribution, next_hidden, attention

    def forward(self, observations: torch.Tensor, hidden: torch.Tensor,
                alive: torch.Tensor, episode_start: torch.Tensor, return_attention: bool = False):
        distribution, next_hidden, attention = self.distribution_step(observations, hidden, alive, episode_start)
        result = (torch.tanh(distribution.mean) * alive[..., None], next_hidden)
        return (*result, attention) if return_attention else result

    def distribution_sequence(self, observations: torch.Tensor, initial_hidden: torch.Tensor,
                              alive: torch.Tensor, episode_starts: torch.Tensor):
        if observations.ndim != 4 or episode_starts.shape != observations.shape[:2]:
            raise ValueError('sequences must be [batch,length,agents,52], starts [batch,length]')
        means, scales, hidden_rows, attention_rows = [], [], [], []
        hidden = initial_hidden
        for step in range(observations.shape[1]):
            distribution, hidden, attention = self.distribution_step(
                observations[:, step], hidden, alive[:, step], episode_starts[:, step])
            means.append(distribution.mean); scales.append(distribution.scale)
            hidden_rows.append(hidden); attention_rows.append(attention)
        attention = {key: torch.stack([row[key] for row in attention_rows], 1) for key in attention_rows[0]}
        return Normal(torch.stack(means, 1), torch.stack(scales, 1)), torch.stack(hidden_rows, 1), attention

    def evaluate_sequence(self, observations: torch.Tensor, actions: torch.Tensor,
                          raw_actions: torch.Tensor, initial_hidden: torch.Tensor,
                          alive: torch.Tensor, episode_starts: torch.Tensor,
                          compute_entropy: bool = True):
        distribution, hidden_rows, attention = self.distribution_sequence(
            observations, initial_hidden, alive, episode_starts)
        log_prob = self._squashed_log_prob(distribution, raw_actions, actions) * alive
        if compute_entropy:
            sampled_raw = distribution.rsample()
            entropy = -self._squashed_log_prob(distribution, sampled_raw, torch.tanh(sampled_raw)) * alive
        else:
            entropy = torch.zeros_like(log_prob)
        return log_prob, entropy, hidden_rows, attention, distribution.scale.log()


def attention_diagnostics(attention: dict, hidden: torch.Tensor, alive: torch.Tensor) -> dict[str, torch.Tensor]:
    denominator = alive.sum().clamp_min(1.)
    result = {'gru_hidden_norm_mean': (hidden.norm(dim=-1) * alive).sum() / denominator}
    for side in ('ally', 'enemy'):
        weights = attention[f'{side}_attention_weights']
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1).mean(-1)
        top1 = weights.max(-1).values.mean(-1)
        result[f'{side}_attention_entropy'] = (entropy * alive).sum() / denominator
        result[f'{side}_attention_top1_mean'] = (top1 * alive).sum() / denominator
    return result
