"""Typed entity tokens and spatial self-query attention; baseline Gaussian math."""
import math
import torch
from torch import nn
from torch.distributions import Normal
from algorithm.mappo.networks import SharedMAPPOActor

OBSERVATION_LAYOUT = dict(self_dim=7, ally_count=3, ally_dim=6, enemy_count=4, enemy_dim=5)


def split_v338_entities(observation):
    if observation.shape[-1] != 45:
        raise ValueError('v3.38 entity parser requires exactly 45 features')
    own, allies, enemies = observation.split((7, 18, 20), dim=-1)
    return own, allies.reshape(*observation.shape[:-1], 3, 6), enemies.reshape(*observation.shape[:-1], 4, 5)


class EntityAttentionMAPPOActor(SharedMAPPOActor):
    """Shared decentralized policy; all entities attend, policy uses only self row."""
    def __init__(self, log_std_min=-5., log_std_max=.5, log_std_init=-.5,
                 mean_head_init_gain=.01, mask_epsilon=1e-8):
        # Reuse initialization and every sampling/log-prob/entropy method unchanged.
        super().__init__(128, 3, 256, log_std_min, log_std_max, 'relu',
                         'state_independent', log_std_init, mean_head_init_gain)
        self.mask_epsilon = mask_epsilon
        self.entity_embedding_dim, self.attention_heads, self.head_dim = 64, 4, 16
        self.self_encoder = nn.Sequential(nn.Linear(7, 64), nn.ReLU())
        self.ally_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU())
        self.enemy_encoder = nn.Sequential(nn.Linear(5, 64), nn.ReLU())
        self.wq, self.wk, self.wv = (nn.Linear(64, 64) for _ in range(3))

    def encode_entities(self, observations):
        own, allies, enemies = split_v338_entities(observations)
        valid = torch.cat((torch.ones_like(own[..., :1], dtype=torch.bool),
            (allies.abs() > self.mask_epsilon).any(-1),
            (enemies.abs() > self.mask_epsilon).any(-1)), dim=-1)
        tokens = torch.cat((self.self_encoder(own).unsqueeze(-2),
            self.ally_encoder(allies), self.enemy_encoder(enemies)), dim=-2)
        return tokens, valid

    def entity_attention(self, tokens, valid):
        leading = tokens.shape[:-2]
        def heads(x):
            return x.reshape(*leading, 8, 4, 16).transpose(-3, -2)
        query, key, value = heads(self.wq(tokens)), heads(self.wk(tokens)), heads(self.wv(tokens))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        # Self is always valid, even for dead-agent rows; trainer masks their actions.
        weights = torch.softmax(scores.masked_fill(~valid[..., None, None, :], -torch.inf), -1)
        context = (weights @ value).transpose(-3, -2).reshape(*leading, 8, 64)
        return context, weights

    def representation(self, observations, return_attention=False):
        tokens, valid = self.encode_entities(observations)
        context, weights = self.entity_attention(tokens, valid)
        fused = torch.cat((tokens[..., 0, :], context[..., 0, :]), -1)
        return (fused, weights[..., 0, :]) if return_attention else fused

    def distribution(self, observations, return_attention=False):
        fused, attention = self.representation(observations, True)
        features = self.backbone(fused)
        distribution = Normal(self.mean(features), self._policy_log_std(features).exp())
        return (distribution, attention) if return_attention else distribution

    def forward(self, observations, return_attention=False):
        distribution, weights = self.distribution(observations, True)
        actions = torch.tanh(distribution.mean)
        return (actions, weights) if return_attention else actions
