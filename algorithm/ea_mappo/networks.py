"""STEA spatial frontend, followed by a stateless actor head."""
import torch
from torch import nn
from torch.distributions import Normal
from algorithm.mappo.networks import SharedMAPPOActor
from algorithm.stea_mappo.networks import MaskedEntityAttention, SpatioTemporalEntityAttentionActor
from algorithm.common.entity_layout import entity_feature_dims


class EntityAttentionActor(nn.Module):
    def __init__(self, entity_dim=64, entity_attention_heads=2, spatial_hidden_dim=128,
                 action_dim=3, log_std_min=-5., log_std_max=.5,
                 policy_std_mode="state_independent", log_std_init=-.5, mean_head_init_gain=.01, self_feature_dim=7,
                 observation_layout='legacy'):
        super().__init__()
        if (entity_dim,entity_attention_heads,spatial_hidden_dim,action_dim) != (64,2,128,3):
            raise ValueError("EA-MAPPO requires entity64/heads2/spatial128/action3")
        if policy_std_mode != "state_independent" or log_std_min > log_std_max:
            raise ValueError("EA-MAPPO requires valid state-independent Gaussian")
        self.log_std_min,self.log_std_max = log_std_min,log_std_max
        if self_feature_dim not in (7,8): raise ValueError('self_feature_dim must be 7 or 8')
        self.self_feature_dim = self_feature_dim
        self.observation_layout = observation_layout
        _, ally_dim, enemy_dim = entity_feature_dims(observation_layout, self_feature_dim)
        self.self_encoder = nn.Sequential(nn.Linear(self_feature_dim,64),nn.ReLU())
        self.ally_encoder = nn.Sequential(nn.Linear(ally_dim,64),nn.ReLU())
        self.enemy_encoder = nn.Sequential(nn.Linear(enemy_dim,64),nn.ReLU())
        self.ally_attention = MaskedEntityAttention(64,2)
        self.enemy_attention = MaskedEntityAttention(64,2)
        self.spatial_fusion = nn.Sequential(nn.Linear(192,128),nn.ReLU())
        self.actor_head = nn.Sequential(nn.Linear(128,128),nn.ReLU())
        self.mean = nn.Linear(128,3)
        self._initialize_policy_std(128,policy_std_mode,log_std_init,mean_head_init_gain)

    # Reuse only the pure spatial and Gaussian methods, never recurrent methods.
    encode_entities = SpatioTemporalEntityAttentionActor.encode_entities
    spatial = SpatioTemporalEntityAttentionActor.spatial
    _squashed_log_prob = staticmethod(SharedMAPPOActor._squashed_log_prob)
    _initialize_policy_std = SharedMAPPOActor._initialize_policy_std
    _policy_log_std = SharedMAPPOActor._policy_log_std
    sample = SharedMAPPOActor.sample
    evaluate_actions = SharedMAPPOActor.evaluate_actions
    deterministic = SharedMAPPOActor.deterministic
    forward = SharedMAPPOActor.forward

    def distribution(self, observations):
        # Dead-agent observations are all zero; action/log-prob masking remains
        # external. The 56D layout uses whole-slot zeroing, with no alive feature.
        fused,_ = self.spatial(observations,torch.ones_like(observations[...,0]))
        features = self.actor_head(fused)
        return Normal(self.mean(features),self._policy_log_std(features).exp())


def spatial_diagnostics(attention, alive):
    result = {}
    for side in ('ally','enemy'):
        weights = attention[f'{side}_attention_weights']
        entropy = -(weights*weights.clamp_min(1e-12).log()).sum(-1).mean(-1)
        top1 = weights.max(-1).values.mean(-1)
        for name,value in (('entropy',entropy),('top1_mean',top1)):
            result[f'{side}_attention_{name}'] = (value*alive).sum()/alive.sum().clamp_min(1.)
    return result
