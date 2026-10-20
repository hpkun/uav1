"""Strict local-observation shared value function."""
import torch
from torch import nn
from algorithm.mappo.networks import SharedMAPPOActor

class LocalValueCritic(nn.Module):
    def __init__(self, observation_dim=45, hidden_dim=256, activation='relu'):
        super().__init__()
        if observation_dim != 45 or activation != 'relu':
            raise ValueError('IPPO local critic requires 45 inputs and ReLU')
        self.value_network = nn.Sequential(nn.Linear(45, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, observations, alive_mask=None):
        if observations.ndim < 2 or observations.shape[-1] != 45:
            raise ValueError('LocalValueCritic expects [...,45] local observations')
        values = self.value_network(observations).squeeze(-1)
        if alive_mask is not None:
            if alive_mask.shape != values.shape:
                raise ValueError('local critic alive mask shape mismatch')
            values = values * alive_mask
        return values
