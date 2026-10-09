"""Plain local deterministic actors and individual centralized scalar Q critics."""
import torch
from torch import nn


class DeterministicActor(nn.Module):
    def __init__(self, observation_dim, action_dim=3):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(observation_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim), nn.Tanh())

    def forward(self, own_observation):
        return self.network(own_observation)


class CentralizedQCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, num_agents):
        super().__init__()
        self.joint_input_dim = num_agents*(observation_dim+action_dim)
        self.network = nn.Sequential(nn.Linear(self.joint_input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    @staticmethod
    def joint_input(observations, actions, alive):
        mask = alive.unsqueeze(-1).bool()
        # where also prevents non-finite data in a dead slot from influencing Q.
        obs = torch.where(mask, observations, torch.zeros_like(observations))
        act = torch.where(mask, actions, torch.zeros_like(actions))
        return torch.cat((obs.flatten(start_dim=-2), act.flatten(start_dim=-2)), dim=-1)

    def forward(self, observations, actions, alive):
        return self.network(self.joint_input(observations, actions, alive)).squeeze(-1)
