"""Local shared actor and context-only self-inclusive attention Q critic."""
import torch
from torch import nn


class SharedLocalActor(nn.Module):
    def __init__(self, observation_dim=56, action_dim=3, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim), nn.Tanh())

    def forward(self, observations):
        return self.network(observations)


class CentralizedAttentionCritic(nn.Module):
    def __init__(self, observation_dim=56, action_dim=3, num_agents=5, hidden_dim=256, attention_heads=2):
        super().__init__()
        if hidden_dim % attention_heads:
            raise ValueError('hidden_dim must be divisible by attention_heads')
        self.observation_dim, self.action_dim, self.num_agents = observation_dim, action_dim, num_agents
        self.hidden_dim, self.heads = hidden_dim, attention_heads
        self.encoder = nn.Sequential(nn.Linear(observation_dim+action_dim, hidden_dim), nn.LeakyReLU(.01))
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(.01))
        self.q_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(.01), nn.Linear(hidden_dim, 1))

    def forward(self, observations, actions, alive_masks, return_attention=False):
        if (observations.ndim != 3 or observations.shape[1:] != (self.num_agents, self.observation_dim)
                or actions.shape != (*observations.shape[:2], self.action_dim)
                or alive_masks.shape != observations.shape[:2]):
            raise ValueError('critic requires obs [B,5,56], actions [B,5,3], alive [B,5]')
        mask = alive_masks.bool()
        # Mask before encoding to isolate dead agents even with extreme inputs.
        z = torch.cat((observations, actions), dim=-1).masked_fill(~mask[..., None], 0.)
        embedding = self.encoder(z)
        def heads(x):
            return x.reshape(x.shape[0], self.num_agents, self.heads, -1).transpose(1, 2)
        q, k, v = heads(self.query(embedding)), heads(self.key(embedding)), heads(self.value(embedding))
        # Unscaled dot product follows the user-specified variant contract.
        scores = q @ k.transpose(-1, -2)
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = scores.softmax(dim=-1) * mask[:, None, None, :]
        attention = attention * mask[:, None, :, None]
        context = (attention @ v).transpose(1, 2).reshape(observations.shape[0], self.num_agents, self.hidden_dim)
        # No self shortcut: Q reads context only. All-dead rows remain finite.
        values = self.q_head(context) * mask[..., None]
        return (values, attention) if return_attention else values
