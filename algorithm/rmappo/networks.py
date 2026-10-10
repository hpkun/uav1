"""Flat configured observation encoder and independent per-agent GRU128."""
import torch
from torch import nn
from torch.distributions import Normal
from algorithm.mappo.networks import SharedMAPPOActor


class FlatRecurrentActor(nn.Module):
    def __init__(self, observation_dim=65, action_dim=3, flat_encoder_dim=128,
                 gru_hidden_dim=128, gru_layers=1, log_std_min=-5., log_std_max=.5,
                 policy_std_mode="state_independent", log_std_init=-.5, mean_head_init_gain=.01):
        super().__init__()
        if observation_dim not in (65,66,104) or (action_dim, flat_encoder_dim, gru_hidden_dim, gru_layers) != (3,128,128,1):
            raise ValueError("RMAPPO requires obs65 or obs104, action3 and encoder128/GRU128/layers1")
        if policy_std_mode != "state_independent" or log_std_min > log_std_max:
            raise ValueError("RMAPPO requires valid state-independent Gaussian")
        self.observation_dim = observation_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.flat_encoder = nn.Sequential(nn.Linear(observation_dim,128),nn.ReLU())
        self.gru = nn.GRU(128,128,num_layers=1,batch_first=True)
        self.actor_head = nn.Sequential(nn.Linear(128,128),nn.ReLU())
        self.mean = nn.Linear(128,action_dim)
        self._initialize_policy_std(128,policy_std_mode,log_std_init,mean_head_init_gain)

    _squashed_log_prob = staticmethod(SharedMAPPOActor._squashed_log_prob)
    _initialize_policy_std = SharedMAPPOActor._initialize_policy_std
    _policy_log_std = SharedMAPPOActor._policy_log_std

    def spatial(self, observations, alive):
        return self.flat_encoder(observations), {}

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
        distribution = Normal(self.mean(features), self._policy_log_std(features).exp())
        return distribution, next_hidden, attention

    def forward(self, observations: torch.Tensor, hidden: torch.Tensor,
                alive: torch.Tensor, episode_start: torch.Tensor, return_attention: bool = False):
        distribution, next_hidden, attention = self.distribution_step(observations, hidden, alive, episode_start)
        result = (torch.tanh(distribution.mean) * alive[..., None], next_hidden)
        return (*result, attention) if return_attention else result

    def distribution_sequence(self, observations: torch.Tensor, initial_hidden: torch.Tensor,
                              alive: torch.Tensor, episode_starts: torch.Tensor):
        if observations.ndim != 4 or observations.shape[-1] != self.observation_dim or episode_starts.shape != observations.shape[:2]:
            raise ValueError(f'sequences must be [batch,length,agents,{self.observation_dim}], starts [batch,length]')
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



def recurrent_diagnostics(auxiliary, hidden, alive):
    """Read-only hidden norm; there are no entity or attention modules."""
    return {'gru_hidden_norm_mean': (hidden.norm(dim=-1)*alive).sum()/alive.sum().clamp_min(1.)}
