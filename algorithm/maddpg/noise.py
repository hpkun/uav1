"""Seeded independent OU states for each environment/agent/action coordinate."""
from copy import deepcopy
import numpy as np


class OUNoise:
    def __init__(self, num_envs, num_agents, action_dim, seed, theta=.15, sigma=.20, mu=0.):
        self.theta, self.sigma, self.mu = float(theta), float(sigma), float(mu)
        if self.theta < 0 or self.sigma < 0:
            raise ValueError('OU theta and sigma must be nonnegative')
        self.state = np.zeros((num_envs, num_agents, action_dim), np.float32)
        streams = np.random.SeedSequence(seed).spawn(num_envs*num_agents)
        self.rngs = [[np.random.default_rng(streams[e*num_agents+i]) for i in range(num_agents)]
                     for e in range(num_envs)]

    def sample(self):
        for e, rngs in enumerate(self.rngs):
            for i, rng in enumerate(rngs):
                self.state[e, i] += self.theta*(self.mu-self.state[e, i])+self.sigma*rng.normal(size=self.state.shape[-1])
        return self.state.copy()

    def reset(self, environments):
        self.state[np.asarray(environments, dtype=np.int64)] = 0.

    def state_dict(self):
        return {'state': self.state.copy(), 'rng_states': [[deepcopy(rng.bit_generator.state) for rng in row] for row in self.rngs],
                'theta': self.theta, 'sigma': self.sigma, 'mu': self.mu}

    def load_state_dict(self, state):
        if (state['state'].shape != self.state.shape or
            (state['theta'], state['sigma'], state['mu']) != (self.theta, self.sigma, self.mu)):
            raise ValueError('OU state shape/config mismatch')
        self.state[:] = state['state']
        for row, saved in zip(self.rngs, state['rng_states']):
            for rng, value in zip(row, saved):
                rng.bit_generator.state = deepcopy(value)


def exploration_scale(step, start=1., end=.1, decay_steps=3_000_000):
    if decay_steps <= 0:
        raise ValueError('exploration decay must be positive')
    fraction = np.clip(float(step)/decay_steps, 0., 1.)
    return float(start+(end-start)*fraction)
