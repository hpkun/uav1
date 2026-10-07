"""CPU joint-transition replay for four homogeneous agents."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    alive_masks: np.ndarray
    next_alive_masks: np.ndarray


class JointReplayBuffer:
    def __init__(self, capacity=1_000_000, num_agents=4, observation_dim=52,
                 action_dim=3, seed=0):
        self.capacity = int(capacity)
        self.num_agents = int(num_agents)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        if self.capacity <= 0:
            raise ValueError("replay capacity must be positive")
        shape = (self.capacity, self.num_agents)
        self.observations = np.empty((*shape, self.observation_dim), np.float32)
        self.actions = np.empty((*shape, self.action_dim), np.float32)
        self.rewards = np.empty(shape, np.float32)
        self.next_observations = np.empty((*shape, self.observation_dim), np.float32)
        self.dones = np.empty(self.capacity, np.bool_)
        self.alive_masks = np.empty(shape, np.float32)
        self.next_alive_masks = np.empty(shape, np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add_batch(self, observations, actions, rewards, next_observations,
                  dones, alive_masks, next_alive_masks):
        arrays = [
            np.asarray(observations, np.float32), np.asarray(actions, np.float32),
            np.asarray(rewards, np.float32), np.asarray(next_observations, np.float32),
            np.asarray(dones, np.bool_), np.asarray(alive_masks, np.float32),
            np.asarray(next_alive_masks, np.float32),
        ]
        batch = len(arrays[0])
        expected = [
            (batch, self.num_agents, self.observation_dim),
            (batch, self.num_agents, self.action_dim), (batch, self.num_agents),
            (batch, self.num_agents, self.observation_dim), (batch,),
            (batch, self.num_agents), (batch, self.num_agents),
        ]
        if any(value.shape != shape for value, shape in zip(arrays, expected)):
            raise ValueError(f"invalid joint replay shapes: {[value.shape for value in arrays]}")
        if batch > self.capacity:
            arrays = [value[-self.capacity:] for value in arrays]
            batch = self.capacity
        indices = (np.arange(batch) + self.position) % self.capacity
        stores = (self.observations, self.actions, self.rewards, self.next_observations,
                  self.dones, self.alive_masks, self.next_alive_masks)
        for store, value in zip(stores, arrays):
            store[indices] = value
        self.position = (self.position + batch) % self.capacity
        self.size = min(self.capacity, self.size + batch)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self.size < int(batch_size):
            raise RuntimeError("replay does not contain enough transitions")
        indices = self.rng.integers(0, self.size, size=int(batch_size))
        return ReplayBatch(*(
            value[indices].copy() for value in (
                self.observations, self.actions, self.rewards,
                self.next_observations, self.dones, self.alive_masks,
                self.next_alive_masks,
            )
        ))


__all__ = ["JointReplayBuffer", "ReplayBatch"]

