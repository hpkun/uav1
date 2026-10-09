"""Uniform one-step joint transition replay with lazy CPU storage chunks.

Capacity counts environment transitions. Lazy allocation avoids committing the
entire million-transition buffer before any samples have been collected.
"""
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
    def __init__(self, capacity, num_agents, observation_dim, action_dim=3, seed=0, chunk_size=4096):
        if min(capacity, num_agents, observation_dim, action_dim, chunk_size) <= 0:
            raise ValueError('replay dimensions and capacity must be positive')
        self.capacity, self.chunk_size = int(capacity), min(int(chunk_size), int(capacity))
        self.shapes = ((num_agents, observation_dim), (num_agents, action_dim), (num_agents,),
                       (num_agents, observation_dim), (), (num_agents,), (num_agents,))
        self.chunks = {}
        self.position = self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    @property
    def storage_nbytes(self):
        return sum(value.nbytes for chunk in self.chunks.values() for value in chunk)

    def _chunk(self, index):
        if index not in self.chunks:
            length = min(self.chunk_size, self.capacity-index*self.chunk_size)
            self.chunks[index] = tuple(np.empty((length, *shape), np.bool_ if i == 4 else np.float32)
                for i, shape in enumerate(self.shapes))
        return self.chunks[index]

    def add_batch(self, observations, actions, rewards, next_observations, dones, alive_masks, next_alive_masks):
        arrays = tuple(np.asarray(value, np.bool_ if i == 4 else np.float32) for i, value in enumerate(
            (observations, actions, rewards, next_observations, dones, alive_masks, next_alive_masks)))
        batch = len(arrays[0])
        if any(value.shape != (batch, *shape) for value, shape in zip(arrays, self.shapes)):
            raise ValueError('invalid joint replay shapes')
        if not np.isfinite(arrays[1]).all() or (np.abs(arrays[1]) > 1).any():
            raise ValueError('replay must contain clipped executed actions')
        skip = max(0, batch-self.capacity)
        indices = (self.position+np.arange(skip, batch)) % self.capacity
        for chunk_id in np.unique(indices//self.chunk_size):
            selected = indices//self.chunk_size == chunk_id
            slots = indices[selected] % self.chunk_size
            for store, values in zip(self._chunk(int(chunk_id)), arrays):
                store[slots] = values[skip:][selected]
        self.position = (self.position+batch) % self.capacity
        self.size = min(self.capacity, self.size+batch)

    def get_batch(self, indices):
        indices = np.asarray(indices, np.int64)
        if indices.ndim != 1 or (indices < 0).any() or (indices >= self.size).any():
            raise ValueError('invalid replay indices')
        arrays = [np.empty((len(indices), *shape), np.bool_ if i == 4 else np.float32)
                  for i, shape in enumerate(self.shapes)]
        for chunk_id in np.unique(indices//self.chunk_size):
            selected = indices//self.chunk_size == chunk_id
            slots = indices[selected] % self.chunk_size
            for values, store in zip(arrays, self._chunk(int(chunk_id))):
                values[selected] = store[slots]
        return ReplayBatch(*arrays)

    def sample(self, batch_size):
        if batch_size <= 0 or self.size < batch_size:
            raise RuntimeError('replay does not contain enough transitions')
        return self.get_batch(self.rng.integers(0, self.size, size=batch_size))
