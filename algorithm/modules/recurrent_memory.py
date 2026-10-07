"""Configuration and hidden-state lifecycle for true recurrent MAPPO."""
from __future__ import annotations

from typing import Any
import numpy as np

from .base import CapabilityModule


class RecurrentMemoryModule(CapabilityModule):
    name = "recurrent_memory"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.mode = str(self.config.get("mode", "actor_critic_gru"))
        if self.mode not in {"actor_gru", "critic_gru", "actor_critic_gru", "wave_segmented_actor_gru", "wave_state_memory_actor_gru"}:
            raise ValueError(f"invalid recurrent mode: {self.mode}")
        self.hidden_dim = int(self.config.get("hidden_dim", 128))
        self.sequence_length = int(self.config.get("sequence_length", 32))
        if min(self.hidden_dim, self.sequence_length) <= 0:
            raise ValueError("recurrent hidden_dim and sequence_length must be positive")

    @property
    def actor_enabled(self) -> bool:
        return self.enabled and self.mode in {"actor_gru", "actor_critic_gru", "wave_segmented_actor_gru", "wave_state_memory_actor_gru"}

    @property
    def wave_segmented(self) -> bool:
        return self.enabled and self.mode == "wave_segmented_actor_gru"

    @property
    def state_memory(self) -> bool:
        return self.enabled and self.mode == "wave_state_memory_actor_gru"

    @property
    def wave_boundary_reset(self) -> bool:
        return self.wave_segmented or self.state_memory

    @property
    def critic_enabled(self) -> bool:
        return self.enabled and self.mode in {"critic_gru", "actor_critic_gru"}

    def zeros(self, num_envs: int, num_agents: int, actor: bool) -> np.ndarray | None:
        active = self.actor_enabled if actor else self.critic_enabled
        return np.zeros((num_envs, num_agents, self.hidden_dim), dtype=np.float32) if active else None

    @staticmethod
    def reset_for_episode(hidden: np.ndarray | None, done: np.ndarray) -> np.ndarray | None:
        if hidden is not None: hidden[np.asarray(done, dtype=bool)] = 0.0
        return hidden

    @staticmethod
    def apply_alive(hidden: np.ndarray | None, alive: np.ndarray) -> np.ndarray | None:
        return None if hidden is None else hidden * np.asarray(alive, dtype=np.float32)[..., None]


__all__ = ["RecurrentMemoryModule"]
