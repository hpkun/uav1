"""Algorithm-side wave context; environment observations remain untouched."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from .base import CapabilityModule


class WaveContextModule(CapabilityModule):
    name = "wave_context"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.target = str(self.config.get("context_target", "critic_only"))
        if self.target not in {"actor_only", "critic_only", "actor_critic"}:
            raise ValueError(f"invalid wave context target: {self.target}")
        self.max_waves = int(self.config.get("max_waves", 3))
        if self.max_waves < 1: raise ValueError("max_waves must be positive")
        self.encoding = str(self.config.get("encoding", "rich"))
        if self.encoding not in {"rich", "scalar_round", "mission_markov"}:
            raise ValueError(f"invalid wave context encoding: {self.encoding}")

    @property
    def context_dim(self) -> int:
        if not self.enabled:
            return 0
        if self.encoding == "scalar_round":
            return 1
        return self.max_waves + 2

    @property
    def actor_enabled(self) -> bool:
        return self.enabled and self.target in {"actor_only", "actor_critic"}

    @property
    def critic_enabled(self) -> bool:
        return self.enabled and self.target in {"critic_only", "actor_critic"}

    def encode_numpy(
        self, wave_index, total_waves, *, mission_progress=None,
        episode_step=None, max_steps=None,
    ) -> np.ndarray:
        wave = np.asarray(wave_index, dtype=np.int64)
        total = np.asarray(total_waves, dtype=np.int64)
        wave, total = np.broadcast_arrays(wave, total)
        if self.encoding == "scalar_round":
            return wave.astype(np.float32)[..., None]
        clipped = np.clip(wave, 1, self.max_waves)
        one_hot = np.eye(self.max_waves, dtype=np.float32)[clipped - 1]
        if self.encoding == "mission_markov":
            if mission_progress is None or episode_step is None or max_steps is None:
                raise ValueError("mission_markov requires mission_progress, episode_step, and max_steps")
            progress = np.broadcast_to(np.asarray(mission_progress, dtype=np.float32), wave.shape)
            step = np.broadcast_to(np.asarray(episode_step, dtype=np.float32), wave.shape)
            horizon = np.broadcast_to(np.asarray(max_steps, dtype=np.float32), wave.shape)
            remaining = np.clip((horizon - step) / np.maximum(horizon, 1.0), 0.0, 1.0)
            return np.concatenate([
                one_hot, np.clip(progress, 0.0, 1.0)[..., None],
                remaining.astype(np.float32)[..., None],
            ], axis=-1)
        denom = np.maximum(total - 1, 1)
        progress = np.where(total > 1, (wave - 1) / denom, 0.0).astype(np.float32)
        remaining = np.where(total > 1, (total - wave) / denom, 0.0).astype(np.float32)
        return np.concatenate([one_hot, progress[..., None], remaining[..., None]], axis=-1)

    def encode_tensor(
        self, wave_index: torch.Tensor, total_waves: torch.Tensor, *,
        mission_progress: torch.Tensor | None = None,
        episode_step: torch.Tensor | None = None,
        max_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        wave = wave_index.long(); total = total_waves.long()
        if self.encoding == "scalar_round":
            return wave.to(dtype=torch.float32).unsqueeze(-1)
        one_hot = torch.nn.functional.one_hot(
            wave.clamp(1, self.max_waves) - 1, self.max_waves
        ).to(dtype=torch.float32)
        if self.encoding == "mission_markov":
            if mission_progress is None or episode_step is None or max_steps is None:
                raise ValueError("mission_markov requires mission_progress, episode_step, and max_steps")
            progress = torch.broadcast_to(mission_progress.float(), wave.shape).clamp(0.0, 1.0)
            step = torch.broadcast_to(episode_step.float(), wave.shape)
            horizon = torch.broadcast_to(max_steps.float(), wave.shape)
            remaining = ((horizon - step) / horizon.clamp_min(1.0)).clamp(0.0, 1.0)
            return torch.cat([one_hot, progress.unsqueeze(-1), remaining.unsqueeze(-1)], dim=-1)
        denom = (total - 1).clamp_min(1)
        progress = torch.where(total > 1, (wave - 1).float() / denom, torch.zeros_like(wave, dtype=torch.float32))
        remaining = torch.where(total > 1, (total - wave).float() / denom, torch.zeros_like(wave, dtype=torch.float32))
        return torch.cat([one_hot, progress.unsqueeze(-1), remaining.unsqueeze(-1)], dim=-1)


__all__ = ["WaveContextModule"]
