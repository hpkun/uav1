"""Optional PPO actor KL guard and deterministic learning-rate schedule."""
from __future__ import annotations

from typing import Any

from .base import CapabilityModule


PPO_STABILIZATION_VERSION = 1


class PPOStabilizationModule(CapabilityModule):
    name = "ppo_stabilization"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.target_kl = float(self.config.get("target_kl", 0.015))
        self.hard_kl = float(self.config.get("hard_kl", 0.030))
        self.actor_early_stop = bool(self.config.get("actor_early_stop", True))
        self.actor_lr_schedule = str(self.config.get("actor_lr_schedule", "linear"))
        self.actor_lr_start = float(self.config.get("actor_lr_start", 3e-4))
        self.actor_lr_end = float(self.config.get("actor_lr_end", 1e-4))
        self.critic_lr_schedule = str(self.config.get("critic_lr_schedule", "none"))
        if self.target_kl <= 0 or self.hard_kl <= self.target_kl:
            raise ValueError("KL thresholds require 0 < target_kl < hard_kl")
        if self.actor_lr_schedule != "linear" or self.critic_lr_schedule != "none":
            raise ValueError("first-version stabilization requires linear actor LR and constant critic LR")
        if self.actor_lr_start <= 0 or self.actor_lr_end <= 0:
            raise ValueError("learning rates must be positive")

    def actor_learning_rate(self, sampled_steps: int, total_sampled_steps: int) -> float:
        if total_sampled_steps <= 0:
            raise ValueError("total_sampled_steps must be positive")
        progress = min(1.0, max(0.0, float(sampled_steps) / float(total_sampled_steps)))
        return self.actor_lr_end + (self.actor_lr_start - self.actor_lr_end) * (1.0 - progress)

    def should_stop_actor(self, epoch_kl: float) -> bool:
        return bool(self.enabled and self.actor_early_stop and epoch_kl > self.hard_kl)


__all__ = ["PPO_STABILIZATION_VERSION", "PPOStabilizationModule"]
