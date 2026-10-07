"""Optional actor-only delayed linear learning-rate decay."""
from __future__ import annotations

from typing import Any

from .base import CapabilityModule


ACTOR_LR_DECAY_VERSION = 1


class ActorLRDecayModule(CapabilityModule):
    """A stateless schedule determined only by global sampled steps."""

    name = "actor_lr_decay"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.schedule = str(self.config.get("schedule", "delayed_linear"))
        self.start_step = int(self.config.get("start_step", 600_000))
        self.end_step = int(self.config.get("end_step", 900_000))
        self.start_lr = float(self.config.get("start_lr", 3e-4))
        self.end_lr = float(self.config.get("end_lr", 1e-4))
        if self.schedule != "delayed_linear":
            raise ValueError("actor_lr_decay requires schedule=delayed_linear")
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("actor_lr_decay requires 0 <= start_step < end_step")
        if self.start_lr <= 0 or self.end_lr <= 0:
            raise ValueError("actor_lr_decay learning rates must be positive")

    def learning_rate(self, sampled_steps: int, disabled_lr: float) -> float:
        if not self.enabled:
            return float(disabled_lr)
        step = int(sampled_steps)
        if step <= self.start_step:
            return self.start_lr
        if step >= self.end_step:
            return self.end_lr
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        return self.start_lr + progress * (self.end_lr - self.start_lr)

    def apply(self, optimizer: Any, sampled_steps: int, disabled_lr: float) -> float:
        lr = self.learning_rate(sampled_steps, disabled_lr)
        if self.enabled:
            for group in optimizer.param_groups:
                group["lr"] = lr
        return lr


__all__ = ["ACTOR_LR_DECAY_VERSION", "ActorLRDecayModule"]
