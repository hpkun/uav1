"""Actor-only PPO epoch guard with no learning-rate or optimizer side effects."""
from __future__ import annotations

from typing import Any

from .base import CapabilityModule


ACTOR_KL_GUARD_VERSION = 1


class ActorKLEpochGuardModule(CapabilityModule):
    """Stop remaining actor epochs after a completed epoch reaches hard KL."""

    name = "actor_kl_guard"
    version = ACTOR_KL_GUARD_VERSION

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.hard_kl = float(self.config.get("hard_kl", 0.05))
        self.actor_early_stop = bool(self.config.get("actor_early_stop", True))
        if self.hard_kl <= 0:
            raise ValueError("actor_kl_guard requires hard_kl > 0")
        if not self.actor_early_stop:
            raise ValueError("actor_kl_guard requires actor_early_stop=true")

    def should_stop_actor(self, epoch_kl: float) -> bool:
        return bool(self.enabled and self.actor_early_stop and float(epoch_kl) >= self.hard_kl)


__all__ = ["ACTOR_KL_GUARD_VERSION", "ActorKLEpochGuardModule"]
