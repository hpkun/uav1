"""Bounded, identity-initialized mission conditioning for the MAPPO actor."""
from __future__ import annotations

from typing import Any

from .base import CapabilityModule

MISSION_FILM_VERSION = 1


class MissionFiLMModule(CapabilityModule):
    name = "mission_film"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.mode = str(self.config.get("mode", "bounded_augmented_film"))
        self.encoder_hidden_dim = int(self.config.get("encoder_hidden_dim", 32))
        self.alpha = float(self.config.get("alpha", 0.2))
        self.augmented_residual = bool(self.config.get("augmented_residual", True))
        self.identity_init = bool(self.config.get("identity_init", True))
        if self.enabled:
            if self.mode != "bounded_augmented_film":
                raise ValueError("mission_film v1 only supports bounded_augmented_film")
            if self.encoder_hidden_dim <= 0:
                raise ValueError("mission_film encoder_hidden_dim must be positive")
            if not 0.0 < self.alpha <= 1.0:
                raise ValueError("mission_film alpha must be in (0, 1]")
            if not self.identity_init:
                raise ValueError("mission_film v1 requires identity_init=true")
            if not self.augmented_residual:
                raise ValueError("mission_film v1 requires augmented_residual=true")


__all__ = ["MISSION_FILM_VERSION", "MissionFiLMModule"]
