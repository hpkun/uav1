"""Nearest-target Blue pursuit in the paper increment action space."""
from __future__ import annotations

import numpy as np

from .math_utils import wrap_angle
from .models import AircraftState


class NearestTargetPursuitPolicy:
    def __init__(self, config: dict, action_config: dict) -> None:
        self.config = config
        self.command_config = action_config["command"]

    @staticmethod
    def nearest_target_index(
        own: AircraftState, targets: list[AircraftState]
    ) -> int | None:
        candidates = [
            (
                (target.x - own.x) ** 2
                + (target.y - own.y) ** 2
                + (target.z - own.z) ** 2,
                index,
            )
            for index, target in enumerate(targets) if target.alive
        ]
        return min(candidates)[1] if candidates else None

    def action_toward(
        self,
        own: AircraftState,
        desired_heading: float,
        desired_elevation: float,
        desired_speed: float,
    ) -> np.ndarray:
        cfg = self.command_config
        return np.clip(np.asarray([
            wrap_angle(desired_heading - own.psi) / float(cfg["heading_delta_max"]),
            (desired_elevation - own.theta) / float(cfg["pitch_delta_max"]),
            (desired_speed - own.v) / float(cfg["speed_delta_max"]),
        ], dtype=np.float32), -1.0, 1.0)

    def action(self, own: AircraftState, targets: list[AircraftState]) -> np.ndarray:
        target_index = self.nearest_target_index(own, targets) if own.alive else None
        if target_index is None:
            return np.zeros(3, dtype=np.float32)
        target = targets[target_index]
        dx, dy = target.x - own.x, target.y - own.y
        horizontal = float(np.hypot(dx, dy))
        return self.action_toward(
            own,
            float(np.arctan2(dy, dx)),
            float(np.arctan2(own.z - target.z, horizontal)),
            float(self.config["desired_speed"]),
        )

    def team_actions(
        self, team: list[AircraftState], targets: list[AircraftState]
    ) -> np.ndarray:
        return np.stack([self.action(state, targets) for state in team])




__all__ = ["NearestTargetPursuitPolicy"]
