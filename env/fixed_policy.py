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


class GroundAwareNearestTargetPursuitPolicy(NearestTargetPursuitPolicy):
    """Nearest-target pursuit with a stateless time-to-ground pitch guard."""

    def __init__(self, config: dict, action_config: dict,
                 aircraft_config: dict) -> None:
        super().__init__(config, action_config)
        guard = config.get("ground_avoidance", {})
        self.guard_time_constants = float(guard["guard_time_constants"])
        self.downward_speed_epsilon = float(
            guard.get("downward_speed_epsilon", 1e-6)
        )
        self.pitch_time_constant = float(
            action_config["controller"]["pitch_time_constant"]
        )
        self.theta_max = float(aircraft_config["theta_max"])
        if self.guard_time_constants <= 0.0:
            raise ValueError("guard_time_constants must be positive")
        if self.downward_speed_epsilon < 0.0:
            raise ValueError("downward_speed_epsilon must be non-negative")
        self.reset_diagnostics()

    def reset_diagnostics(self) -> None:
        self.total_decision_steps = 0
        self.override_steps = 0
        self.activation_count = 0
        self.maximum_activation_duration_steps = 0
        self._previous_override = np.zeros(4, dtype=bool)
        self._current_duration = np.zeros(4, dtype=np.int64)
        self.last_override_mask = np.zeros(4, dtype=bool)

    def reset_transient_state(self) -> None:
        self._previous_override.fill(False)
        self._current_duration.fill(0)
        self.last_override_mask.fill(False)

    def ground_risk(
        self, own: AircraftState, commanded_pitch: float,
        guard_time_constants: float | None = None,
    ) -> tuple[bool, float | None]:
        downward_current = max(-own.v * np.sin(own.theta), 0.0)
        downward_commanded = max(-own.v * np.sin(commanded_pitch), 0.0)
        downward_speed = max(downward_current, downward_commanded)
        if downward_speed <= self.downward_speed_epsilon:
            return False, None
        time_to_ground = own.altitude / downward_speed
        multiplier = (
            self.guard_time_constants
            if guard_time_constants is None else float(guard_time_constants)
        )
        return bool(time_to_ground <= multiplier * self.pitch_time_constant), float(
            time_to_ground
        )

    def _action_and_override(
        self, own: AircraftState, targets: list[AircraftState]
    ) -> tuple[np.ndarray, bool]:
        target_index = self.nearest_target_index(own, targets) if own.alive else None
        if target_index is None:
            return np.zeros(3, dtype=np.float32), False
        target = targets[target_index]
        dx, dy = target.x - own.x, target.y - own.y
        horizontal = float(np.hypot(dx, dy))
        desired_heading = float(np.arctan2(dy, dx))
        desired_pitch = float(np.arctan2(own.z - target.z, horizontal))
        baseline = self.action_toward(
            own, desired_heading, desired_pitch, float(self.config["desired_speed"])
        )
        override, _ = self.ground_risk(own, desired_pitch)
        if override:
            baseline = self.action_toward(
                own, desired_heading, self.theta_max,
                float(self.config["desired_speed"]),
            )
        return baseline, override

    def team_actions(
        self, team: list[AircraftState], targets: list[AircraftState]
    ) -> np.ndarray:
        rows = [self._action_and_override(state, targets) for state in team]
        actions = np.stack([row[0] for row in rows])
        mask = np.asarray([row[1] for row in rows], dtype=bool)
        alive = np.asarray([state.alive for state in team], dtype=bool)
        self.total_decision_steps += int(alive.sum())
        self.override_steps += int(mask.sum())
        self.activation_count += int(np.sum(mask & ~self._previous_override))
        self._current_duration = np.where(mask, self._current_duration + 1, 0)
        if self._current_duration.size:
            self.maximum_activation_duration_steps = max(
                self.maximum_activation_duration_steps,
                int(self._current_duration.max()),
            )
        self._previous_override = mask
        self.last_override_mask = mask
        return actions

    def diagnostics(self) -> dict[str, int | float]:
        return {
            "blue_ground_guard_decision_steps": self.total_decision_steps,
            "blue_ground_guard_override_steps": self.override_steps,
            "blue_ground_guard_activations": self.activation_count,
            "blue_ground_guard_activation_ratio": (
                self.override_steps / max(self.total_decision_steps, 1)
            ),
            "blue_ground_guard_max_duration_steps": (
                self.maximum_activation_duration_steps
            ),
        }


__all__ = [
    "GroundAwareNearestTargetPursuitPolicy", "NearestTargetPursuitPolicy",
]
