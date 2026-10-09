"""Nearest-target Blue pursuit in the paper increment action space."""
from __future__ import annotations

import numpy as np
from enum import Enum
from dataclasses import dataclass

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




class SternPhase(Enum):
    PURE_PURSUIT = "PURE_PURSUIT"
    RELATIVE_BEARING = "RELATIVE_BEARING"
    OFFSET_RECIPROCAL = "OFFSET_RECIPROCAL"
    CONVERT = "CONVERT"


@dataclass
class SternState:
    locked_target_index: int | None = None
    phase: SternPhase = SternPhase.PURE_PURSUIT
    relative_bearing_heading: float | None = None
    turn_side: int | None = None


class SternConversionPolicy(NearestTargetPursuitPolicy):
    """Independent deterministic target locks and four heading phases.

    State transitions are evaluated once per decision; no weapon logic lives here.
    A reacquisition decision uses pure pursuit before any new conversion begins.
    """
    def __init__(self, config: dict, action_config: dict, team_size: int) -> None:
        super().__init__(config, action_config)
        self.team_size = team_size
        self.reset()

    def reset(self) -> None:
        self.states = [SternState() for _ in range(self.team_size)]

    @staticmethod
    def target_coordinates(own: AircraftState, target: AircraftState) -> tuple[float, float]:
        dx, dy = own.x - target.x, own.y - target.y
        c, s = np.cos(target.psi), np.sin(target.psi)
        return float(dx*c + dy*s), float(-dx*s + dy*c)

    def action(self, own: AircraftState, targets: list[AircraftState], aircraft_index: int = 0) -> np.ndarray:
        state = self.states[aircraft_index]
        if not own.alive:
            return np.zeros(3, dtype=np.float32)
        j = state.locked_target_index
        reacquired = j is None or j >= len(targets) or not targets[j].alive
        if reacquired:
            state = SternState(locked_target_index=self.nearest_target_index(own, targets))
            self.states[aircraft_index] = state
        if state.locked_target_index is None:
            return np.zeros(3, dtype=np.float32)
        target = targets[state.locked_target_index]
        dx, dy, dz = target.x-own.x, target.y-own.y, target.z-own.z
        distance = float(np.sqrt(dx*dx+dy*dy+dz*dz))
        longitudinal, lateral = self.target_coordinates(own, target)
        cfg = self.config
        if not reacquired:
            if state.phase is SternPhase.PURE_PURSUIT and distance <= cfg["turn_range"]:
                parity_side = 1 if aircraft_index % 2 == 0 else -1
                candidate_plus = float(wrap_angle(own.psi + cfg["turn_angle"]))
                candidate_minus = float(wrap_angle(own.psi - cfg["turn_angle"]))
                state.turn_side = parity_side
                if abs(lateral) > 1e-6:
                    # Project each existing heading candidate onto the target's
                    # left axis, choosing the larger outward lateral motion.
                    score_plus = np.sign(lateral) * np.sin(candidate_plus - target.psi)
                    score_minus = np.sign(lateral) * np.sin(candidate_minus - target.psi)
                    if score_plus > score_minus:
                        state.turn_side = 1
                    elif score_minus > score_plus:
                        state.turn_side = -1
                state.relative_bearing_heading = candidate_plus if state.turn_side == 1 else candidate_minus
                state.phase = SternPhase.RELATIVE_BEARING
            elif state.phase is SternPhase.RELATIVE_BEARING and abs(lateral) >= cfg["required_lateral_displacement"]:
                state.phase = SternPhase.OFFSET_RECIPROCAL
            elif state.phase is SternPhase.OFFSET_RECIPROCAL and longitudinal <= 0 and distance <= cfg["conversion_range"]:
                state.phase = SternPhase.CONVERT
        heading = float(np.arctan2(dy, dx))
        if state.phase is SternPhase.RELATIVE_BEARING:
            heading = state.relative_bearing_heading
        elif state.phase is SternPhase.OFFSET_RECIPROCAL:
            heading = float(wrap_angle(target.psi + np.pi))
        return self.action_toward(own, heading, float(np.arctan2(own.z-target.z, np.hypot(dx,dy))), float(cfg["desired_speed"]))

    def team_actions(self, team: list[AircraftState], targets: list[AircraftState]) -> np.ndarray:
        return np.stack([self.action(state, targets, index) for index, state in enumerate(team)])


__all__ = ["NearestTargetPursuitPolicy", "SternConversionPolicy", "SternPhase", "SternState"]
