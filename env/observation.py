"""Exact 52-dimensional paper-constrained local observation contract."""
from __future__ import annotations

import numpy as np

from .math_utils import wrap_angle
from .models import AircraftState
from .geometry import engagement_geometry

BASE_OBSERVATION_DIM = 52
# Backward-compatible name for the frozen paper observation.
OBSERVATION_DIM = BASE_OBSERVATION_DIM
OWN_FIRE_READY_DIM = 1


def observation_dim_from_config(cfg: dict) -> int:
    """Resolve the opt-in schema without changing the legacy 52D default."""
    return BASE_OBSERVATION_DIM + (
        OWN_FIRE_READY_DIM if bool(cfg.get("include_own_fire_ready", False)) else 0
    )


def flight_path_frame(state: AircraftState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = state.velocity_vector()
    forward = forward / max(float(np.linalg.norm(forward)), 1e-12)
    right = np.array([-np.sin(state.psi), np.cos(state.psi), 0.0], dtype=float)
    up = np.cross(right, forward)
    up = up / max(float(np.linalg.norm(up)), 1e-12)
    return forward, right, up


def _relative_position(
    own: AircraftState,
    other: AircraftState,
    frame: tuple[np.ndarray, np.ndarray, np.ndarray],
    scale: float,
) -> np.ndarray:
    displacement = np.array(
        [other.x - own.x, other.y - own.y, other.z - own.z], dtype=float
    )
    return np.asarray([np.dot(displacement, axis) / scale for axis in frame])


def _ally_slot(
    own: AircraftState,
    ally: AircraftState,
    frame: tuple[np.ndarray, np.ndarray, np.ndarray],
    cfg: dict,
) -> np.ndarray:
    if not ally.alive:
        return np.zeros(7, dtype=np.float32)
    values = np.concatenate((
        _relative_position(
            own, ally, frame, float(cfg["relative_position_scale"])
        ),
        [
            (ally.v - float(cfg["speed_center"])) / float(cfg["speed_scale"]),
            wrap_angle(ally.psi - own.psi) / np.pi,
            ally.theta / (np.pi / 3.0),
            1.0,
        ],
    ))
    return values.astype(np.float32)


def _enemy_slot(own: AircraftState, enemy: AircraftState, cfg: dict) -> np.ndarray:
    if not enemy.alive:
        return np.zeros(6, dtype=np.float32)
    geometry = engagement_geometry(own, enemy)
    return np.asarray([
        geometry.distance / float(cfg["relative_position_scale"]),
        (enemy.v - float(cfg["speed_center"])) / float(cfg["speed_scale"]),
        geometry.aa / np.pi,
        geometry.ata / np.pi,
        geometry.ha / (np.pi / 2.0),
        1.0,
    ], dtype=np.float32)


def build_team_observations(
    team: list[AircraftState],
    opponents: list[AircraftState],
    cfg: dict,
    last_executed_phi: np.ndarray | None = None,
    own_fire_ready: np.ndarray | None = None,
) -> np.ndarray:
    include_ready = bool(cfg.get("include_own_fire_ready", False))
    observation_dim = observation_dim_from_config(cfg)
    phis = (
        np.zeros(len(team), dtype=float)
        if last_executed_phi is None else np.asarray(last_executed_phi, dtype=float)
    )
    if phis.shape != (len(team),):
        raise ValueError("last_executed_phi must match team size")
    ready = None
    if include_ready:
        if own_fire_ready is None:
            raise ValueError("FireReady observation requires own_fire_ready")
        ready = np.asarray(own_fire_ready, dtype=np.float32)
        if ready.shape != (len(team),) or not np.all(np.isin(ready, (0.0, 1.0))):
            raise ValueError("own_fire_ready must be a binary vector matching team size")
    observations = []
    for own_index, own in enumerate(team):
        if not own.alive:
            observations.append(np.zeros(observation_dim, dtype=np.float32))
            continue
        frame = flight_path_frame(own)
        self_features = np.asarray([
            own.x / float(cfg["horizontal_position_scale"]),
            own.y / float(cfg["horizontal_position_scale"]),
            own.altitude / float(cfg["altitude_scale"]),
            (own.v - float(cfg["speed_center"])) / float(cfg["speed_scale"]),
            phis[own_index] / (np.pi / 2.0),
            own.psi / np.pi,
            own.theta / (np.pi / 3.0),
        ], dtype=np.float32)
        allies = [state for index, state in enumerate(team) if index != own_index]
        ally_slots = [_ally_slot(own, ally, frame, cfg) for ally in allies]
        enemy_slots = [_enemy_slot(own, enemy, cfg) for enemy in opponents]
        values = np.concatenate([self_features, *ally_slots, *enemy_slots])
        if include_ready:
            values = np.concatenate((values, np.asarray([ready[own_index]], np.float32)))
        observations.append(values.astype(np.float32))
    result = np.stack(observations)
    if result.shape != (4, observation_dim) or not np.all(np.isfinite(result)):
        raise FloatingPointError("invalid combat observation")
    return result


__all__ = [
    "BASE_OBSERVATION_DIM", "OBSERVATION_DIM", "OWN_FIRE_READY_DIM",
    "observation_dim_from_config", "build_team_observations", "flight_path_frame",
]
