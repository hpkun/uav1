"""Paper action increments mapped to Eq. (2) point-mass controls."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .math_utils import wrap_angle
from .models import AircraftState, ControlCommand


@dataclass(frozen=True)
class ManeuverTarget:
    heading: float
    pitch: float
    speed: float


def action_to_target(
    state: AircraftState, action: np.ndarray, config: dict
) -> ManeuverTarget:
    """Decode normalized ``[a_psi, a_theta, a_v]`` as relative increments."""
    a_psi, a_theta, a_v = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
    return ManeuverTarget(
        heading=float(wrap_angle(
            state.psi + float(config["heading_delta_max"]) * a_psi
        )),
        pitch=float(state.theta + float(config["pitch_delta_max"]) * a_theta),
        speed=float(state.v + float(config["speed_delta_max"]) * a_v),
    )


def target_to_control(
    state: AircraftState,
    target: ManeuverTarget,
    controller: dict,
    gravity: float = 9.81,
) -> ControlCommand:
    """Invert Eq. (2), with A/B feasibility projection and an 8-g nz cap."""
    psi_rate = wrap_angle(target.heading - state.psi) / float(
        controller["heading_time_constant"]
    )
    theta_rate = (target.pitch - state.theta) / float(
        controller["pitch_time_constant"]
    )
    acceleration = (target.speed - state.v) / float(
        controller["speed_time_constant"]
    )
    a_vertical = max(
        float(np.cos(state.theta) + state.v * theta_rate / gravity), 0.0
    )
    b_lateral = float(state.v * np.cos(state.theta) * psi_rate / gravity)
    raw_nz = float(np.hypot(a_vertical, b_lateral))
    nz_max = float(controller["normal_load_max"])
    if raw_nz > nz_max:
        scale = nz_max / raw_nz
        a_vertical *= scale
        b_lateral *= scale
    nz = float(np.hypot(a_vertical, b_lateral))
    phi = float(np.arctan2(b_lateral, a_vertical)) if nz > 0.0 else 0.0
    nx = float(np.sin(state.theta) + acceleration / gravity)
    return ControlCommand(nx=nx, nz=nz, phi=phi)


def action_to_control(
    state: AircraftState, action: np.ndarray, config: dict, gravity: float = 9.81
) -> ControlCommand:
    return target_to_control(
        state,
        action_to_target(state, action, config["command"]),
        config["controller"],
        gravity,
    )


__all__ = [
    "ManeuverTarget", "action_to_control", "action_to_target", "target_to_control"
]
