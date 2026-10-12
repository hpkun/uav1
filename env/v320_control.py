"""Feasible A/B maneuver allocation; historical speed control is unchanged."""
import numpy as np
from .models import ControlCommand


def action_to_control_v320(state, action, action_config, aircraft_spec, gravity=9.81):
    a_heading, a_pitch, a_speed = np.clip(np.asarray(action, dtype=float), -1., 1.)
    controller = action_config['controller']
    nz_max = float(controller['normal_load_max'])
    a0 = float(np.cos(state.theta))
    b_max = float(np.sqrt(max(nz_max**2 - a0**2, 0.)))
    b = float(a_heading * b_max)
    a_max = float(np.sqrt(max(nz_max**2 - b**2, 0.)))
    a = float(a0 + a_pitch * (a_max - a0) if a_pitch >= 0 else a0 * (1. + a_pitch))
    nz = float(np.hypot(a, b))
    phi = float(np.arctan2(b, a)) if nz != 0. else 0.
    target_speed = float(np.clip(state.v + float(action_config['command']['speed_delta_max']) * a_speed,
                                 aircraft_spec.v_min, aircraft_spec.v_max))
    acceleration = (target_speed - state.v) / float(controller['speed_time_constant'])
    nx = float(np.sin(state.theta) + acceleration / gravity)
    return ControlCommand(nx=nx, nz=nz, phi=phi)
