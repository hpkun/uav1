"""Direct normalized increments followed by one NED kinematic position step."""
import numpy as np
from .math_utils import wrap_angle


def advance_aircraft(state, action, dt, command, spec):
    result = state.copy()
    if not state.alive:
        return result
    action = np.asarray(action, dtype=float)
    if action.shape != (3,) or not np.isfinite(action).all():
        raise ValueError('action must be finite shape (3,)')
    a = np.clip(action, -1., 1.)
    result.psi = float(wrap_angle(state.psi + command['heading_delta_max'] * a[0]))
    result.theta = float(np.clip(state.theta + command['pitch_delta_max'] * a[1],
                                spec.theta_min, spec.theta_max))
    result.v = float(np.clip(state.v + command['speed_delta_max'] * a[2], spec.v_min, spec.v_max))
    result.x += result.v * np.cos(result.theta) * np.cos(result.psi) * dt
    result.y += result.v * np.cos(result.theta) * np.sin(result.psi) * dt
    result.z -= result.v * np.sin(result.theta) * dt
    return result
