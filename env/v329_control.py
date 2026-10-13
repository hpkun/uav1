"""Paper increment actions; physically feasible controller reconstruction.

The 2 s tracking constants, target clipping, nonnegative vertical load and
radial load projection are project calibration, not a published controller.
"""
import numpy as np
from .math_utils import wrap_angle
from .models import ControlCommand


def target_state_v329(state, action, action_config, aircraft_spec):
    a = np.clip(np.asarray(action, dtype=float), -1., 1.)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('v3.29 action must be finite shape (3,)')
    cfg = action_config['command']
    return (
        wrap_angle(state.psi + cfg['heading_delta_max'] * a[0]),
        float(np.clip(state.theta + cfg['pitch_delta_max'] * a[1],
                      aircraft_spec.theta_min, aircraft_spec.theta_max)),
        float(np.clip(state.v + cfg['speed_delta_max'] * a[2],
                      aircraft_spec.v_min, aircraft_spec.v_max)),
    )


def action_to_control_v329(state, action, action_config, aircraft_spec, gravity=9.81):
    psi_d, theta_d, v_d = target_state_v329(state, action, action_config, aircraft_spec)
    cfg = action_config['controller']
    psi_error = wrap_angle(psi_d - state.psi)
    # Exactly 180 degrees has two equally short paths. Preserve the requested
    # increment's sign instead of making +1 reverse at wrap's half-open seam.
    if abs(abs(psi_error) - np.pi) <= 1e-12:
        psi_error = float(np.copysign(np.pi, action[0]))
    psi_rate = psi_error / cfg['heading_time_constant']
    theta_rate = (theta_d - state.theta) / cfg['pitch_time_constant']
    acceleration = (v_d - state.v) / cfg['speed_time_constant']
    a = max(float(np.cos(state.theta) + state.v * theta_rate / gravity), 0.)
    b = float(state.v * np.cos(state.theta) * psi_rate / gravity)
    raw_nz = float(np.hypot(a, b))
    if raw_nz > cfg['normal_load_max']:
        scale = cfg['normal_load_max'] / raw_nz
        a *= scale
        b *= scale
    nz = float(np.hypot(a, b))
    phi = float(np.arctan2(b, a)) if nz else 0.
    nx = float(np.sin(state.theta) + acceleration / gravity)
    return ControlCommand(nx=nx, nz=nz, phi=phi)
