"""v3.15 feasible targets; the historical controller remains authoritative."""
import numpy as np
from .control import ManeuverTarget, action_to_target, target_to_control


def action_to_target_v315(state, action, action_config, aircraft_spec):
    """Decode the original relative command, then clip pitch and speed targets."""
    raw = action_to_target(state, action, action_config['command'])
    return ManeuverTarget(
        heading=raw.heading,
        pitch=float(np.clip(raw.pitch, aircraft_spec.theta_min, aircraft_spec.theta_max)),
        speed=float(np.clip(raw.speed, aircraft_spec.v_min, aircraft_spec.v_max)),
    )


def action_to_control_v315(state, action, action_config, aircraft_spec, gravity=9.81):
    target = action_to_target_v315(state, action, action_config, aircraft_spec)
    return target_to_control(state, target, action_config['controller'], gravity)


__all__ = ['action_to_target_v315', 'action_to_control_v315']
