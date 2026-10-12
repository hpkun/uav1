"""Encode existing nearest-pursuit intentions into v3.20 feasible actions."""
import numpy as np
from .math_utils import wrap_angle
from .v310_policy import PaperNearestTargetBluePolicy


class V321NearestTargetBluePolicy(PaperNearestTargetBluePolicy):
    def __init__(self, config, action_config, aircraft_spec):
        super().__init__(config, action_config)
        self.controller_config = action_config['controller']
        self.aircraft_spec = aircraft_spec

    def action_toward(self, own, desired_heading, desired_elevation, desired_speed):
        command, controller = self.command_config, self.controller_config
        gravity, epsilon = 9.81, 1e-12
        a0 = float(np.cos(own.theta))
        nz_max = float(controller['normal_load_max'])
        b_max = float(np.sqrt(max(nz_max**2 - a0**2, 0.)))
        psi_rate = wrap_angle(desired_heading - own.psi) / float(controller['heading_time_constant'])
        b_required = own.v * a0 * psi_rate / gravity
        a_heading = float(np.clip(b_required / b_max, -1., 1.)) if b_max > epsilon else 0.
        # Match the float32 action that the shared controller will receive.
        a_heading = float(np.float32(a_heading))
        b = a_heading * b_max
        a_max = float(np.sqrt(max(nz_max**2 - b**2, 0.)))
        pitch_delta = float(command['pitch_delta_max'])
        legacy_pitch = float(np.clip((desired_elevation - own.theta) / pitch_delta, -1., 1.))
        target_pitch = float(np.clip(own.theta + pitch_delta * legacy_pitch,
                                     self.aircraft_spec.theta_min, self.aircraft_spec.theta_max))
        theta_rate = (target_pitch - own.theta) / float(controller['pitch_time_constant'])
        a_required = a0 + own.v * theta_rate / gravity
        a_feasible = float(np.clip(a_required, 0., a_max))
        if a_feasible >= a0:
            a_pitch = (a_feasible - a0) / (a_max - a0) if a_max - a0 > epsilon else 0.
        else:
            a_pitch = a_feasible / a0 - 1.
        a_speed = (desired_speed - own.v) / float(command['speed_delta_max'])
        return np.clip(np.asarray([a_heading, a_pitch, a_speed], dtype=np.float32), -1., 1.)
