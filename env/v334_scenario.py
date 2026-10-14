"""Rotated point-symmetric 2D positions; no policy or assigned target roles."""
import numpy as np
from .models import AircraftState
from .math_utils import wrap_angle


def random_combat_states_v334(rng, center_radius, formation_local_offsets,
                              altitude_center, speed_center, team_size):
    alpha = float(rng.uniform(-np.pi, np.pi))
    u = np.array([np.cos(alpha), np.sin(alpha)])
    t = np.array([-u[1], u[0]])
    offsets = np.asarray(formation_local_offsets, dtype=float)

    def team(side, heading):
        center = side * center_radius * u
        positions = center - side * (offsets[:, :1] * u + offsets[:, 1:] * t)
        return [AircraftState(float(p[0]), float(p[1]), -float(altitude_center),
                              float(speed_center), 0., float(wrap_angle(heading))) for p in positions]

    return team(-1, alpha), team(1, alpha + np.pi), alpha
