"""One random diameter orientation; otherwise fixed compact five-aircraft teams."""
import numpy as np
from .models import AircraftState
from .math_utils import wrap_angle


def random_combat_states_v311(rng, team_size, center_radius, formation_offsets,
                              altitude_center, speed_center):
    alpha = float(rng.uniform(-np.pi, np.pi))
    u = np.array([np.cos(alpha), np.sin(alpha)])
    t = np.array([-np.sin(alpha), np.cos(alpha)])
    def team(side, heading):
        states = []
        for offset in formation_offsets:
            p = side * center_radius * u + offset * t
            states.append(AircraftState(float(p[0]), float(p[1]), -altitude_center,
                                        speed_center, 0., float(wrap_angle(heading))))
        return states
    return team(-1, alpha), team(1, alpha + np.pi), alpha
