"""Independent diversified initialization; no combat protocol changes."""
import numpy as np
from .math_utils import wrap_angle
from .models import AircraftState

MAX_SAMPLING_ATTEMPTS = 10000

def sample_laterals(rng, team_size, lateral_min, lateral_max, minimum_same_team_lateral_separation):
    # Reject whole independent uniform proposals, avoiding sequential-placement bias.
    for _ in range(MAX_SAMPLING_ATTEMPTS):
        values = rng.uniform(lateral_min, lateral_max, team_size)
        if np.all(np.diff(np.sort(values)) >= minimum_same_team_lateral_separation):
            return rng.permutation(values)
    raise RuntimeError('v3.1 lateral rejection sampling exhausted max attempts')

def random_combat_states_v31(rng, center_radius, lateral_min, lateral_max,
        minimum_same_team_lateral_separation, altitude_center, altitude_perturbation_max,
        speed_center, speed_perturbation_max, heading_perturbation_max, team_size=5):
    alpha = float(rng.uniform(-np.pi, np.pi))
    longitudinal = np.array([np.cos(alpha), np.sin(alpha)])
    transverse = np.array([-np.sin(alpha), np.cos(alpha)])
    def team(side, nominal):
        offsets = sample_laterals(rng, team_size, lateral_min, lateral_max,
                                  minimum_same_team_lateral_separation)
        states = []
        for offset in offsets:
            p = side * center_radius * longitudinal + offset * transverse
            states.append(AircraftState(float(p[0]), float(p[1]),
                -float(altitude_center + rng.uniform(-altitude_perturbation_max, altitude_perturbation_max)),
                float(speed_center + rng.uniform(-speed_perturbation_max, speed_perturbation_max)),
                0., float(wrap_angle(nominal + rng.uniform(-heading_perturbation_max, heading_perturbation_max)))))
        return states
    return team(-1, alpha), team(1, alpha + np.pi), alpha
