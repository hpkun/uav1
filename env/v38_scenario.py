"""Project 5v5 chord placement around opposite paper-scale diameter endpoints."""
import numpy as np
from .math_utils import wrap_angle
from .models import AircraftState
from .v31_scenario import sample_laterals

# Floating-point safety only, in metres; not a physical/configurable setback.
SPAWN_EPSILON=1e-7

def random_combat_states_v38(rng,arena_radius,lateral_min,lateral_max,
        minimum_same_team_lateral_separation,altitude_center,altitude_perturbation_max,
        speed_center,speed_perturbation_max,heading_perturbation_max,team_size=5):
    alpha=float(rng.uniform(-np.pi,np.pi))
    longitudinal=np.array([np.cos(alpha),np.sin(alpha)])
    transverse=np.array([-np.sin(alpha),np.cos(alpha)])
    def team(side,nominal):
        offsets=sample_laterals(rng,team_size,lateral_min,lateral_max,
                               minimum_same_team_lateral_separation)
        states=[]
        for offset in offsets:
            radius=np.sqrt((arena_radius-SPAWN_EPSILON)**2-offset**2)
            p=side*radius*longitudinal+offset*transverse
            # Keep v3.7 draw order: altitude, speed, heading, then next aircraft.
            states.append(AircraftState(float(p[0]),float(p[1]),
                -float(altitude_center+rng.uniform(-altitude_perturbation_max,altitude_perturbation_max)),
                float(speed_center+rng.uniform(-speed_perturbation_max,speed_perturbation_max)),
                0.,float(wrap_angle(nominal+rng.uniform(-heading_perturbation_max,heading_perturbation_max)))))
        return states
    return team(-1,alpha),team(1,alpha+np.pi),alpha
