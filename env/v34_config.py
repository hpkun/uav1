"""Frozen v3.4 Blue schema, all other fields validated as v3.3."""
from copy import deepcopy
import math
from .v33_config import validate_v33

EXTRA = dict(evade_enter_distance=1200., evade_exit_distance=1600., threat_radius=1500., support_radius=1500.)
BLUE = dict(desired_speed=250., guard_radius=4500., guard_altitude_min=500., guard_altitude_max=5500., **EXTRA)

def validate_v34(config):
    blue = config.get('blue_policy')
    if not isinstance(blue, dict) or set(blue) != set(BLUE):
        raise ValueError('v3.4 blue_policy schema mismatch')
    for key, expected in BLUE.items():
        v = blue[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v != expected:
            raise ValueError(f'v3.4 frozen blue_policy.{key} must equal {expected}')
    if not (0 < blue['evade_enter_distance'] and config['weapon']['range_max'] < blue['evade_enter_distance']
            < blue['threat_radius'] <= blue['evade_exit_distance'] <= config['sensor']['range_max']
            and 0 < blue['support_radius'] <= config['sensor']['range_max']):
        raise ValueError('v3.4 Blue distance ordering invalid')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.3'
    for key in EXTRA:
        legacy['blue_policy'].pop(key)
    validate_v33(legacy)
    return config
