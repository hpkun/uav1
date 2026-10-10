"""Frozen paper Blue schema; other configuration remains under v3.3 validation."""
from copy import deepcopy
import math
from .v33_config import validate_v33

BLUE=dict(desired_speed=250.,close_distance=1800.,dispersion_threshold=1500.)

def validate_v35(config):
    blue=config.get('blue_policy')
    if not isinstance(blue,dict) or set(blue)!=set(BLUE):raise ValueError('v3.5 blue_policy schema mismatch')
    for key,expected in BLUE.items():
        value=blue[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value!=expected:
            raise ValueError(f'v3.5 frozen blue_policy.{key} must equal {expected}')
    if not config['weapon']['range_max']<blue['close_distance']<config['sensor']['range_max']:
        raise ValueError('v3.5 close distance ordering invalid')
    legacy=deepcopy(config);legacy['environment_version']='3.3'
    legacy['blue_policy']=dict(desired_speed=250.,guard_radius=4500.,guard_altitude_min=500.,guard_altitude_max=5500.)
    validate_v33(legacy)
    return config
