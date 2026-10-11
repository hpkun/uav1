"""Strict v3.9 radar/reward extension; all other v3.8 fields frozen."""
from copy import deepcopy
from .v38_config import validate_v38
from .v37_config import REWARD as V37_REWARD
from .v36_config import frozen
import math

REWARD = dict(V37_REWARD, guide_distance_min=4000., tactical_distance_max=4000.)
SENSOR = dict(range_max=10000., off_boresight_angle_max=math.pi / 2)


def validate_v39(config):
    frozen(config.get('sensor'), SENSOR, 'v3.9 sensor')
    frozen(config.get('reward'), REWARD, 'v3.9 reward')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.8'
    legacy['sensor']['range_max'] = 4000.
    legacy['reward'] = dict(V37_REWARD)
    validate_v38(legacy)
    return config
