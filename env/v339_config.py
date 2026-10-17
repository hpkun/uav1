"""Strict distance-tier schema over the unchanged v3.38 baseline."""
from copy import deepcopy
import math

from .v338_config import validate_v338


def validate_v339(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.39':
        raise ValueError('v3.39 requires environment_version 3.39')
    policy = config.get('blue_policy')
    fields = {'near_distance_max', 'far_distance_min', 'near_desired_speed',
              'mid_desired_speed', 'far_desired_speed'}
    if not isinstance(policy, dict) or set(policy) != fields:
        raise ValueError('v3.39 requires exactly the five distance-tier Blue policy fields')
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in policy.values()):
        raise ValueError('v3.39 Blue policy values must be finite numbers')
    if not 0 < policy['near_distance_max'] < policy['far_distance_min']:
        raise ValueError('v3.39 requires 0 < near_distance_max < far_distance_min')
    if any(not 150 <= policy[name] <= 300 for name in fields if name.endswith('desired_speed')):
        raise ValueError('v3.39 desired speeds must be in [150, 300] m/s')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.38'
    baseline['blue_policy'] = {'desired_speed': 225.0}
    validate_v338(baseline)
    return config
