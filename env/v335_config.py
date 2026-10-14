"""Strict v3.34 baseline with only normalized hit-distance scale changed."""
from copy import deepcopy
import math
from .v334_config import validate_v334

EFFECTIVE_HIT_DISTANCE = 1000. / math.log(6.)
BASELINE_EFFECTIVE_HIT_DISTANCE = 2232.442506204989


def validate_v335(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.35':
        raise ValueError('v3.35 requires environment_version 3.35')
    if config.get('weapon', {}).get('effective_hit_distance') != EFFECTIVE_HIT_DISTANCE:
        raise ValueError('v3.35 requires effective_hit_distance=1000/ln(6)=558.1106265512473')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.34'
    baseline['weapon']['effective_hit_distance'] = BASELINE_EFFECTIVE_HIT_DISTANCE
    validate_v334(baseline)
    return config
