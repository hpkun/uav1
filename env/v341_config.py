"""Strict one-parameter calibration experiment over the frozen v3.40 schema."""
from copy import deepcopy
import math
from .v340_config import validate_v340

EFFECTIVE_HIT_DISTANCE = 2000. / math.log(6.)


def validate_v341(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.41':
        raise ValueError('v3.41 requires environment_version 3.41')
    weapon = config.get('weapon')
    if not isinstance(weapon, dict) or weapon.get('effective_hit_distance') != EFFECTIVE_HIT_DISTANCE:
        raise ValueError('v3.41 requires weapon.effective_hit_distance=2000/ln(6)')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.40'
    baseline['weapon']['effective_hit_distance'] = 1000. / math.log(6.)
    validate_v340(baseline)
    return config
