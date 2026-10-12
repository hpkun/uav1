"""Freeze v3.15 values, adding only a pi/6 true-boresight threshold."""
from copy import deepcopy
import math
from .v315_config import validate_v315


def validate_v316(config):
    if not isinstance(config, dict) or str(config.get('environment_version')) != '3.16':
        raise ValueError('v3.16 requires version 3.16')
    legacy = deepcopy(config)
    weapon = legacy.get('weapon')
    if not isinstance(weapon, dict) or 'boresight_max' not in weapon:
        raise ValueError('v3.16 requires weapon.boresight_max')
    try:
        boresight_max = float(weapon.pop('boresight_max'))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('weapon.boresight_max must be finite and in (0, pi]') from error
    if not math.isfinite(boresight_max) or not 0 < boresight_max <= math.pi:
        raise ValueError('weapon.boresight_max must be finite and in (0, pi]')
    if boresight_max != math.pi / 6:
        raise ValueError('formal v3.16 weapon.boresight_max must equal pi/6')
    legacy['environment_version'] = '3.15'
    validate_v315(legacy)
    return config
