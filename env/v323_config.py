"""Frozen v3.22 fields plus one calibrated attacker-wide firing interval."""
from copy import deepcopy
import math
from .v322_config import validate_v322


def validate_v323(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.23':
        raise ValueError('v3.23 requires environment_version 3.23')
    fire = config.get('fire_control')
    if (not isinstance(fire, dict) or set(fire) != {'cooldown_seconds'}
            or isinstance(fire['cooldown_seconds'], bool) or fire['cooldown_seconds'] != 1.0):
        raise ValueError('v3.23 fire_control requires only cooldown_seconds=1.0')
    dt = float(config['simulation']['dt'])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    ratio = float(fire['cooldown_seconds']) / dt
    if not math.isclose(ratio, round(ratio), rel_tol=0., abs_tol=1e-10):
        raise ValueError('cooldown_seconds must correspond to an integer number of steps')
    legacy = deepcopy(config)
    del legacy['fire_control']
    legacy['environment_version'] = '3.22'
    validate_v322(legacy)
    return config
