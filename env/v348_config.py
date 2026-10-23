"""Freeze all v3.47 conditions except its weapon and required fire resources."""
from copy import deepcopy
import math
from .v347_config import validate_v347
from .v344_config import FIRE_CONTROL as DWELL_FIRE_CONTROL
from .v341_config import EFFECTIVE_HIT_DISTANCE

WEAPON = dict(range_min=0., range_max=2000., off_boresight_max=math.pi/6,
              ammo_per_aircraft=6, effective_hit_distance=EFFECTIVE_HIT_DISTANCE,
              azimuth_noise_scale=.9984711359155588,
              elevation_noise_scale=.9984711359155588)
FIRE_CONTROL = dict(cooldown_seconds=1.)


def validate_v348(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.48':
        raise ValueError('v3.48 requires environment_version 3.48')
    if config.get('weapon') != WEAPON or type(config['weapon'].get('ammo_per_aircraft')) is not int:
        raise ValueError('v3.48 requires the frozen v3.42 cone/probability calibration and 6 rounds')
    if config.get('fire_control') != FIRE_CONTROL:
        raise ValueError('v3.48 requires 1.0 second cooldown and no dwell fields')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.47'
    baseline['weapon'] = {k: WEAPON[k] for k in ('range_min', 'range_max', 'off_boresight_max')}
    baseline['fire_control'] = deepcopy(DWELL_FIRE_CONTROL)
    validate_v347(baseline)
    return config
