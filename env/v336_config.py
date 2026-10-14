"""Strict v3.35 numerical baseline, with explicitly named ATA/HA limits."""
from copy import deepcopy
import math
from .v335_config import validate_v335


def validate_v336(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.36':
        raise ValueError('v3.36 requires environment_version 3.36')
    weapon = config.get('weapon', {})
    for name in ('ata_max', 'ha_max'):
        if weapon.get(name) != math.pi / 6:
            raise ValueError(f'v3.36 requires weapon.{name}=pi/6')
    if 'boresight_azimuth_max' in weapon or 'boresight_elevation_max' in weapon:
        raise ValueError('v3.36 uses ata_max/ha_max, not body-frame limit fields')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.35'
    baseline['weapon']['boresight_azimuth_max'] = baseline['weapon'].pop('ata_max')
    baseline['weapon']['boresight_elevation_max'] = baseline['weapon'].pop('ha_max')
    validate_v335(baseline)
    return config
