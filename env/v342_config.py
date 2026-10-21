"""Isolated cone experiment; all non-geometric v3.41 constraints are retained."""
from copy import deepcopy
import math
from .v341_config import validate_v341


def validate_v342(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.42':
        raise ValueError('v3.42 requires environment_version 3.42')
    weapon = config.get('weapon', {})
    if weapon.get('off_boresight_max') != math.pi / 6:
        raise ValueError('v3.42 requires weapon.off_boresight_max=pi/6')
    if 'ata_max' in weapon or 'ha_max' in weapon:
        raise ValueError('v3.42 replaces ATA/HA gates with one cone')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.41'
    baseline['weapon'].pop('off_boresight_max')
    baseline['weapon'].update(ata_max=math.pi/6, ha_max=math.pi/6)
    validate_v341(baseline)
    return config
