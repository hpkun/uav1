"""Freeze v3.26 parameters except the shared load cap and Blue desired speed."""
from copy import deepcopy
from .v326_config import validate_v326


def validate_v328(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.28':
        raise ValueError('v3.28 requires environment_version 3.28')
    if config.get('action', {}).get('controller', {}).get('normal_load_max') != 10.0:
        raise ValueError('v3.28 requires normal_load_max=10.0')
    if config.get('blue_policy', {}).get('desired_speed') != 225.0:
        raise ValueError('v3.28 requires Blue desired_speed=225.0')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.26'
    legacy['action']['controller']['normal_load_max'] = 8.0
    legacy['blue_policy']['desired_speed'] = 250.0
    validate_v326(legacy)
    return config
