"""Observation correction with every v3.23 formal parameter frozen."""
from copy import deepcopy
from .v323_config import validate_v323


def validate_v324(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.24':
        raise ValueError('v3.24 requires environment_version 3.24')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.23'
    validate_v323(legacy)
    return config
