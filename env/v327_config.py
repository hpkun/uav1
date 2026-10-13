"""Persistent targeting with frozen v3.26 formal parameters."""
from copy import deepcopy
from .v326_config import validate_v326


def validate_v327(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.27':
        raise ValueError('v3.27 requires environment_version 3.27')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.26'
    validate_v326(legacy)
    return config
