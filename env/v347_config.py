"""Validate v3.47 against the unchanged v3.46 environment contract."""
from copy import deepcopy
from .v346_config import validate_v346


def validate_v347(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.47':
        raise ValueError('v3.47 requires environment_version 3.47')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.46'
    validate_v346(baseline)
    return config
