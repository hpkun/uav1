"""Numerically identical v3.36 baseline, with a distinct reward identity."""
from copy import deepcopy
from .v336_config import validate_v336


def validate_v337(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.37':
        raise ValueError('v3.37 requires environment_version 3.37')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.36'
    validate_v336(baseline)
    return config
