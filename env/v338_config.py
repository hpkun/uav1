"""Exact v3.37 configuration, with an isolated observation identity."""
from copy import deepcopy
from .v337_config import validate_v337


def validate_v338(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.38':
        raise ValueError('v3.38 requires environment_version 3.38')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.37'
    validate_v337(baseline)
    return config
