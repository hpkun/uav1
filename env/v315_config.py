"""No new parameters: freeze every v3.15 value against the v3.14 contract."""
from copy import deepcopy
from .v314_config import validate_v314


def validate_v315(config):
    if not isinstance(config, dict) or str(config.get('environment_version')) != '3.15':
        raise ValueError('v3.15 requires version 3.15')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.14'
    validate_v314(legacy)
    return config
