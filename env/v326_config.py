"""v3.26 retains the entire frozen v3.25 configuration."""
from copy import deepcopy
from .v325_config import validate_v325


def validate_v326(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.26':
        raise ValueError('v3.26 requires environment_version 3.26')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.25'
    validate_v325(legacy)
    return config
