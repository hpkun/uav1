"""v3.21 freezes every v3.20 formal parameter."""
from copy import deepcopy
from .v320_config import validate_v320


def validate_v321(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.21':
        raise ValueError('v3.21 requires environment_version 3.21')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.20'
    validate_v320(legacy)
    return config
