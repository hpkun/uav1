"""v3.20 changes controller implementation, never formal numeric parameters."""
from copy import deepcopy
from .v319_config import validate_v319


def validate_v320(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.20':
        raise ValueError('v3.20 requires environment_version 3.20')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.19'
    validate_v319(legacy)
    return config
