"""v3.22 freezes the entire v3.21 schema and formal parameters."""
from copy import deepcopy
from .v321_config import validate_v321


def validate_v322(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.22':
        raise ValueError('v3.22 requires environment_version 3.22')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.21'
    validate_v321(legacy)
    return config
