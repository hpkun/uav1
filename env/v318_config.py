"""v3.18 changes cadence in code, adding no configuration parameters."""
from copy import deepcopy
from .v317_config import validate_v317


def validate_v318(config):
    if not isinstance(config, dict) or str(config.get('environment_version')) != '3.18':
        raise ValueError('v3.18 requires version 3.18')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.17'
    validate_v317(legacy)
    return config
