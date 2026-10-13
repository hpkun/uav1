"""v3.29 freezes every v3.28 parameter; only action semantics change."""
from copy import deepcopy
from .v328_config import validate_v328


def validate_v329(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.29':
        raise ValueError('v3.29 requires environment_version 3.29')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.28'
    validate_v328(baseline)
    return config
