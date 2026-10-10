"""v3.2 has exactly the v3.1 schema and numeric configuration."""
from copy import deepcopy
from .v31_config import validate_v31

def validate_v32(config):
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.1'
    validate_v31(legacy)
    return config
