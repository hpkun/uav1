"""Freeze every v3.12 parameter; extend only the version contract."""
from copy import deepcopy
from .v312_config import validate_v312


def validate_v313(config):
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.12'
    validate_v312(legacy)
    return config
