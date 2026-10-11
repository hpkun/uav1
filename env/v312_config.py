"""Observation-only version extension; v3.11 mechanics remain frozen."""
from copy import deepcopy
from .v311_config import validate_v311, _frozen, EXPECTED

OBSERVATION = dict(EXPECTED['observation'], altitude_scale=6000.)


def validate_v312(config):
    _frozen(config.get('observation'), OBSERVATION, 'observation')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.11'
    legacy['observation'] = dict(EXPECTED['observation'])
    validate_v311(legacy)
    return config
