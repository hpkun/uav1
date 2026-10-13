"""Freeze all v3.24 fields except the disabled Safety scale."""
from copy import deepcopy
from .v324_config import validate_v324


def validate_v325(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.25':
        raise ValueError('v3.25 requires environment_version 3.25')
    reward = config.get('reward')
    if (not isinstance(reward, dict) or 'safety_penalty_scale' not in reward
            or isinstance(reward['safety_penalty_scale'], bool)
            or reward['safety_penalty_scale'] != 0.0):
        raise ValueError('v3.25 requires safety_penalty_scale=0.0')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.24'
    legacy['reward']['safety_penalty_scale'] = 0.2
    validate_v324(legacy)
    return config
