"""Independent candidate reward identity with frozen v3.42 mechanics."""
from copy import deepcopy
from .v342_config import validate_v342
from .v343_reward import RESEARCH_REWARD


def validate_v343(config):
    if not isinstance(config,dict) or config.get('environment_version') != '3.43':
        raise ValueError('v3.43 requires environment_version 3.43')
    if config.get('research_reward') != RESEARCH_REWARD:
        raise ValueError('v3.43 requires the explicit candidate research_reward parameters')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.42'
    baseline.pop('research_reward')
    validate_v342(baseline)
    return config


def validate_shaping_gamma(env_config, algorithm_config):
    if env_config.get('environment_version') == '3.43':
        if algorithm_config['training']['gamma'] != env_config['research_reward']['gamma']:
            raise ValueError('v3.43 PBRS gamma must equal training gamma')
