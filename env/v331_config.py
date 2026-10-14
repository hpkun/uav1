"""Freeze v3.30 after mapping the six doubled R4 coefficients back."""
from copy import deepcopy
from .v330_config import validate_v330

R4_BASELINE = {
    'offense_reward_30deg': .01,
    'offense_reward_15deg': .02,
    'offense_reward_5deg': .10,
    'threat_penalty_30deg': -.015,
    'threat_penalty_15deg': -.025,
    'threat_penalty_5deg': -.15,
}


def validate_v331(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.31':
        raise ValueError('v3.31 requires environment_version 3.31')
    reward = config.get('reward', {})
    for key, value in R4_BASELINE.items():
        if reward.get(key) != 2 * value:
            raise ValueError(f'v3.31 requires {key}={2 * value}')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.30'
    baseline['reward'].update(R4_BASELINE)
    validate_v330(baseline)
    return config
