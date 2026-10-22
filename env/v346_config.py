"""Strict paper R4 identity, retaining all v3.44 constraints and PBRS gamma."""
from copy import deepcopy
from .v344_config import validate_v344
from .v343_reward import RESEARCH_REWARD
from .v346_reward import PAPER_R4

PBRS = {k: v for k, v in RESEARCH_REWARD.items()
        if k == 'gamma' or k.startswith('potential_')}


def validate_v346(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.46':
        raise ValueError('v3.46 requires environment_version 3.46')
    if config.get('research_reward') != PBRS:
        raise ValueError('v3.46 research_reward must contain only unchanged PBRS parameters')
    reward = config.get('reward', {})
    if any(reward.get(k) != v for k, v in PAPER_R4.items()):
        raise ValueError('v3.46 requires exact paper Eq. (25) R4 parameters in reward')
    # Transform ONLY a validation copy; no legacy R4 coefficient reaches runtime.
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.44'
    baseline['research_reward'] = deepcopy(RESEARCH_REWARD)
    validate_v344(baseline)
    return config
