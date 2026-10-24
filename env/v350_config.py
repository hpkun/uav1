"""Freeze v3.48 mechanics; permit only the specified paper reward/outcome change."""
from copy import deepcopy

from .v348_config import validate_v348
from .v346_config import PBRS


def validate_v350(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.50':
        raise ValueError('v3.50 requires matching environment identity')
    if 'research_reward' in config or 'candidate_reward' in config:
        raise ValueError('v3.50 forbids PBRS/candidate reward configuration')
    if config.get('reward', {}).get('guide_distance_min') != 4000.:
        raise ValueError('v3.50 paper R3 requires 4000 metres')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.48'
    baseline['reward']['guide_distance_min'] = 3000.
    # Validation only: these fields are never placed into the live v3.50 config.
    baseline['research_reward'] = deepcopy(PBRS)
    validate_v348(baseline)
    return config
