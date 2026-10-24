"""Opt-in tactical reward experiment; every v3.48 mechanism remains frozen."""
from copy import deepcopy
import math
from .v348_config import validate_v348

CANDIDATE_REWARD=dict(tactical_outer_distance=4000., tactical_span=2000.,
    alignment_max=math.pi, offense_base=.02, offense_tail=0., defense_scale=.01)


def validate_v349(config):
    if not isinstance(config,dict) or config.get('environment_version')!='3.49':
        raise ValueError('v3.49 requires matching version')
    if config.get('candidate_reward')!=CANDIDATE_REWARD:
        raise ValueError('v3.49 requires explicit frozen candidate_reward')
    baseline=deepcopy(config);baseline['environment_version']='3.48';baseline.pop('candidate_reward')
    validate_v348(baseline)
    return config
