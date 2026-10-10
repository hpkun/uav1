"""Independent v3.3 reward schema; frozen v3.2 common validation."""
from copy import deepcopy
import math
from .v32_config import validate_v32

REWARD=dict(kill_reward=10.,loss_penalty=-10.,boundary_penalty=-10.,team_casualty_penalty=-2.,
    win_reward=20.,lose_penalty=-20.,draw_reward=0.,potential_gamma=.99,potential_scale=1.,
    distance_weight=.5,angle_weight=.5,safe_distance=200.,safety_penalty_scale=.2)

def validate_v33(config):
    reward=config.get('reward')
    if not isinstance(reward,dict) or set(reward)!=set(REWARD):raise ValueError('v3.3 reward schema mismatch')
    for key,expected in REWARD.items():
        value=reward[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value!=expected:
            raise ValueError(f'v3.3 frozen reward.{key} must equal {expected}')
    legacy=deepcopy(config);legacy['environment_version']='3.2'
    legacy['reward'].pop('team_casualty_penalty')
    validate_v32(legacy)
    return config
