"""Strict scaled hardware/tactical schema, frozen inherited v3.5 mechanisms."""
from copy import deepcopy
import math
from .v35_config import validate_v35

TACTICAL=dict(guide_reward=.001,offense_reward_30deg=.01,offense_reward_15deg=.02,offense_reward_5deg=.10,
    threat_penalty_30deg=-.015,threat_penalty_15deg=-.025,threat_penalty_5deg=-.15,tactical_aspect_max=math.pi/6)
EVENT=dict(kill_reward=10.,loss_penalty=-10.,boundary_penalty=-10.,team_casualty_penalty=-2.,win_reward=20.,lose_penalty=-20.,draw_reward=0.,safe_distance=200.,safety_penalty_scale=.2)
WEAPON=dict(range_min=0.,range_max=2000.,ata_max=math.pi/6,ha_max=math.pi/6,hit_probability=.7,ammo_per_aircraft=6)

def frozen(actual,expected,label):
    if not isinstance(actual,dict) or set(actual)!=set(expected):raise ValueError(f'v3.6 {label} schema mismatch')
    for key,value in actual.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value!=expected[key]:
            raise ValueError(f'v3.6 frozen {label}.{key} must equal {expected[key]}')

def validate_v36(config):
    frozen(config.get('weapon'),WEAPON,'weapon')
    frozen(config.get('sensor'),dict(range_max=4000.,off_boresight_angle_max=math.pi/2),'sensor')
    frozen(config.get('reward'),dict(**EVENT,**TACTICAL),'reward')
    legacy=deepcopy(config);legacy['environment_version']='3.5'
    legacy['sensor']['range_max']=3000.
    legacy['weapon']=dict(range_min=0.,range_max=1000.,off_boresight_angle_max=math.pi/2,hit_probability=.7,ammo_per_aircraft=6)
    legacy['reward']=dict(**EVENT,potential_gamma=.99,potential_scale=1.,distance_weight=.5,angle_weight=.5)
    validate_v35(legacy)
    return config
