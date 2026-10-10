"""Freeze v3.7 fidelity changes; map copies to v3.6 solely for validation."""
from copy import deepcopy
from .v36_config import validate_v36,frozen,WEAPON as V36_WEAPON,EVENT,TACTICAL
from .v37_weapon import EFFECTIVE_HIT_DISTANCE,NOISE_SCALE

WEAPON={k:v for k,v in V36_WEAPON.items() if k!='hit_probability'}
WEAPON.update(effective_hit_distance=EFFECTIVE_HIT_DISTANCE,ata_noise_scale=NOISE_SCALE,ha_noise_scale=NOISE_SCALE)
REWARD=dict(**EVENT,**TACTICAL)
for key in ('team_casualty_penalty','win_reward','lose_penalty','draw_reward'):REWARD[key]=0.

def validate_v37(config):
    frozen(config.get('weapon'),WEAPON,'v3.7 weapon')
    frozen(config.get('reward'),REWARD,'v3.7 reward')
    for key in ('effective_hit_distance','ata_noise_scale','ha_noise_scale'):
        if config['weapon'][key]<=0:raise ValueError(f'v3.7 weapon.{key} must be positive')
    legacy=deepcopy(config);legacy['environment_version']='3.6'
    legacy['weapon']=dict(V36_WEAPON)
    legacy['reward']=dict(**EVENT,**TACTICAL)
    validate_v36(legacy)
    return config
