"""Initialization-only v3.8 schema with frozen v3.7 inherited mechanisms."""
from copy import deepcopy
import math
from .v31_config import SCENARIO as V31_SCENARIO
from .v37_config import validate_v37

SCENARIO={k:v for k,v in V31_SCENARIO.items() if k!='center_radius'}

def validate_v38(config):
    arena=config.get('arena')
    expected_arena=dict(radius=5000.,altitude_min=0.,altitude_max=6000.)
    if not isinstance(arena,dict) or set(arena)!=set(expected_arena):raise ValueError('v3.8 arena schema mismatch')
    for key,value in arena.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value!=expected_arena[key]:
            raise ValueError(f'v3.8 frozen arena.{key} must equal {expected_arena[key]}')
    scenario=config.get('scenario')
    if not isinstance(scenario,dict) or set(scenario)!=set(SCENARIO):raise ValueError('v3.8 scenario schema mismatch')
    for key,value in scenario.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value!=SCENARIO[key]:
            raise ValueError(f'v3.8 frozen scenario.{key} must equal {SCENARIO[key]}')
    if type(scenario['team_size']) is not int:raise ValueError('v3.8 team_size must be integer')
    legacy=deepcopy(config);legacy['environment_version']='3.7'
    # Adapter for validation only; v3.8 runtime uses arena radius/chord geometry.
    legacy['scenario']['center_radius']=2500.
    validate_v37(legacy)
    return config
