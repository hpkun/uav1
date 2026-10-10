"""Independent strict scenario schema, sharing frozen v3.0 combat validation."""
from copy import deepcopy
import math
from .v30_config import validate_v30

SCENARIO = dict(team_size=5, center_radius=2500., lateral_min=-2500., lateral_max=2500.,
    minimum_same_team_lateral_separation=400., altitude_center=3000., altitude_perturbation_max=100.,
    speed_center=225., speed_perturbation_max=10., heading_perturbation_max=0.3490658503988659)

def validate_v31(config):
    s = config.get('scenario')
    if not isinstance(s, dict) or set(s) != set(SCENARIO):
        raise ValueError('v3.1 scenario schema mismatch')
    for key, expected in SCENARIO.items():
        value = s[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != expected:
            raise ValueError(f'v3.1 frozen scenario.{key} must equal {expected}')
    if type(s['team_size']) is not int:
        raise ValueError('v3.1 team_size must be integer')
    # Validation adapter only: these offsets are never used to generate v3.1 states.
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.0'
    legacy['scenario'] = {k:v for k,v in s.items() if k not in
        ('lateral_min','lateral_max','minimum_same_team_lateral_separation')}
    legacy['scenario']['formation_offsets'] = [-600.,-300.,0.,300.,600.]
    validate_v30(legacy)
    if math.hypot(s['center_radius'], max(abs(s['lateral_min']), abs(s['lateral_max']))) > config['arena']['radius']:
        raise ValueError('v3.1 initialization outside arena')
    if not (config['arena']['altitude_min'] < s['altitude_center']-s['altitude_perturbation_max']
            <= s['altitude_center']+s['altitude_perturbation_max'] < config['arena']['altitude_max']):
        raise ValueError('v3.1 initial altitude outside bounds')
    if not (config['aircraft']['v_min'] <= s['speed_center']-s['speed_perturbation_max']
            <= s['speed_center']+s['speed_perturbation_max'] <= config['aircraft']['v_max']):
        raise ValueError('v3.1 initial speed outside bounds')
    return config
