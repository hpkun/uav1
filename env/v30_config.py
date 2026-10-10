"""Strict independent schema; legacy validation is deliberately untouched."""
import math


def validate_v30(config):
    fields={'environment_version','simulation','action','aircraft','arena','scenario',
            'sensor','weapon','reward','observation','blue_policy'}
    if set(config)!=fields: raise ValueError('v3.0 top-level schema mismatch')
    schemas={
        'simulation':{'dt','max_steps'},
        'arena':{'radius','altitude_min','altitude_max'},
        'sensor':{'range_max','off_boresight_angle_max'},
        'weapon':{'range_min','range_max','off_boresight_angle_max','hit_probability','ammo_per_aircraft'},
        'reward':{'kill_reward','loss_penalty','boundary_penalty','win_reward','lose_penalty','draw_reward',
                  'potential_gamma','potential_scale','distance_weight','angle_weight','safe_distance','safety_penalty_scale'},
        'blue_policy':{'desired_speed','guard_radius','guard_altitude_min','guard_altitude_max'},
        'scenario':{'team_size','center_radius','formation_offsets','altitude_center',
                    'altitude_perturbation_max','speed_center','speed_perturbation_max','heading_perturbation_max'},
        'observation':{'horizontal_position_scale','altitude_scale','speed_center','speed_scale','relative_position_scale'},
        'aircraft':{'v_min','v_max','theta_min','theta_max'}}
    for name,keys in schemas.items():
        if not isinstance(config[name],dict) or set(config[name])!=keys:
            raise ValueError(f'v3.0 {name} schema mismatch')
    def numeric(value):
        return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)
    for name in schemas:
        for key,value in config[name].items():
            if key!='formation_offsets' and not numeric(value):
                raise ValueError(f'v3.0 {name}.{key} must be finite numeric')
    s,w,a,r,b=config['sensor'],config['weapon'],config['arena'],config['reward'],config['blue_policy']
    scenario=config['scenario'];simulation=config['simulation']
    if scenario['team_size']!=5 or len(scenario['formation_offsets'])!=5 or not all(numeric(x) for x in scenario['formation_offsets']):
        raise ValueError('v3.0 requires five agents and five finite formation offsets')
    if w['range_min']!=0 or w['range_max']!=1000 or not 0<=w['hit_probability']<=1:
        raise ValueError('v3.0 weapon range/probability invalid')
    if type(w['ammo_per_aircraft']) is not int or w['ammo_per_aircraft']<=0:
        raise ValueError('v3.0 ammo must be positive integer')
    if not (s['range_max']>w['range_max'] and 0<w['off_boresight_angle_max']<=s['off_boresight_angle_max']<=math.pi):
        raise ValueError('v3.0 sensor must contain weapon envelope')
    if not (0<=a['altitude_min']<a['altitude_max'] and a['radius']>scenario['center_radius']>0):
        raise ValueError('v3.0 arena/scenario invalid')
    if not (simulation['dt']>0 and type(simulation['max_steps']) is int and simulation['max_steps']>0):
        raise ValueError('v3.0 simulation invalid')
    if not (0<=r['potential_gamma']<=1 and 0<r['safe_distance']<w['range_max']
            and min(r['potential_scale'],r['distance_weight'],r['angle_weight'],r['safety_penalty_scale'])>=0):
        raise ValueError('v3.0 reward potential/safety invalid')
    if not (0<b['guard_radius']<a['radius'] and a['altitude_min']<b['guard_altitude_min']<b['guard_altitude_max']<a['altitude_max']
            and config['aircraft']['v_min']<=b['desired_speed']<=config['aircraft']['v_max']):
        raise ValueError('v3.0 Blue guard invalid')
    from .models import AircraftSpec
    AircraftSpec(**config['aircraft'])
    # Keep the existing command/controller schema and mathematical implementation.
    if set(config['action'])!={'command','controller'} or set(config['action']['command'])!={'heading_delta_max','pitch_delta_max','speed_delta_max'} or set(config['action']['controller'])!={'heading_time_constant','pitch_time_constant','speed_time_constant','normal_load_max'}:
        raise ValueError('v3.0 action schema mismatch')
    if not all(numeric(x) and x>0 for part in config['action'].values() for x in part.values()):
        raise ValueError('v3.0 action values invalid')
    if not (config['aircraft']['v_min']>0 and config['aircraft']['v_max']>config['aircraft']['v_min'] and config['aircraft']['theta_min']<config['aircraft']['theta_max']):
        raise ValueError('v3.0 aircraft bounds invalid')
    if min(config['observation'][key] for key in ('horizontal_position_scale','altitude_scale','speed_scale','relative_position_scale'))<=0:
        raise ValueError('v3.0 observation scales invalid')
    return config
