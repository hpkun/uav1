"""Independent frozen v3.11 schema; controller and old spawn fields rejected."""
import math
from .v39_config import REWARD
from .v311_weapon import EFFECTIVE_HIT_DISTANCE, NOISE_SCALE

COMMAND = dict(heading_delta_max=math.pi, pitch_delta_max=math.pi/3, speed_delta_max=50.)
SCENARIO = dict(team_size=5, center_radius=4800., formation_offsets=[-400., -200., 0., 200., 400.],
                altitude_center=3000., speed_center=225.)
SENSOR = dict(range_max=10000., off_boresight_angle_max=math.pi/3)
WEAPON = dict(range_min=0., range_max=4000., ata_max=math.pi/6, ha_max=math.pi/6,
              ammo_per_aircraft=6, effective_hit_distance=EFFECTIVE_HIT_DISTANCE,
              ata_noise_scale=NOISE_SCALE, ha_noise_scale=NOISE_SCALE)
EXPECTED = dict(simulation=dict(dt=.1, max_steps=1000), action=dict(command=COMMAND),
    aircraft=dict(v_min=150., v_max=300., theta_min=-math.pi/3, theta_max=math.pi/3),
    arena=dict(radius=5000., altitude_min=0., altitude_max=6000.), scenario=SCENARIO,
    sensor=SENSOR, weapon=WEAPON, reward=REWARD,
    observation=dict(horizontal_position_scale=5000., altitude_scale=10000., speed_center=225.,
                     speed_scale=75., relative_position_scale=10000.), blue_policy=dict(desired_speed=250.))


def _frozen(actual, expected, path):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f'v3.11 {path} schema mismatch')
        for key, value in expected.items():
            _frozen(actual[key], value, f'{path}.{key}')
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f'v3.11 {path} must equal {expected}')
        for i, value in enumerate(expected):
            _frozen(actual[i], value, f'{path}[{i}]')
    elif isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual != expected:
        raise ValueError(f'v3.11 frozen {path} must equal {expected}')


def validate_v311(config):
    if not isinstance(config, dict) or set(config) != set(EXPECTED) | {'environment_version'}:
        raise ValueError('v3.11 top-level schema mismatch')
    if str(config['environment_version']) != '3.11':
        raise ValueError('v3.11 version mismatch')
    for key, expected in EXPECTED.items():
        _frozen(config[key], expected, key)
    for section, key in (('simulation', 'max_steps'), ('scenario', 'team_size'), ('weapon', 'ammo_per_aircraft')):
        if type(config[section][key]) is not int:
            raise ValueError(f'v3.11 {section}.{key} must be integer')
    return config
