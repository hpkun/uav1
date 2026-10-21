"""Fail-closed schema and dimensions for navigation3d-v1 only."""
import copy
from pathlib import Path
import numpy as np
import yaml

IDENTITY = 'navigation3d-v1'
OBSERVATION_SCHEMA = 'local_navigation3d_20_plus_10_per_peer_v1'


def observation_dim(config):
    return 20 + 10 * (config['num_agents'] - 1)


def load_config(path):
    return validate_config(yaml.safe_load(Path(path).read_text(encoding='utf-8')))


def validate_config(config):
    expected = {'environment_id', 'observation_schema', 'num_agents', 'dt', 'max_steps',
        'initial_center_ned', 'formation_offsets_ned', 'initial_jitter_m', 'initial_speed',
        'active_agents', 'waypoint_centers_ned', 'goal_radius_m', 'visibility_radius_m',
        'distance_scale_m', 'bounds', 'motion', 'reward'}
    if set(config) != expected or config['environment_id'] != IDENTITY or config['observation_schema'] != OBSERVATION_SCHEMA:
        raise ValueError('invalid navigation environment identity/schema/keys')
    n = config['num_agents']
    if type(n) is not int or not 1 <= n <= 8 or type(config['max_steps']) is not int or config['max_steps'] <= 0:
        raise ValueError('invalid navigation count/horizon')
    active = config['active_agents']
    if len(active) != n or not all(type(x) is bool for x in active) or not any(active):
        raise ValueError('at least one explicitly active navigation agent required')
    for key in ('dt', 'goal_radius_m', 'visibility_radius_m', 'distance_scale_m'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'invalid {key}')
    if not np.isfinite(config['initial_jitter_m']) or config['initial_jitter_m'] < 0:
        raise ValueError('invalid initial jitter')
    b, m, r = (config[k] for k in ('bounds', 'motion', 'reward'))
    if set(b) != {'horizontal_limit_m', 'min_altitude_m', 'max_altitude_m'} or not np.isfinite(list(b.values())).all():
        raise ValueError('invalid navigation bounds')
    if b['horizontal_limit_m'] <= 0 or not 0 < b['min_altitude_m'] < b['max_altitude_m']:
        raise ValueError('invalid navigation bounds order')
    if set(m) != {'v_min', 'v_max', 'pitch_limit_rad', 'heading_delta_max', 'pitch_delta_max', 'speed_delta_max'} or not np.isfinite(list(m.values())).all():
        raise ValueError('invalid navigation motion')
    if not 0 < m['v_min'] < m['v_max'] or not 0 < m['pitch_limit_rad'] < np.pi/2 or any(m[k] <= 0 for k in ('heading_delta_max', 'pitch_delta_max', 'speed_delta_max')):
        raise ValueError('invalid navigation motion limits')
    if not np.isfinite(config['initial_speed']) or not m['v_min'] <= config['initial_speed'] <= m['v_max']:
        raise ValueError('invalid initial speed')
    if set(r) != {'gamma', 'distance_weight', 'alignment_weight', 'formation_weight', 'formation_scale_m',
        'danger_radius_m', 'collision_radius_m', 'danger_penalty', 'failure_penalty', 'time_penalty', 'completion_bonus'} or not np.isfinite(list(r.values())).all():
        raise ValueError('invalid navigation reward schema')
    if not 0 < r['gamma'] < 1 or any(v < 0 for k, v in r.items() if k != 'gamma'):
        raise ValueError('invalid navigation reward coefficients')
    if not 0 < r['collision_radius_m'] < r['danger_radius_m'] or r['formation_scale_m'] <= 0:
        raise ValueError('invalid navigation safety distances')
    if r['completion_bonus'] <= sum(r[k] for k in ('distance_weight', 'alignment_weight', 'formation_weight')):
        raise ValueError('completion bonus must dominate the bounded potential')
    offsets = np.asarray(config['formation_offsets_ned'], dtype=float)
    center = np.asarray(config['initial_center_ned'], dtype=float)
    route = np.asarray(config['waypoint_centers_ned'], dtype=float)
    if offsets.shape != (n, 3) or center.shape != (3,) or route.ndim != 2 or route.shape[1] != 3 or not len(route):
        raise ValueError('invalid navigation route/formation dimensions')
    if not all(np.isfinite(v).all() for v in (offsets, center, route)):
        raise ValueError('nonfinite navigation route')
    points = np.concatenate([center[None], route])[:, None] + offsets[None]
    margin = config['initial_jitter_m']
    if (np.abs(points[..., :2]) + margin >= b['horizontal_limit_m']).any() or (-points[..., 2]-margin <= b['min_altitude_m']).any() or (-points[..., 2]+margin >= b['max_altitude_m']).any():
        raise ValueError('navigation route must lie inside safety bounds')
    return copy.deepcopy(config)
