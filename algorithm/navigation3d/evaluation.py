"""Deterministic navigation evaluation and pose traces, always requiring CUDA."""
from pathlib import Path
import json
import numpy as np
import torch
from env.navigation3d import Navigation3DEnv
from env.navigation3d.geometry import waypoint_geometry, wrap
from .trainer import NavigationTrainer, require_cuda


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_value, allow_nan=False), encoding='utf-8')


def waypoint_controller(env):
    actions = np.zeros((env.num_agents, 3))
    m = env.config['motion']
    for i in np.flatnonzero(env.valid_mask):
        s = env.states[i]
        g = waypoint_geometry(env.positions[i], s.theta, s.psi, env.goals[i])
        desired_speed = np.clip(g.distance, 100., env.config['initial_speed'])
        actions[i] = [g.heading_error/m['heading_delta_max'],
            wrap(g.world_elevation-s.theta)/m['pitch_delta_max'], (desired_speed-s.v)/m['speed_delta_max']]
    return np.clip(actions, -1, 1)


def run_episode(env_config, seed, trainer=None, controller=None):
    env = Navigation3DEnv(env_config)
    obs, info = env.reset(seed)
    states = [np.stack([s.as_array() for s in env.states])]
    goals, masks = [env.goals.copy()], [env.valid_mask.copy()]
    components, actions, phases = [], [], [env.waypoint_index]
    while True:
        action = trainer.act(obs[None], env.valid_mask[None], deterministic=True)[0] if trainer else controller(env)
        obs, _, terminated, truncated, info = env.step(action)
        states.append(np.stack([s.as_array() for s in env.states]))
        goals.append(env.goals.copy()); masks.append(env.valid_mask.copy()); phases.append(env.waypoint_index)
        components.append(np.stack(list(info['reward_components'].values())))
        actions.append(action)
        if terminated or truncated:
            break
    info['simulation_time_s'] = env.steps * env.config['dt']
    trace = dict(states=np.array(states), goals=np.array(goals), valid_masks=np.array(masks),
        phases=np.array(phases), actions=np.array(actions), reward_components=np.array(components),
        component_names=np.array(list(env.component_totals)), dt=env.config['dt'])
    return info, trace


def evaluate(env_config, trainer, seeds):
    require_cuda(trainer.device)
    records = [run_episode(env_config, seed, trainer=trainer)[0] for seed in seeds]
    if not records:
        raise ValueError('navigation evaluation needs at least one seed')
    return dict(episodes=len(records), success_rate=float(np.mean([r['success'] for r in records])),
        average_steps=float(np.mean([r['episode_steps'] for r in records])),
        average_return=float(np.mean([r['episode_return'] for r in records])), records=records)


def load_checkpoint(path, env_config=None, algorithm_config=None, device='cuda'):
    require_cuda(device)
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get('checkpoint_family') != 'navigation3d-v1':
        raise RuntimeError('checkpoint is not an independent navigation3d model')
    extra = state['extra']
    trainer = NavigationTrainer(env_config or extra['environment_config'],
        algorithm_config or extra['algorithm_config'], device, extra['training_seed'])
    trainer.load(path)
    return trainer
