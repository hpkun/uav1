"""Internal deterministic recorder. User CLI: tools/replay_combat.py."""
import argparse
from pathlib import Path
from typing import Any
import sys
import time
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
from algorithm.common.protocol import config_sha256
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.madsac.factory import build_madsac_trainer
from algorithm.madsac.protocol import validate_madsac_checkpoint
from tools.combat_visualization import (FEATURE_NAMES, FRAME_FIELDS, TRANSITION_FIELDS,
    TRACE_SCHEMA_VERSION, COOLDOWN_FIELDS, RecordingCombatEnv, append_frame, checkpoint_sha256,
    dump_metadata, ensure_fresh_output, write_trace)


def resolved(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def record(trainer: Any, env_config: dict[str, Any], seed: int, output_dir: str | Path,
           metadata: dict[str, Any] | None = None, wall_timeout_s: float = 0) -> dict[str, Any]:
    output_dir = Path(output_dir)
    ensure_fresh_output(output_dir)
    env = RecordingCombatEnv(env_config)
    observation, _ = env.reset(seed)
    frames = {key: [] for key in FRAME_FIELDS}
    transitions = {key: [] for key in TRANSITION_FIELDS}
    append_frame(frames, env)
    initial_ammo = {side: getattr(env,side+'_ammo',None) for side in ('red','blue')}
    initial_ammo = {k:None if v is None else np.asarray(v).tolist() for k,v in initial_ammo.items()}
    has_cooldown = all(hasattr(env,side+'_fire_cooldown_steps') for side in ('red','blue'))
    if has_cooldown:
        transitions.update({name:[] for name in COOLDOWN_FIELDS})
    targets_available = None
    start = time.monotonic()
    recurrent = hasattr(getattr(trainer, 'actor', None), 'gru_hidden_dim')
    if recurrent:
        hidden = np.zeros((env.team_size, trainer.actor.gru_hidden_dim), dtype=np.float32)
        episode_start = np.array(1., dtype=np.float32)
    while True:
        if recurrent:
            action, _, _, hidden = trainer.act(observation, env.red_alive_mask,
                hidden, episode_start, deterministic=True)
        else:
            action = trainer.act(observation[None], env.red_alive_mask[None], deterministic=True)[0]
        observation, reward, terminated, truncated, info = env.step(action)
        if recurrent:
            hidden *= env.red_alive_mask[:, None]
            episode_start = np.array(0., dtype=np.float32)
        transitions['red_actions'].append(info['executed_red_actions'])
        transitions['blue_actions'].append(np.asarray(info.get('executed_blue_actions',env.observed_actions.get('blue'))).copy())
        available = 'blue_pursuit_target_indices' in info
        if targets_available is not None and available != targets_available:
            raise RuntimeError('Blue target availability changed during recording')
        targets_available = available
        transitions['blue_pursuit_target_indices'].append(np.asarray(info['blue_pursuit_target_indices'],dtype=np.int64).copy() if available else np.full(env.team_size,-1,dtype=np.int64))
        for side in ('red','blue'):
            ammo = info.get(side+'_ammo',getattr(env,side+'_ammo',None))
            transitions[side+'_ammo'].append(np.full(env.team_size,-1,dtype=np.int64) if ammo is None else np.asarray(ammo,dtype=np.int64).copy())
            if has_cooldown:transitions[side+'_cooldown'].append(getattr(env,side+'_fire_cooldown_steps').copy())
        transitions['local_rewards'].append(np.asarray(reward).copy())
        transitions['reward_components'].append(np.stack([info[f'{name}_rewards'] for name in ('r1', 'r2', 'r3', 'r4')]))
        transitions['terminated'].append(terminated)
        transitions['truncated'].append(truncated)
        append_frame(frames, env)
        if terminated or truncated:
            if recurrent:
                hidden.fill(0.)
            break
        if wall_timeout_s > 0 and time.monotonic() - start > wall_timeout_s:
            raise RuntimeError('recording exceeded wall timeout')
    shapes = write_trace(output_dir / 'episode_trace.npz', frames, transitions, dt=env.dt)
    result = dict(metadata or {})
    result.update({'trace_schema_version': TRACE_SCHEMA_VERSION, 'feature_names': FEATURE_NAMES,
         'reward_component_names': ['R1','R2','R3','R4'], 'reward_component_axis': 1, 'agent_axis': 2,
        'transition_state_alignment': 'transition[t]: frame[t] -> frame[t+1]; ammo/cooldown are post-transition',
        'blue_target_source': 'info.blue_pursuit_target_indices' if targets_available else 'unavailable_in_this_environment',
        'blue_target_available': bool(targets_available), 'initial_ammo': initial_ammo,
        'cooldown_units': 'simulation_steps', 'cooldown_available': has_cooldown,
        'weapon_config': env.config['weapon'], 'arena_config': env.config['arena'],
        'weapon_class': type(env.weapon).__name__,
        'episode_seed': int(seed), 'dt': env.dt, 'max_steps': env.max_steps,
        'arena_radius': env.arena_radius, 'environment_version': env.environment_version,
        'environment_config_sha256': config_sha256(env_config),
        'deterministic_policy': True, 'episode_steps': env.steps,
        'simulation_time_s': env.steps * env.dt, 'termination_reason': info['termination_reason'],
        'red_win': info['red_win'], 'blue_win': info['blue_win'], 'draw': info['draw'],
        'red_survivors': info['red_survivors'], 'blue_survivors': info['blue_survivors'],
        'episode_return': float(np.asarray(transitions['local_rewards']).sum()),
        'events': env.events, 'trace_shapes': shapes,
        **{f'{side}_{event}': info[f'{side}_{event}'] for side in ('red', 'blue')
           for event in ('fire_attempts', 'weapon_hits', 'attack_kills', 'boundary_exits', 'ground_losses')},
        **{f'episode_{name}_total': info[f'episode_{name}_total'] for name in ('r1', 'r2', 'r3', 'r4')},
    })
    result['combat_counts'] = {side:dict(env.combat_counts[side]) for side in ('red','blue')}
    for side in ('red','blue'):
        for field in ('ammo_used','ceiling_losses'):
            result[side+'_'+field] = env.combat_counts[side].get(field)
    from tools.combat_trace import read_trace
    read_trace(output_dir / 'episode_trace.npz',result)
    dump_metadata(output_dir / 'metadata.json', result)
    return result


def load_recording_policy(checkpoint, env_config, config, device='cuda'):
    """Use official validation, factory and load; no alternate actor implementation."""
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for checkpoint recording')
    checkpoint = Path(checkpoint)
    original_hash = checkpoint_sha256(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    extra = state.get('extra', {})
    algorithm = str(state['algorithm']).lower()
    if state.get('checkpoint_family') in ('eaa_mappo_v338', 'eaa_mappo_v339', 'eaa_mappo_v341'):
        from algorithm.eaa_mappo.evaluation import load_checkpoint
        trainer = load_checkpoint(checkpoint, config, env_config, device)
    elif algorithm == 'matd3':
        from algorithm.matd3.protocol import validate_checkpoint
        from algorithm.matd3.factory import build_matd3_trainer
        validate_checkpoint(state, env_config, config)
        trainer = build_matd3_trainer(config, device, extra['training_seed'])
        trainer.load(checkpoint, env_config, config)
    elif algorithm == 'maddpg':
        from algorithm.maddpg.protocol import validate_checkpoint, require_cuda
        from algorithm.maddpg.factory import build_maddpg_trainer
        require_cuda(device)
        validate_checkpoint(state, env_config, config)
        trainer = build_maddpg_trainer(config, device, extra['training_seed'])
        trainer.load(checkpoint, env_config, config)
    elif algorithm == 'mappo':
        validate_checkpoint_for_evaluation(state, env_config, config)
        if extra.get('algorithm_config_sha256') != config_sha256(config):
            raise RuntimeError('checkpoint algorithm config fingerprint mismatch')
        from algorithm.common.critic_protocol import checkpoint_widths
        actor_width, critic_width = checkpoint_widths(state)
        trainer = build_mappo_trainer(config, device, actor_width, critic_width)
        trainer.load(checkpoint)
    elif algorithm == 'maac':
        from algorithm.maac.protocol import validate_checkpoint
        from algorithm.maac.trainer import MAACTrainer
        validate_checkpoint(state, env_config, config)
        trainer = MAACTrainer(env_config, config, device, state['training_seed'], extra['training_smoke'])
        trainer.load(checkpoint)
    elif algorithm == 'madsac':
        validate_madsac_checkpoint(state, env_config, config, expected_training_seed=extra['training_seed'])
        trainer = build_madsac_trainer(config, device, extra['network_architecture']['hidden_dim'], extra['training_seed'])
        trainer.load_for_evaluation(checkpoint)
    else:
        import importlib
        modules = {'rmappo': 'rmappo', 'ea-mappo': 'ea_mappo', 'stea-mappo': 'stea_mappo'}
        if algorithm not in modules:
            raise ValueError(f'unsupported algorithm: {algorithm}')
        name = modules[algorithm]
        protocol = importlib.import_module(f'algorithm.{name}.protocol')
        factory = importlib.import_module(f'algorithm.{name}.factory')
        protocol.validate_checkpoint(state, env_config, config)
        trainer = getattr(factory, f'build_{name}_trainer')(config, device,
            seed=extra['training_seed'], smoke=bool(extra['training_smoke']))
        trainer.load(checkpoint)
    if 'environment_config_sha256' in extra and extra['environment_config_sha256'] != config_sha256(env_config):
        raise RuntimeError('checkpoint environment config fingerprint mismatch')
    if extra.get('algorithm_config_sha256') != config_sha256(config):
        raise RuntimeError('checkpoint algorithm config fingerprint mismatch')
    if checkpoint_sha256(checkpoint) != original_hash:
        raise RuntimeError('checkpoint changed during load; use a stable checkpoint snapshot')
    return trainer, state, {'algorithm': state['algorithm'], 'checkpoint_path': str(checkpoint),
        'checkpoint_sha256': original_hash,
        'environment_config_sha256': config_sha256(env_config),
        'checkpoint_sampled_steps': int(state['sampled_steps']),
        'training_seed': extra.get('training_seed',state.get('training_seed')),
        'observation_dim': extra.get('observation_dim'), 'action_dim': extra.get('action_dim'),
        'num_agents': extra.get('num_agents'),
        'algorithm_config_sha256': config_sha256(config)}



if __name__ == '__main__':
    raise SystemExit('Internal module. Use the official entry: python -u tools/replay_combat.py --help')
