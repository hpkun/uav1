"""Record a deterministic combat episode from either baseline checkpoint."""
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
    TRACE_SCHEMA_VERSION, RecordingCombatEnv, append_frame, checkpoint_sha256,
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
    start = time.monotonic()
    while True:
        action = trainer.act(observation[None], env.red_alive_mask[None], deterministic=True)[0]
        observation, reward, terminated, truncated, info = env.step(action)
        transitions['red_actions'].append(info['executed_red_actions'])
        transitions['local_rewards'].append(reward)
        transitions['reward_components'].append(np.stack([info[f'{name}_rewards'] for name in ('r1', 'r2', 'r3', 'r4')]))
        transitions['terminated'].append(terminated)
        transitions['truncated'].append(truncated)
        append_frame(frames, env)
        if terminated or truncated:
            break
        if wall_timeout_s > 0 and time.monotonic() - start > wall_timeout_s:
            raise RuntimeError('recording exceeded wall timeout')
    shapes = write_trace(output_dir / 'episode_trace.npz', frames, transitions)
    result = dict(metadata or {})
    result.update({'trace_schema_version': TRACE_SCHEMA_VERSION, 'feature_names': FEATURE_NAMES,
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
    dump_metadata(output_dir / 'metadata.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--env-config', default='configs/combat_environment.yaml')
    parser.add_argument('--algorithm-config')
    parser.add_argument('--episode-seed', required=True, type=int)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--wall-timeout-s', type=float, default=0.0)
    args = parser.parse_args()
    if args.device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for checkpoint recording')
    checkpoint = resolved(args.checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    extra = state.get('extra', {})
    algorithm = str(state['algorithm']).lower()
    config_path = args.algorithm_config or f'configs/{algorithm}.yaml'
    config = yaml.safe_load(resolved(config_path).read_text(encoding='utf-8'))
    env_config = yaml.safe_load(resolved(args.env_config).read_text(encoding='utf-8'))
    if algorithm == 'mappo':
        validate_checkpoint_for_evaluation(state, env_config, config)
        if extra.get('algorithm_config_sha256') != config_sha256(config):
            raise RuntimeError('checkpoint algorithm config fingerprint mismatch')
        from algorithm.common.critic_protocol import checkpoint_widths
        actor_width, critic_width = checkpoint_widths(state)
        trainer = build_mappo_trainer(config, args.device, actor_width, critic_width)
        trainer.load(checkpoint)
    elif algorithm == 'madsac':
        validate_madsac_checkpoint(state, env_config, config, expected_training_seed=extra['training_seed'])
        trainer = build_madsac_trainer(config, args.device, extra['network_architecture']['hidden_dim'], extra['training_seed'])
        trainer.load_for_evaluation(checkpoint)
    else:
        raise ValueError(f'unsupported algorithm: {algorithm}')
    result = record(trainer, env_config, args.episode_seed, resolved(args.output_dir),
        {'algorithm': algorithm.upper(), 'checkpoint_path': str(checkpoint),
         'checkpoint_sha256': checkpoint_sha256(checkpoint),
         'checkpoint_sampled_steps': int(state['sampled_steps']),
         'algorithm_config_sha256': config_sha256(config)}, args.wall_timeout_s)
    print(result)


if __name__ == '__main__':
    main()
