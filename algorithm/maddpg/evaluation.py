"""Noise-free local actors; reuse existing combat metric definitions."""
import torch
from algorithm.common.evaluator import evaluate as evaluate_combat
from .protocol import require_cuda, validate_checkpoint
from .factory import build_maddpg_trainer


def validate_seeds(seeds, extra=None):
    seeds = tuple(int(s) for s in seeds)
    if not seeds or any(b != a+1 for a, b in zip(seeds, seeds[1:])):
        raise ValueError('evaluation seeds must be nonempty, increasing and contiguous')
    if seeds[0] < 10_000_000 or seeds[-1] >= 70_000_000:
        raise ValueError('use development seeds below the reserved 70M holdout')
    if extra:
        start = extra['training_seed']
        end = start+extra['training_total_sampled_steps']+extra['training_num_envs']
        if any(start <= seed <= end for seed in seeds):
            raise ValueError('evaluation seeds overlap training scenarios')
    return seeds


def evaluate(trainer, env_config, seeds):
    require_cuda(trainer.device)
    return evaluate_combat(trainer, env_config, validate_seeds(seeds))


def evaluate_checkpoint(path, env_config, config, device, seeds):
    require_cuda(device)
    state = torch.load(path, map_location='cpu', weights_only=False)
    extra = validate_checkpoint(state, env_config, config)
    seeds = validate_seeds(seeds, extra)
    trainer = build_maddpg_trainer(config, device, extra['training_seed'])
    trainer.load(path, env_config, config)
    return {'algorithm': 'maddpg', 'implementation_version': state['implementation_version'],
        'checkpoint': str(path), 'checkpoint_sampled_steps': state['sampled_steps'],
        'training_seed': extra['training_seed'], 'environment_version': extra['environment_version'],
        'environment_config_sha256': extra['environment_config_sha256'], 'algorithm_config_sha256': extra['algorithm_config_sha256'],
        'network_architecture': state['network_architecture'], 'parameter_counts': extra['parameter_counts'],
        'evaluation_mode': 'deterministic', 'evaluation_seed_base': seeds[0], 'evaluation_seed_end': seeds[-1],
        'protocol_complete': True, **evaluate(trainer, env_config, seeds)}
