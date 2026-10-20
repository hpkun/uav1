"""Strict deterministic IPPO evaluation with explicit seed provenance."""
from pathlib import Path
import torch
from algorithm.common.evaluator import evaluate
from algorithm.common.protocol import config_sha256
from .factory import build_ippo_trainer
from .protocol import require_cuda, validate_checkpoint

def load_checkpoint(path, algorithm_config, environment_config, device='cuda'):
    require_cuda(device)
    state = torch.load(path, map_location='cpu', weights_only=False)
    validate_checkpoint(state, environment_config, algorithm_config)
    trainer = build_ippo_trainer(algorithm_config, device)
    trainer.load(path, environment_config, algorithm_config)
    return trainer

def evaluate_ippo_checkpoint(checkpoint_path, algorithm_config, environment_config,
                             device, evaluation_seeds, split='development'):
    seeds = [int(s) for s in evaluation_seeds]
    if not seeds or any(b != a+1 for a,b in zip(seeds,seeds[1:])):
        raise ValueError('evaluation requires nonempty contiguous seeds')
    if split not in ('development','holdout'):
        raise ValueError('split must be development or holdout')
    if split == 'holdout' and seeds[0] < 70_000_000:
        raise ValueError('holdout seeds must be >=70000000')
    if split == 'development' and not 10_000_000 <= seeds[0] <= seeds[-1] < 20_000_000:
        raise ValueError('development seeds must be in [10000000,20000000)')
    trainer = load_checkpoint(checkpoint_path, algorithm_config, environment_config, device)
    result = evaluate(trainer, environment_config, seeds)
    result.update(algorithm='ippo', ippo_impl_version=1, checkpoint=str(Path(checkpoint_path).resolve()),
        checkpoint_sampled_steps=trainer.sampled_steps, evaluation_split=split,
        evaluation_seed_base=seeds[0], evaluation_seed_end=seeds[-1],
        evaluation_seeds=seeds, deterministic_policy=True, device=str(device),
        environment_config_sha256=config_sha256(environment_config),
        algorithm_config_sha256=config_sha256(algorithm_config), protocol_complete=True,
        parameter_sharing=True, critic_scope='local_observation')
    return result
