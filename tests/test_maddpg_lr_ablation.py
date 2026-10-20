"""Controlled v3.39 LR screening: config parity, CUDA optimizers and provenance."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from algorithm.common.protocol import config_sha256
from algorithm.maddpg.factory import build_maddpg_trainer
from algorithm.maddpg.noise import exploration_scale
from algorithm.maddpg.protocol import require_cuda, validate_config, validate_checkpoint
from algorithm.maddpg.runner import MADDPGTrainingRunner

ROOT = Path(__file__).resolve().parents[1]
CASES = [('a1', 1e-4, 3e-4), ('a2', 3e-4, 1e-4), ('a3', 3e-4, 3e-4)]


def load(name):
    return yaml.safe_load((ROOT / 'configs' / name).read_text(encoding='utf-8'))


@pytest.mark.parametrize('name,actor_lr,critic_lr', CASES)
def test_only_learning_rates_and_metadata_differ(name, actor_lr, critic_lr):
    baseline = load('maddpg_4v4_v339.yaml')
    config = load(f'maddpg_4v4_v339_lr_{name}.yaml')
    validate_config(load('combat_environment_v339.yaml'), config)
    assert config['training']['actor_learning_rate'] == actor_lr
    assert config['training']['critic_learning_rate'] == critic_lr
    metadata = config['metadata']['lr_ablation']
    assert metadata['experiment_name'] == f'maddpg_v339_lr_{name}'
    assert metadata['baseline_config'] == 'configs/maddpg_4v4_v339.yaml'
    assert metadata['actor_learning_rate'] == actor_lr
    assert metadata['critic_learning_rate'] == critic_lr
    assert f'actor_lr={actor_lr:.4f}' in config['metadata']['project_training_choice']
    assert f'critic_lr={critic_lr:.4f}' in config['metadata']['project_training_choice']
    normalized = deepcopy(config)
    normalized['metadata'] = baseline['metadata']
    for key in ('actor_learning_rate', 'critic_learning_rate'):
        normalized['training'][key] = baseline['training'][key]
    assert normalized == baseline
    assert config['training']['total_sampled_steps'] == 3_000_000
    assert config['implementation']['exploration_decay_sampled_steps'] == 3_000_000
    hashes = {config_sha256(load(f'maddpg_4v4_v339_lr_{n}.yaml')) for n, _, _ in CASES}
    assert len(hashes | {config_sha256(baseline)}) == 4


@pytest.mark.parametrize('name,actor_lr,critic_lr', CASES)
def test_all_cuda_optimizer_groups_use_requested_rates(name, actor_lr, critic_lr):
    require_cuda('cuda')
    config = load(f'maddpg_4v4_v339_lr_{name}.yaml')
    trainer = build_maddpg_trainer(config, 'cuda', 1)
    assert next(trainer.actors.parameters()).is_cuda
    assert len(trainer.actor_optimizers) == len(trainer.critic_optimizers) == 4
    for optimizers, rate in [(trainer.actor_optimizers, actor_lr),
                             (trainer.critic_optimizers, critic_lr)]:
        assert all(group['lr'] == rate for opt in optimizers for group in opt.param_groups)


@pytest.mark.parametrize('name,actor_lr,critic_lr', CASES)
def test_1m_budget_keeps_config_ou_evaluation_and_checkpoint_identity(
        name, actor_lr, critic_lr, tmp_path, monkeypatch):
    require_cuda('cuda')
    # Isolate constructor/provenance checks; real workers run in the CLI CUDA checks.
    class VectorStub:
        backend = 'test_stub'
        worker_pids = []

        def __init__(self, num_envs, config, seed, forbidden_seeds):
            self.current_alive_masks = np.ones((num_envs, 4), np.float32)
            self.episode_indices = np.zeros(num_envs, np.int64)

        def reset(self):
            return np.zeros((16, 4, 45), np.float32)

        def close(self):
            pass

    monkeypatch.setattr('algorithm.maddpg.runner.ParallelVectorEnv', VectorStub)
    env = load('combat_environment_v339.yaml')
    config = load(f'maddpg_4v4_v339_lr_{name}.yaml')
    original = deepcopy(config)
    output = tmp_path / f'maddpg_v339_lr_{name}_seed1_1m'
    runner = MADDPGTrainingRunner(env, config, 16, 1_000_000, 'cuda', 1, output)
    try:
        assert config == original == runner.algorithm_config == runner.requested_algorithm_config
        startup = runner.startup_summary()
        assert startup['total_sampled_steps'] == 1_000_000
        assert startup['effective_training']['total_sampled_steps'] == 3_000_000
        assert startup['algorithm_config_sha256'] == config_sha256(config)
        assert startup['requested_algorithm_config_sha256'] == config_sha256(config)
        assert 'steps=1.0M' in runner.startup_log_line()
        assert runner.evaluation_seeds == tuple(range(10_000_000, 10_000_020))
        assert runner.algorithm_config['implementation']['evaluation_mode'] == 'deterministic'
        runner.trainer.sampled_steps = 1_000_000
        assert runner.current_exploration_scale() == pytest.approx(.7)
        assert runner.current_exploration_scale() == exploration_scale(1_000_000, 1., .1, 3_000_000)
        runner.trainer.sampled_steps = 0
        state = runner.trainer.checkpoint_state(runner.checkpoint_extra())
        validate_checkpoint(state, env, config)
        assert state['extra']['training_total_sampled_steps'] == 1_000_000
        for other in ['maddpg_4v4_v339.yaml'] + [
                f'maddpg_4v4_v339_lr_{n}.yaml' for n, _, _ in CASES if n != name]:
            with pytest.raises(RuntimeError, match='algorithm_config_sha256 mismatch'):
                validate_checkpoint(state, env, load(other))
        with pytest.raises(FileExistsError, match='non-empty'):
            MADDPGTrainingRunner(env, config, 16, 1_000_000, 'cuda', 1, output)
    finally:
        runner.vector.close()
