"""Real CUDA updates, local masks, multiprocessing and strict identity reloads."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from env.navigation3d import load_config
from algorithm.mappo.trainer import MAPPOTrainer
from algorithm.ippo.trainer import IPPOTrainer
from algorithm.navigation3d.trainer import NavigationTrainer, validate_algorithm, require_cuda
from algorithm.navigation3d.runner import NavigationRunner
from algorithm.navigation3d.vector_env import NavigationVectorEnv
from algorithm.navigation3d.evaluation import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module', autouse=True)
def cuda_context():
    require_cuda()
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def configs(kind='ippo'):
    env = load_config(ROOT/'configs/navigation3d_v1.yaml')
    algorithm = yaml.safe_load((ROOT/f'configs/{kind}_navigation3d_v1.yaml').read_text(encoding='utf-8'))
    return env, algorithm


@pytest.mark.parametrize('kind', ['ippo', 'mappo'])
def test_historical_hyperparameters_unchanged(kind):
    _, c = configs(kind)
    historical = yaml.safe_load((ROOT/f'configs/{kind}_4v4_v341.yaml').read_text(encoding='utf-8'))
    assert c['training'] == historical['training']
    assert c['implementation'] == historical['implementation']
    assert c['network']['actor_hidden_layers'] == historical['network']['actor_hidden_layers']
    assert NavigationTrainer._ppo_update is IPPOTrainer._ppo_update


@pytest.mark.parametrize('kind', ['ippo', 'mappo'])
def test_real_cuda_update_and_exact_checkpoint_roundtrip(kind, tmp_path):
    e, a = configs(kind)
    runner = NavigationRunner(e, a, num_envs=2, seed=11)
    try:
        assert len(set(runner.vector.worker_pids)) == 2
        trainer = runner.trainer
        before_actor = {k: v.clone() for k, v in trainer.actor.state_dict().items()}
        before_critic = {k: v.clone() for k, v in trainer.critic.state_dict().items()}
        batch = runner.collect_rollout(8)
        assert batch.observations.shape == (8, 2, 4, 50)
        assert np.isfinite(batch.rewards).all()
        row = trainer.update(batch)
        assert np.isfinite(list(row.values())).all()
        assert row['actor_grad_norm'] > 0 and row['critic_grad_norm'] > 0
        assert trainer.actor_update_count > 0 and trainer.critic_update_count > 0
        for component, before in [('actor', before_actor), ('critic', before_critic)]:
            assert any(not torch.equal(before[k], getattr(trainer, component).state_dict()[k]) for k in before)
        obs, mask = runner.observations, runner.vector.current_alive_masks
        expected_actions = trainer.act(obs, mask, deterministic=True)
        expected_values = trainer.values(obs, mask)
        path = tmp_path/'navigation.pt'
        trainer.save(path)
        reloaded = load_checkpoint(path)
        assert np.array_equal(expected_actions, reloaded.act(obs, mask, deterministic=True))
        assert np.array_equal(expected_values, reloaded.values(obs, mask))
        assert reloaded.sampled_steps == 16 and reloaded.actor_update_count == trainer.actor_update_count
        assert len(reloaded.actor_optimizer.state) > 0 and len(reloaded.critic_optimizer.state) > 0
    finally:
        runner.close()


def test_ippo_locality_and_invalid_masks():
    e, a = configs()
    trainer = NavigationTrainer(e, a)
    obs = np.random.default_rng(1).normal(size=(2, 4, 50)).astype('f')
    mask = np.ones((2, 4), dtype='f')
    old = trainer.values(obs, mask)
    obs[:, 1:] += 10
    assert np.array_equal(old[:, 0], trainer.values(obs, mask)[:, 0])
    mask[:, 1] = 0
    assert not trainer.values(obs, mask)[:, 1].any()
    assert not trainer.act(obs, mask)[:, 1].any()


def test_vector_terminal_frames_and_autoreset_seed_isolation():
    e, _ = configs(); e['max_steps'] = 1
    with NavigationVectorEnv(2, e, 100) as vector:
        vector.reset()
        result = vector.step_batch(np.zeros((2, 4, 3), dtype='f'))
        assert result.truncated.all()
        assert not result.transition_next_observations.any()
        assert not result.next_alive_masks.any()
        assert vector.current_alive_masks.all() and result.observations.any()
        assert vector.used_training_seeds == {100, 101, 102, 103}
        assert [v['seed'] for v in result.infos] == [100, 101]


@pytest.mark.parametrize('field', ['environment_id', 'observation_schema', 'environment_config_sha256',
    'algorithm_config_sha256', 'network_architecture', 'actor_parameter_count', 'training_gamma'])
def test_checkpoint_identity_tampering_is_rejected(field, tmp_path):
    e, a = configs()
    trainer = NavigationTrainer(e, a)
    state = trainer.checkpoint_state()
    state['extra'][field] = 'tampered'
    path = tmp_path/'invalid.pt'; torch.save(state, path)
    with pytest.raises(RuntimeError, match=field):
        trainer.load(path)


def test_checkpoint_rejects_historical_family_and_different_configs(tmp_path):
    e, a = configs()
    trainer = NavigationTrainer(e, a)
    path = tmp_path/'nav.pt'; trainer.save(path)
    changed = deepcopy(e); changed['reward']['completion_bonus'] = 11.
    with pytest.raises(RuntimeError, match='environment_config_sha256'):
        NavigationTrainer(changed, a).load(path)
    state = trainer.checkpoint_state(); state['checkpoint_family'] = '3.41'; torch.save(state, path)
    with pytest.raises(RuntimeError, match='not an independent navigation3d'):
        load_checkpoint(path)


@pytest.mark.parametrize('mutation', ['gamma', 'dim', 'identity', 'cpu'])
def test_invalid_configuration_and_cpu_refused(mutation):
    e, a = configs()
    if mutation == 'cpu':
        with pytest.raises(RuntimeError, match='CUDA'):
            NavigationTrainer(e, a, device='cpu')
        return
    if mutation == 'gamma': a['training']['gamma'] = .98
    elif mutation == 'dim': a['network']['observation_dim'] = 45
    else: a['environment']['environment_id'] = '3.41'
    with pytest.raises(ValueError):
        validate_algorithm(a, e)
