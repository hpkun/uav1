"""The unique combat task, baseline checkpoints, and CUDA integration contract."""
import copy
import inspect
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
import env
from env.combat_env import MultiUAVCombatEnv
from env.factory import make_combat_environment
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation, validate_checkpoint_for_resume
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.madsac.runner import MADSACTrainingRunner
from algorithm.madsac.protocol import validate_madsac_checkpoint
from algorithm.madsac.factory import build_madsac_trainer
from algorithm.madsac.evaluation import evaluate_madsac

ROOT = Path(__file__).resolve().parents[1]


def configs(algorithm='mappo'):
    return tuple(yaml.safe_load((ROOT / 'configs' / name).read_text(encoding='utf-8'))
                 for name in ('combat_environment.yaml', f'{algorithm}.yaml'))


def test_factory_and_public_api_are_combat_only():
    config, _ = configs()
    assert type(make_combat_environment()) is MultiUAVCombatEnv
    assert type(make_combat_environment(config)) is MultiUAVCombatEnv
    assert type(make_combat_environment(ROOT / 'configs/combat_environment.yaml')) is MultiUAVCombatEnv
    assert env.__all__ == ['AircraftSpec', 'AircraftState', 'ControlCommand', 'EngagementGeometry',
        'FireState', 'MultiUAVCombatEnv', 'WeaponEnvelope', 'engagement_geometry', 'make_combat_environment']
    assert sorted(p.name for p in (ROOT / 'configs').glob('*.yaml')) == [
        'combat_environment.yaml', 'combat_environment_v24.yaml', 'combat_environment_v25.yaml', 'combat_environment_v26.yaml', 'combat_environment_v27.yaml', 'combat_environment_v28.yaml', 'combat_environment_v29.yaml', 'combat_environment_v30.yaml', 'combat_environment_v31.yaml', 'combat_environment_v32.yaml', 'combat_environment_v33.yaml',
        'ea_mappo_5v5.yaml', 'ea_mappo_5v5_v30.yaml', 'ea_mappo_5v5_v31.yaml', 'ea_mappo_5v5_v32.yaml', 'ea_mappo_5v5_v33.yaml', 'ea_mappo_8v8.yaml',
        'maddpg_5v5.yaml', 'maddpg_8v8.yaml',
        'madsac.yaml', 'mappo.yaml', 'mappo_5v5.yaml', 'mappo_5v5_v30.yaml', 'mappo_5v5_v31.yaml', 'mappo_5v5_v32.yaml', 'mappo_5v5_v33.yaml', 'mappo_8v8.yaml',
        'mappo_8v8_formal.yaml', 'rmappo_5v5.yaml', 'rmappo_5v5_v30.yaml', 'rmappo_5v5_v31.yaml', 'rmappo_5v5_v32.yaml', 'rmappo_5v5_v33.yaml', 'rmappo_8v8.yaml',
        'stea_mappo.yaml', 'stea_mappo_5v5.yaml', 'stea_mappo_5v5_v30.yaml', 'stea_mappo_5v5_v31.yaml', 'stea_mappo_5v5_v32.yaml', 'stea_mappo_5v5_v33.yaml', 'stea_mappo_8v8.yaml',
        'stea_mappo_8v8_formal.yaml']
    assert list(inspect.signature(validate_checkpoint_for_evaluation).parameters) == [
        'state', 'env_config', 'algorithm_config']
    assert list(inspect.signature(evaluate_mappo_checkpoint).parameters) == [
        'checkpoint_path', 'algorithm_config', 'environment_config', 'device', 'evaluation_seeds']


def test_reset_reproducibility_and_environment_action_clipping():
    first, second = make_combat_environment(), make_combat_environment()
    assert np.array_equal(first.reset(77)[0], second.reset(77)[0])
    actions = np.tile([4, -5, 3], (4, 1)).astype(np.float32)
    actual, expected = first.step(actions), second.step(np.clip(actions, -1, 1))
    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    assert actual[0].shape == (4, 52)
    for name in ('r1', 'r2', 'r3', 'r4'):
        assert actual[4][f'{name}_rewards'].shape == (4,)
        assert np.isfinite(actual[4][f'{name}_rewards']).all()


def test_configuration_rejects_unsupported_task_and_observation_fields():
    config, _ = configs()
    config['unsupported_task'] = {}
    with pytest.raises(ValueError, match='unknown combat configuration fields'):
        make_combat_environment(config)
    config, _ = configs()
    config['observation']['unsupported_feature'] = True
    with pytest.raises(ValueError, match='unknown observation fields'):
        make_combat_environment(config)


@pytest.mark.parametrize('red_alive,blue_alive,reason', [
    (False, True, 'blue_win'), (True, False, 'red_win'), (False, False, 'draw_mutual_destruction')])
def test_elimination_outcomes(red_alive, blue_alive, reason):
    combat = make_combat_environment(); combat.reset(31)
    for aircraft in combat.red: aircraft.alive = red_alive
    for aircraft in combat.blue: aircraft.alive = blue_alive
    _, _, terminated, truncated, info = combat.step(np.zeros((4, 3)))
    assert terminated and not truncated
    assert info['termination_reason'] == reason


@pytest.mark.parametrize('algorithm', ['mappo', 'madsac'])
def test_cuda_training_checkpoint_and_evaluation_closed_loop(tmp_path, algorithm):
    if not torch.cuda.is_available():
        pytest.fail('CUDA required for baseline training/checkpoint audits')
    environment, config = configs(algorithm)
    environment['simulation']['max_steps'] = 2
    config['training']['evaluation_episodes'] = 2
    config['training']['evaluation_interval_sampled_steps'] = 16
    config['implementation']['checkpoint_interval_sampled_steps'] = 16
    runner_class = MAPPOTrainingRunner if algorithm == 'mappo' else MADSACTrainingRunner
    runner = runner_class(environment, config, num_envs=1, total_sampled_steps=32,
                         device='cuda', output_dir=tmp_path, smoke=True)
    summary = runner.run()
    checkpoint = tmp_path / 'latest.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert summary['sampled_steps'] == 32
    assert (tmp_path / 'best_eval.pt').is_file()
    assert state['extra']['environment_version'] == '2.3'
    expected_fields = ['environment_config_sha256', 'environment_version']
    if algorithm == 'madsac':
        expected_fields.insert(0, 'environment_config')
    assert sorted(key for key in state['extra'] if key.startswith('environment_')) == expected_fields
    seeds = range(10_000_000, 10_000_002)
    if algorithm == 'mappo':
        validate_checkpoint_for_resume(state, environment, config)
        result = evaluate_mappo_checkpoint(checkpoint, config, environment, 'cuda', seeds)
        changed = copy.deepcopy(environment); changed['simulation']['dt'] = .2
        with pytest.raises(RuntimeError, match='sha256'):
            validate_checkpoint_for_evaluation(state, changed, config)
    else:
        assert summary['actor_update_count'] > 0 and summary['critic_update_count'] > 0
        validate_madsac_checkpoint(state, environment, config)
        trainer = build_madsac_trainer(config, 'cuda', hidden_dim=32)
        trainer.load_for_evaluation(checkpoint)
        result = evaluate_madsac(trainer, environment, seeds)
        changed = copy.deepcopy(state); changed['extra']['environment_version'] = '0'
        with pytest.raises(RuntimeError, match='environment_version'):
            validate_madsac_checkpoint(changed, environment, config)
    assert result['evaluation_episodes'] == 2
    assert result['red_win_rate'] + result['blue_win_rate'] + result['draw_rate'] + result['timeout_rate'] == pytest.approx(1)
    for key in ('episode_return', 'episode_length', 'fire_attempts', 'weapon_hits',
                'red_losses', 'blue_losses', 'red_survivors', 'blue_survivors'):
        assert np.isfinite(result[key])
    assert result['red_survivors'] + result['red_losses'] == 4
    assert result['blue_survivors'] + result['blue_losses'] == 4
