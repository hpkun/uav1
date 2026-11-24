"""The reduced Actor input is isolated from existing Jia checkpoints."""

from dataclasses import replace
import json

import numpy as np
import pytest

from algorithm.maddpg_config import MADDPGConfig
from env.interception import InterceptionEnv
from experiment.interception import load_run, read_config, train
from tools.replay import collect, html, validate_trace


CONFIG = 'configs/interception_ideal_jia_low_level_obs24.yaml'


def test_reduced_actor_input_keeps_attacker_and_target_features():
    config, parameters, _ = read_config(CONFIG, 'configs/maddpg_low_level_interval10.yaml')
    old, _, _ = read_config('configs/interception_ideal_jia_straight_oob_exit.yaml')
    assert config == replace(old, visible_neighbors=0,
                             observation_layout='relative_assignment_target_velocity_no_defenders',
                             approach_weight=0.2)
    assert parameters.update_interval_steps == 10
    new_env, old_env = InterceptionEnv(config), InterceptionEnv(old)
    new_obs, new_info = new_env.reset(seed=7)
    old_obs, old_info = old_env.reset(seed=7)
    assert new_env.contract['version'] == 'interception_8'
    assert old_env.contract['version'] == 'interception_7'
    assert new_obs.shape == (5, 24)
    np.testing.assert_array_equal(new_obs[:, :4], old_obs[:, :4])
    np.testing.assert_array_equal(new_obs[:, 4:19], old_obs[:, 16:31])
    np.testing.assert_array_equal(new_obs[:, 19:], old_obs[:, 31:])
    np.testing.assert_array_equal(new_info['central_state'], old_info['central_state'])
    actions = np.zeros((5, 2))
    _, _, _, _, new_step = new_env.step(actions)
    _, _, _, _, old_step = old_env.step(actions)
    assert new_step['reward_components']['approach'] == pytest.approx(
        4 * old_step['reward_components']['approach'])
    for name in ('capture', 'penetration', 'heading', 'boundary'):
        assert new_step['reward_components'][name] == pytest.approx(old_step['reward_components'][name])


@pytest.mark.parametrize(('interval', 'expected_updates'), [(5, 20), (10, 10), (100, 1)])
def test_parallel_update_schedule_and_checkpoint(tmp_path, interval, expected_updates):
    config, _, _ = read_config(CONFIG)
    config = replace(config, episode_steps=2)
    parameters = MADDPGConfig(hidden_dim=16, batch_size=4, buffer_capacity=16,
                              learning_starts=4, update_interval_steps=interval,
                              updates_per_interval=1)
    output = tmp_path / f'interval_{interval}'
    train(config, parameters, 8, 'maddpg', steps=100, num_envs=4, seed=7,
          device='cpu', attack_mode='jia_straight', output=output,
          eval_interval=0, eval_episodes=1, log_every_steps=100)
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['updates'] == expected_updates
    rows = [json.loads(row) for row in (output / 'updates.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == expected_updates
    if interval == 10:
        restored, policy, _, _, _ = load_run(output, 'Last', 'cpu')
        assert policy.environment_spec['version'] == 'interception_8'
        trace = validate_trace(collect(restored, policy, 'jia_straight', 900001))
        assert trace['summary']['steps'] == 2
        assert '__DATA__' not in html(trace, 1)
