"""Reduced-scale radial scenarios preserve the five-slot MADDPG interface."""
from dataclasses import replace

import numpy as np
import pytest

from algorithm.maddpg_config import MADDPGConfig
from env import make_environment
from experiment.interception import load_run, read_config, train
from tools.replay import collect, html, validate_trace


@pytest.mark.parametrize('count', (1, 2))
def test_radial_scale_geometry_assignment_and_capture(count):
    config, _, _ = read_config(f'configs/interception_radial_{count}v{count}.yaml')
    env = make_environment(config)
    obs, info = env.reset(seed=7)
    assert env.contract['schema'] == 'Interception/12'
    assert (env.contract['active_defenders'], env.contract['n_attackers']) == (count, count)
    assert obs.shape == (5, 24) and info['central_state'].shape == (75,)
    np.testing.assert_array_equal(info['active_masks'], [1]*count+[0]*(5-count))
    np.testing.assert_array_equal(env.attacker_alive, [1]*count+[0]*(5-count))
    np.testing.assert_array_equal(env.assignment.targets, list(range(count))+[-1]*(5-count))
    np.testing.assert_array_equal(obs[count:], 0)
    np.testing.assert_array_equal(env.central_state()[25:50].reshape(5, 5)[:, 4],
                                  [1]*count+[0]*(5-count))
    attacker_angles = np.arctan2(env.positions[1, :count, 1], env.positions[1, :count, 0])
    defender_angles = np.arctan2(env.positions[0, :count, 1], env.positions[0, :count, 0])
    np.testing.assert_allclose(np.mod(defender_angles-attacker_angles, 2*np.pi), np.pi/9)
    if count == 2:
        assert abs(np.mod(attacker_angles[1]-attacker_angles[0], 2*np.pi)-np.pi) <= np.pi/30
    env.positions[0, :count] = env.positions[1, :count]
    _, individual, terminated, truncated, step_info = env.step(np.zeros((5, 2), np.float32))
    assert terminated and not truncated
    assert (step_info['episode_summary']['captured_count'],
            step_info['episode_summary']['penetrated_count'],
            step_info['episode_summary']['n_attackers']) == (count, 0, count)
    assert step_info['episode_summary']['defender_retired_count'] == count
    assert step_info['reward_components']['capture'] == 20*count
    assert step_info['global_reward'] == 20*count
    np.testing.assert_allclose(individual, step_info['individual_rewards'], atol=1e-5)
    assert np.isclose(step_info['team_reward'],
                      step_info['global_reward']+step_info['local_rewards'].sum())
    assert np.isclose(sum(step_info['reward_components'].values()), step_info['team_reward'])


@pytest.mark.parametrize('count', (1, 2))
def test_radial_scale_fixed_straight_attackers_breach_without_control(count):
    config, _, _ = read_config(f'configs/interception_radial_{count}v{count}.yaml')
    env = make_environment(config)
    env.reset(seed=17)
    start = env.positions[1, :count].copy()
    directions = env.attacker_course_directions[:count].copy()
    events = []
    penetration_reward = 0.0
    while not env.finished:
        _, _, _, _, info = env.step(np.zeros((5, 2), np.float32))
        events.extend(info['events'])
        penetration_reward += info['reward_components']['penetration']
        displacement = env.positions[1, :count] - start
        np.testing.assert_allclose(displacement[:, 0]*directions[:, 1] -
                                   displacement[:, 1]*directions[:, 0], 0, atol=1e-8)
    assert (env.captured_count, env.penetrated_count) == (0, count)
    assert sum(e['type']=='penetration' for e in events) == count
    assert penetration_reward == -50*count
    assert info['episode_summary']['global_return'] == -50*count


@pytest.mark.parametrize('count', (1, 2))
def test_radial_scale_short_training_checkpoint_and_replay(count, tmp_path):
    config, _, _ = read_config(f'configs/interception_radial_{count}v{count}.yaml')
    short = replace(config, episode_steps=2)
    parameters = MADDPGConfig(hidden_dim=16, batch_size=2, buffer_capacity=8,
                              learning_starts=2, update_interval_steps=2,
                              updates_per_interval=1)
    output = tmp_path / f'radial_{count}v{count}'
    train(short, parameters, 2, 'maddpg', steps=4, num_envs=1, seed=7,
          device='cpu', attack_mode='radial_straight', output=output,
          eval_interval=0, eval_episodes=1, log_every_steps=4)
    restored, policy, _, _, settings = load_run(output, 'Last', 'cpu')
    assert settings['environment_contract']['active_attackers'] == count
    assert policy.critics[0].net[0].in_features == 85
    trace = validate_trace(collect(restored, policy, 'radial_straight', 900001))
    assert sum(trace['frames'][0]['attacker_alive']) == count
    page = html(trace, 1)
    assert '中心型${cfg.active_pairs}向拦截回放' in page
    assert f'"active_pairs":{count}' in page
