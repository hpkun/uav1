"""The isolated one-on-one task must resolve and reward exactly one aircraft pair."""
from dataclasses import replace
import numpy as np
import pytest

from env import make_environment
from experiment.interception import read_config, summarize


CONFIG = 'configs/interception_one_vs_one_jia.yaml'


def _environment(seed=7):
    config, _, _ = read_config(CONFIG)
    env = make_environment(config)
    observation, info = env.reset(seed=seed)
    return env, observation, info


def test_one_vs_one_reset_and_assignment():
    env, observation, info = _environment()
    assert env.contract['schema'] == 'Interception/10'
    assert observation.shape == (5, 24)
    assert info['central_state'].shape == (75,)
    np.testing.assert_array_equal(info['active_masks'], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(env.attacker_alive, [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(env.assignment.targets, [0, -1, -1, -1, -1])
    target = np.array([env.config.base_x, env.config.base_y + env.config.breach_radius])
    direction = target - env.positions[1, 0]
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(env.attacker_course_directions[0], direction)
    assert np.count_nonzero(observation[1:]) == 0


def test_one_vs_one_capture_and_reward():
    env, _, _ = _environment()
    env.positions[0, 0] = env.positions[1, 0]
    _, rewards, terminated, truncated, info = env.step(np.zeros((5, 2)))
    assert terminated and not truncated
    summary = info['episode_summary']
    assert summary['outcome'] == 'all_captured'
    assert (summary['captured_count'], summary['penetrated_count'], summary['n_attackers']) == (1, 0, 1)
    assert info['reward_components']['capture'] == 20.0
    assert info['reward_components']['penetration'] == 0.0
    np.testing.assert_allclose(info['team_reward'], rewards[0])
    np.testing.assert_allclose(sum(info['reward_components'].values()), summary['team_return'])
    assert summarize([summary])['capture_efficiency'] == 1.0


def test_one_vs_one_penetration_and_reward():
    env, _, _ = _environment()
    env.positions[0, 0] = [0.0, -95.0]
    env.positions[1, 0] = [0.0, env.config.base_y + env.config.breach_radius + 0.1]
    env.velocities[1, 0] = [0.0, -5.0]
    _, rewards, terminated, truncated, info = env.step(np.zeros((5, 2)))
    assert terminated and not truncated
    summary = info['episode_summary']
    assert summary['outcome'] == 'penetrated'
    assert (summary['captured_count'], summary['penetrated_count']) == (0, 1)
    assert info['reward_components']['penetration'] == -100.0
    np.testing.assert_allclose(info['team_reward'], rewards[0])
    np.testing.assert_allclose(sum(info['reward_components'].values()), summary['team_return'])
    assert summarize([summary])['capture_efficiency'] == 0.0


def test_target_velocity_views_change_only_assigned_velocity_slots():
    baseline, _, _ = read_config('configs/interception_one_vs_one_jia_no_heading.yaml')
    observations = {}
    states = {}
    environments = {}
    for view in ('relative', 'absolute', 'none'):
        env = make_environment(replace(baseline, target_velocity_view=view))
        observation, info = env.reset(seed=17)
        environments[view] = env
        observations[view] = observation
        states[view] = info['central_state']
        assert observation.shape == (5, 24)
        assert env.contract['observation_dim'] == 24
        assert env.contract['state_dim'] == 75
    relative = observations['relative']
    absolute = observations['absolute']
    absent = observations['none']
    np.testing.assert_array_equal(states['relative'], states['absolute'])
    np.testing.assert_array_equal(states['relative'], states['none'])
    for other in (absolute, absent):
        np.testing.assert_array_equal(relative[:, :-3], other[:, :-3])
        np.testing.assert_array_equal(relative[:, -1], other[:, -1])
    defender_velocity = environments['relative'].velocities[0, 0]
    attacker_velocity = environments['relative'].velocities[1, 0]
    np.testing.assert_allclose(relative[0, -3:-1], (attacker_velocity-defender_velocity)/10)
    np.testing.assert_allclose(absolute[0, -3:-1], attacker_velocity/10)
    np.testing.assert_array_equal(absent[:, -3:-1], 0)
    np.testing.assert_array_equal(relative[1:], 0)


def test_target_velocity_view_rejects_unknown_value():
    baseline, _, _ = read_config('configs/interception_one_vs_one_jia_no_heading.yaml')
    with pytest.raises(ValueError, match='target velocity view'):
        replace(baseline, target_velocity_view='other')
