"""Boundary retirement is confined to the new interception contract."""

from dataclasses import replace

import numpy as np
import pytest

from algorithm.maddpg_config import MADDPGConfig
from env.interception import InterceptionEnv
from experiment.interception import load_run, read_config, train
from tools.replay import validate_trace


CONFIG = 'configs/interception_ideal_jia_straight_oob_exit.yaml'


def test_new_config_preserves_previous_jia_experiment():
    new, _, _ = read_config(CONFIG)
    old, _, _ = read_config('configs/interception_ideal_jia_straight_heading_approach005.yaml')
    assert new == replace(old, defender_oob_retire=True)
    assert InterceptionEnv(new).contract['version'] == 'interception_7'
    assert InterceptionEnv(old).contract['version'] == 'interception_6'


def test_out_of_bounds_defender_retires_once_and_reassigns():
    config, _, _ = read_config(CONFIG)
    env = InterceptionEnv(config)
    env.reset(seed=7)
    env.positions[0, 0] = [99.9, 0]
    env.velocities[0, 0] = [5, 0]
    initial_target = int(env.assignment.targets[0])
    _, rewards, done, truncated, info = env.step(np.zeros((5, 2)))
    assert not done and not truncated
    assert not env.defender_alive[0] and info['active_masks'][0] == 0
    assert env.velocities[0, 0].tolist() == [0, 0]
    assert info['reward_target_ids'][0] == initial_target
    assert env.assignment.targets[0] == -1
    assert env.attacker_alive.all()
    assert info['local_rewards'][0] == pytest.approx(-2)
    assert info['reward_components']['boundary'] == pytest.approx(-.4)
    assert sum(info['reward_components'].values()) == pytest.approx(info['team_reward'])
    position = env.positions[0, 0].copy()
    actions = np.zeros((5, 2))
    actions[0] = [2, 0]
    _, _, _, _, next_info = env.step(actions)
    np.testing.assert_array_equal(env.positions[0, 0], position)
    np.testing.assert_array_equal(next_info['executed_actions'][0, 0], [0, 0])
    assert next_info['reward_components']['boundary'] == 0
    assert env.defender_oob_count == 1


def test_replay_accepts_boundary_retirement():
    config, _, _ = read_config(CONFIG)
    env = InterceptionEnv(replace(config, episode_steps=1))
    env.reset(seed=7)
    env.positions[0, 0] = [99.9, 0]
    env.velocities[0, 0] = [5, 0]
    first = dict(step=0, defender_alive=env.defender_alive.tolist())
    _, _, done, _, info = env.step(np.zeros((5, 2)))
    assert done
    last = dict(step=1, defender_alive=env.defender_alive.tolist())
    trace = dict(frames=[first,last], events=info['events'], summary=info['episode_summary'],
                 config=env.contract['config'])
    assert validate_trace(trace) is trace
    assert trace['summary']['defender_retired_count'] == 1


def test_short_training_and_checkpoint_use_new_contract(tmp_path):
    config, _, _ = read_config(CONFIG)
    config = replace(config, episode_steps=2)
    parameters = MADDPGConfig(hidden_dim=16, batch_size=2, buffer_capacity=8,
                              learning_starts=2, update_interval_steps=2,
                              updates_per_interval=1)
    output = tmp_path / 'boundary'
    train(config, parameters, 2, 'maddpg', steps=4, num_envs=2, seed=7,
          device='cpu', attack_mode='jia_straight', output=output,
          eval_interval=4, eval_episodes=1, log_every_steps=4)
    restored, policy, _, _, _ = load_run(output, 'Last', 'cpu')
    assert restored.defender_oob_retire
    assert policy.environment_spec['version'] == 'interception_7'
