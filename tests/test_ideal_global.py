"""The ideal baseline sees every aircraft and assigns at reset."""

from dataclasses import replace
import json

import numpy as np
import pytest

from algorithm.maddpg_config import MADDPGConfig
from env.interception import InterceptionConfig, InterceptionEnv
from experiment.interception import load_run, read_config, train


CONFIG = 'configs/interception_ideal_global_initial_heading_approach005.yaml'


def test_initial_matching_and_global_actor_visibility():
    config, _, _ = read_config(CONFIG)
    env = InterceptionEnv(config)
    observation, info = env.reset(seed=7)
    targets = info['assigned_target_ids']
    assert env.contract['version'] == info['version'] == 'interception_5'
    assert set(targets) == set(range(5))
    assert np.all(observation[:, [18, 21, 24, 27, 30]] == 1)
    assert np.all(observation[:, -1] == 1)
    for i, j in enumerate(targets):
        assert env.central_state()[50 + 5*i + j] == 1


def test_assignment_stays_until_resolution_event():
    config, _, _ = read_config(CONFIG)
    env = InterceptionEnv(config)
    env.reset(seed=7)
    initial = env.assignment.targets.copy()
    env.positions[1] = env.positions[1, ::-1].copy()
    targets, replanned = env._update_assignment(20)
    assert not replanned
    np.testing.assert_array_equal(targets, initial)
    i, j = 0, int(initial[0])
    env.defender_alive[i] = False
    env.attacker_alive[j] = False
    targets, replanned = env._update_assignment(21, [{'type': 'capture'}])
    assert replanned and targets[i] == -1
    assert set(targets[targets >= 0]) == set(range(5)) - {j}


def test_approach_reward_uses_pre_step_assigned_target():
    config, _, _ = read_config(CONFIG)
    env = InterceptionEnv(config)
    env.reset(seed=7)
    j = int(env.assignment.targets[0])
    env.defender_alive[1:] = False
    env.attacker_alive[:] = False
    env.attacker_alive[j] = True
    env.assignment.targets[1:] = -1
    env.positions[0, 0] = [0, -20]
    env.positions[1, j] = [0, 20]
    env.velocities.fill(0)
    env.velocities[0, 0] = [0, 1]
    _, _, _, _, info = env.step(np.zeros((5, 2)))
    assert info['reward_target_ids'][0] == j
    assert info['reward_components']['approach'] == pytest.approx(.05 * .1 / 5)
    assert info['reward_components']['heading'] == pytest.approx(.03 / 5)
    assert sum(info['reward_components'].values()) == pytest.approx(info['team_reward'])


def test_ideal_requires_full_arena_sensing():
    with pytest.raises(ValueError, match='full-arena'):
        InterceptionConfig(assignment_method='ideal_global_distance_matching',
                           observation_layout='relative_assignment_target_velocity',
                           sensing_radius=100)


def test_ideal_short_training_and_checkpoint(tmp_path):
    config, _, _ = read_config(CONFIG)
    config = replace(config, episode_steps=2)
    parameters = MADDPGConfig(hidden_dim=16, batch_size=2, buffer_capacity=8,
                              learning_starts=2, update_interval_steps=2,
                              updates_per_interval=1)
    output = tmp_path / 'ideal'
    train(config, parameters, 2, 'maddpg', steps=4, num_envs=2, seed=7,
          device='cpu', attack_mode='direct_rush', output=output,
          eval_interval=4, eval_episodes=1, log_every_steps=4)
    evaluation = json.loads((output / 'evaluations.jsonl').read_text(encoding='utf-8').strip())
    assert set(evaluation['modes']['direct_rush']['mean_reward_components']) >= {'approach', 'heading'}
    restored, policy, _, _, _ = load_run(output, 'Last', 'cpu')
    assert restored.assignment_method == 'ideal_global_distance_matching'
    env = InterceptionEnv(restored)
    obs, info = env.reset(seed=8)
    assert policy.policy_actions(obs[None], info['active_masks'][None]).shape == (1, 5, 2)
