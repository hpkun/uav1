"""The assigned-target heading term is isolated from historical rewards."""

from dataclasses import replace
import json

import numpy as np
import pytest

from env.interception import InterceptionConfig, InterceptionEnv
from algorithm.maddpg_config import MADDPGConfig
from experiment.interception import load_run, read_config, train
from experiment.prioritized_replay import ReplayConfig


def _setup(weight, *, target_y=20.0, target_assigned=True):
    base, _, _ = read_config('configs/interception_huang_global_target_velocity_heading.yaml')
    env = InterceptionEnv(replace(base, heading_weight=weight))
    env.reset(seed=7)
    env.positions[0, 0] = [0, -20]
    env.positions[1, 0] = [0, target_y]
    env.velocities[0, 0] = [0, 0]
    env.velocities[1, 0] = [0, 0]
    env.assignment.targets[:] = -1
    if target_assigned:
        env.assignment.targets[0] = 0
    return env


@pytest.mark.parametrize('acceleration,expected_cosine', ((2.0, 1.0), (-2.0, -1.0)))
def test_heading_uses_executed_velocity_and_logs_team_mean(acceleration, expected_cosine):
    new = _setup(0.3)
    old = _setup(0.0)
    actions = np.zeros((5, 2))
    actions[0, 1] = acceleration
    _, new_rewards, _, _, new_info = new.step(actions)
    _, old_rewards, _, _, old_info = old.step(actions)
    assert new.contract['version'] == 'interception_4'
    assert old.contract['version'] == 'interception_3'
    assert new_info['reward_components']['heading'] == pytest.approx(0.3 * expected_cosine / 5)
    assert new_rewards[0] - old_rewards[0] == pytest.approx(0.3 * expected_cosine, abs=1e-6)
    np.testing.assert_allclose(new_rewards[1:], old_rewards[1:])
    assert 'heading' not in old_info['reward_components']
    assert sum(new_info['reward_components'].values()) == pytest.approx(new_info['team_reward'])


@pytest.mark.parametrize('assigned,target_y', ((False, 20.0), (True, 90.0)))
def test_heading_requires_visible_assigned_target(assigned, target_y):
    env = _setup(0.3, target_y=target_y, target_assigned=assigned)
    actions = np.zeros((5, 2))
    actions[0, 1] = 2.0
    _, _, _, _, info = env.step(actions)
    assert info['reward_components']['heading'] == 0.0


def test_capture_step_uses_velocity_before_retirement():
    env = _setup(0.3, target_y=-17.0)
    env.attacker_alive[1:] = False
    env.defender_alive[1:] = False
    actions = np.zeros((5, 2))
    actions[0, 1] = 2.0
    _, _, terminated, _, info = env.step(actions)
    assert terminated
    assert not env.defender_alive[0]
    assert info['reward_components']['heading'] == pytest.approx(0.3 / 5)


def test_heading_requires_assignment_layout():
    with pytest.raises(ValueError, match='assigned-target'):
        InterceptionConfig(heading_weight=0.3)


@pytest.mark.parametrize('replay_mode', ('uniform', 'critic_per_with_uniform'))
def test_heading_training_and_checkpoint_with_both_replay_modes(tmp_path, replay_mode):
    config, _, _ = read_config('configs/interception_huang_global_target_velocity_heading.yaml')
    config = replace(config, episode_steps=2)
    parameters = MADDPGConfig(hidden_dim=16, batch_size=2, buffer_capacity=8,
                              learning_starts=2, update_interval_steps=2,
                              updates_per_interval=1)
    output = tmp_path / replay_mode
    train(config, parameters, 2, 'maddpg', steps=4, num_envs=2, seed=7,
          device='cpu', attack_mode='direct_rush', output=output,
          eval_interval=4, eval_episodes=1, log_every_steps=4,
          replay_config=ReplayConfig(mode=replay_mode))
    settings = json.loads((output / 'settings.json').read_text(encoding='utf-8'))
    assert settings['environment_contract']['version'] == 'interception_4'
    evaluation = json.loads((output / 'evaluations.jsonl').read_text(encoding='utf-8').strip())
    assert 'heading' in evaluation['modes']['direct_rush']['mean_reward_components']
    restored, policy, name, _, _ = load_run(output, 'Last', 'cpu')
    assert name == 'maddpg' and restored.heading_weight == 0.03
    env = InterceptionEnv(restored)
    observations, info = env.reset(seed=8)
    assert policy.policy_actions(observations[None], info['active_masks'][None]).shape == (1, 5, 2)
