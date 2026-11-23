"""The smaller Actor observation preserves full centralized training state."""

from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from algorithm.mappo_config import MAPPOConfig
from algorithm.maddpg_config import MADDPGConfig
from env.interception import InterceptionEnv
from experiment.interception import load_run, read_config, train


def test_actor_slots_and_critic_state_are_separate():
    old, _, _ = read_config('configs/interception_huang_global.yaml')
    new, _, _ = read_config('configs/interception_huang_global_target_velocity.yaml')
    old_config = asdict(old)
    new_config = asdict(new)
    old_config.pop('observation_layout')
    new_config.pop('observation_layout')
    assert old_config == new_config
    old_env, new_env = InterceptionEnv(old), InterceptionEnv(new)
    old_env.reset(seed=7)
    new_env.reset(seed=7)
    for env in (old_env, new_env):
        env.defender_alive[2:] = False
        env.attacker_alive[1:] = False
        env.positions[0, 0] = [0, -20]
        env.positions[0, 1] = [10, -20]
        env.positions[1, 0] = [0, 30]
        env.velocities[0, 0] = [1, 2]
        env.velocities[0, 1] = [-1, 3]
        env.velocities[1, 0] = [0, -2]
        env.assignment.targets[0] = 0
    observation = new_env.observations()[0]
    assert new_env.contract['schema'] == 'Interception/3'
    assert new_env.contract['version'] == 'interception_3'
    assert observation.shape == (36,)
    np.testing.assert_allclose(observation[:4], [0, -.2, .2, .4])
    np.testing.assert_allclose(observation[4:7], [.1, 0, 1])
    np.testing.assert_allclose(observation[16:19], [0, .5, 1])
    np.testing.assert_allclose(observation[-5:], [0, .5, -.1, -.4, 1])
    np.testing.assert_array_equal(new_env.central_state(), old_env.central_state())
    assert new_env.central_state().shape == (75,)
    assert old_env.observations().shape == (5, 54)


def test_actor_ablation_keeps_assignment_rewards_and_dynamics():
    old, _, _ = read_config('configs/interception_huang_global.yaml')
    new, _, _ = read_config('configs/interception_huang_global_target_velocity.yaml')
    old_env, new_env = InterceptionEnv(old), InterceptionEnv(new)
    old_env.reset(seed=11)
    new_env.reset(seed=11)
    for _ in range(25):
        old_transition = old_env.step(np.zeros((5, 2)))
        new_transition = new_env.step(np.zeros((5, 2)))
        np.testing.assert_allclose(old_env.positions, new_env.positions)
        np.testing.assert_allclose(old_env.velocities, new_env.velocities)
        np.testing.assert_array_equal(old_env.assignment.targets, new_env.assignment.targets)
        np.testing.assert_allclose(old_env.central_state(), new_env.central_state())
        assert old_transition[1] == pytest.approx(new_transition[1])
        assert old_transition[4]['reward_components'] == pytest.approx(
            new_transition[4]['reward_components'])


@pytest.mark.parametrize('name', ('mappo', 'maddpg'))
def test_new_observation_trains_and_restores(tmp_path, name):
    config, _, _ = read_config('configs/interception_huang_global_target_velocity.yaml')
    config = replace(config, episode_steps=2)
    parameters = (MAPPOConfig(hidden_size=16, update_epochs=1, minibatch_size=4)
                  if name == 'mappo' else MADDPGConfig(
                      hidden_dim=16, batch_size=2, buffer_capacity=4,
                      learning_starts=2, update_interval_steps=2, updates_per_interval=1))
    output = tmp_path / name
    train(config, parameters, 2, name, steps=4, num_envs=2, seed=7,
          device='cpu', attack_mode='direct_rush', output=output,
          eval_interval=0, log_every_steps=4)
    settings = json.loads((output/'settings.json').read_text(encoding='utf-8'))
    assert settings['environment_contract']['observation_dim'] == 36
    assert settings['environment_contract']['state_dim'] == 75
    restored, policy, algorithm, _, _ = load_run(output, 'Last', 'cpu')
    assert algorithm == name and restored.observation_layout == 'relative_assignment_target_velocity'
    env = InterceptionEnv(restored)
    obs, info = env.reset(seed=8)
    assert policy.policy_actions(obs[None], info['active_masks'][None]).shape == (1, 5, 2)
