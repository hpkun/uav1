"""Unavoidable breach feedback remains learnable without ending the episode early."""

from dataclasses import replace
import json

import numpy as np
import pytest

from env.interception import InterceptionEnv
from algorithm.maddpg_config import MADDPGConfig
from experiment.interception import load_run, read_config, train
from tools.diagnose import analyze_trace
from tools.replay import collect, validate_trace


CONFIG = 'configs/interception_ideal_jia_low_level_credit_fix.yaml'


class ConstantPolicy:
    def policy_actions(self, observations, active_masks, *, deterministic):
        actions = np.zeros((len(observations), 5, 2), np.float32)
        actions[:, :, 0] = 2
        return actions


def test_last_defender_boundary_exit_gives_single_trainable_failure_reward():
    config, _, _ = read_config(CONFIG)
    old, _, _ = read_config('configs/interception_ideal_jia_low_level_obs24.yaml')
    assert config == replace(old, prepay_unavoidable_breach=True)
    env = InterceptionEnv(replace(config, episode_steps=2))
    observations, info = env.reset(seed=7)
    assert observations.shape == (5, 24)
    assert info['central_state'].shape == (76,)
    assert info['central_state'][-1] == 0
    env.defender_alive[:4] = False
    env.positions[0, 4] = [99.9, 0]
    env.velocities[0, 4] = [5, 0]
    before_mask = env.defender_alive.copy()
    _, rewards, done, truncated, info = env.step(np.zeros((5, 2)))
    assert before_mask[4] and info['active_masks'].sum() == 0
    assert rewards[4] == pytest.approx(-102)
    assert info['reward_components']['penetration'] == -100
    assert info['central_state'][-1] == 1
    assert any(event['type'] == 'defense_lost' for event in info['events'])
    assert not done and not truncated
    _, _, done, _, final = env.step(np.zeros((5, 2)))
    assert done and final['reward_components']['penetration'] == 0
    assert final['episode_summary']['penetrated_count'] == 5
    assert final['episode_summary']['global_return'] == -100


def test_first_real_breach_remains_single_penalty():
    config, _, _ = read_config(CONFIG)
    env = InterceptionEnv(config)
    env.reset(seed=7)
    env.positions[1, 0] = env.base.copy()
    env.velocities[1, 0] = 0
    _, _, _, _, info = env.step(np.zeros((5, 2)))
    assert info['reward_components']['penetration'] == -100
    assert info['central_state'][-1] == 1
    env.defender_alive[:] = False
    _, _, _, _, info = env.step(np.zeros((5, 2)))
    assert info['reward_components']['penetration'] == 0


def test_replay_and_diagnostics_account_for_heading_and_retirement():
    config, _, _ = read_config(CONFIG)
    trace = validate_trace(collect(config, ConstantPolicy(), 'jia_straight', 600000))
    assert trace['summary']['defender_oob_count'] > 0
    result = analyze_trace(trace)
    assert result['approach']['defender_motion'] + result['approach']['attacker_motion'] == pytest.approx(
        result['approach']['total'])


def test_new_state_trains_and_loads_without_changing_old_interface(tmp_path):
    config, _, _ = read_config(CONFIG)
    parameters = MADDPGConfig(hidden_dim=16, batch_size=2, buffer_capacity=8,
                              learning_starts=2, update_interval_steps=2,
                              updates_per_interval=1)
    output = tmp_path / 'credit'
    train(replace(config, episode_steps=2), parameters, 2, 'maddpg', steps=4,
          num_envs=2, seed=7, device='cpu', attack_mode='jia_straight',
          output=output, eval_interval=0, eval_episodes=1, log_every_steps=4)
    restored, policy, _, _, _ = load_run(output, 'Last', 'cpu')
    assert policy.environment_spec['version'] == 'interception_9'
    env = InterceptionEnv(restored)
    env.reset(seed=7)
    assert env.central_state().shape == (76,)
    update = json.loads((output / 'updates.jsonl').read_text(encoding='utf-8').splitlines()[0])
    assert update['critic_grad_norm_mean_agent'] <= update['critic_grad_norm_max_agent']
    assert update['critic_grad_norm_max_agent'] <= update['critic_grad_norm'] + 1e-5
