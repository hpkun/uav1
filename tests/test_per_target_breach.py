"""Per-target breach penalties in the existing five-on-five retirement protocol."""
from dataclasses import replace
import json

import numpy as np
import pytest

from env import make_environment
from experiment.interception import episode_summary_with_outcome, read_config, summarize, train
from algorithm.maddpg_config import MADDPGConfig


CONFIG = 'configs/interception_ideal_jia_low_level_obs24_no_heading_per_target_breach50.yaml'
ACTION = np.zeros((5, 2), np.float32)


def _environment(*, steps=5000, per_target=True):
    config, _, _ = read_config(CONFIG)
    if not per_target:
        config = replace(config, breach_penalty_mode='first_breach', penetration_penalty=100)
    env = make_environment(replace(config, episode_steps=steps))
    env.reset(seed=7)
    env.positions[0] = np.array([[-80, 0], [-40, 0], [0, 0], [40, 0], [80, 0]])
    env.positions[1] = np.array([[-80, 80], [-40, 80], [0, 80], [40, 80], [80, 80]])
    env.velocities.fill(0)
    return env


def _step(env, *, captures=(), breaches=()):
    for i in captures:
        env.positions[1, i] = env.positions[0, i]
    for j in breaches:
        env.positions[1, j] = [float(j-2), -80.0]
    _, rewards, _, _, info = env.step(ACTION)
    np.testing.assert_allclose(rewards, info['individual_rewards'])
    np.testing.assert_allclose(info['individual_rewards']-info['local_rewards'], info['global_reward'])
    assert info['team_reward'] == pytest.approx(info['global_reward']+info['local_rewards'].mean())
    assert sum(info['reward_components'].values()) == pytest.approx(info['team_reward'])
    return info


@pytest.mark.parametrize(('captures', 'breaches', 'global_reward'), [
    ((0,), (), 20),
    ((), (0,), -50),
    ((), (0, 1), -100),
    ((0, 1, 2, 3, 4), (), 100),
    ((0, 1, 2, 3), (4,), 30),
    ((), (0, 1, 2, 3, 4), -250),
])
def test_single_step_events(captures, breaches, global_reward):
    env = _environment()
    info = _step(env, captures=captures, breaches=breaches)
    assert info['global_reward'] == global_reward
    assert info['reward_components']['capture'] == 20*len(captures)
    assert info['reward_components']['penetration'] == -50*len(breaches)
    assert env.captured_count == len(captures)
    assert env.penetrated_count == len(breaches)
    if len(captures)+len(breaches)==5:
        assert info['episode_summary']['global_return'] == global_reward


def test_shared_global_reward_allows_different_local_distance_rewards():
    env = _environment()
    for i, target in enumerate(env.assignment.targets):
        env.positions[1, target] = env.positions[0, i] + [0, 40]
    env.velocities[0, :, 1] = [0, 1, 2, -1, -2]
    info = _step(env, breaches=(0,))
    assert info['global_reward']==-50
    assert np.ptp(info['local_rewards'])>0.01
    np.testing.assert_allclose(info['individual_rewards']-info['local_rewards'], -50)


def test_sequential_breaches_are_charged_once_each_and_shared():
    env = _environment()
    first = _step(env, breaches=(0,))
    assert first['global_reward'] == -50 and not env.attacker_alive[0]
    second = _step(env)
    assert second['global_reward'] == 0
    third = _step(env, breaches=(1,))
    assert third['global_reward'] == -50
    assert env.global_return == -100
    assert env.penetrated_count == 2


def test_timeout_charges_each_unresolved_target_once_without_real_breach():
    env = _environment(steps=1)
    info = _step(env)
    assert info['global_reward'] == -250
    assert info['reward_components']['penetration'] == -250
    assert len([event for event in info['events'] if event['type']=='penetration'
                and event.get('reason')=='time_limit']) == 5
    row = episode_summary_with_outcome(info)
    assert row['time_limit_occurred'] and row['actual_penetrated_count']==0
    result = summarize([row])
    assert (result['failure_rate'], result['penetration_rate'], result['time_limit_rate']) == (1, 0, 1)
    assert row['global_return']==-250


def test_old_default_mode_still_only_charges_first_breach():
    env = _environment(per_target=False)
    assert _step(env, breaches=(0,))['global_reward']==-100
    assert _step(env, breaches=(1, 2))['global_reward']==0
    assert env.global_return==-100


def test_bad_mode_is_rejected():
    config, _, _ = read_config(CONFIG)
    with pytest.raises(ValueError, match='breach penalty mode'):
        replace(config, breach_penalty_mode='other')


def test_short_maddpg_training_and_evaluation_use_new_reward(tmp_path):
    config, _, _ = read_config(CONFIG)
    parameters = MADDPGConfig(hidden_dim=16, batch_size=2, buffer_capacity=8,
                              learning_starts=2, update_interval_steps=2)
    output = tmp_path/'per_target'
    train(replace(config, episode_steps=1), parameters, 2, 'maddpg', steps=4,
          num_envs=1, seed=7, device='cpu', attack_mode='jia_straight',
          output=output, eval_interval=2, eval_episodes=1, log_every_steps=4)
    rows=[json.loads(line) for line in (output/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==4
    assert all(row['reward_components']['penetration']==-250 for row in rows)
    assert all(sum(row['reward_components'].values())==pytest.approx(row['team_return']) for row in rows)
    evaluations=[json.loads(line)['modes']['jia_straight'] for line in
                 (output/'evaluations.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(evaluations)==2
    assert all(row['mean_reward_components']['penetration']==-250 for row in evaluations)
    assert all(sum(row['mean_reward_components'].values())==pytest.approx(row['mean_team_return'])
               for row in evaluations)
    assert len((output/'updates.jsonl').read_text(encoding='utf-8').splitlines())>0
