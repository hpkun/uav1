"""Paper-inspired per-agent credit assignment for the radial 5v5 baseline.

Unlike legacy team=global+mean(local), this isolated reward contract uses
team=sum_i(individual_reward_i), matching the original MADDPG experiment
training-curve convention. The physical one-capture-per-defender rule is fixed.
"""
from dataclasses import replace, asdict

import numpy as np
import pytest

from algorithm.maddpg import MADDPG
from algorithm.maddpg_config import MADDPGConfig
from env.interception import InterceptionEnv, PaperRewardInterceptionConfig
from experiment.interception import read_config, train, load_run, evaluate_policy, summarize


CONFIG = 'configs/interception_radial_paper_reward.yaml'
ZERO_ACTION = np.zeros((5, 2), dtype=np.float32)


def make_env():
    config, _, _ = read_config(CONFIG)
    env = InterceptionEnv(config)
    env.reset(seed=7)
    return env


def checked_step(env):
    _, reward, _, _, info = env.step(ZERO_ACTION)
    np.testing.assert_allclose(reward, info['individual_rewards'].astype(np.float32))
    assert info['reward_aggregation'] == 'sum_agent'
    assert info['team_reward'] == pytest.approx(info['individual_rewards'].sum())
    assert sum(info['reward_components'].values()) == pytest.approx(info['team_reward'], abs=1e-9)
    return reward, info


def test_contract_is_distinct_and_old_radial_stays_unchanged():
    new, _, _ = read_config(CONFIG)
    old, _, _ = read_config('configs/interception_ideal_radial_straight.yaml')
    assert isinstance(new, PaperRewardInterceptionConfig)
    assert InterceptionEnv(new).contract['schema'] == 'Interception/12'
    assert InterceptionEnv(old).contract['schema'] == 'Interception/11'
    assert new.capture_reward == 20.0
    assert new.teammate_capture_reward == 5.0
    assert new.all_captured_reward == 50.0
    assert new.approach_weight == old.approach_weight == 0.2
    with pytest.raises(ValueError, match='Paper reward profile'):
        replace(new, attack_mode='jia_straight')


def test_single_capture_is_self_reward_and_four_peer_rewards():
    env = make_env()
    env.positions[1, 0] = env.positions[0, 0].copy()
    before_defenders = env.positions[0].copy()
    before_attackers = env.positions[1].copy()
    assigned = env.assignment.targets.copy()
    reward, info = checked_step(env)
    assert env.captured_count == 1
    assert not env.defender_alive[0] and not env.attacker_alive[0]
    assert info['global_reward'] == pytest.approx(0.0)
    c = info['reward_components']
    assert c['capture_self'] == pytest.approx(20.0)
    assert c['capture_teammate'] == pytest.approx(20.0)
    assert c['all_captured'] == 0.0
    assert c['penetration'] == 0.0
    approach = np.array([
        np.linalg.norm(before_defenders[i]-before_attackers[assigned[i]]) -
        np.linalg.norm(env.positions[0,i]-env.positions[1,assigned[i]])
        for i in range(5)
    ])
    np.testing.assert_allclose(info['local_rewards']-0.2*approach,
                               np.array([20.,5.,5.,5.,5.]),atol=1e-9)
    assert c['capture_self'] + c['capture_teammate'] == pytest.approx(40.0)


def test_per_target_failure_is_shared_in_each_critic_reward():
    env = make_env()
    env.positions[1, 0] = [0.0, 30.1]
    reward, info = checked_step(env)
    assert env.penetrated_count == 1 and env.captured_count == 0
    assert info['global_reward'] == pytest.approx(-50.0)
    assert info['reward_components']['penetration'] == pytest.approx(-250.0)
    np.testing.assert_allclose(
        reward-info['local_rewards'].astype(np.float32),
        np.full(5, -50, dtype=np.float32))
    # One physical breach is counted once in the state, five times in the
    # sum of five agents' learning rewards.


def test_clean_five_capture_win_emits_bonus_only_once_per_agent():
    env = make_env()
    env.positions[1] = env.positions[0].copy()
    reward, info = checked_step(env)
    assert env.captured_count == 5 and env.penetrated_count == 0 and env.finished
    assert not env.defender_alive.any() and not env.attacker_alive.any()
    c = info['reward_components']
    assert c['capture_self'] == 100.0
    assert c['capture_teammate'] == 100.0
    assert c['all_captured'] == 250.0
    assert c['penetration'] == 0.0
    assert info['global_reward'] == 50.0
    assert info['episode_summary']['reward_aggregation'] == 'sum_agent'
    assert info['episode_summary']['team_return'] == pytest.approx(reward.sum())


def test_short_training_and_checkpoint_keep_reward_contract(tmp_path):
    config, _, _ = read_config(CONFIG)
    short = replace(config, episode_steps=1)
    parameters = MADDPGConfig(hidden_dim=16, batch_size=2, buffer_capacity=8,
                              learning_starts=2, update_interval_steps=2,
                              updates_per_interval=1)
    output = tmp_path/'paper_reward'
    train(short, parameters, 2, 'maddpg', steps=4, num_envs=1, seed=7,
          device='cpu', attack_mode='radial_straight', output=output,
          eval_interval=0, eval_episodes=1, log_every_steps=4)
    restored, policy, name, _, settings = load_run(output, 'Last', 'cpu')
    assert name == 'maddpg'
    assert settings['environment_contract']['version'] == 'interception_12'
    assert isinstance(restored, PaperRewardInterceptionConfig)
    rows = evaluate_policy(restored, policy, 'radial_straight', 1, 600000)
    stats = summarize(rows)
    assert stats['reward_aggregation'] == 'sum_agent'
    assert stats['mean_team_return'] == pytest.approx(
        sum(stats['mean_reward_components'].values()))


def test_retired_actor_stays_off_but_team_critic_learns_future_rewards():
    config, _, _ = read_config(CONFIG)
    env = InterceptionEnv(config)
    learner = MADDPG(MADDPGConfig(hidden_dim=16, batch_size=2),
                     'cpu', 7, environment_config=dict(environment=asdict(config)),
                     environment_contract=env.contract)
    batch = dict(
        observations=np.zeros((2,5,24),np.float32),
        next_observations=np.zeros((2,5,24),np.float32),
        states=np.zeros((2,75),np.float32),
        next_states=np.zeros((2,75),np.float32),
        actions=np.zeros((2,5,2),np.float32),
        rewards=-50*np.ones((2,5),np.float32),
        dones=np.array([0.,1.],np.float32),
        active_masks=np.zeros((2,5),np.float32),
        next_active_masks=np.zeros((2,5),np.float32),
    )
    metrics = learner.update(batch)
    assert metrics['critic_full_team_episode'] == 1.0
    assert metrics['active_critic_count'] == 5
    assert metrics['critic_loss'] > 0
    assert learner.actor_optimizer_steps == [0]*5
    # When every defender has retired, the team Critic nevertheless sees
    # the subsequent failure event while the Actor remains ineligible.
    old, _, _ = read_config('configs/interception_ideal_radial_straight.yaml')
    legacy = MADDPG(MADDPGConfig(hidden_dim=16, batch_size=2),
                    'cpu', 7, environment_config=dict(environment=asdict(old)),
                    environment_contract=InterceptionEnv(old).contract)
    old_metrics = legacy.update(batch)
    assert old_metrics['critic_full_team_episode'] == 0.0
    assert old_metrics['active_critic_count'] == 0
