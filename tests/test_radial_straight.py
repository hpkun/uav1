"""Geometry and replay checks for the isolated central radial battlefield."""

from dataclasses import replace

import numpy as np

from algorithm.maddpg_config import MADDPGConfig
from env.interception import InterceptionEnv
from experiment.interception import load_run, read_config, train
from tools.replay import collect, html, validate_trace


CONFIG = 'configs/interception_ideal_radial_straight.yaml'


def test_radial_geometry_and_contract():
    config, _, _ = read_config(CONFIG)
    assert config.attack_mode == 'radial_straight'
    assert config.capture_radius == 5 and config.heading_weight == 0
    env = InterceptionEnv(config)
    observation, info = env.reset(seed=7)
    assert env.contract['schema'] == 'Interception/11'
    assert observation.shape == (5, 24) and env.central_state().shape == (75,)
    assert sorted(env.assignment.targets.tolist()) == list(range(5))
    assert np.max(np.abs(env.positions)) < config.arena_half_extent
    assert np.all((160 <= np.linalg.norm(env.positions[1], axis=1)) &
                  (np.linalg.norm(env.positions[1], axis=1) <= 180))
    assert np.all((50 <= np.linalg.norm(env.positions[0], axis=1)) &
                  (np.linalg.norm(env.positions[0], axis=1) <= 60))
    assert np.allclose(env.velocities[0], 0)
    attacker_angles = np.mod(np.arctan2(env.positions[1,:,1], env.positions[1,:,0]), 2*np.pi)
    defender_angles = np.mod(np.arctan2(env.positions[0,:,1], env.positions[0,:,0]), 2*np.pi)
    np.testing.assert_allclose(np.mod(defender_angles-attacker_angles, 2*np.pi), np.pi/9)
    unit = env.positions[1] / np.linalg.norm(env.positions[1], axis=1, keepdims=True)
    targets = config.breach_radius * unit
    distances = np.linalg.norm(targets[:, None] - targets[None, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= 2*config.breach_radius*np.sin(np.deg2rad(33)) - 1e-9
    np.testing.assert_allclose(env.attacker_course_directions, -unit)
    assert info['version'] == 'interception_11'
    legacy = InterceptionEnv(replace(config, attack_mode='jia_straight', base_y=-80,
                                     breach_radius=15, arena_half_extent=100,
                                     sensing_radius=300, defender_initial_speed=2))
    assert legacy.contract['version'] == 'interception_8'


def test_radial_attackers_remain_straight_and_all_breach_without_defender_actions():
    config, _, _ = read_config(CONFIG)
    for seed in (7, 17, 27):
        env = InterceptionEnv(config)
        env.reset(seed=seed)
        start = env.positions[1].copy()
        directions = env.attacker_course_directions.copy()
        all_events = []
        while not env.finished:
            _, _, _, _, info = env.step(np.zeros((5, 2)))
            all_events.extend(info['events'])
            displacement = env.positions[1, env.attacker_alive] - start[env.attacker_alive]
            remaining = directions[env.attacker_alive]
            np.testing.assert_allclose(displacement[:, 0]*remaining[:, 1] -
                                       displacement[:, 1]*remaining[:, 0], 0, atol=1e-8)
        assert env.penetrated_count == 5 and env.captured_count == 0
        assert sum(event['type'] == 'penetration' for event in all_events) == 5
        assert not any(event['type'] == 'capture' for event in all_events)


def test_radial_replay_has_central_view():
    class StationaryPolicy:
        def policy_actions(self, observations, masks, *, deterministic):
            return np.zeros((len(observations), 5, 2))

    config, _, _ = read_config(CONFIG)
    trace = validate_trace(collect(config, StationaryPolicy(), 'radial_straight', 900001))
    assert trace['schema'] == 'InterceptionReplay/2'
    page = html(trace, 5)
    assert '中心型五向拦截回放' in page
    assert '五向径向直线进攻' in page
    assert '中心防护区' in page


def test_radial_short_training_checkpoint_and_replay(tmp_path):
    config, _, _ = read_config(CONFIG)
    short = replace(config, episode_steps=2)
    parameters = MADDPGConfig(hidden_dim=16, batch_size=2, buffer_capacity=8,
                              learning_starts=2, update_interval_steps=2,
                              updates_per_interval=1)
    output = tmp_path / 'radial'
    train(short, parameters, 2, 'maddpg', steps=4, num_envs=1, seed=7,
          device='cpu', attack_mode='radial_straight', output=output,
          eval_interval=0, eval_episodes=1, log_every_steps=4)
    restored, policy, _, _, settings = load_run(output, 'Last', 'cpu')
    assert settings['attack_mode'] == restored.attack_mode == 'radial_straight'
    assert validate_trace(collect(restored, policy, 'radial_straight', 900001))['summary']['steps'] == 2


def test_radial_local_sum_reward_accounting():
    """Only reporting/team scalar sums local terms; MADDPG agent rewards stay full."""
    config, _, _ = read_config(CONFIG)
    env = InterceptionEnv(config)
    env.reset(seed=7)
    _, individual, _, _, info = env.step(np.zeros((5, 2), np.float32))
    local = info['local_rewards']
    global_reward = info['global_reward']
    np.testing.assert_allclose(individual,global_reward+local,rtol=0,atol=1e-5)
    np.testing.assert_allclose(info['individual_rewards'],global_reward+local)
    assert np.isclose(info['team_reward'],global_reward+local.sum())
    assert np.isclose(info['reward_components']['approach'],
                      config.approach_weight*(
                          (local/config.approach_weight).sum()))  # no boundary in this step
    assert np.isclose(sum(info['reward_components'].values()),info['team_reward'])
    assert np.isclose(env.team_return,info['team_reward'])

    # Legacy straight-line scenario keeps the earlier mean-agent accounting.
    from dataclasses import replace
    old = InterceptionEnv(replace(config, attack_mode='jia_straight',base_y=-80,
                                  breach_radius=15,arena_half_extent=100,
                                  sensing_radius=300,defender_initial_speed=2))
    old.reset(seed=7)
    _, old_individual, _, _, old_info = old.step(np.zeros((5, 2), np.float32))
    np.testing.assert_allclose(old_individual,old_info['global_reward']+old_info['local_rewards'])
    assert np.isclose(old_info['team_reward'],
                      old_info['global_reward']+old_info['local_rewards'].mean())
