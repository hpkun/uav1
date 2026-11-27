"""Tests for Interception/14: 51D fixed-ID full-entity observations and nearest-target shaping."""
from dataclasses import replace
import json

import numpy as np
import pytest

from algorithm.maddpg import BoxActor
from algorithm.maddpg_config import NormalizedMADDPGConfig
from algorithm.matd3_config import NormalizedMATD3Config
from env import JiaHeadingNearestConfig, make_environment
from env.jia_heading import wrap_heading
from experiment.cli import main as cli_main
from experiment.interception import evaluate_policy, load_run, make_policy, read_config, train
from tools.replay import collect, html, validate_trace

ENV_TIME01 = 'configs/interception_radial_5v5_jia_heading_nearest_time01.yaml'
ENV_D08 = 'configs/interception_radial_5v5_jia_heading_nearest_d08_h003_time0015.yaml'
ALG_MATD3 = 'configs/matd3_jia_heading_interval10.yaml'
ALG_MADDPG = 'configs/maddpg_jia_heading_interval10.yaml'
ENV_13 = 'configs/interception_radial_5v5_jia_heading_time01.yaml'


def test_config_contract_and_historical_13_preservation():
    cfg1, params1, rollout1 = read_config(ENV_TIME01, ALG_MATD3)
    cfg2, params2, rollout2 = read_config(ENV_D08, ALG_MADDPG)
    assert isinstance(cfg1, JiaHeadingNearestConfig) and isinstance(cfg2, JiaHeadingNearestConfig)
    assert isinstance(params1, NormalizedMATD3Config) and isinstance(params2, NormalizedMADDPGConfig)
    assert rollout1 == 128 and rollout2 == 128

    env = make_environment(cfg1)
    contract = env.contract
    assert contract['schema'] == 'Interception/14'
    assert contract['version'] == 'interception_14'
    assert contract['observation_dim'] == 51
    assert contract['state_dim'] == 60
    assert contract['action_dim'] == 2
    assert contract['action_schema'] == 'normalized_speed_turn_box'
    assert contract['assignment_schema'] is None
    assert contract['reward_schema'].count('fixed_shared_time_cost') == 1
    assert make_environment(replace(cfg1, time_penalty_per_step=0.0)).contract['reward_schema'].count('fixed_shared_time_cost') == 0

    with pytest.raises(ValueError):
        JiaHeadingNearestConfig(assignment_method='ideal_global_distance_matching')
    with pytest.raises(ValueError):
        JiaHeadingNearestConfig(communication_radius=-10.0)
    with pytest.raises(ValueError, match='MADDPG/MATD3'):
        read_config(ENV_TIME01, 'configs/mappo_interception.yaml')

    # Historical Interception/13 contract & implementation digest must remain byte-for-byte untouched.
    cfg13, _, _ = read_config(ENV_13, ALG_MATD3)
    env13 = make_environment(cfg13)
    assert env13.contract['version'] == 'interception_13'
    assert env13.observation_dim == 26 and env13.state_dim == 85
    assert env13.contract['implementation_sha256'] == '46230752bf44cca962b0061c62f56c8bd70ae0e13cd21ced226b2a6ee0dfe21d'

    # Initial radial spawn geometry matches Interception/13 for identical seeds.
    env.reset(seed=7)
    env13.reset(seed=7)
    np.testing.assert_allclose(env.positions, env13.positions, atol=1e-12)
    np.testing.assert_allclose(env.headings, env13.headings, atol=1e-12)
    np.testing.assert_allclose(env.velocities, env13.velocities, atol=1e-12)


def test_observation_normalization_masks_fixed_id_order_and_central_state():
    cfg, params, _ = read_config(ENV_TIME01, ALG_MATD3)
    env = make_environment(cfg)
    obs, info = env.reset(seed=7)
    assert obs.shape == (5, 51) and obs.dtype == np.float32
    assert info['central_state'].shape == (60,) and info['central_state'].dtype == np.float32
    assert info['assigned_target_ids'] is None

    # Verify exact own, ally, and enemy slices for defender 2.
    i = 2
    own_pos = env.positions[0, i]
    own_vel = env.velocities[0, i]
    np.testing.assert_allclose(obs[i, 0:2], (own_pos - env.base) / cfg.arena_half_extent, atol=1e-6)
    np.testing.assert_allclose(obs[i, 2:4], own_vel / cfg.defender_speed_max, atol=1e-6)
    np.testing.assert_allclose(obs[i, 4:6], [np.cos(env.headings[0, i]), np.sin(env.headings[0, i])], atol=1e-6)

    expected_allies = [0, 1, 3, 4]
    for slot, j in enumerate(expected_allies):
        k = 6 + slot * 5
        np.testing.assert_allclose(obs[i, k:k + 2], (env.positions[0, j] - own_pos) / (2 * cfg.arena_half_extent), atol=1e-6)
        np.testing.assert_allclose(obs[i, k + 2:k + 4], (env.velocities[0, j] - own_vel) / (2 * cfg.defender_speed_max), atol=1e-6)
        assert obs[i, k + 4] == 1.0

    for j in range(5):
        k = 26 + j * 5
        np.testing.assert_allclose(obs[i, k:k + 2], (env.positions[1, j] - own_pos) / (2 * cfg.arena_half_extent), atol=1e-6)
        np.testing.assert_allclose(obs[i, k + 2:k + 4], (env.velocities[1, j] - own_vel) / (cfg.defender_speed_max + cfg.attacker_speed_max), atol=1e-6)
        assert obs[i, k + 4] == 1.0

    # Verify fixed ID slot stability even when distance ordering reverses dramatically.
    env.positions[0, 0] = [0.0, 0.0]
    env.positions[1, 0] = [180.0, 0.0]
    env.positions[1, 4] = [10.0, 0.0]
    obs_swapped = env.observations()
    # Blue0 remains in enemy slot 0 (26:31) even though Blue4 is far closer; Blue4 remains in slot 4 (46:51).
    np.testing.assert_allclose(obs_swapped[0, 26:28], [180.0 / 400.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(obs_swapped[0, 46:48], [10.0 / 400.0, 0.0], atol=1e-6)

    # Test communication_radius & sensing_radius masking plus retired defender zeroing.
    limited_cfg = replace(cfg, communication_radius=50.0, sensing_radius=100.0)
    limited_env = make_environment(limited_cfg)
    limited_env.reset(seed=7)
    limited_env.positions[0] = [[0, 0], [30, 0], [80, 0], [0, 30], [0, -30]]
    limited_env.positions[1] = [[60, 0], [150, 0], [0, 80], [0, -150], [90, 0]]
    limited_env.defender_alive[4] = False
    limited_env.attacker_alive[2] = False
    lobs = limited_env.observations()
    # For defender 0: ally 1 (dist 30) visible, ally 2 (dist 80 > 50) masked, ally 3 (dist 30) visible, ally 4 (dead) masked.
    assert lobs[0, 6 + 0 * 5 + 4] == 1.0
    np.testing.assert_array_equal(lobs[0, 6 + 1 * 5:6 + 2 * 5], 0.0)
    assert lobs[0, 6 + 2 * 5 + 4] == 1.0
    np.testing.assert_array_equal(lobs[0, 6 + 3 * 5:6 + 4 * 5], 0.0)
    # Enemies for defender 0: Blue0 (60) visible, Blue1 (150>100) masked, Blue2 (dead) masked, Blue3 (150>100) masked, Blue4 (90) visible.
    assert lobs[0, 26 + 0 * 5 + 4] == 1.0
    np.testing.assert_array_equal(lobs[0, 26 + 1 * 5:26 + 2 * 5], 0.0)
    np.testing.assert_array_equal(lobs[0, 26 + 2 * 5:26 + 3 * 5], 0.0)
    np.testing.assert_array_equal(lobs[0, 26 + 3 * 5:26 + 4 * 5], 0.0)
    assert lobs[0, 26 + 4 * 5 + 4] == 1.0
    # Retired defender 4 gets all-zero obs and zero central_state slice + heading.
    np.testing.assert_array_equal(lobs[4], 0.0)
    state = limited_env.central_state()
    np.testing.assert_array_equal(state[20:25], 0.0)
    np.testing.assert_array_equal(state[58:60], 0.0)
    # Dead attacker 2 has zero central_state slice.
    np.testing.assert_array_equal(state[35:40], 0.0)

    # Check MATD3 & MADDPG network shapes.
    policy_matd3 = make_policy('matd3', cfg, params, 'cpu', 7)
    assert isinstance(policy_matd3.actors[0], BoxActor)
    assert policy_matd3.actors[0].net[0].in_features == 51
    assert policy_matd3.critics_1[0].net[0].in_features == 70
    assert policy_matd3.critics_2[0].net[0].in_features == 70


def test_nearest_selection_many_to_one_signs_and_target_switching():
    cfg = JiaHeadingNearestConfig(time_penalty_per_step=0.0, approach_weight=1.0, heading_weight=0.5)
    env = make_environment(cfg)
    env.reset(seed=7)

    # Place Red0 and Red1 both closest to Blue2 -> many-to-one selection allowed!
    env.positions[0] = [[0, 0], [0, 20], [-150, -150], [-150, 150], [150, -150]]
    env.positions[1] = [[-160, -160], [-160, 160], [40, 10], [160, -160], [160, 160]]
    env.headings[0] = [0.0, np.pi, 0.0, 0.0, 0.0]  # Red0 heads +x (approaching Blue2), Red1 heads -x (retreating from Blue2)
    env.headings[1] = [0.0, 0.0, 0.0, 0.0, 0.0]
    env.velocities[0] = 6.0 * np.column_stack((np.cos(env.headings[0]), np.sin(env.headings[0])))
    env.velocities[1] = 6.0 * np.column_stack((np.cos(env.headings[1]), np.sin(env.headings[1])))
    # Freeze attackers far except Blue2 moving +x at 6 m/s while Red0 flies +x at 10 m/s (closes 0.4 m) and Red1 flies -x at 10 m/s (opens ~1.6 m).
    actions = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    _, _, _, _, info = env.step(actions)
    assert info['reward_target_ids'][0] == 2
    assert info['reward_target_ids'][1] == 2
    # Red0 closes distance and aligns toward Blue2 (+); Red1 retreats and points away (-).
    assert info['local_rewards'][0] > 0.0
    assert info['local_rewards'][1] < 0.0
    assert np.isclose(sum(info['reward_components'].values()), info['team_reward'])

    # Heading alignment unit check (aligned / perpendicular / opposite / zero distance).
    env.positions[0, 0] = [0.0, 0.0]
    env.positions[1, 0] = [50.0, 0.0]
    for angle, expected in ((0.0, 1.0), (np.pi / 2, 0.0), (np.pi, -1.0)):
        env.headings[0, 0] = angle
        assert env._heading_alignment(0, 0) == pytest.approx(expected, abs=1e-12)
    env.positions[1, 0] = [0.0, 0.0]
    assert env._heading_alignment(0, 0) == 0.0

    # Target-switching test: Blue0 initially closer to Red0, after one step Blue1 becomes closer.
    env.reset(seed=7)
    env.defender_alive[:] = [True, False, False, False, False]
    env.attacker_alive[:] = [True, True, False, False, False]
    # Red0 at (0,0) heading +x at speed 10 (moves +1.0 in x to (1.0, 0.0)).
    # Blue0 at (-19.8, 0) heading -x at speed 6 (moves to (-20.4, 0): dist before=19.8, after=21.4).
    # Blue1 at (20.0, 0) heading -x at speed 6 (moves to (19.4, 0): dist before=20.0, after=18.4 -> NEW NEAREST!).
    env.positions[0, 0] = [0.0, 0.0]
    env.headings[0, 0] = 0.0
    env.positions[1, 0] = [-19.8, 0.0]
    env.headings[1, 0] = np.pi
    env.positions[1, 1] = [20.0, 0.0]
    env.headings[1, 1] = np.pi
    _, _, _, _, info_sw = env.step(np.array([[1.0, 0.0], [0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float64))
    assert info_sw['reward_target_ids'][0] == 1
    expected_approach = 19.8 - 18.4
    expected_heading = 1.0  # Red0 at (1,0) heading +x points directly at Blue1 at (19.4,0), NOT Blue0 at (-20.4,0)!
    assert info_sw['reward_components']['approach'] == pytest.approx(expected_approach, abs=1e-6)
    assert info_sw['reward_components']['heading'] == pytest.approx(0.5 * expected_heading, abs=1e-6)
    assert np.isclose(sum(info_sw['reward_components'].values()), info_sw['team_reward'])


def test_no_visible_enemies_and_post_capture_reselection():
    # No visible enemies -> zero approach & heading reward.
    blind_cfg = JiaHeadingNearestConfig(sensing_radius=10.0, time_penalty_per_step=0.1)
    env = make_environment(blind_cfg)
    env.reset(seed=7)
    _, individual, _, _, info = env.step(np.zeros((5, 2)))
    assert info['reward_components']['approach'] == 0.0
    assert info['reward_components']['heading'] == 0.0
    assert info['reward_components']['time'] == pytest.approx(-0.1)
    np.testing.assert_array_equal(info['reward_target_ids'], -1)
    assert np.isclose(sum(info['reward_components'].values()), info['team_reward'])
    np.testing.assert_allclose(individual, -0.1, atol=1e-6)

    # Post-capture reselection: Red0 captures Blue0 while Red1 also had Blue0 as nearest;
    # on the next step Red1 must never select dead Blue0.
    cfg = JiaHeadingNearestConfig(time_penalty_per_step=0.0)
    env2 = make_environment(cfg)
    env2.reset(seed=7)
    env2.defender_alive[:] = [True, True, False, False, False]
    env2.attacker_alive[:] = [True, True, False, False, False]
    env2.positions[0, 0] = [100.0, 0.0]
    env2.positions[0, 1] = [110.0, 0.0]
    env2.positions[1, 0] = [101.0, 0.0]
    env2.positions[1, 1] = [-100.0, 0.0]
    env2.headings[0, :2] = [0.0, np.pi]
    env2.headings[1, :2] = [np.pi, 0.0]
    _, _, _, _, info1 = env2.step(np.zeros((5, 2)))
    assert any(e['type'] == 'capture' and e['defender'] == 0 and e['attacker'] == 0 for e in info1['events'])
    assert not env2.defender_alive[0] and not env2.attacker_alive[0]
    assert info1['reward_target_ids'][1] == 0  # Blue0 was in pre-step candidate set
    assert np.isfinite(info1['team_reward'])

    _, _, _, _, info2 = env2.step(np.zeros((5, 2)))
    assert info2['reward_target_ids'][0] == -1  # Retired Red0
    assert info2['reward_target_ids'][1] == 1   # Re-selected surviving Blue1, never dead Blue0
    assert info2['local_rewards'][0] == 0.0
    assert np.isclose(sum(info2['reward_components'].values()), info2['team_reward'])


def test_full_episode_nearest_pursuit_and_unopposed_breach_ledgers():
    for time_pen in (0.0, 0.1):
        cfg = JiaHeadingNearestConfig(time_penalty_per_step=time_pen)
        env = make_environment(cfg)
        env.reset(seed=7)
        totals = {}
        while not env.finished:
            actions = np.zeros((5, 2))
            alive_attackers = np.flatnonzero(env.attacker_alive)
            for i in np.flatnonzero(env.defender_alive):
                target = int(min(alive_attackers, key=lambda j: np.linalg.norm(env.positions[1, j] - env.positions[0, i])))
                delta = env.positions[1, target] - env.positions[0, i]
                error = wrap_heading(np.arctan2(delta[1], delta[0]) - env.headings[0, i])
                actions[i, 0] = -1.0 if abs(error) > 0.35 else 1.0
                actions[i, 1] = np.clip(4.0 * error, -1.0, 1.0)
            _, _, _, _, info = env.step(actions)
            for key, value in info['reward_components'].items():
                totals[key] = totals.get(key, 0.0) + value
        assert (env.captured_count, env.penetrated_count) == (5, 0)
        assert np.isclose(sum(totals.values()), env.team_return, atol=1e-6)
        assert np.isclose(env.team_return, info['episode_summary']['team_return'], atol=1e-6)
        if time_pen:
            assert totals['time'] == pytest.approx(-time_pen * env.step_count, abs=1e-6)
        else:
            assert 'time' not in totals

    # Unopposed breach ledger check.
    cfg = JiaHeadingNearestConfig()
    env = make_environment(cfg)
    env.reset(seed=7)
    totals = {}
    while not env.finished:
        _, _, _, _, info = env.step(np.zeros((5, 2)))
        for key, value in info['reward_components'].items():
            totals[key] = totals.get(key, 0.0) + value
    assert (env.captured_count, env.penetrated_count) == (0, 5)
    assert totals['penetration'] == pytest.approx(-250.0)
    assert np.isclose(sum(totals.values()), env.team_return, atol=1e-6)


@pytest.mark.parametrize('algorithm,param_cls', [
    ('matd3', NormalizedMATD3Config),
    ('maddpg', NormalizedMADDPGConfig),
])
def test_short_train_preflight_load_eval_and_replay(tmp_path, algorithm, param_cls):
    config, _, _ = read_config(ENV_TIME01, ALG_MATD3 if algorithm == 'matd3' else ALG_MADDPG)
    out = tmp_path / f'{algorithm}_nearest'
    if algorithm == 'matd3':
        preflight = cli_main([
            'preflight',
            '--env-config', ENV_TIME01,
            '--algorithm-config', ALG_MATD3,
            '--algorithm', 'matd3',
            '--attack-mode', 'radial_straight',
            '--total-sampled-steps', '8',
            '--num-envs', '1',
            '--seed', '7',
            '--device', 'cpu',
            '--output-dir', str(out),
        ])
        assert preflight['status'] == 'supported'
        assert preflight['environment_contract']['schema'] == 'Interception/14'
        assert preflight['environment_contract']['observation_dim'] == 51
        assert preflight['environment_contract']['state_dim'] == 60
    params = param_cls(
        hidden_dim=16,
        batch_size=2,
        buffer_capacity=16,
        learning_starts=2,
        update_interval_steps=2,
        updates_per_interval=1,
        exploration_std_mps2=0.0,
    )
    train(
        replace(config, episode_steps=3),
        params,
        2,
        algorithm,
        steps=6,
        num_envs=1,
        seed=7,
        device='cpu',
        attack_mode='radial_straight',
        output=out,
        eval_interval=3,
        eval_episodes=1,
        log_every_steps=6,
        update_log_interval=2,
    )
    assert (out / 'updates.jsonl.gz').exists()
    assert (out / 'updates_summary.jsonl').exists()
    assert not (out / 'updates.jsonl').exists()
    restored, policy, name, _, settings = load_run(out, 'Best', 'cpu')
    assert name == algorithm
    assert settings['environment_contract']['version'] == 'interception_14'
    assert settings['environment_contract']['observation_dim'] == 51
    assert settings['environment_contract']['state_dim'] == 60
    assert json.loads((out / 'summary.json').read_text(encoding='utf-8'))['updates'] == 3

    rows = evaluate_policy(restored, policy, 'radial_straight', 1, 900001)
    assert np.isclose(sum(rows[0]['reward_components'].values()), rows[0]['team_return'], atol=1e-6)
    trace = validate_trace(collect(restored, policy, 'radial_straight', 900001))
    assert trace['schema'] == 'InterceptionReplay/3'
    assert trace['frames'][0]['assigned_target_ids'] is None
    assert trace['frames'][1]['physical_commands'] is not None
    rendered = html(trace, 1)
    assert "cfg.observation_layout==='relative_nearest_entities'&&cfg.communication_radius!=null" in rendered
