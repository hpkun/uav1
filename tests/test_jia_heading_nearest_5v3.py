"""Tests for Interception/15: 5v3 radial speed-heading with 41D fixed-ID entity obs, nearest shaping, and replay."""
from dataclasses import replace
import json

import numpy as np
import pytest

from algorithm.maddpg import BoxActor
from algorithm.maddpg_config import NormalizedMADDPGConfig
from algorithm.matd3_config import NormalizedMATD3Config
from env import JiaHeadingNearest5v3Config, make_environment
from env.jia_heading import wrap_heading
from experiment.cli import main as cli_main
from experiment.interception import evaluate_policy, load_run, make_policy, read_config, summarize, train
from tools.replay import collect, html, main as replay_main, validate_trace

ENV_5V3 = 'configs/interception_radial_5v3_jia_heading_nearest_d08_h003_time0015.yaml'
ENV_14 = 'configs/interception_radial_5v5_jia_heading_nearest_d08_h003_time0015.yaml'
ENV_13 = 'configs/interception_radial_5v5_jia_heading_time01.yaml'
ALG_MATD3 = 'configs/matd3_jia_heading_interval10.yaml'
ALG_MADDPG = 'configs/maddpg_jia_heading_interval10.yaml'


def test_5v3_config_contract_and_historical_preservation():
    cfg_t, params_t, rollout_t = read_config(ENV_5V3, ALG_MATD3)
    cfg_d, params_d, rollout_d = read_config(ENV_5V3, ALG_MADDPG)
    assert isinstance(cfg_t, JiaHeadingNearest5v3Config) and isinstance(cfg_d, JiaHeadingNearest5v3Config)
    assert isinstance(params_t, NormalizedMATD3Config) and isinstance(params_d, NormalizedMADDPGConfig)
    assert rollout_t == 128 and rollout_d == 128
    assert (cfg_t.n_defenders, cfg_t.n_attackers) == (5, 3)
    assert (cfg_t.approach_weight, cfg_t.heading_weight, cfg_t.time_penalty_per_step) == (0.8, 0.03, 0.015)

    env = make_environment(cfg_t)
    contract = env.contract
    assert contract['schema'] == 'Interception/15'
    assert contract['version'] == 'interception_15'
    assert contract['n_agents'] == 5
    assert contract['n_attackers'] == 3
    assert contract['observation_dim'] == 41
    assert contract['state_dim'] == 50
    assert contract['action_dim'] == 2
    assert contract['action_schema'] == 'normalized_speed_turn_box'
    assert contract['assignment_schema'] is None
    assert contract['reward_schema'].count('fixed_shared_time_cost') == 1
    assert make_environment(replace(cfg_t, time_penalty_per_step=0.0)).contract['reward_schema'].count('fixed_shared_time_cost') == 0

    with pytest.raises(ValueError):
        JiaHeadingNearest5v3Config(n_attackers=5)
    with pytest.raises(ValueError):
        JiaHeadingNearest5v3Config(visible_attackers=5)
    with pytest.raises(ValueError):
        JiaHeadingNearest5v3Config(assignment_method='ideal_global_distance_matching')
    with pytest.raises(ValueError, match='MADDPG/MATD3'):
        read_config(ENV_5V3, 'configs/mappo_interception.yaml')

    # Historical Interception/13 and Interception/14 contracts and implementation digests must remain untouched.
    env13 = make_environment(read_config(ENV_13, ALG_MATD3)[0])
    env14 = make_environment(read_config(ENV_14, ALG_MATD3)[0])
    assert env13.contract['implementation_sha256'] == '46230752bf44cca962b0061c62f56c8bd70ae0e13cd21ced226b2a6ee0dfe21d'
    assert env14.contract['implementation_sha256'] == 'e71343f91623db71ad9172d753adae4391a4e4ac8ca3c2f7529b7839b3c065a0'


def test_5v3_radial_geometry_observation_state_and_network_shapes():
    cfg, params_t, _ = read_config(ENV_5V3, ALG_MATD3)
    _, params_d, _ = read_config(ENV_5V3, ALG_MADDPG)
    env = make_environment(cfg)
    obs, info = env.reset(seed=7)

    assert obs.shape == (5, 41) and obs.dtype == np.float32
    assert info['central_state'].shape == (50,) and info['central_state'].dtype == np.float32
    assert info['active_masks'].shape == (5,) and np.all(info['active_masks'] == 1.0)
    assert env.defender_alive.shape == (5,) and env.attacker_alive.shape == (3,)
    assert info['assigned_target_ids'] is None

    # Check 3 attackers on 120-deg outer ring heading inward, and 5 defenders on 72-deg inner ring heading outward.
    att_r = np.linalg.norm(env.attacker_positions, axis=1)
    def_r = np.linalg.norm(env.defender_positions, axis=1)
    assert np.all((160.0 <= att_r) & (att_r <= 180.0))
    assert np.all((50.0 <= def_r) & (def_r <= 60.0))
    att_angles = np.arctan2(env.attacker_positions[:, 1], env.attacker_positions[:, 0])
    def_angles = np.arctan2(env.defender_positions[:, 1], env.defender_positions[:, 0])
    for k in range(3):
        gap = np.mod(att_angles[(k + 1) % 3] - att_angles[k], 2 * np.pi)
        assert abs(gap - 2 * np.pi / 3) <= np.pi / 30 + 1e-9
    for k in range(5):
        gap = np.mod(def_angles[(k + 1) % 5] - def_angles[k], 2 * np.pi)
        assert abs(gap - 2 * np.pi / 5) <= np.pi / 30 + 1e-9
    np.testing.assert_allclose(env.attacker_headings, wrap_heading(att_angles + np.pi), atol=1e-12)
    np.testing.assert_allclose(env.defender_headings, wrap_heading(def_angles), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(env.defender_velocities, axis=1), 6.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(env.attacker_velocities, axis=1), 6.0, atol=1e-12)

    # Verify exact own, 4-ally, and 3-enemy slices for defender 2.
    i = 2
    own_pos = env.defender_positions[i]
    own_vel = env.defender_velocities[i]
    np.testing.assert_allclose(obs[i, 0:2], (own_pos - env.base) / cfg.arena_half_extent, atol=1e-6)
    np.testing.assert_allclose(obs[i, 2:4], own_vel / cfg.defender_speed_max, atol=1e-6)
    np.testing.assert_allclose(obs[i, 4:6], [np.cos(env.defender_headings[i]), np.sin(env.defender_headings[i])], atol=1e-6)
    for slot, j in enumerate((0, 1, 3, 4)):
        k = 6 + slot * 5
        np.testing.assert_allclose(obs[i, k:k + 2], (env.defender_positions[j] - own_pos) / (2 * cfg.arena_half_extent), atol=1e-6)
        np.testing.assert_allclose(obs[i, k + 2:k + 4], (env.defender_velocities[j] - own_vel) / (2 * cfg.defender_speed_max), atol=1e-6)
        assert obs[i, k + 4] == 1.0
    for j in range(3):
        k = 26 + j * 5
        np.testing.assert_allclose(obs[i, k:k + 2], (env.attacker_positions[j] - own_pos) / (2 * cfg.arena_half_extent), atol=1e-6)
        np.testing.assert_allclose(obs[i, k + 2:k + 4], (env.attacker_velocities[j] - own_vel) / (cfg.defender_speed_max + cfg.attacker_speed_max), atol=1e-6)
        assert obs[i, k + 4] == 1.0

    # Verify communication_radius / sensing_radius masking and central_state zeroing for dead entities.
    limited_env = make_environment(replace(cfg, communication_radius=50.0, sensing_radius=100.0))
    limited_env.reset(seed=7)
    limited_env.defender_positions[:] = [[0, 0], [30, 0], [80, 0], [0, 30], [0, -30]]
    limited_env.attacker_positions[:] = [[60, 0], [150, 0], [90, 0]]
    limited_env.defender_alive[4] = False
    limited_env.attacker_alive[1] = False
    lobs = limited_env.observations()
    assert lobs[0, 6 + 0 * 5 + 4] == 1.0
    np.testing.assert_array_equal(lobs[0, 6 + 1 * 5:6 + 2 * 5], 0.0)
    assert lobs[0, 6 + 2 * 5 + 4] == 1.0
    np.testing.assert_array_equal(lobs[0, 6 + 3 * 5:6 + 4 * 5], 0.0)
    assert lobs[0, 26 + 0 * 5 + 4] == 1.0
    np.testing.assert_array_equal(lobs[0, 26 + 1 * 5:26 + 2 * 5], 0.0)
    assert lobs[0, 26 + 2 * 5 + 4] == 1.0
    np.testing.assert_array_equal(lobs[4], 0.0)
    state = limited_env.central_state()
    np.testing.assert_array_equal(state[20:25], 0.0)
    np.testing.assert_array_equal(state[30:35], 0.0)
    np.testing.assert_array_equal(state[48:50], 0.0)

    # Check MATD3 & MADDPG network shapes (obs 41, critic state 50 + 10 joint actions = 60).
    p_matd3 = make_policy('matd3', cfg, params_t, 'cpu', 7)
    p_maddpg = make_policy('maddpg', cfg, params_d, 'cpu', 7)
    assert isinstance(p_matd3.actors[0], BoxActor) and isinstance(p_maddpg.actors[0], BoxActor)
    assert p_matd3.actors[0].net[0].in_features == 41 and p_maddpg.actors[0].net[0].in_features == 41
    assert p_matd3.critics_1[0].net[0].in_features == 60 and p_matd3.critics_2[0].net[0].in_features == 60
    assert p_maddpg.critics[0].net[0].in_features == 60


def test_5v3_full_episode_all_capture_and_unopposed_breach():
    cfg, _, _ = read_config(ENV_5V3)
    env = make_environment(cfg)
    env.reset(seed=7)
    totals = {}
    while not env.finished:
        actions = np.zeros((5, 2))
        alive_attackers = np.flatnonzero(env.attacker_alive)
        for i in np.flatnonzero(env.defender_alive):
            target = int(min(alive_attackers, key=lambda j: np.linalg.norm(env.attacker_positions[j] - env.defender_positions[i])))
            delta = env.attacker_positions[target] - env.defender_positions[i]
            error = wrap_heading(np.arctan2(delta[1], delta[0]) - env.defender_headings[i])
            actions[i, 0] = -1.0 if abs(error) > 0.35 else 1.0
            actions[i, 1] = np.clip(4.0 * error, -1.0, 1.0)
        _, _, _, _, info = env.step(actions)
        for key, value in info['reward_components'].items():
            totals[key] = totals.get(key, 0.0) + value
    assert (env.captured_count, env.penetrated_count) == (3, 0)
    assert int(env.defender_alive.sum()) == 2
    assert info['episode_summary']['n_attackers'] == 3
    assert info['episode_summary']['outcome'] == 'all_captured'
    assert totals['capture'] == pytest.approx(60.0)
    assert np.isclose(sum(totals.values()), env.team_return, atol=1e-6)
    summary = summarize([dict(info['episode_summary'], reward_components=totals)])
    assert summary['all_capture_rate'] == 1.0
    assert summary['capture_efficiency'] == 1.0

    # Unopposed breach: all 3 attackers breach (-150 penalty).
    env2 = make_environment(cfg)
    env2.reset(seed=7)
    totals2 = {}
    while not env2.finished:
        _, _, _, _, info2 = env2.step(np.zeros((5, 2)))
        for key, value in info2['reward_components'].items():
            totals2[key] = totals2.get(key, 0.0) + value
    assert (env2.captured_count, env2.penetrated_count) == (0, 3)
    assert info2['episode_summary']['n_attackers'] == 3
    assert totals2['penetration'] == pytest.approx(-150.0)
    assert np.isclose(sum(totals2.values()), env2.team_return, atol=1e-6)


@pytest.mark.parametrize('algorithm,param_cls', [
    ('matd3', NormalizedMATD3Config),
    ('maddpg', NormalizedMADDPGConfig),
])
def test_5v3_preflight_train_eval_and_replay(tmp_path, algorithm, param_cls):
    alg_path = ALG_MATD3 if algorithm == 'matd3' else ALG_MADDPG
    config, _, _ = read_config(ENV_5V3, alg_path)
    out = tmp_path / f'{algorithm}_5v3'
    if algorithm == 'matd3':
        preflight = cli_main([
            'preflight',
            '--env-config', ENV_5V3,
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
        assert preflight['environment_contract']['schema'] == 'Interception/15'
        assert preflight['environment_contract']['n_agents'] == 5
        assert preflight['environment_contract']['n_attackers'] == 3
        assert preflight['environment_contract']['observation_dim'] == 41
        assert preflight['environment_contract']['state_dim'] == 50

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
        eval_episodes=2,
        log_every_steps=6,
        update_log_interval=2,
    )
    restored, policy, name, _, settings = load_run(out, 'Best', 'cpu')
    assert name == algorithm
    assert settings['environment_contract']['version'] == 'interception_15'
    assert settings['environment_contract']['n_attackers'] == 3
    assert settings['environment_contract']['observation_dim'] == 41
    assert settings['environment_contract']['state_dim'] == 50

    # Verify evaluation summary uses attacker_count=3 and replay CLI + HTML handle 5v3 cleanly.
    rows = evaluate_policy(restored, policy, 'radial_straight', 2, 900001)
    assert all(r['n_attackers'] == 3 for r in rows)
    eval_summary = summarize(rows)
    assert eval_summary['capture_efficiency'] == pytest.approx(sum(r['captured_count'] for r in rows) / (3 * 2))

    replay_dir = tmp_path / f'{algorithm}_5v3_replay'
    trace = replay_main([
        '--checkpoint', str(out),
        '--label', 'Best',
        '--attack-mode', 'radial_straight',
        '--seed', '900001',
        '--device', 'cpu',
        '--frame-stride', '1',
        '--html',
        '--output', str(replay_dir),
    ])
    assert trace['schema'] == 'InterceptionReplay/3'
    assert len(trace['frames'][0]['positions'][0]) == 5
    assert len(trace['frames'][0]['positions'][1]) == 3
    assert len(trace['frames'][0]['defender_alive']) == 5
    assert len(trace['frames'][0]['attacker_alive']) == 3
    assert len(trace['frames'][1]['physical_commands'][0]) == 5
    assert len(trace['frames'][1]['physical_commands'][1]) == 3
    assert (replay_dir / 'trace.json.gz').exists()
    assert (replay_dir / 'replay.html').exists()
    rendered = (replay_dir / 'replay.html').read_text(encoding='utf-8')
    assert '"n_defenders":5' in rendered and '"n_attackers":3' in rendered
    assert '中心型 ${nDef}v${nAtt}（${nAtt}向）拦截回放' in rendered
    assert '${summary.captured_count}/${nAtt}' in rendered

    # Tampering with attacker_alive length must fail replay validation.
    bad_trace = json.loads(json.dumps(trace))
    bad_trace['frames'][0]['attacker_alive'].append(False)
    with pytest.raises(ValueError, match='attacker status'):
        validate_trace(bad_trace)
