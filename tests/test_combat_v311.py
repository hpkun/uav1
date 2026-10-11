"""Geometry, direct increments, repeat fire and isolated v3.11 acceptance."""
from copy import deepcopy
import json
import math
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.combat_v310 import CombatEnvironmentV310
from env.combat_v311 import CombatEnvironmentV311
from env.combat_v35 import CombatEnvironmentV35
from env.config import load_config, validate_config, environment_dimensions
from env.models import AircraftState
from env.math_utils import wrap_angle
from env.sensor import sensor_geometry
from env.v30_observation import build_v30_observations
from env.v311_config import COMMAND, SCENARIO, SENSOR, WEAPON, REWARD
from env.v311_kinematics import advance_aircraft
from env.v311_scenario import random_combat_states_v311
from env.v311_weapon import MADSACFourKmWeapon, EFFECTIVE_HIT_DISTANCE
from env.v311_reward import tactical_rewards
from tools._prepare_v311 import capture310

CONFIG = 'configs/combat_environment_v311.yaml'


def state(x=0., y=0., altitude=3000., psi=0., theta=0., v=225., alive=True):
    return AircraftState(x, y, -altitude, v, theta, psi, alive)


def env():
    e = MultiUAVCombatEnv(CONFIG)
    e.reset(1)
    return e


def one_pair(e, own=None, target=None):
    e.red = [own or state()] + [state(alive=False) for _ in range(4)]
    e.blue = [target or state(x=1000., psi=np.pi)] + [state(alive=False) for _ in range(4)]


@pytest.mark.parametrize('action,theta,v', [([0, 0, 0], .2, 225.),
    ([1, 0, 0], .2, 225.), ([-1, 0, 0], .2, 225.),
    ([0, 1, 0], np.pi/3, 225.), ([0, -1, 0], -np.pi/3, 225.),
    ([0, 0, 1], .2, 300.), ([0, 0, -1], .2, 150.)])
def test_exact_direct_motion(action, theta, v):
    e = env()
    initial_theta = -.2 if action[1] < 0 else .2
    original = state(x=10., y=20., theta=initial_theta, psi=.4,
                     v=290. if action[2] > 0 else 160. if action[2] < 0 else 225.)
    result = advance_aircraft(original, action, .1, COMMAND, e.spec)
    expected_psi = wrap_angle(.4 + np.pi*action[0])
    assert result.psi == expected_psi
    assert result.theta == pytest.approx(theta)
    assert result.v == v
    np.testing.assert_allclose(result.as_array()[:3], original.as_array()[:3] +
        np.array([v*np.cos(theta)*np.cos(expected_psi), v*np.cos(theta)*np.sin(expected_psi),
                  -v*np.sin(theta)])*.1, atol=1e-12)
    assert original.theta == initial_theta and original.x == 10.


def test_dead_motion_clipping_no_controller_or_integrator_and_phi():
    e = env()
    assert not hasattr(e, 'dynamics') and not hasattr(e, 'integrator')
    assert 'controller' not in e.config['action']
    s = state(alive=False)
    np.testing.assert_array_equal(advance_aircraft(s, [1, 1, 1], .1, COMMAND, e.spec).as_array(), s.as_array())
    with (patch('env.combat_env.action_to_control', side_effect=AssertionError('controller used')),
          patch('env.integrator.RK4Integrator.step', side_effect=AssertionError('RK4 used'))):
        obs, _, _, _, info = e.step(np.ones((5, 3))*10.)
    assert not info['executed_red_phi'].any()
    assert not e.red_last_executed_phi.any() and not e.blue_last_executed_phi.any()
    assert not obs[:, 4].any()
    assert np.isfinite(obs).all()


@pytest.mark.parametrize('seed', range(100))
def test_single_draw_compact_initialization(seed):
    rng = np.random.default_rng(seed)
    reference = np.random.default_rng(seed)
    alpha = reference.uniform(-np.pi, np.pi)
    red, blue, actual = random_combat_states_v311(rng, **SCENARIO)
    assert actual == alpha and rng.bit_generator.state == reference.bit_generator.state
    u = np.array([np.cos(alpha), np.sin(alpha)])
    t = np.array([-np.sin(alpha), np.cos(alpha)])
    for team, side in ((red, -1), (blue, 1)):
        positions = np.array([[s.x, s.y] for s in team])
        np.testing.assert_allclose(positions.mean(0), side*4800*u, atol=1e-10)
        np.testing.assert_allclose((positions-side*4800*u)@t, SCENARIO['formation_offsets'], atol=1e-10)
        assert np.all(np.linalg.norm(positions, axis=1) < 5000)
        for s in team:
            assert s.altitude == 3000. and s.v == 225. and s.theta == 0.
            assert s.psi == wrap_angle(alpha if side == -1 else alpha+np.pi)
    a = env()
    o1, i1 = a.reset(seed)
    o2, i2 = a.reset(seed)
    np.testing.assert_array_equal(o1, o2)
    assert i1['nominal_anchor_separation'] == i2['nominal_anchor_separation'] == 9600.
    assert i1['minimum_inter_team_distance'] == pytest.approx(9600.)


@pytest.mark.parametrize('distance,degrees,visible', [(9000, 0, True), (10000, 0, True),
    (10001, 0, False), (5000, 59, True), (5000, 61, False)])
@pytest.mark.parametrize('plane', ['horizontal', 'vertical'])
def test_true_3d_simple_radar(distance, degrees, visible, plane):
    angle = np.deg2rad(degrees)
    target = state(x=distance*np.cos(angle),
        y=distance*np.sin(angle) if plane == 'horizontal' else 0.,
        altitude=3000.+distance*np.sin(angle) if plane == 'vertical' else 3000.)
    assert bool(sensor_geometry([state()], [target], SENSOR)[0][0, 0]) is visible


@pytest.mark.parametrize('degrees,positive,negative', [(25, .01, -.015), (10, .02, -.025), (3, .1, -.15)])
def test_reward_levels_with_forced_invisible_sensor(degrees, positive, negative):
    # Real FOV-external geometry cannot simultaneously have ATA/HA<=30 in a
    # 60-degree cone. Force a false mask and assert the reward never queries it.
    target = state(x=1000*np.cos(np.deg2rad(degrees)), y=1000*np.sin(np.deg2rad(degrees)), psi=0.)
    with patch('env.sensor.sensor_geometry', side_effect=AssertionError('reward consulted radar')):
        offense = tactical_rewards([state()], [target], REWARD)
        threat = tactical_rewards([state(x=target.x, y=target.y)], [state()], REWARD)
    assert offense['tactical_offense'][0] == positive
    assert threat['tactical_defense'][0] == negative


def test_guide_outside_radar_range_and_rewards_sensor_independent_bounded_alive():
    target = state(x=11000.)
    assert not sensor_geometry([state()], [target], SENSOR)[0][0, 0]
    assert tactical_rewards([state()], [target], REWARD)['guide'][0] == .001
    assert tactical_rewards([state()], [state(x=4000.)], REWARD)['guide'][0] == .001
    assert tactical_rewards([state()], [state(x=3999.)], REWARD)['guide'][0] == 0.
    assert tactical_rewards([state(alive=False)], [target], REWARD)['guide'][0] == 0.
    multiple = [state(x=1000.+i) for i in range(5)]
    assert tactical_rewards([state()], multiple, REWARD)['tactical_offense'][0] == .1
    assert tactical_rewards([state()], [state(x=1000., alive=False)], REWARD)['tactical'][0] == 0.
    e = env()
    one_pair(e, target=state(x=1000., psi=0.))
    _, expected = e.dense_combat_reward()
    e.config['sensor'] = dict(range_max=0., off_boresight_angle_max=0.)
    _, actual = e.dense_combat_reward()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])


@pytest.mark.parametrize('distance,ata,ha,ammo,alive_a,alive_t,qualifies', [
    (4000, 30, 30, 6, True, True, True), (4000.001, 0, 0, 6, True, True, False),
    (3000, 30.001, 0, 6, True, True, False), (3000, 0, 30.001, 6, True, True, False),
    (3000, 0, 0, 0, True, True, False), (3000, 0, 0, 6, False, True, False),
    (3000, 0, 0, 6, True, False, False)])
def test_equation7(distance, ata, ha, ammo, alive_a, alive_t, qualifies):
    # LOS along +x, ATA/HA from attacker attitude, AA=180 deg target heading.
    weapon = MADSACFourKmWeapon(**WEAPON)
    assert weapon.qualifies(state(psi=np.deg2rad(ata), theta=np.deg2rad(ha), alive=alive_a),
                            state(x=distance, psi=np.pi, alive=alive_t), ammo) is qualifies


def test_repeat_fire_no_pair_state_and_nearest_tie_selection():
    e = env()
    one_pair(e)
    with patch.object(e, 'attempt_hit_for_pair', return_value=False):
        for step in range(1, 4):
            # Poison compatibility slot: v3.11 must never consult it.
            e.red_fire_states = object()
            _, _, _, _, info = e.step(np.zeros((5, 3)), np.zeros((5, 3)))
            assert info['red_step_fire_attempts'] == 1
            assert e.red_ammo[0] == 6-step
    e.blue = [state(x=2000., psi=np.pi), state(x=1000., psi=np.pi), state(x=1000., psi=np.pi)] + [state(alive=False)]*2
    e.red[0] = state()
    with patch.object(e, 'attempt_hit_for_pair', return_value=False):
        attempts = e._entry_attempts(e.red, e.blue, None, 'red')
    assert [(i, j) for i, j, _ in attempts] == [(0, 1)]
    assert e.blue_fire_states is None


def test_six_rounds_exhaustion_each_step_one_attempt():
    e = env()
    one_pair(e)
    with patch.object(e, 'attempt_hit_for_pair', return_value=False):
        for step in range(8):
            _, _, _, _, info = e.step(np.zeros((5, 3)), np.zeros((5, 3)))
            assert info['red_step_fire_attempts'] == (1 if step < 6 else 0)
    assert e.red_ammo[0] == e.blue_ammo[0] == 0
    assert e.combat_counts['red']['ammo_used'] == 6


def test_simultaneous_mutual_destruction_and_zero_terminal_reward():
    e = env()
    one_pair(e)
    with patch.object(e, 'attempt_hit_for_pair', return_value=True):
        _, reward, terminated, truncated, info = e.step(np.zeros((5, 3)), np.zeros((5, 3)))
    assert terminated and not truncated
    assert info['red_step_fire_attempts'] == info['blue_step_fire_attempts'] == 1
    assert info['red_step_attack_kills'] == info['blue_step_attack_kills'] == 1
    assert info['red_win'] and not info['blue_win'] and not info['draw']
    assert not info['outcome_rewards'].any() and not info['team_casualty_rewards'].any()
    assert info['pbrs_enabled'] is False
    np.testing.assert_array_equal(reward, sum(info[f'r{i}_rewards'] for i in range(1,5)) + info['safety_rewards'])


def test_equation8_calibration_monotonic_shared_single_gaussian():
    w = MADSACFourKmWeapon(**WEAPON)
    assert EFFECTIVE_HIT_DISTANCE == 4000/math.log(6)
    assert w.expected_hit_probability(4000, 0, 0) == pytest.approx(.7, abs=1e-12)
    for dimension in ('distance', 'ata', 'ha'):
        probabilities = []
        for value in (0., .1, .5, 1.):
            args = dict(distance=2000., ata=.1, ha=.1)
            args[dimension] = value*4000 if dimension == 'distance' else value
            probabilities.append(w.expected_hit_probability(**args))
        assert all(a >= b for a, b in zip(probabilities, probabilities[1:]))
    class RNG:
        calls = 0
        def normal(self, loc, scale):
            assert (loc, scale) == (0., 1.)
            self.calls += 1
            return .4
    rng = RNG()
    threshold = np.pi*np.exp(-3000/EFFECTIVE_HIT_DISTANCE)
    assert w.attempt_hit(rng, 3000, .1, .8) == (.1+w.ata_noise_scale*.4 <= threshold and .8+w.ha_noise_scale*.4 <= threshold)
    assert rng.calls == 1


@pytest.mark.parametrize('red,blue,truncated,winner', [(5, 0, False, 'red'), (0, 5, False, 'blue'),
    (0, 0, False, 'red'), (5, 1, True, 'blue'), (1, 5, True, 'blue'), (5, 5, True, 'blue')])
def test_paper_outcome(red, blue, truncated, winner):
    e = env()
    for i in range(5):
        e.red[i].alive = i < red
        e.blue[i].alive = i < blue
    win, loss, draw, reason = e._outcome(truncated)
    assert win == (winner == 'red') and loss == (winner == 'blue') and not draw
    assert ('red_win' if winner == 'red' else 'blue_win') in reason


def test_actual_timeout_no_survivor_comparison():
    e = env()
    e.blue[1].alive = e.blue[2].alive = e.blue[3].alive = e.blue[4].alive = False
    e.steps = 999
    _, _, terminated, truncated, info = e.step(np.zeros((5,3)), np.zeros((5,3)))
    assert not terminated and truncated and info['blue_win'] and not info['draw']
    assert not info['outcome_rewards'].any()


@pytest.mark.parametrize('x,altitude', [(5100., 3000.), (0., -1.), (0., 6001.), (5100., -1.)])
def test_blue_boundary_return_identical_method_and_result(x, altitude):
    e = env()
    old = MultiUAVCombatEnv('configs/combat_environment_v310.yaml')
    old.reset(1)
    assert CombatEnvironmentV311._resolve_noncombat_losses is CombatEnvironmentV35._resolve_noncombat_losses
    for a in (e, old):
        a.red = [state() for _ in range(5)]
        a.blue = [state(x=x, altitude=altitude, theta=-.2)] + [state(alive=False) for _ in range(4)]
    assert e._resolve_noncombat_losses() == old._resolve_noncombat_losses()
    np.testing.assert_array_equal(e.blue[0].as_array(), old.blue[0].as_array())
    assert e.blue[0].alive and e.blue_returns == old.blue_returns


@pytest.mark.parametrize('cause,expected_component', [('horizontal', 'r2'), ('ceiling', 'r2'), ('ground', 'r1')])
def test_red_boundary_reward(cause, expected_component):
    e = env()
    e.red[0] = state(x=5100. if cause == 'horizontal' else 0.,
                     altitude=-1. if cause == 'ground' else 6001. if cause == 'ceiling' else 3000.)
    e.red_ammo.fill(0)
    e.blue_ammo.fill(0)
    _, _, _, _, info = e.step(np.zeros((5,3)), np.zeros((5,3)))
    assert not e.red[0].alive
    assert info[f'{expected_component}_rewards'][0] == -10.


@pytest.mark.parametrize('section,key,value', [('action', 'controller', dict(heading_time_constant=2)),
    ('action', 'normal_load_max', 8), ('scenario', 'heading_perturbation_max', .1),
    ('scenario', 'lateral_min', -2500), ('scenario', 'speed_perturbation_max', 10),
    ('weapon', 'range_max', 2000), ('reward', 'potential_scale', 1),
    ('reward', 'win_reward', 20), ('sensor', 'noise', .1), ('blue_policy', 'desired_speed', 260)])
def test_validator_rejects_engineering_parameters_and_drift(section, key, value):
    c = deepcopy(load_config(CONFIG))
    c[section][key] = value
    with pytest.raises(ValueError):
        validate_config(c)


def test_frozen_observation_layout_blue_strategy_and_algorithms():
    e = env()
    assert isinstance(e, CombatEnvironmentV311)
    assert environment_dimensions(e.config) == (66, 3, 5)
    assert CombatEnvironmentV311._observations is CombatEnvironmentV310._observations
    assert e.fixed_policy.__class__ is env().fixed_policy.__class__
    e.red_last_executed_phi.fill(0.)
    expected = build_v30_observations(e.red, e.blue, e.config['observation'], e.config['sensor'], e.red_ammo, 6, np.zeros(5))
    np.testing.assert_array_equal(e._observations(), expected)
    assert expected.shape == (5, 66) and not expected[:,4].any()
    assert np.all(expected[:,7] == 1.)
    for algorithm in ('mappo', 'rmappo', 'ea_mappo', 'stea_mappo'):
        assert Path(f'configs/{algorithm}_5v5_v311.yaml').read_bytes() == Path(f'configs/{algorithm}_5v5_v310.yaml').read_bytes()
    old = load_config('configs/combat_environment_v310.yaml')
    assert 'controller' in old['action'] and old['weapon']['range_max'] == 2000.


def test_previous_v310_full_trajectory_reference():
    assert capture310() == json.loads(Path('tests/v311_v310_reference.json').read_text())
