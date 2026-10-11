"""Desired-command dynamics, frozen combat semantics and version isolation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from env.combat_env import MultiUAVCombatEnv
from env.combat_v313 import CombatEnvironmentV313
from env.combat_v314 import CombatEnvironmentV314
from env.config import load_config, validate_config, environment_dimensions
from env.control import action_to_control, action_to_target
from env.models import AircraftState
from env.math_utils import wrap_angle
from env.dynamics import PointMassDynamics
from env.integrator import RK4Integrator
from env.v313_observation import build_team_observations_v313
from env.v314_observation import build_team_observations_v314
from env.v313_geometry import combat_geometry_v313
from env.v313_weapon import MADSACPaperHAWeapon
from tools._capture_v33_reference import canonical, states

CONFIG = 'configs/combat_environment_v314.yaml'


def env(version='314', seed=1):
    e = MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml')
    e.reset(seed)
    return e


def state(psi=0., theta=0., v=225., x=0., y=0., altitude=3000., alive=True):
    return AircraftState(x, y, -altitude, v, theta, psi, alive)


def advance(action, s=None, version='314'):
    e = env(version)
    original = state() if s is None else s.copy()
    aircraft = [original.copy() for _ in range(5)]
    phis = e._advance(aircraft, np.tile(action, (5, 1)))
    return aircraft[0], phis[0], action_to_control(original, action, e.config['action']) if version == '314' else None


def capture313():
    result = {}
    for mode in ('ZERO', 'PURSUIT'):
        for seed in (1, 31, 10000000):
            e = env('313', seed)
            obs, info = e.reset(seed)
            h = hashlib.sha256(json.dumps(canonical((states(e), obs, info)), sort_keys=True).encode())
            while True:
                action = np.zeros((5, 3)) if mode == 'ZERO' else e.fixed_policy.team_actions(e.red, e.blue)
                obs, reward, term, trunc, info = e.step(action)
                h.update(json.dumps(canonical((states(e), obs, reward, term, trunc, info,
                    e.rng.bit_generator.state)), sort_keys=True).encode())
                if term or trunc:
                    break
            result[f'{mode}_{seed}'] = h.hexdigest()
    return result


def test_version_factory_minimal_inheritance_and_config_frozen():
    assert issubclass(CombatEnvironmentV314, CombatEnvironmentV313)
    assert set(CombatEnvironmentV314.__dict__) - {'__module__', '__doc__'} == {
        'environment_version', '__init__', '_advance', '_observations'}
    for name in ('step', 'reset', 'build_weapon', 'attempt_hit_for_pair',
                 'dense_combat_reward', 'initial_states', '_entry_attempts',
                 '_resolve_combat', '_resolve_noncombat_losses', '_outcome'):
        assert getattr(CombatEnvironmentV314, name) is getattr(CombatEnvironmentV313, name)
    new, old = env(), env('313')
    assert type(new) is CombatEnvironmentV314
    assert isinstance(new.dynamics, PointMassDynamics)
    assert isinstance(new.integrator, RK4Integrator)
    assert environment_dimensions(new.config) == (56, 3, 5)
    cfg = deepcopy(new.config)
    controller = cfg['action'].pop('controller')
    assert controller == dict(heading_time_constant=2., pitch_time_constant=2., speed_time_constant=2., normal_load_max=8.)
    cfg['environment_version'] = '3.13'
    assert cfg == old.config


@pytest.mark.parametrize('key', ['heading_time_constant', 'pitch_time_constant', 'speed_time_constant', 'normal_load_max'])
@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf'), True, '2', 3.])
def test_controller_strict_contract(key, value):
    cfg = load_config(CONFIG)
    cfg['action']['controller'][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize('change', ['extra', 'missing', 'none', 'missing_controller', 'extra_action', 'reward', 'scenario'])
def test_validator_schema_and_frozen_sections(change):
    cfg = load_config(CONFIG)
    if change == 'extra': cfg['action']['controller']['lag'] = 1.
    if change == 'missing': del cfg['action']['controller']['normal_load_max']
    if change == 'none': cfg['action']['controller'] = None
    if change == 'missing_controller': del cfg['action']['controller']
    if change == 'extra_action': cfg['action']['extra'] = 1.
    if change == 'reward': cfg['reward']['guide_reward'] = .1
    if change == 'scenario': cfg['scenario']['center_radius'] = 4700.
    with pytest.raises(ValueError): validate_config(cfg)


@pytest.mark.parametrize('psi', [0., .7, -2., np.pi-1e-5])
def test_zero_action_preserves_level_speed_heading_and_moves_forward(psi):
    before = state(psi=psi)
    after, phi, _ = advance([0, 0, 0], before)
    np.testing.assert_allclose(after.as_array(), before.as_array()+np.r_[before.velocity_vector()*.1, 0, 0, 0], atol=1e-12, rtol=0)
    assert phi == pytest.approx(0., abs=1e-12)


def test_heading_half_command_and_symmetry():
    pos, _, _ = advance([.5, 0, 0])
    neg, _, _ = advance([-.5, 0, 0])
    target = action_to_target(state(), [.5, 0, 0], load_config(CONFIG)['action']['command'])
    assert target.heading == pytest.approx(np.pi/2)
    assert 0 < pos.psi < np.radians(10)
    assert neg.psi == pytest.approx(-pos.psi, abs=1e-13)
    assert neg.x == pytest.approx(pos.x, abs=1e-13)
    assert neg.y == pytest.approx(-pos.y, abs=1e-13)


def test_full_heading_is_bounded_and_pi_tie_uses_existing_wrap():
    target = action_to_target(state(), [1, 0, 0], load_config(CONFIG)['action']['command'])
    assert abs(target.heading) == np.pi
    after, phi, c = advance([1, 0, 0])
    assert np.isfinite(after.as_array()).all()
    assert 0 < abs(after.psi) < np.radians(10)
    assert after.psi < 0  # +pi wraps to -pi under the historical [-pi, pi) convention.
    assert -np.pi/2 <= phi <= np.pi/2 and c.nz <= 8.+1e-12


@pytest.mark.parametrize('axis', [1, 2])
def test_pitch_and_speed_are_partial_first_step_responses(axis):
    positive, negative = np.zeros(3), np.zeros(3)
    positive[axis], negative[axis] = .5, -.5
    pos, _, _ = advance(positive)
    neg, _, _ = advance(negative)
    if axis == 1:
        assert 0 < pos.theta < np.pi/6
        assert -np.pi/6 < neg.theta < 0
        assert pos.altitude > 3000 and neg.altitude < 3000
    else:
        assert 0 < pos.v-225 < 25
        assert -25 < neg.v-225 < 0
        assert pos.v-225 == pytest.approx(225-neg.v, abs=1e-12)


@pytest.mark.parametrize('action', [[0,0,0],[.5,0,0],[-.5,0,0],[0,.5,0],[0,0,.5],[.43,.6,-.31],[.5,-1,0]])
@pytest.mark.parametrize('s', [state(), state(psi=3.13, theta=.2, v=280), state(psi=-2, theta=-.3, v=170)])
def test_exact_controller_rk4_path_and_load_projection(action, s):
    e = env()
    c = action_to_control(s, action, e.config['action'])
    assert 0 <= c.nz <= 8.+1e-12 and abs(c.phi) <= np.pi/2
    expected = e.integrator.step(s, c, e.dynamics, e.spec)
    aircraft = [s.copy() for _ in range(5)]
    with patch.object(e.dynamics, 'derivatives', wraps=e.dynamics.derivatives) as derivatives:
        phis = e._advance(aircraft, np.tile(action, (5, 1)))
    assert derivatives.call_count == 20  # four actual RK4 stages for each live aircraft.
    for actual in aircraft:
        np.testing.assert_array_equal(actual.as_array(), expected.as_array())
    np.testing.assert_allclose(phis, c.phi, atol=1e-7, rtol=0)
    if abs(action[0]) >= .43 and s.theta == 0:
        assert c.nz == pytest.approx(8.)


def test_phi_exact_control_precision_at_vertical_projection_boundary():
    e = env()
    action = [.5, -1, 0]  # vertical A projected to zero, phi exactly pi/2.
    c = action_to_control(state(), action, e.config['action'])
    aircraft = [state() for _ in range(5)]
    phis = e._advance(aircraft, np.tile(action, (5, 1)))
    assert phis[0] == c.phi
    obs = build_team_observations_v314(aircraft, [state(x=4000)]*5, e.config['observation'], phis)
    assert np.max(np.abs(obs[:,4])) <= 1.


@pytest.mark.parametrize('magnitude', [.1, .25, .5, 1.])
def test_twenty_step_recomputed_relative_heading_response(magnitude):
    e = env()
    sides = [[state() for _ in range(5)], [state() for _ in range(5)]]
    histories = []
    for aircraft, sign in zip(sides, [1, -1]):
        history = []
        for _ in range(20):
            before = aircraft[0].copy()
            action = [sign*magnitude, 0, 0]
            target = action_to_target(before, action, e.config['action']['command'])
            assert abs(wrap_angle(target.heading-before.psi)) == pytest.approx(np.pi*magnitude)
            phis = e._advance(aircraft, np.tile(action, (5,1)))
            after = aircraft[0]
            assert np.isfinite(after.as_array()).all() and np.isfinite(phis).all()
            assert abs(wrap_angle(after.psi-before.psi)) < np.radians(10)
            assert 150 <= after.v <= 300 and abs(after.theta) <= np.pi/3
            history.append(after.as_array())
        histories.append(np.array(history))
    np.testing.assert_allclose(histories[0][:,[0,2,3,4]], histories[1][:,[0,2,3,4]], atol=1e-10, rtol=0)
    # 180deg is a tie: both signs choose -pi under wrap, so compare that direction directly.
    multiplier = 1 if magnitude == 1 else -1
    np.testing.assert_allclose(histories[0][:,[1,5]], multiplier*histories[1][:,[1,5]], atol=1e-10, rtol=0)


def test_v313_v314_action_execution_difference_and_old_trace_exact():
    old, _, _ = advance([.43, 0, 0], version='313')
    new, _, _ = advance([.43, 0, 0])
    assert np.degrees(old.psi) == pytest.approx(77.4)
    assert 0 < np.degrees(new.psi) < 10
    assert capture313() == json.loads(Path('tests/v314_v313_reference.json').read_text())


def test_dead_not_advanced_phi_observation_and_other_55_features():
    e = env()
    assert not e._observations()[:,4].any()
    e.red[1].alive = False
    before = e.red[1].as_array().copy()
    obs, *_ = e.step(np.tile([.5,0,0],(5,1)), np.zeros((5,3)))
    np.testing.assert_array_equal(before, e.red[1].as_array())
    assert e.red_last_executed_phi[1] == 0 and not obs[1].any()
    live = e.red_alive_mask.astype(bool)
    assert np.all(e.red_last_executed_phi[live] != 0)
    np.testing.assert_array_equal(obs[live,4], e.red_last_executed_phi[live]/(np.pi/2))
    assert np.max(np.abs(obs[:,4])) <= 1
    old = build_team_observations_v313(e.red,e.blue,e.config['observation'])
    obs[:,4] = 0
    np.testing.assert_array_equal(obs, old)
    e.reset(1)
    assert not e._observations()[:,4].any()


@pytest.mark.parametrize('phis', [np.zeros(4), np.zeros((5,1)), np.array([0,0,0,0,np.nan]), np.array([0,0,0,0,np.inf])])
def test_phi_input_strict_shape_finite(phis):
    e = env()
    with pytest.raises(ValueError):
        build_team_observations_v314(e.red,e.blue,e.config['observation'],phis)


def test_initialization_both_sides_zero_actions_approach():
    old, e = env('313',31), env('314',31)
    for side in ('red','blue'):
        np.testing.assert_array_equal([s.as_array() for s in getattr(old,side)],[s.as_array() for s in getattr(e,side)])
    distances = []
    for _ in range(20):
        for own, target in [(e.red,e.blue),(e.blue,e.red)]:
            center = np.mean([s.as_array()[:2] for s in target],axis=0)
            for s in own:
                assert abs(wrap_angle(np.arctan2(*(center-s.as_array()[:2])[::-1])-s.psi)) < .05
        centers = [np.mean([s.as_array()[:2] for s in team],axis=0) for team in (e.red,e.blue)]
        distances.append(np.linalg.norm(centers[1]-centers[0]))
        e.step(np.zeros((5,3)),np.zeros((5,3)))
    assert np.all(np.diff(distances)<0) and distances[-1]>4000


@pytest.mark.parametrize('heading,elevation,speed', [(.7,.2,250.),(-2.,-.3,250.),(3.1,.1,250.)])
def test_blue_policy_reconstructs_desired_targets(heading,elevation,speed):
    e = env()
    own = state(psi=2.9,theta=.1,v=225)
    action = e.fixed_policy.action_toward(own,heading,elevation,speed)
    target = action_to_target(own,action,e.config['action']['command'])
    assert abs(wrap_angle(target.heading-heading)) < 2e-7
    assert target.pitch == pytest.approx(elevation,abs=1e-7)
    assert target.speed == 250.


def test_red_blue_same_advance_and_full_step_symmetry():
    e = env()
    action = np.tile([.3,.1,-.2],(5,1))
    e.red=[state(x=100*i) for i in range(5)]
    e.blue=deepcopy(e.red)
    red_phi=e._advance(e.red,action)
    blue_phi=e._advance(e.blue,action)
    np.testing.assert_array_equal(red_phi,blue_phi)
    np.testing.assert_array_equal([s.as_array() for s in e.red],[s.as_array() for s in e.blue])
    # Both calls made by inherited step also use the same hook and RK4.
    e.red=[state(x=-4400+100*i) for i in range(5)]
    e.blue=deepcopy(e.red)
    with patch.object(e.integrator,'step',wraps=e.integrator.step) as integrate:
        e.step(action,action)
    assert integrate.call_count == 10
    np.testing.assert_array_equal(e.red_last_executed_phi,e.blue_last_executed_phi)
    np.testing.assert_array_equal([s.as_array() for s in e.red],[s.as_array() for s in e.blue])


@pytest.mark.parametrize('distance', [100., 2500., 5000.])
def test_static_reward_weapon_geometry_sampler_and_safety_regression(distance):
    old, new = env('313'),env()
    old.red=[state(x=30*i,theta=.1*i) for i in range(5)]
    old.blue=[state(x=distance+30*i,psi=np.pi,altitude=3000+50*i) for i in range(5)]
    new.red,new.blue=deepcopy(old.red),deepcopy(old.blue)
    assert new.config['reward'] == old.config['reward']
    assert new.config['weapon'] == old.config['weapon']
    assert type(new.weapon) is type(old.weapon) is MADSACPaperHAWeapon
    reward_old, diag_old=old.dense_combat_reward()
    reward_new, diag_new=new.dense_combat_reward()
    np.testing.assert_array_equal(reward_new,reward_old)
    for k in diag_old: np.testing.assert_array_equal(diag_new[k],diag_old[k])
    for k in combat_geometry_v313(old.red,old.blue):
        np.testing.assert_array_equal(combat_geometry_v313(old.red,old.blue)[k],combat_geometry_v313(new.red,new.blue)[k])
    for a,b in zip(old.weapon.eligibility(old.red,old.blue,old.red_ammo),new.weapon.eligibility(new.red,new.blue,new.red_ammo)):
        np.testing.assert_array_equal(a,b)
    old.rng=np.random.default_rng(314);new.rng=np.random.default_rng(314)
    for _ in range(50):
        assert old.attempt_hit_for_pair(old.red[0],old.blue[0]) == new.attempt_hit_for_pair(new.red[0],new.blue[0])
    assert old.rng.bit_generator.state==new.rng.bit_generator.state


@pytest.mark.parametrize('dead_side,truncated', [('blue',False),('red',False),(None,True)])
def test_outcome_regression(dead_side,truncated):
    old,new=env('313'),env()
    if dead_side:
        for e in (old,new):
            for s in getattr(e,dead_side): s.alive=False
    assert old._outcome(truncated)==new._outcome(truncated)


@pytest.mark.parametrize('name',['mappo','rmappo','ea_mappo','stea_mappo'])
def test_algorithm_configs_byte_identical(name):
    assert Path(f'configs/{name}_5v5_v314.yaml').read_bytes()==Path(f'configs/{name}_5v5_v313.yaml').read_bytes()


def test_recording_factory_dispatch():
    from tools.combat_visualization import RecordingCombatEnv
    e=RecordingCombatEnv(CONFIG)
    assert isinstance(e,CombatEnvironmentV314)
    assert e.reset(1)[0].shape==(5,56)


def test_existing_four_checkpoint_isolation_cuda():
    from tools.validate_combat_v314 import check_checkpoint_isolation
    rows=check_checkpoint_isolation()
    assert len(rows)==4
    assert all(r['old_checkpoint_accepted'] and r['cross_version_rejected'] for r in rows.values())


def test_real_vector_auto_reset_phi_does_not_leak():
    from tools.validate_combat_v314 import check_vector_reset
    result=check_vector_reset()
    assert result['auto_reset_phi_zero'] and result['reset_equal_fresh'] and result['first_step_equal_fresh']


@pytest.mark.parametrize('module_name,class_name', [
    ('mappo','MAPPOTrainingRunner'),('rmappo','RMAPPOTrainingRunner'),
    ('ea_mappo','EAMAPPOTrainingRunner'),('stea_mappo','STEAMAPPOTrainingRunner')])
def test_training_evaluation_record_and_start_log_version(module_name,class_name):
    import importlib
    from types import SimpleNamespace
    module=importlib.import_module(f'algorithm.{module_name}.runner')
    cls=getattr(module,class_name)
    runner=object.__new__(cls)
    runner.env_config=load_config(CONFIG)
    runner.trainer=SimpleNamespace(sampled_steps=512)
    runner.evaluation_seeds=[10000000,10000001]
    with patch.object(module,'evaluate',return_value={'evaluation_episodes':2}):
        row=runner._evaluation_record()
    assert row['environment_version']=='3.14' and row['sampled_steps']==512


def test_no_direct_motion_called_and_explicit_rk4_instance():
    e=env()
    with patch('env.combat_v311.advance_aircraft',side_effect=AssertionError('direct-motion called')):
        obs,*_=e.step(np.tile([.5,0,0],(5,1)),np.zeros((5,3)))
    assert obs.shape==(5,56)
