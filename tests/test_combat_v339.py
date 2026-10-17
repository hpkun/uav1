"""Blue speed-only isolation, exact boundaries and canonical observer protection."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest

from env.config import load_config, validate_config, environment_dimensions
from env.combat_env import MultiUAVCombatEnv
from env.combat_v338 import CombatEnvironmentV338
from env.combat_v339 import CombatEnvironmentV339
from env.models import AircraftState
from env.v310_policy import PaperNearestTargetBluePolicy
from env.v339_policy import DistanceTierNearestTargetBluePolicy
from tools.combat_visualization import RecordingCombatEnv, states_array

ROOT = Path(__file__).resolve().parents[1]


def config(version='339'):
    return load_config(ROOT / f'configs/combat_environment_v{version}.yaml')


def state(x=0, v=225, **kwargs):
    return AircraftState(x, kwargs.get('y', 0), kwargs.get('z', -3000),
                         v, kwargs.get('theta', 0), kwargs.get('psi', 0),
                         kwargs.get('alive', True))


def policy():
    c = config()
    return DistanceTierNearestTargetBluePolicy(c['blue_policy'], c['action'])


def exact(a, b):
    if isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            if key != 'environment_version':
                exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b


def internal_exact(a, b):
    exact(states_array(a.red + a.blue), states_array(b.red + b.blue))
    exact(a.rng.bit_generator.state, b.rng.bit_generator.state)
    for side in ('red', 'blue'):
        for suffix in ('_alive_mask', '_ammo', '_fire_cooldown_steps'):
            exact(getattr(a, side + suffix), getattr(b, side + suffix))


def test_dispatch_inheritance_config_and_frozen_methods():
    e = MultiUAVCombatEnv(config())
    assert type(e) is CombatEnvironmentV339
    assert CombatEnvironmentV339.__bases__ == (CombatEnvironmentV338,)
    assert set(CombatEnvironmentV339.__dict__) - {'__module__', '__doc__'} == {'environment_version', '__init__'}
    for name in ('reset', 'step', '_advance', '_observations', 'initial_states',
                 'dense_combat_reward', 'build_weapon', '_outcome', '_resolve_combat',
                 '_resolve_noncombat_losses', 'attempt_hit_for_pair', '_entry_attempts'):
        assert getattr(CombatEnvironmentV339, name) is getattr(CombatEnvironmentV338, name)
    old, new = config('338'), config()
    assert {k for k in old if old[k] != new[k]} == {'environment_version', 'blue_policy'}
    assert old['blue_policy'] == {'desired_speed': 225.0}
    assert new['blue_policy'] == dict(near_distance_max=1000., far_distance_min=3000.,
        near_desired_speed=200., mid_desired_speed=220., far_desired_speed=240.)
    assert environment_dimensions(new) == (45, 3, 4)


@pytest.mark.parametrize('distance,speed,av', [
    (4000,225,1),(2000,225,-1),(500,225,-1),
    (3000,220,0),(1000,200,0),(4000,240,0),(2000,220,0),(500,200,0),
    (3000.001,240,0),(1000.001,220,0)])
def test_exact_speed_commands_and_boundaries(distance, speed, av):
    p = policy()
    own = state(v=speed)
    before = own.as_array().copy()
    exact(p.action(own, [state(distance)]), np.array([0, 0, av], np.float32))
    exact(own.as_array(), before)


def test_distance_is_3d_and_not_horizontal():
    # Horizontal distance is near, but 3D distance is mid.
    assert policy().action(state(v=220), [state(500, z=-4200)])[2] == 0


def test_no_target_dead_blue_mro_switches_and_reset():
    p = policy()
    assert p.team_actions.__func__ is PaperNearestTargetBluePolicy.team_actions
    own, targets = state(), [state(4000), state(5000)]
    with patch.object(p, 'action', wraps=p.action) as called:
        assert p.team_actions([own], targets)[0, 2] == 1
        assert called.call_count == 1
    targets[1].x = 500
    assert p.team_actions([own], targets)[0, 2] == -1
    assert p.diagnostics() == dict(blue_paper_nearest_agent_steps=2,
        blue_target_switches=1, blue_last_target_indices=[1])
    for t in targets:
        t.alive = False
    exact(p.team_actions([own], targets), np.zeros((1,3), np.float32))
    exact(p.team_actions([state(alive=False)], [state(500)]), np.zeros((1,3), np.float32))
    p.reset(5)
    assert p.diagnostics() == dict(blue_paper_nearest_agent_steps=0,
        blue_target_switches=0, blue_last_target_indices=[])


def test_heading_pitch_exactly_match_old_policy():
    p = policy()
    c = config('338')
    old = PaperNearestTargetBluePolicy(c['blue_policy'], c['action'])
    rng = np.random.default_rng(9)
    for _ in range(100):
        own = state(theta=rng.uniform(-1,1), psi=rng.uniform(-3,3))
        targets = [state(rng.uniform(-4000,4000), y=rng.uniform(-4000,4000),
                         z=rng.uniform(-5000,-1000)) for _ in range(4)]
        exact(p.action(own, targets)[:2], old.action(own, targets)[:2])


@pytest.mark.parametrize('distance,desired', [(4000,240),(2000,220),(500,200)])
def test_automatic_environment_gradual_convergence(distance, desired):
    e = MultiUAVCombatEnv(config()); e.reset(1)
    speeds = []
    # Reposition the artificial target before each step to isolate speed control.
    # The actual policy, transition and aircraft speed update remain authoritative.
    for _ in range(12):
        e.red = [state(e.blue[0].x + distance, y=e.blue[0].y,
                       z=e.blue[0].z)] + [state(alive=False) for _ in range(3)]
        for b in e.blue[1:]:
            b.alive = False
        e.blue[0].psi = 0
        before = e.blue[0].v
        e.step(np.zeros((4,3)))
        speeds.append(e.blue[0].v)
        assert abs(e.blue[0].v - before) <= 2.5
        assert 150 <= e.blue[0].v <= 300
    assert speeds[-1] == desired
    if desired == 200:
        exact(speeds, [222.5,220,217.5,215,212.5,210,207.5,205,202.5,200,200,200])


def test_automatic_boundary_crossing_and_dead_actions():
    e = RecordingCombatEnv(config()); e.reset(4)
    for distance, av in [(3001,1),(3000,-1),(1001,-1),(1000,-1),(3001,1)]:
        e.blue = [state()] + [state(alive=False) for _ in range(3)]
        e.red = [state(distance)] + [state(alive=False) for _ in range(3)]
        e.step(np.zeros((4,3)))
        actions = e.observed_actions['blue']
        assert actions[0,2] == av
        assert np.all(np.abs(actions) <= 1)
        exact(actions[1:], np.zeros((3,3),np.float32))


@pytest.mark.parametrize('seed', [0,1,19,199])
def test_reset_exact_initial_state_observation_rng(seed):
    a,b = MultiUAVCombatEnv(config('338')), MultiUAVCombatEnv(config())
    exact(a.reset(seed), b.reset(seed)); internal_exact(a,b)
    assert all(s.v == 225 for s in a.red+a.blue+b.red+b.blue)
    assert a.weapon == b.weapon


@pytest.mark.parametrize('scenario', ['initial','fire','noncombat'])
def test_explicit_identical_actions_full_transition_isolation(scenario):
    a,b = MultiUAVCombatEnv(config('338')), MultiUAVCombatEnv(config())
    a.reset(11); b.reset(11)
    if scenario != 'initial':
        for e in (a,b):
            e.red = [state(0,y=i*150) for i in range(4)]
            e.blue = [state(500,y=i*150,psi=np.pi) for i in range(4)]
            if scenario == 'noncombat':
                e.red[0].z = -5999; e.red[0].theta = 1
    rng = np.random.default_rng(31)
    for _ in range(1000):
        red = rng.uniform(-1,1,(4,3)).astype(np.float32)
        blue = rng.uniform(-1,1,(4,3)).astype(np.float32)
        x,y = a.step(red,blue),b.step(red.copy(),blue.copy())
        exact(x,y); internal_exact(a,b)
        if x[2] or x[3]:
            break
    else:
        pytest.fail('episode did not terminate')
    if scenario == 'fire':
        assert (a.red_ammo < 6).any() and (a.blue_ammo < 6).any()


def test_canonical_observer_full_episode_exact_and_single_policy_call():
    a,b = MultiUAVCombatEnv(config()),RecordingCombatEnv(config())
    assert isinstance(b,CombatEnvironmentV339)
    exact(a.reset(20001000),b.reset(20001000))
    rng = np.random.default_rng(41)
    with patch.object(b.fixed_policy,'team_actions',wraps=b.fixed_policy.team_actions) as called:
        for step in range(1000):
            red = rng.uniform(-1,1,(4,3)).astype(np.float32)
            x,y = a.step(red),b.step(red.copy())
            exact(x,y); internal_exact(a,b)
            assert called.call_count == step+1
            exact(a.fixed_policy.diagnostics(),b.fixed_policy.diagnostics())
            exact(b.blue_step_target_indices,y[4]['blue_pursuit_target_indices'])
            exact(b.observed_actions['blue'], y[4]['executed_blue_actions'])
            assert b.observed_actions['blue'].shape == (4,3)
            if x[2] or x[3]:
                break
        else:
            pytest.fail('episode did not terminate')
    b.reset(2)
    assert b.fixed_policy.diagnostics()['blue_paper_nearest_agent_steps'] == 0


def test_canonical_trace_records_executed_actions_speed_and_targets(tmp_path):
    from tools.record_combat_episode import record
    from tools.combat_trace import read_trace

    class ZeroActor:
        def act(self, observations, alive, deterministic=True):
            return np.zeros((*observations.shape[:-1], 3), dtype=np.float32)

    record(ZeroActor(), config(), 7, tmp_path / 'episode')
    trace = read_trace(tmp_path / 'episode' / 'episode_trace.npz')
    e = MultiUAVCombatEnv(config()); e.reset(7)
    exact(trace['blue_kinematics'][0], states_array(e.blue))
    for t in range(len(trace['blue_actions'])):
        expected = [e.fixed_policy.nearest_target_index(s, e.red) if s.alive else None for s in e.blue]
        exact(trace['blue_pursuit_target_indices'][t], np.array([-1 if i is None else i for i in expected]))
        _, _, terminated, truncated, info = e.step(np.zeros((4,3), np.float32))
        exact(trace['blue_actions'][t], info['executed_blue_actions'])
        exact(trace['blue_kinematics'][t+1], states_array(e.blue))
    assert terminated or truncated


def test_existing_checkpoint_version_guard_and_config_hash_isolation():
    # Pure metadata validation; no checkpoint/model evaluation or device fallback.
    from algorithm.common.checkpoint import validate_checkpoint_environment
    from algorithm.common.protocol import config_sha256
    old, new = config('338'), config()
    assert config_sha256(old) != config_sha256(new)
    with pytest.raises(RuntimeError, match='environment_version mismatch'):
        validate_checkpoint_environment({'extra': {'environment_version': '3.38'}}, new)


@pytest.mark.parametrize('key,value', [
    ('near_distance_max',0),('near_distance_max',3000),('far_distance_min',500),
    ('near_desired_speed',149.99),('mid_desired_speed',300.01),
    ('far_desired_speed',float('nan')),('far_distance_min',float('inf')),
    ('near_distance_max',True),('mid_desired_speed','220'),('desired_speed',225)])
def test_invalid_blue_policy_rejected(key,value):
    c=config(); c['blue_policy'][key]=value
    with pytest.raises(ValueError): validate_config(c)


@pytest.mark.parametrize('key',list(config()['blue_policy']))
def test_missing_policy_field_rejected(key):
    c=config(); del c['blue_policy'][key]
    with pytest.raises(ValueError): validate_config(c)


@pytest.mark.parametrize('section,key,value',[
    ('weapon','range_max',4000),('reward','guide_reward',.002),
    ('scenario','speed_center',240),('action','command',{}),
    ('observation','speed_scale',50),('fire_control','cooldown_seconds',2)])
def test_unrelated_configuration_drift_rejected(section,key,value):
    c=config(); c[section][key]=value
    with pytest.raises(ValueError): validate_config(c)
