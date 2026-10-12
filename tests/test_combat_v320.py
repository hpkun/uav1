"""Feasible maneuver allocation and isolated v3.19 behavior contracts."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v319 import CombatEnvironmentV319
from env.combat_v320 import CombatEnvironmentV320
from env.config import environment_dimensions, validate_config
from env.models import AircraftState
from env.v315_control import action_to_control_v315
from env.v320_control import action_to_control_v320
from tools.audit_combat_v317_v318 import body_pair, FixedNormal, window_env


def env(version='320', seed=31):
    e = MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml')
    e.reset(seed)
    return e


def control(e, action, state=None):
    return action_to_control_v320(state or AircraftState(0, 0, -3000, 225, 0, 0), action, e.config['action'], e.spec)


def rates(e, action):
    s = AircraftState(0, 0, -3000, 225, 0, 0)
    c = control(e, action, s)
    return np.degrees(e.dynamics.derivatives(s, c)[4:]), c


def test_minimal_direct_inheritance_and_frozen_config():
    old, new = env('319'), env()
    assert CombatEnvironmentV320.__bases__ == (CombatEnvironmentV319,)
    assert set(CombatEnvironmentV320.__dict__) - {'__module__', '__doc__'} == {'environment_version', '_advance'}
    for name in ('reset', 'step', '_observations', 'build_weapon', 'attempt_hit_for_pair', '_entry_attempts', '_new_fire_states', '_resolve_combat', '_outcome', '_resolve_noncombat_losses', 'dense_combat_reward'):
        assert getattr(CombatEnvironmentV320, name) is getattr(CombatEnvironmentV319, name)
    cfg = deepcopy(new.config)
    validate_config(cfg)
    assert cfg == new.config
    cfg['environment_version'] = '3.19'
    assert cfg == old.config
    assert environment_dimensions(new.config) == (56, 3, 5)
    assert Path('configs/mappo_5v5_v320.yaml').read_bytes() == Path('configs/mappo_5v5_v319.yaml').read_bytes()


def test_zero_action_equilibrium():
    e = env()
    c = control(e, [0, 0, 0])
    assert (c.nx, c.nz, c.phi) == (0, 1, 0)
    np.testing.assert_array_equal(e.dynamics.derivatives(AircraftState(0, 0, -3000, 225, 0, 0), c)[3:], 0)


def test_heading_linear_strict_and_symmetric_without_old_saturation():
    e = env()
    actions = (0, .2, .5, 1)
    headings = [rates(e, [a, 0, 0])[0][1] for a in actions]
    assert np.all(np.diff(headings) > 0)
    for a, heading in zip(actions, headings):
        assert heading == pytest.approx(math.degrees(9.81 / 225 * math.sqrt(63) * a))
        assert rates(e, [-a, 0, 0])[0][1] == pytest.approx(-heading)
        assert rates(e, [a, 0, 0])[0][0] == pytest.approx(0, abs=1e-12)
    s = AircraftState(0, 0, -3000, 225, 0, 0)
    old = [abs(np.degrees(e.dynamics.derivatives(s, action_to_control_v315(s, [a, 0, 0], e.config['action'], e.spec))[5])) for a in (.3, .5, 1)]
    assert max(old) / min(old) < 1.04
    assert rates(e, [.3, 0, 0])[0][1] < headings[2] < headings[3]


def test_negative_pitch_linear_and_zero_load_convention():
    e = env()
    values = [rates(e, [0, a, 0])[0][0] for a in (0, -.2, -.5, -1)]
    assert np.all(np.diff(values) < 0)
    np.testing.assert_allclose(values, np.degrees(9.81 / 225 * np.array([0, -.2, -.5, -1])), atol=1e-12)
    c = control(e, [0, -1, 0])
    assert c.nz == c.phi == 0


def test_positive_pitch_monotonic_and_full_budget():
    e = env()
    values = [rates(e, [0, a, 0])[0][0] for a in (0, .25, .5, 1)]
    assert np.all(np.diff(values) > 0)
    assert values[-1] == pytest.approx(math.degrees(9.81 * 7 / 225))
    assert control(e, [0, 1, 0]).nz == 8
    assert control(e, [0, 1, 0]).phi == 0


@pytest.mark.parametrize('heading,pitch',[(.5,.5),(.8,.5),(1,1),(1,-1)])
def test_combined_budget_without_rescaling(heading, pitch):
    e = env()
    c = control(e, [heading, pitch, 0])
    b = heading * math.sqrt(63)
    amax = math.sqrt(max(64-b*b, 0))
    a = 1 + pitch*(amax-1) if pitch >= 0 else 1+pitch
    assert c.nz <= 8 + 1e-12
    assert c.nz*math.cos(c.phi) == pytest.approx(a, abs=1e-12)
    assert c.nz*math.sin(c.phi) == pytest.approx(b, abs=1e-12)


def test_load_budget_across_domain_and_action_clipping():
    e = env()
    for theta in np.linspace(-math.pi/3, math.pi/3, 9):
        s = AircraftState(0, 0, -3000, 225, theta, 0)
        for heading in np.linspace(-1, 1, 21):
            assert e.dynamics.derivatives(s, control(e, [heading, 0, 0], s))[4] == pytest.approx(0, abs=1e-12)
            for pitch in np.linspace(-1, 1, 21):
                c = control(e, [heading, pitch, 0], s)
                assert 0 <= c.nz <= 8 + 1e-12
                assert math.isfinite(c.phi)
    assert control(e, [2, -2, 2]) == control(e, [1, -1, 1])


@pytest.mark.parametrize('speed',[150,160,225,290,300])
@pytest.mark.parametrize('theta',[-math.pi/3,0,math.pi/3])
@pytest.mark.parametrize('av',[-1,-.5,0,.5,1])
def test_speed_exact_historical_semantics(speed, theta, av):
    e = env()
    s = AircraftState(0, 0, -3000, speed, theta, 0)
    new = control(e, [.5, -.5, av], s)
    old = action_to_control_v315(s, [.5, -.5, av], e.config['action'], e.spec)
    target = float(np.clip(speed+50*av,150,300))
    acceleration = (target-speed)/2
    assert new.nx == old.nx == float(np.sin(theta)+acceleration/9.81)
    assert e.dynamics.derivatives(s,new)[3] == pytest.approx(acceleration,abs=1e-12)


@pytest.mark.parametrize('side',['red','blue'])
def test_advance_fixed_control_four_rk4_stages_and_dead_mask(side):
    e = env()
    states = [AircraftState(0,0,-3000,300,math.pi/3,math.pi-.001)] + [AircraftState(0,0,-3000,150,-math.pi/3,0,False) for _ in range(4)]
    actions = np.tile([1,1,1], (5,1))
    setattr(e, side, states)
    before = deepcopy(states)
    with patch.object(e.dynamics,'derivatives',wraps=e.dynamics.derivatives) as derivatives:
        phis = e._advance(states,actions)
    assert derivatives.call_count == 4
    controls = [call.args[1] for call in derivatives.call_args_list]
    assert all(c is controls[0] for c in controls)
    assert controls[0] == control(e,actions[0],before[0])
    assert controls[0].nz <= 8+1e-12
    assert phis[0] == controls[0].phi
    assert np.isfinite(states[0].as_array()).all()
    assert 150 <= states[0].v <= 300
    assert -math.pi/3 <= states[0].theta <= math.pi/3
    assert -math.pi <= states[0].psi < math.pi
    assert states[1:] == before[1:]
    # The inherited step calls this same method for each team.
    with patch.object(e,'_advance',wraps=e._advance) as advance:
        e.step(np.zeros((5,3)),np.zeros((5,3)))
    assert advance.call_count == 2
    assert advance.call_args_list[0].args[0] is e.red
    assert advance.call_args_list[1].args[0] is e.blue


@pytest.mark.parametrize('seed',[1,31,10000000])
def test_reset_and_all_noncontroller_components_exact(seed):
    old,new = env('319',seed),env(seed=seed)
    oo,oi = old.reset(seed)
    no,ni = new.reset(seed)
    np.testing.assert_array_equal(oo,no)
    assert oi['nominal_anchor_separation'] == ni['nominal_anchor_separation'] == 8000
    for side in ('red','blue'):
        assert getattr(old,side) == getattr(new,side)
        np.testing.assert_array_equal(getattr(old,side+'_ammo'),getattr(new,side+'_ammo'))
    assert old.config['weapon'] == new.config['weapon']
    assert old.config['reward'] == new.config['reward']
    for name in ('weapon','fixed_policy','dynamics','integrator'):
        assert type(getattr(old,name)) is type(getattr(new,name))
    np.testing.assert_array_equal(old.fixed_policy.team_actions(old.blue,old.red),new.fixed_policy.team_actions(new.blue,new.red))
    old._advance(old.red,np.tile([.5,.5,0],(5,1)))
    new._advance(new.red,np.tile([.5,.5,0],(5,1)))
    assert not np.array_equal(old.red[0].as_array(),new.red[0].as_array())


@pytest.mark.parametrize('distance',[100,2500,5000])
def test_static_full_reward_r1_r2_r3_r4_safety_exact(distance):
    old,new = env('319'),env()
    for e in (old,new):
        e.red = [AircraftState(0,0,-3000,225,0,0)] + [AircraftState(0,0,-3000,225,0,0,False) for _ in range(4)]
        e.blue = [AircraftState(distance,0,-3000,225,0,0)] + [AircraftState(0,0,-3000,225,0,0,False) for _ in range(4)]
    with patch.object(old,'_advance',return_value=np.zeros(5)),patch.object(new,'_advance',return_value=np.zeros(5)),patch.object(old,'_entry_attempts',return_value=[(0,0,True)]),patch.object(new,'_entry_attempts',return_value=[(0,0,True)]):
        _,ro,to,xo,io = old.step(np.zeros((5,3)),np.zeros((5,3)))
        _,rn,tn,xn,inn = new.step(np.zeros((5,3)),np.zeros((5,3)))
    assert (to,xo) == (tn,xn)
    for key in ('r1_rewards','r2_rewards','r3_rewards','r4_rewards','safety_rewards'):
        np.testing.assert_array_equal(io[key],inn[key])
    np.testing.assert_array_equal(ro,rn)


def test_weapon_same_inputs_rng_and_repeat_fire():
    old,new = env('319'),env()
    for distance in (799,999,1000,1001):
        for az,el in ((0,0),(30,0),(31,0),(25,25)):
            a,b = body_pair(distance=distance,azimuth=az,elevation=el)
            assert old.weapon.qualifies(a,b,6) == new.weapon.qualifies(a,b,6)
    for epsilon in (0,.5,-.5,2,-2):
        assert old.weapon.attempt_hit(FixedNormal(epsilon),900,.2,-.1) == new.weapon.attempt_hit(FixedNormal(epsilon),900,.2,-.1)
    fixture = window_env('317')
    for e in (old,new):
        e.red,e.blue = deepcopy(fixture.red),deepcopy(fixture.blue)
        e.rng = FixedNormal(4)
    for _ in range(8):
        assert old._entry_attempts(old.red,old.blue,None,'red') == new._entry_attempts(new.red,new.blue,None,'red')
    assert old.red_ammo[0] == new.red_ammo[0] == 0
    assert old.rng.calls == new.rng.calls == 6


def test_version_dispatch_checkpoint_rejection_and_logging():
    from tools.combat_visualization import RecordingCombatEnv
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation, validate_checkpoint_for_resume
    from algorithm.mappo.runner import MAPPOTrainingRunner
    e = RecordingCombatEnv('configs/combat_environment_v320.yaml')
    assert isinstance(e,CombatEnvironmentV320)
    cfg = yaml.safe_load(Path('configs/mappo_5v5_v320.yaml').read_bytes())
    for validator in (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume):
        with pytest.raises(RuntimeError,match='environment_version'):
            validator({'algorithm':'MAPPO','extra':{'environment_version':'3.19'}},e.config,cfg)
    runner = object.__new__(MAPPOTrainingRunner)
    runner.env_config = e.config
    runner.trainer = SimpleNamespace(sampled_steps=0)
    runner.evaluation_seeds = [10000000]
    with patch('algorithm.mappo.runner.evaluate',return_value={}):
        assert runner._evaluation_record()['environment_version'] == '3.20'
