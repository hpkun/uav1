"""Blue intention inversion; all physical and tactical contracts stay inherited."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v320 import CombatEnvironmentV320
from env.combat_v321 import CombatEnvironmentV321
from env.v310_policy import PaperNearestTargetBluePolicy
from env.v321_policy import V321NearestTargetBluePolicy
from env.v320_control import action_to_control_v320
from env.v315_control import action_to_control_v315
from env.models import AircraftState
from env.config import validate_config, environment_dimensions
from tools.audit_combat_v317_v318 import body_pair, FixedNormal, window_env


def env(version='321'):
    e = MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml')
    e.reset(31)
    return e


def response(e, heading=0, pitch=0, state=None):
    s = state or AircraftState(0,0,-3000,225,0,0)
    action = e.fixed_policy.action_toward(s,s.psi+math.radians(heading),s.theta+math.radians(pitch),250)
    mapper = action_to_control_v315 if e.environment_version == '3.19' else action_to_control_v320
    c = mapper(s,action,e.config['action'],e.spec)
    return np.degrees(e.dynamics.derivatives(s,c)[4:]),action,c


def test_minimal_inheritance_and_config_protocol():
    assert CombatEnvironmentV321.__bases__ == (CombatEnvironmentV320,)
    assert set(CombatEnvironmentV321.__dict__)-{'__module__','__doc__'} == {'environment_version','__init__'}
    assert V321NearestTargetBluePolicy.__bases__ == (PaperNearestTargetBluePolicy,)
    assert set(V321NearestTargetBluePolicy.__dict__)-{'__module__','__doc__'} == {'__init__','action_toward'}
    for name in ('_advance','reset','step','_observations','dense_combat_reward','build_weapon','_entry_attempts','_resolve_combat','_outcome','_resolve_noncombat_losses'):
        assert getattr(CombatEnvironmentV321,name) is getattr(CombatEnvironmentV320,name)
    assert CombatEnvironmentV321._advance.__globals__['action_to_control_v320'] is action_to_control_v320
    for name in ('nearest_target_index','action','team_actions','diagnostics','reset'):
        assert getattr(V321NearestTargetBluePolicy,name) is getattr(PaperNearestTargetBluePolicy,name)
    cfg = deepcopy(env().config)
    validate_config(cfg)
    assert cfg == env().config
    assert environment_dimensions(cfg) == (56,3,5)
    cfg['environment_version'] = '3.20'
    assert cfg == env('320').config
    assert Path('configs/mappo_5v5_v321.yaml').read_bytes() == Path('configs/mappo_5v5_v320.yaml').read_bytes()


@pytest.mark.parametrize('seed',[1,31,10000000])
def test_reset_exact_and_red_controller_unchanged(seed):
    old,new = env('320'),env()
    oo,oi = old.reset(seed)
    no,ni = new.reset(seed)
    np.testing.assert_array_equal(oo,no)
    assert oi['nominal_anchor_separation'] == ni['nominal_anchor_separation'] == 8000
    for side in ('red','blue'):
        assert getattr(old,side) == getattr(new,side)
        np.testing.assert_array_equal(getattr(old,side+'_ammo'),getattr(new,side+'_ammo'))
    for key in ('reward','weapon','scenario','observation','blue_policy'):
        assert old.config[key] == new.config[key]
    for name in ('weapon','dynamics','integrator'):
        assert vars(getattr(old,name)) == vars(getattr(new,name))
        assert type(getattr(old,name)) is type(getattr(new,name))
    actions = np.tile([.5,-.2,.7],(5,1))
    np.testing.assert_array_equal(old._advance(old.red,actions),new._advance(new.red,actions))
    assert old.red == new.red


def test_selection_geometry_diagnostics_immediate_switch_and_no_target():
    old,new = env('320').fixed_policy,env().fixed_policy
    own = AircraftState(0,0,-3000,225,.1,.2)
    targets = [AircraftState(1000,200,-3100,225,0,0),AircraftState(300,100,-3200,225,0,0),AircraftState(200,0,-3000,225,0,0,False)]
    for policy in (old,new):
        assert policy.nearest_target_index(own,targets) == 1
        with patch.object(policy,'action_toward',wraps=policy.action_toward) as encoder:
            policy.team_actions([own],targets)
        args = encoder.call_args.args
        assert args[1:] == (math.atan2(100,300),math.atan2(200,math.hypot(300,100)),250)
    assert old.diagnostics() == new.diagnostics()
    targets[1].alive = False
    for policy in (old,new):
        assert policy.nearest_target_index(own,targets) == 0
        policy.team_actions([own],targets)
        assert policy.diagnostics()['blue_target_switches'] == 1
    assert old.diagnostics() == new.diagnostics()
    targets[0].alive = False
    for policy in (old,new):
        np.testing.assert_array_equal(policy.team_actions([own],targets),np.zeros((1,3)))
    assert old.diagnostics() == new.diagnostics()


@pytest.mark.parametrize('angle',[5,10,20,40,90,180,-5,-10,-20,-40,-90,-179.999])
def test_heading_tracking_and_natural_budget(angle):
    from env.math_utils import wrap_angle
    rate,action,c = response(env(),heading=angle)
    expected = np.clip(math.degrees(wrap_angle(math.radians(angle)))/2,-19.82802219648885,19.82802219648885)
    assert rate[1] == pytest.approx(expected,abs=2e-6)
    assert abs(rate[0]) < 1e-12
    assert c.nz <= 8+1e-12


@pytest.mark.parametrize('angle',[-1,-2,-5,-10,-20,2,10,20,40])
def test_pitch_tracking_and_physical_saturation(angle):
    rate,action,c = response(env(),pitch=angle)
    assert rate[0] == pytest.approx(np.clip(angle/2,-2.498095986770389,17.486671907392726),abs=2e-6)
    assert c.nz <= 8+1e-12


@pytest.mark.parametrize('heading,pitch',[(20,10),(60,10),(90,-10)])
def test_combined_heading_priority_and_remaining_budget(heading,pitch):
    rate,action,c = response(env(),heading,pitch)
    b = action[0]*math.sqrt(63)
    amax = math.sqrt(max(64-b*b,0))
    required = 1+225*math.radians(pitch)/2/9.81
    feasible = np.clip(required,0,amax)
    assert np.isfinite(action).all() and (abs(action)<=1).all()
    assert c.nz <= 8+1e-12
    assert c.nz*math.sin(c.phi) == pytest.approx(b,abs=1e-12)
    assert c.nz*math.cos(c.phi) == pytest.approx(feasible,abs=1e-6)


@pytest.mark.parametrize('speed',[150,225,300])
def test_speed_encoding_exact(speed):
    state = AircraftState(0,0,-3000,speed,0,0)
    assert response(env(),20,-2,state)[1][2] == response(env('320'),20,-2,state)[1][2]


@pytest.mark.parametrize('heading,pitch',[(5,0),(-10,0),(20,0),(0,-1),(0,-2),(0,2),(0,10)])
def test_legacy_v319_small_single_axis_tracking_restored(heading,pitch):
    np.testing.assert_allclose(response(env(),heading,pitch)[0],response(env('319'),heading,pitch)[0],atol=2e-6)


def test_v320_diagnostic_comparison():
    assert response(env('320'),20)[0][1] == pytest.approx(2.20311358,abs=1e-6)
    assert response(env(),20)[0][1] == pytest.approx(10,abs=2e-6)


@pytest.mark.parametrize('theta',[-math.pi/3,math.pi/3])
@pytest.mark.parametrize('pitch',[-180,180])
def test_legacy_pitch_target_clipping_at_state_boundaries(theta,pitch):
    s = AircraftState(0,0,-3000,225,theta,0)
    rate,action,c = response(env(),pitch=pitch,state=s)
    target = np.clip(theta+math.pi/3*np.clip(math.radians(pitch)/(math.pi/3),-1,1),-math.pi/3,math.pi/3)
    a0 = math.cos(theta)
    a = np.clip(a0+225*(target-theta)/2/9.81,0,8)
    assert rate[0] == pytest.approx(math.degrees(9.81/225*(a-a0)),abs=2e-6)


def test_numerical_guard_without_new_config():
    e = env()
    cfg = deepcopy(e.config['action'])
    cfg['controller']['normal_load_max'] = 1
    policy = V321NearestTargetBluePolicy(e.config['blue_policy'],cfg,e.spec)
    a = policy.action_toward(AircraftState(0,0,-3000,225,0,0),1,.1,250)
    assert a[0] == a[1] == 0 and np.isfinite(a).all()


@pytest.mark.parametrize('distance',[100,2500,5000])
def test_static_reward_terms_exact(distance):
    old,new = env('320'),env()
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


def test_weapon_rng_repeat_fire_exact():
    old,new = env('320'),env()
    for distance in (999,1000,1001):
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


@pytest.mark.parametrize('case',['blue_dead','red_dead','timeout','boundary','ground','ceiling'])
def test_termination_exact(case):
    old,new = env('320'),env()
    for e in (old,new):
        if case.endswith('_dead'):
            for s in getattr(e,case.split('_')[0]): s.alive = False
        if case == 'boundary': e.red[0].x = 5001
        if case == 'ground': e.red[0].z = 0
        if case == 'ceiling': e.red[0].z = -6001
    assert old._resolve_noncombat_losses() == new._resolve_noncombat_losses()
    assert old._outcome(case=='timeout') == new._outcome(case=='timeout')


def test_checkpoint_recording_and_version_logging():
    from tools.combat_visualization import RecordingCombatEnv
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
    from algorithm.mappo.runner import MAPPOTrainingRunner
    e = RecordingCombatEnv('configs/combat_environment_v321.yaml')
    assert isinstance(e,CombatEnvironmentV321)
    cfg = yaml.safe_load(Path('configs/mappo_5v5_v321.yaml').read_bytes())
    for validator in (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume):
        with pytest.raises(RuntimeError,match='environment_version'):
            validator({'algorithm':'MAPPO','extra':{'environment_version':'3.20'}},e.config,cfg)
    runner = object.__new__(MAPPOTrainingRunner)
    runner.env_config = e.config
    runner.trainer = SimpleNamespace(sampled_steps=0)
    runner.evaluation_seeds = [10000000]
    with patch('algorithm.mappo.runner.evaluate',return_value={}):
        assert runner._evaluation_record()['environment_version'] == '3.21'
