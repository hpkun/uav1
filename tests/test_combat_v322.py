"""Continuous R3 handoff with exact formal R4 and inherited physics."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v321 import CombatEnvironmentV321
from env.combat_v322 import CombatEnvironmentV322
from env.config import validate_config, environment_dimensions
from env.models import AircraftState
from env.v319_reward import tactical_rewards
from env.v322_reward import distance_weights, tactical_rewards_v322
from tools.audit_combat_v317_v318 import body_pair, FixedNormal, window_env


def env(version='322'):
    e = MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml')
    e.reset(31)
    return e


def teams(distance=2000, angle=0, mode='offense'):
    own = AircraftState(0,0,-3000,225,0,math.radians(angle) if mode=='offense' else 0)
    target = AircraftState(distance if mode=='offense' else -distance,0,-3000,225,0,0 if mode=='offense' else math.radians(angle))
    dead = lambda: AircraftState(0,0,-3000,225,0,0,False)
    return [own,*[dead() for _ in range(4)]],[target,*[dead() for _ in range(4)]]


def rewards(red,blue):
    cfg = env().config
    return tactical_rewards_v322(red,blue,cfg['reward'],cfg['weapon'])


def test_minimal_inheritance_and_frozen_configs():
    assert CombatEnvironmentV322.__bases__ == (CombatEnvironmentV321,)
    assert set(CombatEnvironmentV322.__dict__)-{'__module__','__doc__'} == {'environment_version','dense_combat_reward'}
    for name in ('__init__','_advance','reset','step','_observations','build_weapon','attempt_hit_for_pair','_entry_attempts','_resolve_combat','_outcome','_resolve_noncombat_losses'):
        assert getattr(CombatEnvironmentV322,name) is getattr(CombatEnvironmentV321,name)
    cfg = deepcopy(env().config)
    validate_config(cfg)
    assert cfg == env().config
    assert environment_dimensions(cfg) == (56,3,5)
    cfg['environment_version'] = '3.21'
    assert cfg == env('321').config
    assert Path('configs/mappo_5v5_v322.yaml').read_bytes() == Path('configs/mappo_5v5_v321.yaml').read_bytes()


@pytest.mark.parametrize('section,field,value',[
    ('weapon','range_max',999),('reward','guide_distance_min',2999),
    ('reward','tactical_distance_max',2999),('reward','guide_reward',.002)])
def test_formal_parameters_remain_frozen(section,field,value):
    cfg = deepcopy(env().config)
    cfg[section][field] = value
    with pytest.raises(ValueError): validate_config(cfg)


@pytest.mark.parametrize('distance,w3,w4',[(500,0,1),(1000,0,1),(1500,.25,.75),(2000,.5,.5),(2500,.75,.25),(3000,1,0),(3500,1,0)])
def test_weight_boundaries(distance,w3,w4):
    cfg = env().config
    actual = distance_weights(distance,cfg['reward'],cfg['weapon'])
    assert actual == pytest.approx((w3,w4))


def test_complement_and_config_derived_scales_no_magic_distances():
    cfg = env().config
    w3,w4 = distance_weights(np.linspace(1000,3000,501),cfg['reward'],cfg['weapon'])
    np.testing.assert_allclose(w3+w4,1,atol=1e-15)
    assert np.all((w3>=0)&(w3<=1)) and np.all((w4>=0)&(w4<=1))
    # Pure functions intentionally receive nonformal parameters here.
    reward,weapon = deepcopy(cfg['reward']),deepcopy(cfg['weapon'])
    reward['tactical_distance_max'] = 2000
    weapon['range_max'] = 500
    w3,w4 = distance_weights([250,500,875,1250,1625,2000,2500],reward,weapon)
    np.testing.assert_array_equal(w3,[0,0,.25,.5,.75,1,1])
    np.testing.assert_array_equal(w4,[1,1,.75,.5,.25,0,0])
    red,blue = teams(1250)
    result = tactical_rewards_v322(red,blue,reward,weapon)
    assert result['guide'][0] == .0005
    assert result['tactical_offense'][0] == .05


@pytest.mark.parametrize('tactical',[500,400])
def test_invalid_span_rejected(tactical):
    with pytest.raises(ValueError,match='must exceed'):
        distance_weights(600,{'tactical_distance_max':tactical},{'range_max':500})


@pytest.mark.parametrize('distance,expected',[(3001,.001),(3000,.001),(2999,.0009995),(2500,.00075),(2000,.0005),(1500,.00025),(1000,0),(999,0)])
def test_r3_continuous_values(distance,expected):
    red,blue = teams(distance)
    assert rewards(red,blue)['guide'][0] == pytest.approx(expected)


@pytest.mark.parametrize('case',['ata','ha','red_dead','blue_dead'])
def test_r3_angle_and_live_gates(case):
    red,blue = teams(2500)
    if case=='ata': red[0].psi = math.radians(31)
    if case=='ha':
        blue[0].x = 2500*math.cos(math.radians(31))
        blue[0].z = -3000-2500*math.sin(math.radians(31))
    if case=='red_dead': red[0].alive = False
    if case=='blue_dead': blue[0].alive = False
    assert rewards(red,blue)['guide'][0] == 0


def test_r3_no_aa_visibility_or_weapon_gate():
    red,blue = teams(12000)
    blue[0].psi = math.pi
    assert not env().weapon.qualifies(red[0],blue[0],0)
    assert rewards(red,blue)['guide'][0] == .001
    red,blue = teams(2500)
    blue[0].psi = math.pi
    assert rewards(red,blue)['guide'][0] == .00075


@pytest.mark.parametrize('first,second,expected',[(3500,2000,.001),(2500,2000,.00075)])
def test_r3_multitarget_max_not_sum(first,second,expected):
    red,blue = teams(first)
    blue[1] = AircraftState(second,0,-3000,225,0,0)
    assert rewards(red,blue)['guide'][0] == expected


@pytest.mark.parametrize('distance',[500,1000,1500,2000,2500,3000,3500])
@pytest.mark.parametrize('angle',[0,10,20,31])
@pytest.mark.parametrize('mode',['offense','defense'])
def test_r4_exact_formal_regression(distance,angle,mode):
    red,blue = teams(distance,angle,mode)
    cfg = env().config
    old = tactical_rewards(red,blue,cfg['reward'])
    new = tactical_rewards_v322(red,blue,cfg['reward'],cfg['weapon'])
    for key in ('tactical_offense','tactical_defense'):
        np.testing.assert_array_equal(old[key],new[key])


def test_r4_pair_weights_before_max_min_and_random_states():
    cfg = env().config
    rng = np.random.default_rng(322)
    for _ in range(20):
        red,blue = [[AircraftState(*rng.uniform([-4000,-4000,-5000,150,-1,-math.pi],[4000,4000,-1000,300,1,math.pi]),alive=bool(rng.integers(2))) for _ in range(5)] for _ in range(2)]
        old = tactical_rewards(red,blue,cfg['reward'])
        new = tactical_rewards_v322(red,blue,cfg['reward'],cfg['weapon'])
        for key in ('tactical_offense','tactical_defense'):
            np.testing.assert_array_equal(old[key],new[key])
        assert np.all((new['guide']>=0)&(new['guide']<=.001))
    red,blue = teams(2900)
    angle = math.radians(10)
    blue[1] = AircraftState(1000*math.cos(angle),1000*math.sin(angle),-3000,225,0,angle)
    assert rewards(red,blue)['tactical_offense'][0] == pytest.approx(.02)
    red,blue = teams(2900,mode='defense')
    blue[1] = AircraftState(-1000*math.cos(angle),-1000*math.sin(angle),-3000,225,0,0)
    assert rewards(red,blue)['tactical_defense'][0] == pytest.approx(-.025)


@pytest.mark.parametrize('aspect',[0,math.pi])
def test_three_km_handoff_continuity(aspect):
    results = []
    for distance in (3000,2999,3000-1e-6):
        red,blue = teams(distance)
        blue[0].psi = aspect
        results.append(rewards(red,blue))
    assert results[0]['guide'][0] == .001
    assert results[1]['guide'][0] == pytest.approx(.0009995)
    if aspect==0:
        assert results[1]['tactical_offense'][0] == pytest.approx(.00005)
    assert abs(results[2]['tactical'][0]-results[0]['tactical'][0]) < 1e-9
    assert results[1]['tactical'][0] > .0009


@pytest.mark.parametrize('seed',[1,31,10000000])
def test_same_reset_motion_blue_and_nonreward_components(seed):
    old,new = env('321'),env()
    oo,oi = old.reset(seed)
    no,ni = new.reset(seed)
    np.testing.assert_array_equal(oo,no)
    assert oi['nominal_anchor_separation'] == ni['nominal_anchor_separation'] == 8000
    for name in ('weapon','fixed_policy','dynamics','integrator'):
        assert type(getattr(old,name)) is type(getattr(new,name))
    assert vars(old.weapon) == vars(new.weapon)
    actions = np.tile([.5,-.2,.7],(5,1))
    for side in ('red','blue'):
        assert getattr(old,side) == getattr(new,side)
        np.testing.assert_array_equal(getattr(old,side+'_ammo'),getattr(new,side+'_ammo'))
    np.testing.assert_array_equal(old.fixed_policy.team_actions(old.blue,old.red),new.fixed_policy.team_actions(new.blue,new.red))
    for side in ('red','blue'):
        np.testing.assert_array_equal(old._advance(getattr(old,side),actions),new._advance(getattr(new,side),actions))
        assert getattr(old,side) == getattr(new,side)
    np.testing.assert_array_equal(old._observations(),new._observations())


@pytest.mark.parametrize('distance',[100,2500,5000])
def test_full_step_r1_r2_safety_frozen_and_total_correct(distance):
    old,new = env('321'),env()
    old.red,old.blue = teams(distance)
    new.red,new.blue = deepcopy(old.red),deepcopy(old.blue)
    with patch.object(old,'_advance',return_value=np.zeros(5)),patch.object(new,'_advance',return_value=np.zeros(5)),patch.object(old,'_entry_attempts',return_value=[(0,0,True)]),patch.object(new,'_entry_attempts',return_value=[(0,0,True)]):
        _,_,to,xo,io = old.step(np.zeros((5,3)),np.zeros((5,3)))
        _,rn,tn,xn,inn = new.step(np.zeros((5,3)),np.zeros((5,3)))
    assert (to,xo)==(tn,xn)
    for key in ('r1_rewards','r2_rewards','r4_rewards','safety_rewards'):
        np.testing.assert_array_equal(io[key],inn[key])
    np.testing.assert_array_equal(rn,sum(inn[f'r{i}_rewards'] for i in range(1,5))+inn['safety_rewards'])


def test_weapon_sampler_repeat_fire_frozen():
    old,new = env('321'),env()
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
def test_termination_frozen(case):
    old,new = env('321'),env()
    for e in (old,new):
        if case.endswith('_dead'):
            for s in getattr(e,case.split('_')[0]): s.alive = False
        if case=='boundary': e.red[0].x = 5001
        if case=='ground': e.red[0].z = 0
        if case=='ceiling': e.red[0].z = -6001
    assert old._resolve_noncombat_losses() == new._resolve_noncombat_losses()
    assert old._outcome(case=='timeout') == new._outcome(case=='timeout')


def test_recording_checkpoint_isolation_and_logging():
    from tools.combat_visualization import RecordingCombatEnv
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
    from algorithm.mappo.runner import MAPPOTrainingRunner
    e = RecordingCombatEnv('configs/combat_environment_v322.yaml')
    assert isinstance(e,CombatEnvironmentV322)
    cfg = yaml.safe_load(Path('configs/mappo_5v5_v322.yaml').read_bytes())
    for validator in (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume):
        with pytest.raises(RuntimeError,match='environment_version'):
            validator({'algorithm':'MAPPO','extra':{'environment_version':'3.21'}},e.config,cfg)
    runner = object.__new__(MAPPOTrainingRunner)
    runner.env_config = e.config
    runner.trainer = SimpleNamespace(sampled_steps=0)
    runner.evaluation_seeds = [10000000]
    with patch('algorithm.mappo.runner.evaluate',return_value={}):
        assert runner._evaluation_record()['environment_version'] == '3.22'
