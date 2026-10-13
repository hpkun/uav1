"""Signed enemy geometry and invariant physical transitions at fixed actions."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v323 import CombatEnvironmentV323
from env.combat_v324 import CombatEnvironmentV324
from env.config import validate_config, environment_dimensions
from env.models import AircraftState
from env.geometry import engagement_geometry
from env.v315_observation import build_team_observations_v315
from env.v324_observation import build_team_observations_v324
from env.v323_reward import tactical_rewards_v323
from tools.audit_combat_v317_v318 import body_pair, FixedNormal


def env(version='324'):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml')
    e.reset(31)
    return e


def dead():
    return AircraftState(0,0,-3000,225,0,0,False)


def observation(own,enemy,version='324'):
    func=build_team_observations_v324 if version=='324' else build_team_observations_v315
    return func([own]+[dead() for _ in range(4)],[enemy]+[dead() for _ in range(4)],env().config['observation'])[0]


def test_direct_baseline_only_observations_override():
    assert CombatEnvironmentV324.__bases__==(CombatEnvironmentV323,)
    assert set(CombatEnvironmentV324.__dict__)-{'__module__','__doc__'}=={'environment_version','_observations'}
    for name in ('__init__','reset','step','_advance','dense_combat_reward','build_weapon','attempt_hit_for_pair','_entry_attempts','_resolve_combat','_resolve_noncombat_losses','_outcome'):
        assert getattr(CombatEnvironmentV324,name) is getattr(CombatEnvironmentV323,name)
    old,new=env('323'),env()
    cfg=deepcopy(new.config)
    validate_config(cfg)
    assert cfg==new.config and environment_dimensions(cfg)==(56,3,5)
    cfg['environment_version']='3.23'
    assert cfg==old.config
    assert Path('configs/mappo_5v5_v324.yaml').read_bytes()==Path('configs/mappo_5v5_v323.yaml').read_bytes()


@pytest.mark.parametrize('section,field,value',[
    ('observation','relative_position_scale',500),('observation','ammo',True),
    ('fire_control','cooldown_seconds',.5),('weapon','range_max',999),
    ('reward','guide_reward',.002)])
def test_formal_config_still_frozen(section,field,value):
    cfg=deepcopy(env().config);cfg[section][field]=value
    with pytest.raises(ValueError): validate_config(cfg)


def test_horizontal_mirror_reproduces_old_aliasing_and_restores_sign():
    own=AircraftState(0,0,-3000,225,0,0)
    a=AircraftState(1000,1000,-3000,225,0,0)
    b=AircraftState(1000,-1000,-3000,225,0,0)
    old_a,old_b=observation(own,a,'323'),observation(own,b,'323')
    np.testing.assert_array_equal(old_a,old_b)
    new_a,new_b=observation(own,a),observation(own,b)
    assert new_a[31]==new_b[31]
    assert new_a[34]==pytest.approx(.25)
    assert new_b[34]==pytest.approx(-.25)
    assert not np.array_equal(new_a,new_b)
    ga,gb=engagement_geometry(own,a),engagement_geometry(own,b)
    assert ga.boresight_azimuth_error==-gb.boresight_azimuth_error


def test_vertical_mirror_reproduces_old_aliasing_and_restores_sign():
    own=AircraftState(0,0,-3000,225,0,0)
    above=AircraftState(1000,0,-3500,225,0,0)
    below=AircraftState(1000,0,-2500,225,0,0)
    np.testing.assert_array_equal(observation(own,above,'323'),observation(own,below,'323'))
    a,b=observation(own,above),observation(own,below)
    expected=math.atan2(500,1000)/(math.pi/2)
    assert a[31]==b[31]
    assert a[35]==pytest.approx(expected)
    assert b[35]==pytest.approx(-expected)
    assert not np.array_equal(a,b)


def test_signed_aa_and_absolute_reward_symmetry():
    own=AircraftState(0,0,-3000,225,0,0)
    a=AircraftState(2000,0,-3000,225,0,math.radians(20))
    b=AircraftState(2000,0,-3000,225,0,-math.radians(20))
    assert observation(own,a)[33]==pytest.approx(20/180)
    assert observation(own,b)[33]==pytest.approx(-20/180)
    cfg=env().config
    ra=tactical_rewards_v323([own],[a],cfg['reward'],cfg['weapon'])
    rb=tactical_rewards_v323([own],[b],cfg['reward'],cfg['weapon'])
    for key in ra: np.testing.assert_array_equal(ra[key],rb[key])


@pytest.mark.parametrize('pitch,heading,az,el',[
    (0,0,20,15),(58,30,-20,17),(-40,-120,10,-25),(60,179,-170,40),
    (-60,-179,170,-40),(30,90,0,0)])
def test_exact_geometry_contract_general_3d(pitch,heading,az,el):
    own,enemy=body_pair(pitch=pitch,heading=heading,azimuth=az,elevation=el,distance=1900)
    enemy.v=250;enemy.psi=math.radians(-70)
    g=engagement_geometry(own,enemy);obs=observation(own,enemy);cfg=env().config['observation']
    np.testing.assert_allclose(obs[31:36],[g.distance/cfg['relative_position_scale'],
        (enemy.v-cfg['speed_center'])/cfg['speed_scale'],g.aa/math.pi,
        g.boresight_azimuth_error/math.pi,g.boresight_elevation_error/(math.pi/2)],rtol=0,atol=0)
    assert obs[31]*cfg['relative_position_scale']==pytest.approx(g.distance)
    assert obs[33]*math.pi==pytest.approx(g.aa)
    assert obs[34]*math.pi==pytest.approx(g.boresight_azimuth_error)
    assert obs[35]*(math.pi/2)==pytest.approx(g.boresight_elevation_error)


@pytest.mark.parametrize('axis,angle',[('azimuth',20),('elevation',15)])
@pytest.mark.parametrize('distance',[900,2000])
def test_signed_decision_and_absolute_reward_weapon_contract(axis,angle,distance):
    cfg=env().config;results=[];slots=[];eligibility=[]
    for sign in (1,-1):
        own,enemy=body_pair(distance=distance,**{axis:sign*angle})
        enemy.psi=math.atan2(enemy.y-own.y,enemy.x-own.x)
        slots.append(observation(own,enemy)[34 if axis=='azimuth' else 35])
        results.append(tactical_rewards_v323([own],[enemy],cfg['reward'],cfg['weapon']))
        eligibility.append(env().weapon.qualifies(own,enemy,6))
    assert slots[0]>0 and slots[1]<0 and slots[0]==pytest.approx(-slots[1])
    for key in results[0]: np.testing.assert_array_equal(results[0][key],results[1][key])
    assert eligibility==[distance<=1000]*2


@pytest.mark.parametrize('seed',[1,31,10000000])
def test_non_enemy_blocks_and_distance_speed_features_regression(seed):
    old,new=env('323'),env()
    old.reset(seed);new.reset(seed)
    # Include masked allies, varied speed/pitch and general position.
    for e in (old,new):
        e.red[1].alive=False;e.blue[2].alive=False
        e.red[0].theta=.6;e.red[0].psi=-2.;e.red[0].v=177.
    a,b=old._observations(),new._observations()
    assert b.shape==(5,56) and b.dtype==a.dtype==np.float64
    np.testing.assert_array_equal(a[:,:31],b[:,:31])
    for j in range(5):
        np.testing.assert_allclose(a[:,31+5*j:33+5*j],b[:,31+5*j:33+5*j],rtol=1e-15,atol=1e-15)
    np.testing.assert_array_equal(b[:,4],np.zeros(5))


@pytest.mark.parametrize('dead_index',range(5))
def test_dead_enemy_slots_zero_and_fixed_order(dead_index):
    e=env();e.blue[dead_index].alive=False
    obs=e._observations()
    np.testing.assert_array_equal(obs[:,31+5*dead_index:36+5*dead_index],np.zeros((5,5)))
    for i,own in enumerate(e.red):
        for j,enemy in enumerate(e.blue):
            if not enemy.alive: continue
            g=engagement_geometry(own,enemy)
            assert obs[i,34+5*j]==g.boresight_azimuth_error/math.pi


def test_dead_self_zero_no_sensor_or_ammo_cooldown_features():
    e=env();e.red[1].alive=False
    e.blue[0]=AircraftState(20000,0,-3000,250,0,0)
    before=e._observations()
    np.testing.assert_array_equal(before[1],np.zeros(56))
    assert before[0,31]>0 and before[0,32]!=0
    e.red_ammo.fill(0);e.blue_ammo.fill(0)
    e.red_fire_cooldown_steps.fill(10);e.blue_fire_cooldown_steps.fill(10)
    np.testing.assert_array_equal(before,e._observations())


def assert_transition_same(old,new,result_old,result_new):
    _,ro,to,xo,io=result_old;_,rn,tn,xn,inn=result_new
    assert (to,xo)==(tn,xn)
    np.testing.assert_array_equal(ro,rn)
    for side in ('red','blue'):
        assert getattr(old,side)==getattr(new,side)
        for suffix in ('_ammo','_fire_cooldown_steps'):
            np.testing.assert_array_equal(getattr(old,side+suffix),getattr(new,side+suffix))
    assert old.combat_counts==new.combat_counts and old.blue_returns==new.blue_returns
    for key in ('r1_rewards','r2_rewards','r3_rewards','r4_rewards','safety_rewards','outcome_rewards',
                'red_step_fire_attempts','blue_step_fire_attempts','red_step_weapon_hits','blue_step_weapon_hits','termination_reason'):
        np.testing.assert_array_equal(io[key],inn[key])


@pytest.mark.parametrize('seed',[1,31,10000000])
def test_fixed_external_actions_identical_natural_transitions(seed):
    old,new=env('323'),env()
    old.reset(seed);new.reset(seed)
    for name in ('fixed_policy','dynamics','integrator','weapon'):
        assert type(getattr(old,name)) is type(getattr(new,name))
    rng=np.random.default_rng(seed)
    for _ in range(120):
        actions=rng.uniform(-.4,.4,(5,3)).astype(np.float32)
        a,b=old.step(actions),new.step(actions)
        assert_transition_same(old,new,a,b)
        if a[2] or a[3]:break


@pytest.mark.parametrize('scenario',['miss_cooldown','hit','blue_ground','red_ground','red_boundary'])
def test_event_reward_and_termination_transitions_unchanged(scenario):
    old,new=env('323'),env()
    own,enemy=body_pair()
    for e in (old,new):
        e.red=[own.copy()]+[dead() for _ in range(4)]
        e.blue=[enemy.copy()]+[dead() for _ in range(4)]
        e.rng=FixedNormal(0 if scenario=='hit' else 4)
        if scenario=='blue_ground':e.blue[0].z=1
        if scenario=='red_ground':e.red[0].z=1
        if scenario=='red_boundary':e.red[0].x=6000
    with patch.object(old,'_advance',return_value=np.zeros(5)),patch.object(new,'_advance',return_value=np.zeros(5)):
        for _ in range(61 if scenario=='miss_cooldown' else 1):
            a=old.step(np.zeros((5,3)),np.zeros((5,3)))
            b=new.step(np.zeros((5,3)),np.zeros((5,3)))
            assert_transition_same(old,new,a,b)
    assert old.rng.calls==new.rng.calls


def test_recording_checkpoint_isolation_and_version_logging():
    from tools.combat_visualization import RecordingCombatEnv
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
    from algorithm.mappo.runner import MAPPOTrainingRunner
    e=RecordingCombatEnv('configs/combat_environment_v324.yaml')
    assert isinstance(e,CombatEnvironmentV324)
    cfg=yaml.safe_load(Path('configs/mappo_5v5_v324.yaml').read_bytes())
    for validator in (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume):
        with pytest.raises(RuntimeError,match='environment_version'):
            validator({'algorithm':'MAPPO','extra':{'environment_version':'3.23'}},e.config,cfg)
    runner=object.__new__(MAPPOTrainingRunner);runner.env_config=e.config
    runner.trainer=SimpleNamespace(sampled_steps=0);runner.evaluation_seeds=[10000000]
    with patch('algorithm.mappo.runner.evaluate',return_value={}):
        assert runner._evaluation_record()['environment_version']=='3.24'
