"""Symmetric noncombat deaths and actual R1-R4-only training rewards."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v324 import CombatEnvironmentV324
from env.combat_v325 import CombatEnvironmentV325
from env.config import validate_config,environment_dimensions
from env.models import AircraftState
from tools.audit_combat_v317_v318 import body_pair,FixedNormal


def env(version='325'):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml');e.reset(31);return e


def dead():
    return AircraftState(0,0,-3000,225,0,0,False)


def set_violation(state,case):
    if case in ('ground','ground_horizontal'):state.z=1
    if case in ('horizontal','ground_horizontal','ceiling_horizontal'):state.x=6000
    if case in ('ceiling','ceiling_horizontal'):state.z=-6001
    state.theta=-.3;state.psi=.7


def assert_decomposition(reward,info):
    np.testing.assert_array_equal(reward,sum(info[f'r{i}_rewards'] for i in range(1,5)))
    for name in ('safe_rewards','safety_rewards','outcome_rewards'):
        np.testing.assert_array_equal(info[name],np.zeros(5))


def test_minimal_inheritance_and_two_config_differences():
    assert CombatEnvironmentV325.__bases__==(CombatEnvironmentV324,)
    assert set(CombatEnvironmentV325.__dict__)-{'__module__','__doc__'}=={'environment_version','_resolve_noncombat_losses'}
    for name in ('__init__','reset','step','_advance','_observations','dense_combat_reward','build_weapon','attempt_hit_for_pair','_entry_attempts','_resolve_combat','_outcome'):
        assert getattr(CombatEnvironmentV325,name) is getattr(CombatEnvironmentV324,name)
    old,new=env('324'),env();cfg=deepcopy(new.config)
    validate_config(cfg);assert cfg==new.config
    assert environment_dimensions(cfg)==(56,3,5)
    assert cfg['reward']['safety_penalty_scale']==0 and cfg['reward']['safe_distance']==200
    cfg['environment_version']='3.24';cfg['reward']['safety_penalty_scale']=.2
    assert cfg==old.config
    assert Path('configs/mappo_5v5_v325.yaml').read_bytes()==Path('configs/mappo_5v5_v324.yaml').read_bytes()


@pytest.mark.parametrize('scale',[.2,.001,-.1,True,False,'0',float('nan')])
def test_safety_scale_strictly_frozen(scale):
    cfg=deepcopy(env().config);cfg['reward']['safety_penalty_scale']=scale
    with pytest.raises(ValueError):validate_config(cfg)


@pytest.mark.parametrize('key',['win_reward','lose_penalty','draw_reward','team_casualty_penalty'])
def test_terminal_and_team_fields_still_zero(key):
    cfg=deepcopy(env().config);assert cfg['reward'][key]==0;cfg['reward'][key]=20
    with pytest.raises(ValueError):validate_config(cfg)


@pytest.mark.parametrize('side',['red','blue'])
@pytest.mark.parametrize('case',['ground','horizontal','ceiling','ground_horizontal','ceiling_horizontal'])
def test_symmetric_deaths_priority_pose_and_counters(side,case):
    e=env();s=getattr(e,side)[0];set_violation(s,case);before=s.as_array().copy()
    result=e._resolve_noncombat_losses()
    assert not s.alive;np.testing.assert_array_equal(before,s.as_array())
    ground=case.startswith('ground');ceiling=case.startswith('ceiling')
    counts=e.combat_counts[side]
    assert counts['ground_losses']==int(ground)
    assert counts['boundary_exits']==int(not ground)
    assert counts['ceiling_losses']==int(ceiling)
    index=(2 if side=='red' else 3) if ground else (0 if side=='red' else 1)
    assert result[index]==[0]
    assert all(v==0 for v in e.blue_returns.values())
    e._resolve_noncombat_losses()
    assert counts['ground_losses']==int(ground) and counts['boundary_exits']==int(not ground)


@pytest.mark.parametrize('side',['red','blue'])
@pytest.mark.parametrize('radius,altitude,alive',[(4999,3000,True),(5000,3000,True),(5000.001,3000,False),(1000,6000,True),(1000,6000.001,False),(1000,.001,True),(1000,0,False)])
def test_exact_boundary_thresholds(side,radius,altitude,alive):
    e=env();s=getattr(e,side)[0];s.x=radius;s.y=0;s.z=-altitude
    e._resolve_noncombat_losses();assert s.alive==alive


@pytest.mark.parametrize('case',['ground','horizontal','ceiling','ground_horizontal','ceiling_horizontal'])
def test_red_accounting_identical_to_v324(case):
    old,new=env('324'),env()
    for e in (old,new):set_violation(e.red[0],case)
    assert old._resolve_noncombat_losses()==new._resolve_noncombat_losses()
    assert old.red==new.red and old.combat_counts['red']==new.combat_counts['red']


@pytest.mark.parametrize('case',['ground','horizontal','ceiling'])
def test_last_blue_noncombat_death_no_kill_or_terminal_bonus(case):
    e=env();e.blue=[AircraftState(0,0,-3000,225,0,0)]+[dead() for _ in range(4)]
    set_violation(e.blue[0],case);before=e.blue[0].as_array().copy()
    with patch.object(e,'_advance',return_value=np.zeros(5)):
        _,reward,terminated,truncated,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
    assert terminated and not truncated and info['red_win']
    assert info['termination_reason']=='red_win_elimination'
    assert info['red_attack_kills']==info['red_weapon_hits']==0
    np.testing.assert_array_equal(info['r1_rewards'],np.zeros(5))
    np.testing.assert_array_equal(info['r2_rewards'],np.zeros(5))
    np.testing.assert_array_equal(e.blue[0].as_array(),before)
    assert all(v==0 for v in e.blue_returns.values())
    assert info['blue_ground_losses']==int(case=='ground')
    assert info['blue_boundary_exits']==int(case!='ground')
    assert info['blue_ceiling_losses']==int(case=='ceiling')
    assert_decomposition(reward,info)


@pytest.mark.parametrize('distance',[1,50,100,199,200,201,500])
def test_safety_really_zero_with_close_living_aircraft(distance):
    old,new=env('324'),env()
    for e in (old,new):
        e.red=[AircraftState(0,0,-3000,225,0,0),AircraftState(distance,0,-3000,225,0,0)]+[dead() for _ in range(3)]
        e.red_ammo.fill(0);e.blue_ammo.fill(0)
    with patch.object(old,'_advance',return_value=np.zeros(5)),patch.object(new,'_advance',return_value=np.zeros(5)):
        _,ro,_,_,oi=old.step(np.zeros((5,3)),np.zeros((5,3)))
        _,rn,_,_,ni=new.step(np.zeros((5,3)),np.zeros((5,3)))
    if distance<200:assert oi['safety_rewards'][0]<0
    assert_decomposition(rn,ni)
    np.testing.assert_array_equal(ro-oi['safety_rewards'],rn)


@pytest.mark.parametrize('scenario',['normal','guide','offense','defense','red_ground','red_boundary','red_ceiling','weapon_kill','red_weapon_loss','blue_death','timeout'])
def test_actual_reward_r1_r2_r3_r4_exact_for_all_event_types(scenario):
    e=env();a,b=body_pair(distance=3500 if scenario=='guide' else 2000 if scenario in ('offense','defense') else 600)
    if scenario=='normal':b.x=12000
    if scenario=='defense' or scenario=='red_weapon_loss':a,b=b,a
    e.red=[a]+[dead() for _ in range(4)];e.blue=[b]+[dead() for _ in range(4)]
    e.rng=FixedNormal(0 if scenario in ('weapon_kill','red_weapon_loss') else 4)
    if scenario=='red_ground':set_violation(e.red[0],'ground')
    if scenario=='red_boundary':set_violation(e.red[0],'horizontal')
    if scenario=='red_ceiling':set_violation(e.red[0],'ceiling')
    if scenario=='blue_death':set_violation(e.blue[0],'horizontal')
    if scenario=='timeout':e.steps=999
    with patch.object(e,'_advance',return_value=np.zeros(5)):
        _,returned,terminated,truncated,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
    assert_decomposition(returned,info)
    if scenario=='guide':assert info['r3_rewards'][0]==.001
    if scenario=='offense':assert info['r4_rewards'][0]==.05
    if scenario=='defense':assert info['r4_rewards'][0]==-.075
    if scenario in ('red_ground','red_weapon_loss'):assert info['r1_rewards'][0]==-10
    if scenario in ('red_boundary','red_ceiling'):assert info['r2_rewards'][0]==-10
    if scenario=='weapon_kill':assert info['r1_rewards'][0]==10
    if scenario=='timeout':assert truncated and not terminated


@pytest.mark.parametrize('distance',[900,1000,1500,2000,2500,3000,3001])
def test_body_r3_r4_and_continuous_weights_exact(distance):
    old,new=env('324'),env();a,b=body_pair(pitch=58,elevation=17,distance=distance)
    for e in (old,new):e.red=[a.copy()]+[dead() for _ in range(4)];e.blue=[b.copy()]+[dead() for _ in range(4)]
    ro,do=old.dense_combat_reward();rn,dn=new.dense_combat_reward()
    np.testing.assert_array_equal(ro,rn)
    for name in do:np.testing.assert_array_equal(do[name],dn[name])


def test_weapon_cooldown_ammo_and_rng_exact():
    old,new=env('324'),env();a,b=body_pair()
    for e in (old,new):
        e.red=[a.copy()]+[dead() for _ in range(4)];e.blue=[b.copy()]+[dead() for _ in range(4)];e.rng=FixedNormal(4)
    steps=[]
    for step in range(1,62):
        for e in (old,new):e.steps=step
        np.testing.assert_array_equal(old.weapon.eligibility(old.red,old.blue,old.red_ammo)[0],new.weapon.eligibility(new.red,new.blue,new.red_ammo)[0])
        ao=old._entry_attempts(old.red,old.blue,None,'red');an=new._entry_attempts(new.red,new.blue,None,'red')
        assert ao==an
        if an:steps.append(step)
        np.testing.assert_array_equal(old.red_ammo,new.red_ammo)
        np.testing.assert_array_equal(old.red_fire_cooldown_steps,new.red_fire_cooldown_steps)
    assert steps==[1,11,21,31,41,51] and new.rng.calls==old.rng.calls==6
    for eps in (0,.5,-.5,2,-2):
        assert old.weapon.attempt_hit(FixedNormal(eps),900,.2,-.1)==new.weapon.attempt_hit(FixedNormal(eps),900,.2,-.1)


@pytest.mark.parametrize('seed',[1,31,10000000])
def test_observation_policy_and_fixed_actions_transition_regression(seed):
    old,new=env('324'),env();oo,_=old.reset(seed);no,_=new.reset(seed)
    np.testing.assert_array_equal(oo,no);assert no.shape==(5,56)
    for name in ('fixed_policy','dynamics','integrator','weapon'):assert type(getattr(old,name)) is type(getattr(new,name))
    for side in ('red','blue'):
        assert getattr(old,side)==getattr(new,side)
        np.testing.assert_array_equal(getattr(old,side+'_ammo'),getattr(new,side+'_ammo'))
        np.testing.assert_array_equal(getattr(old,side+'_fire_cooldown_steps'),getattr(new,side+'_fire_cooldown_steps'))
    for _ in range(100):
        action=np.tile([.2,0,0],(5,1))
        oo,ro,to,xo,io=old.step(action);no,rn,tn,xn,inn=new.step(action)
        assert (to,xo)==(tn,xn) and old.red==new.red and old.blue==new.blue
        assert all(v==0 for v in old.blue_returns.values())
        np.testing.assert_array_equal(oo,no)
        assert old.combat_counts==new.combat_counts
        for key in ('r1_rewards','r2_rewards','r3_rewards','r4_rewards'):
            np.testing.assert_array_equal(io[key],inn[key])
        for side in ('red','blue'):
            for suffix in ('_ammo','_fire_cooldown_steps'):
                np.testing.assert_array_equal(getattr(old,side+suffix),getattr(new,side+suffix))
        np.testing.assert_array_equal(ro-io['safety_rewards'],rn)
        assert_decomposition(rn,inn)
        if to or xo:break


def test_observation_dead_signed_slots_exact():
    old,new=env('324'),env()
    for e in (old,new):e.red[1].alive=False;e.blue[2].alive=False;e.red[0].theta=.6;e.blue[0].psi=-2.
    np.testing.assert_array_equal(old._observations(),new._observations())


def test_recording_checkpoint_isolation_and_logging():
    from tools.combat_visualization import RecordingCombatEnv
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
    from algorithm.mappo.runner import MAPPOTrainingRunner
    e=RecordingCombatEnv('configs/combat_environment_v325.yaml');assert isinstance(e,CombatEnvironmentV325)
    cfg=yaml.safe_load(Path('configs/mappo_5v5_v325.yaml').read_bytes())
    for validator in (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume):
        with pytest.raises(RuntimeError,match='environment_version'):
            validator({'algorithm':'MAPPO','extra':{'environment_version':'3.24'}},e.config,cfg)
    runner=object.__new__(MAPPOTrainingRunner);runner.env_config=e.config
    runner.trainer=SimpleNamespace(sampled_steps=0);runner.evaluation_seeds=[10000000]
    with patch('algorithm.mappo.runner.evaluate',return_value={}):assert runner._evaluation_record()['environment_version']=='3.25'
