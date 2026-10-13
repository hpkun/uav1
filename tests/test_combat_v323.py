"""Blue ground death, body-frame shaping and attacker-wide firing cadence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v322 import CombatEnvironmentV322
from env.combat_v323 import CombatEnvironmentV323
from env.config import validate_config, environment_dimensions
from env.models import AircraftState
from env.geometry import engagement_geometry
from env.v322_reward import tactical_rewards_v322, distance_weights
from env.v323_reward import tactical_rewards_v323, body_pair_geometry
from tools.audit_combat_v317_v318 import body_pair, FixedNormal


def env(version='323'):
    e = MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml')
    e.reset(31)
    return e


def dead():
    return AircraftState(0,0,-3000,225,0,0,False)


def pair(pitch=0, az=0, el=0, distance=600, aa=0):
    a,b = body_pair(pitch=pitch,azimuth=az,elevation=el,distance=distance)
    b.psi = math.atan2(b.y-a.y,b.x-a.x)+math.radians(aa)
    return a,b


def fixture(side='red',hit=False):
    e = env()
    a,b = pair()
    setattr(e,side,[a]+[dead() for _ in range(4)])
    setattr(e,'blue' if side=='red' else 'red',[b]+[dead() for _ in range(4)])
    e.rng = FixedNormal(0 if hit else 4)
    return e


def reward(a,b,version='323'):
    cfg = env().config
    func = tactical_rewards_v323 if version=='323' else tactical_rewards_v322
    return func([a],[b],cfg['reward'],cfg['weapon'])


def test_inheritance_config_and_frozen_modules():
    assert CombatEnvironmentV323.__bases__ == (CombatEnvironmentV322,)
    assert set(CombatEnvironmentV323.__dict__)-{'__module__','__doc__'} == {'environment_version','__init__','reset','_resolve_noncombat_losses','dense_combat_reward','_entry_attempts'}
    for name in ('step','_advance','_observations','build_weapon','attempt_hit_for_pair','_resolve_combat','_outcome'):
        assert getattr(CombatEnvironmentV323,name) is getattr(CombatEnvironmentV322,name)
    new,old = env(),env('322')
    cfg = deepcopy(new.config)
    validate_config(cfg)
    assert cfg == new.config and environment_dimensions(cfg)==(56,3,5)
    del cfg['fire_control']
    cfg['environment_version']='3.22'
    assert cfg == old.config
    assert new.cooldown_steps==10
    assert Path('configs/mappo_5v5_v323.yaml').read_bytes()==Path('configs/mappo_5v5_v322.yaml').read_bytes()
    assert type(new.fixed_policy) is type(old.fixed_policy)
    assert type(new.weapon) is type(old.weapon) and vars(new.weapon)==vars(old.weapon)
    for key in ('win_reward','lose_penalty','draw_reward','team_casualty_penalty'):
        assert new.config['reward'][key]==0


@pytest.mark.parametrize('fire',[None,{}, {'cooldown_seconds':0},{'cooldown_seconds':.5},{'cooldown_seconds':True},{'cooldown_seconds':1,'extra':0},{'cooldown_seconds':float('nan')}])
def test_invalid_fire_control_rejected(fire):
    cfg = deepcopy(env().config)
    if fire is None: del cfg['fire_control']
    else: cfg['fire_control']=fire
    with pytest.raises(ValueError): validate_config(cfg)


def test_nonintegral_cooldown_steps_rejected():
    cfg=deepcopy(env().config);cfg['simulation']['dt']=.13
    with pytest.raises(ValueError,match='integer'): validate_config(cfg)


@pytest.mark.parametrize('altitude,alive',[(1,True),(0,False),(-1,False)])
def test_blue_ground_death_without_return(altitude,alive):
    e=env();s=e.blue[0];s.z=-altitude;s.theta=-.4
    before=s.as_array().copy()
    result=e._resolve_noncombat_losses()
    assert s.alive==alive
    np.testing.assert_array_equal(s.as_array(),before)
    assert e.combat_counts['blue']['ground_losses']==int(not alive)
    assert result[3]==([] if alive else [0])
    assert e.blue_returns['ground']==0
    e._resolve_noncombat_losses()
    assert e.combat_counts['blue']['ground_losses']==int(not alive)
    if not alive:
        assert not e.weapon.eligibility(e.blue,e.red,e.blue_ammo)[0][0].any()
        np.testing.assert_array_equal(e.fixed_policy.action(s,e.red),np.zeros(3))
        e._advance(e.blue,np.ones((5,3)))
        np.testing.assert_array_equal(s.as_array(),before)


def test_ground_priority_over_horizontal_return():
    e=env();s=e.blue[0];s.x=6000;s.z=1;s.theta=-.4
    before=s.as_array().copy();e._resolve_noncombat_losses()
    np.testing.assert_array_equal(s.as_array(),before)
    assert not s.alive and e.blue_returns['horizontal']==0 and e.blue_returns['boundary']==0


def test_last_blue_ground_death_termination_no_kill_or_terminal_reward():
    e=env();e.blue=[AircraftState(0,0,0,225,-.4,0)]+[dead() for _ in range(4)]
    with patch.object(e,'_advance',return_value=np.zeros(5)):
        _,_,terminated,truncated,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
    assert terminated and not truncated
    assert info['red_win'] and info['termination_reason']=='red_win_elimination'
    assert info['blue_ground_losses']==1 and info['red_attack_kills']==0
    assert info['blue_ground_returns']==0
    np.testing.assert_array_equal(info['r1_rewards'],np.zeros(5))
    np.testing.assert_array_equal(info['outcome_rewards'],np.zeros(5))


@pytest.mark.parametrize('case',['ground','horizontal','ceiling'])
def test_red_boundary_rules_exact(case):
    old,new=env('322'),env()
    for e in (old,new):
        if case=='ground': e.red[0].z=0
        if case=='horizontal': e.red[0].x=6000
        if case=='ceiling': e.red[0].z=-7000
    assert old._resolve_noncombat_losses()==new._resolve_noncombat_losses()
    assert old.red==new.red and old.combat_counts['red']==new.combat_counts['red']


@pytest.mark.parametrize('case',['horizontal','ceiling','both'])
def test_blue_other_returns_exact(case):
    old,new=env('322'),env()
    for e in (old,new):
        if case in ('horizontal','both'): e.blue[0].x=6000
        if case in ('ceiling','both'): e.blue[0].z=-7000;e.blue[0].theta=.4
    assert old._resolve_noncombat_losses()==new._resolve_noncombat_losses()
    assert old.blue==new.blue and old.blue_returns==new.blue_returns


@pytest.mark.parametrize('az,el,level',[(0,0,.1),(5,5,.1),(5.1,0,.02),(15,15,.02),(15.1,0,.01),(29,0,.01),(0,29,.01),(30,30,.01),(31,0,0),(0,31,0),(-29,-29,.01)])
def test_body_angle_levels_and_weapon_gate(az,el,level):
    a,b=pair(az=az,el=el)
    r=reward(a,b)
    assert r['tactical_offense'][0]==pytest.approx(level)
    assert env().weapon.qualifies(a,b,6)==bool(level)
    a,b=pair(az=az,el=el,distance=2000)
    assert reward(a,b)['guide'][0]==pytest.approx(.0005 if level else 0)


@pytest.mark.parametrize('az,el,coefficient',[(0,0,-.15),(10,10,-.025),(20,20,-.015),(-20,-20,-.015),(31,0,0),(0,31,0)])
def test_reverse_body_threat_levels(az,el,coefficient):
    blue,red=pair(az=az,el=el,distance=2000)
    assert reward(red,blue)['tactical_defense'][0]==pytest.approx(coefficient*.5)


@pytest.mark.parametrize('aspect',[-180,180])
def test_aa_gate_kept_and_r3_does_not_require_weapon_eligibility(aspect):
    a,b=pair(aa=aspect)
    assert env().weapon.qualifies(a,b,6)
    assert reward(a,b)['tactical_offense'][0]==0
    a,b=pair(distance=3500,aa=aspect)
    assert not env().weapon.qualifies(a,b,0)
    assert reward(a,b)['guide'][0]==.001


def test_recording_style_3d_world_body_mismatch():
    a,b=pair(pitch=58,el=17,distance=2000)
    g=engagement_geometry(a,b)
    assert math.degrees(g.ha)==pytest.approx(75)
    assert math.degrees(g.boresight_elevation_error)==pytest.approx(17)
    assert abs(g.aa)<=math.pi/6
    old,new=reward(a,b,'322'),reward(a,b)
    assert old['tactical_offense'][0]==old['guide'][0]==0
    assert new['tactical_offense'][0]==pytest.approx(.005)
    assert new['guide'][0]==pytest.approx(.0005)
    a,b=pair(pitch=58,el=17,distance=900)
    assert env().weapon.qualifies(a,b,6)
    assert reward(a,b)['tactical_offense'][0]==.01


def test_pair_geometry_same_as_weapon_and_distance_handoff():
    cfg=env().config
    for d in (500,1000,1500,2000,2500,2999,3000,3001):
        a,b=pair(distance=d)
        g=engagement_geometry(a,b);arrays=body_pair_geometry([a],[b])
        assert arrays['aa'][0,0]==g.aa
        assert arrays['body_azimuth'][0,0]==g.boresight_azimuth_error
        assert arrays['body_elevation'][0,0]==g.boresight_elevation_error
        w3,w4=distance_weights(g.distance,cfg['reward'],cfg['weapon'])
        r=reward(a,b)
        assert r['guide'][0]==pytest.approx(.001*w3)
        assert r['tactical_offense'][0]==pytest.approx(.1*w4)


def test_guide_max_not_sum_live_mask_and_pair_weight_aggregation():
    cfg=env().config
    a,b=pair(distance=2500);_,c=pair(distance=2000)
    r=tactical_rewards_v323([a],[b,c],cfg['reward'],cfg['weapon'])
    assert r['guide'][0]==.00075 and r['tactical_offense'][0]==.05
    b.alive=False;c.alive=False
    r=tactical_rewards_v323([a],[b,c],cfg['reward'],cfg['weapon'])
    assert all(value[0]==0 for value in r.values())


@pytest.mark.parametrize('side',['red','blue'])
@pytest.mark.parametrize('hit',[False,True])
def test_cooldown_exact_cadence_hit_miss_ammo_and_rng(side,hit):
    e=fixture(side,hit);steps=[]
    for step in range(1,62):
        e.steps=step
        attacker,target=getattr(e,side),getattr(e,'blue' if side=='red' else 'red')
        attempts=e._entry_attempts(attacker,target,None,side)
        if attempts:
            steps.append(step)
            assert attempts[0][2]==hit
            assert getattr(e,side+'_fire_cooldown_steps')[0]==10
        if 2<=step<=10: assert not attempts and getattr(e,side+'_ammo')[0]==5
    assert steps==[1,11,21,31,41,51]
    assert np.all(np.diff(steps)*e.dt==1)
    assert getattr(e,side+'_ammo')[0]==0
    assert e.rng.calls==6
    assert e.combat_counts[side]['ammo_used']==e.combat_counts[side]['fire_attempts']==6


@pytest.mark.parametrize('side',['red','blue'])
def test_target_switch_cannot_bypass_attacker_cooldown(side):
    e=fixture(side)
    targets=getattr(e,'blue' if side=='red' else 'red');targets[1]=AircraftState(700,0,-3000,225,0,0)
    attackers=getattr(e,side)
    assert e._entry_attempts(attackers,targets,None,side)[0][1]==0
    targets[1].x=500
    for _ in range(9): assert not e._entry_attempts(attackers,targets,None,side)
    assert e._entry_attempts(attackers,targets,None,side)[0][1]==1


@pytest.mark.parametrize('side',['red','blue'])
def test_attackers_and_sides_independent(side):
    e=fixture(side);own=getattr(e,side);own[1]=own[0].copy()
    getattr(e,side+'_fire_cooldown_steps')[0]=10
    attempts=e._entry_attempts(own,getattr(e,'blue' if side=='red' else 'red'),None,side)
    assert [a[0] for a in attempts]==[1]
    other='blue' if side=='red' else 'red'
    np.testing.assert_array_equal(getattr(e,other+'_fire_cooldown_steps'),np.zeros(5))


def test_fire_window_independent_of_cooldown_full_step():
    e=fixture();e.red_fire_cooldown_steps[0]=10
    with patch.object(e,'_advance',return_value=np.zeros(5)):
        _,_,_,_,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
    assert info['red_fire_window_pairs']==1
    assert info['red_step_fire_attempts']==0
    assert e.red_ammo[0]==6 and e.rng.calls==0
    assert e.combat_counts['red']['fire_window_steps']==1


def test_reset_clears_cooldown():
    e=env();e.red_fire_cooldown_steps[:]=10;e.blue_fire_cooldown_steps[:]=7
    e.reset(31)
    assert e.red_fire_cooldown_steps.dtype==np.int64
    np.testing.assert_array_equal(e.red_fire_cooldown_steps,np.zeros(5))
    np.testing.assert_array_equal(e.blue_fire_cooldown_steps,np.zeros(5))


def test_simultaneous_fire_resolution_unchanged():
    e=fixture(hit=True)
    e.blue[0].psi=math.pi
    with patch.object(e,'_advance',return_value=np.zeros(5)):
        _,_,terminated,_,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
    assert info['red_step_fire_attempts']==info['blue_step_fire_attempts']==1
    assert info['red_attack_kills']==info['blue_attack_kills']==1
    assert not e.red[0].alive and not e.blue[0].alive and terminated
    assert e.red_fire_cooldown_steps[0]==e.blue_fire_cooldown_steps[0]==10


@pytest.mark.parametrize('altitude',[500,3000,5500])
def test_safety_and_event_rewards_static_regression(altitude):
    old,new=env('322'),env()
    for e in (old,new):
        e.red[0].z=-altitude
    with patch.object(old,'_advance',return_value=np.zeros(5)),patch.object(new,'_advance',return_value=np.zeros(5)):
        _,_,_,_,oi=old.step(np.zeros((5,3)),np.zeros((5,3)))
        _,_,_,_,ni=new.step(np.zeros((5,3)),np.zeros((5,3)))
    for key in ('r1_rewards','r2_rewards','safety_rewards','outcome_rewards'):
        np.testing.assert_array_equal(oi[key],ni[key])


@pytest.mark.parametrize('case',['kill','loss','ground','boundary'])
def test_r1_r2_and_zero_terminal_reward(case):
    e=fixture(hit=True)
    if case=='loss': e=fixture('blue',True)
    if case=='ground': e.red[0].z=0
    if case=='boundary': e.red[0].x=6000
    with patch.object(e,'_advance',return_value=np.zeros(5)):
        _,_,_,_,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
    assert info['r1_rewards'][0]==(10 if case=='kill' else -10 if case in ('loss','ground') else 0)
    assert info['r2_rewards'][0]==(-10 if case=='boundary' else 0)
    np.testing.assert_array_equal(info['outcome_rewards'],np.zeros(5))


@pytest.mark.parametrize('seed',[1,31,10000000])
def test_reset_motion_policy_physics_and_hit_sampler_unchanged(seed):
    old,new=env('322'),env()
    oo,_=old.reset(seed);no,_=new.reset(seed)
    np.testing.assert_array_equal(oo,no)
    for name in ('fixed_policy','dynamics','integrator','weapon'):
        assert type(getattr(old,name)) is type(getattr(new,name))
    actions=np.tile([.4,-.2,.5],(5,1))
    for side in ('red','blue'):
        assert getattr(old,side)==getattr(new,side)
        np.testing.assert_array_equal(getattr(old,side+'_ammo'),getattr(new,side+'_ammo'))
    np.testing.assert_array_equal(old.fixed_policy.team_actions(old.blue,old.red),new.fixed_policy.team_actions(new.blue,new.red))
    for side in ('red','blue'):
        np.testing.assert_array_equal(old._advance(getattr(old,side),actions),new._advance(getattr(new,side),actions))
        assert getattr(old,side)==getattr(new,side)
    for eps in (0,.5,-.5,2,-2):
        assert old.weapon.attempt_hit(FixedNormal(eps),900,.2,-.1)==new.weapon.attempt_hit(FixedNormal(eps),900,.2,-.1)


def test_recording_checkpoint_isolation_and_version_logging():
    from tools.combat_visualization import RecordingCombatEnv
    from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
    from algorithm.mappo.runner import MAPPOTrainingRunner
    e=RecordingCombatEnv('configs/combat_environment_v323.yaml')
    assert isinstance(e,CombatEnvironmentV323)
    cfg=yaml.safe_load(Path('configs/mappo_5v5_v323.yaml').read_bytes())
    for validator in (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume):
        with pytest.raises(RuntimeError,match='environment_version'):
            validator({'algorithm':'MAPPO','extra':{'environment_version':'3.22'}},e.config,cfg)
    runner=object.__new__(MAPPOTrainingRunner);runner.env_config=e.config
    runner.trainer=SimpleNamespace(sampled_steps=0);runner.evaluation_seeds=[10000000]
    with patch('algorithm.mappo.runner.evaluate',return_value={}):
        assert runner._evaluation_record()['environment_version']=='3.23'
