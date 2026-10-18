"""One-parameter calibration, shared RNG, real combat and identity invariants."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v340 import CombatEnvironmentV340
from env.combat_v341 import CombatEnvironmentV341
from env.config import load_config,validate_config,environment_dimensions
from env.v337_reward import tactical_rewards_v337
from tools.audit_v339_attack_geometry import pair,states,PreFireObserver,red_actions
from tools.audit_v341_probability import analytic,ANGLES

ROOT=Path(__file__).resolve().parents[1]


def config(v='341'):return load_config(ROOT/f'configs/combat_environment_v{v}.yaml')


def env(v='341',seed=0):
    e=MultiUAVCombatEnv(config(v));e.reset(seed);return e


def algorithm(v='341'):return yaml.safe_load((ROOT/f'configs/mappo_4v4_v{v}.yaml').read_text())


def test_exact_one_parameter_diff_and_method_inheritance():
    c=config();before=deepcopy(c);assert validate_config(c) is c and c==before
    c['environment_version']='3.40';c['weapon']['effective_hit_distance']=1000/math.log(6)
    assert c==config('340')
    assert config()['weapon']['effective_hit_distance']==2000/math.log(6)
    assert CombatEnvironmentV341.__bases__==(CombatEnvironmentV340,)
    assert set(CombatEnvironmentV341.__dict__)-{'__module__','__doc__'}=={'environment_version'}
    for name in ('__init__','step','reset','_advance','initial_states','dense_combat_reward','_entry_attempts',
        '_resolve_combat','_resolve_noncombat_losses','_outcome','_observations','build_weapon','attempt_hit_for_pair'):
        assert getattr(CombatEnvironmentV341,name) is getattr(CombatEnvironmentV340,name)
    assert type(env()) is CombatEnvironmentV341 and environment_dimensions(config())==(45,3,4)
    a=algorithm();a['environment']['environment_version']='3.40';assert a==algorithm('340')


def leaves(c,path=()):
    for k,v in c.items():
        if isinstance(v,dict):yield from leaves(v,path+(k,))
        else:yield path+(k,),v


@pytest.mark.parametrize('path,value',list(leaves(config())))
def test_every_configuration_leaf_frozen(path,value):
    c=config();node=c
    for k in path[:-1]:node=node[k]
    node[path[-1]]=value+1 if isinstance(value,(int,float)) else 'invalid'
    before=deepcopy(c)
    with pytest.raises(ValueError):validate_config(c)
    assert c==before


@pytest.mark.parametrize('v',['339','340'])
def test_historical_calibration_cannot_be_changed(v):
    c=config(v);c['weapon']['effective_hit_distance']=2000/math.log(6)
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('d,old,new',[(500,.8010377188282538,.9556082264),(1000,.4,.8010377188282538),
                                   (1500,.16951970499354874,.5882005801),(2000,.06964649854043319,.4)])
def test_expected_aligned_probabilities(d,old,new):
    assert env('340').weapon.expected_hit_probability(d,0,0)==pytest.approx(old,abs=5e-11,rel=0)
    assert env().weapon.expected_hit_probability(d,0,0)==pytest.approx(new,abs=5e-11,rel=0)


@pytest.mark.parametrize('d',[250,500,750,1000,1500,2000])
@pytest.mark.parametrize('angles',ANGLES)
def test_full_angle_analytic_and_normalized_historical_reference(d,angles):
    a,b,h=env('340').weapon,env().weapon,env('339').weapon
    ata,ha=np.deg2rad(angles)
    for w in (a,b):
        assert w.expected_hit_probability(d,ata,ha)==pytest.approx(analytic(w,d,ata,ha),abs=1e-15,rel=0)
    assert a.expected_hit_probability(d,ata,ha)==b.expected_hit_probability(2*d,ata,ha)
    assert h.expected_hit_probability(d,ata,ha)==b.expected_hit_probability(2*d,ata,ha)
    assert math.pi*math.exp(-d/h.effective_hit_distance)==math.pi*math.exp(-2*d/b.effective_hit_distance)


@pytest.mark.parametrize('angles',ANGLES)
def test_same_rng_consumption_allows_hit_difference(angles):
    class Spy:
        def __init__(self):self.rng=np.random.default_rng(44);self.calls=0
        def normal(self,mean,std):
            assert (mean,std)==(0,1);self.calls+=1;return self.rng.normal(mean,std)
    a,b=env('340').weapon,env().weapon;x,y=Spy(),Spy()
    ata,ha=np.deg2rad(angles);out=[]
    for i in range(200):
        out.append(a.attempt_hit(x,1500,ata,ha)!=b.attempt_hit(y,1500,ata,ha))
        assert x.calls==y.calls==i+1 and x.rng.bit_generator.state==y.rng.bit_generator.state
    assert any(out)


@pytest.mark.parametrize('d',[500,1000,1500,2000,2000.00001])
@pytest.mark.parametrize('angles',[(0,0),(30,30),(30,-30),(31,0),(0,-31)])
def test_eligibility_rewards_and_target_selection_same_state(d,angles):
    a,b=env('340'),env();red,blue=pair(d,ata=angles[0],ha=angles[1])
    for e in (a,b):
        e.red=[red.copy() for _ in range(4)];e.blue=[blue.copy() for _ in range(4)]
    for ammo in (0,1,6):
        assert a.weapon.qualifies(red,blue,ammo)==b.weapon.qualifies(red,blue,ammo)
    x=tactical_rewards_v337(a.red,a.blue,a.config['reward'],a.config['weapon'])
    y=tactical_rewards_v337(b.red,b.blue,b.config['reward'],b.config['weapon'])
    for k in x:np.testing.assert_array_equal(x[k],y[k])
    np.testing.assert_array_equal(a.fixed_policy.team_actions(a.blue,a.red),b.fixed_policy.team_actions(b.blue,b.red))
    np.testing.assert_array_equal(red_actions(a,'pursuit')[1],red_actions(b,'pursuit')[1])


@pytest.mark.parametrize('seed',[0,1,19])
@pytest.mark.parametrize('policy',['zero','pursuit','lateral','vertical'])
def test_real_step_exact_prefix_until_hit_fork(seed,policy):
    a,b=env('340',seed),env('341',seed)
    np.testing.assert_array_equal(a._observations(),b._observations())
    with PreFireObserver(a) as x,PreFireObserver(b) as y:
        for _ in range(1000):
            np.testing.assert_array_equal(states(a),states(b))
            actions,targets=red_actions(a,policy);start=states(a)
            ra,rb=a.step(actions),b.step(actions)
            x.finish(start,actions,targets,ra);y.finish(start,actions,targets,rb)
            for key in ('pre','actions','dense','ammo_pre','ammo_post','cooldown_post','attempt_targets','blue_targets'):
                np.testing.assert_array_equal(x.current[key],y.current[key])
            assert a.rng.bit_generator.state==b.rng.bit_generator.state
            assert ra[0].shape==rb[0].shape==(4,45)
            if not np.array_equal(x.current['hit'],y.current['hit']):break
            np.testing.assert_array_equal(states(a),states(b))
        else:pytest.fail('no hit fork observed')


def one_pair(e,d=1500,headon=False):
    r,b=pair(d);b.psi=math.pi if headon else 0.
    e.red=[r]+[r.copy() for _ in range(3)];e.blue=[b]+[b.copy() for _ in range(3)]
    for s in e.red[1:]+e.blue[1:]:s.alive=False


def test_real_step_probability_fork_and_simultaneous_kill():
    class FixedNoise:
        def normal(self,*args):return .5
    a,b=env('340'),env()
    for e in (a,b):one_pair(e,headon=True);e.rng=FixedNoise()
    results=[e.step(np.zeros((4,3)),np.zeros((4,3))) for e in (a,b)]
    assert not results[0][2] and results[1][2]
    assert a.red[0].alive and a.blue[0].alive and not b.red[0].alive and not b.blue[0].alive
    assert a.red_ammo[0]==b.red_ammo[0]==a.blue_ammo[0]==b.blue_ammo[0]==5


def test_normal_step_cooldown_ammo_and_dead_freeze_equal():
    a,b=env('340'),env();fired=[]
    for e in (a,b):one_pair(e)
    frozen=states(a)[:,1:].copy()
    with patch.object(a,'attempt_hit_for_pair',return_value=False),patch.object(b,'attempt_hit_for_pair',return_value=False):
        for t in range(70):
            old=a.red_ammo[0]
            x,y=a.step(np.zeros((4,3)),np.zeros((4,3))),b.step(np.zeros((4,3)),np.zeros((4,3)))
            if a.red_ammo[0]<old:fired.append(t+1)
            np.testing.assert_array_equal(states(a),states(b));np.testing.assert_array_equal(x[1],y[1])
            np.testing.assert_array_equal(states(b)[:,1:],frozen)
            np.testing.assert_array_equal(a.red_fire_cooldown_steps,b.red_fire_cooldown_steps)
    assert fired==[1,11,21,31,41,51] and a.red_ammo[0]==b.red_ammo[0]==0


@pytest.mark.parametrize('mode',['horizontal','ground','ceiling','timeout'])
def test_boundary_and_termination_unchanged(mode):
    a,b=env('340'),env()
    for e in (a,b):
        one_pair(e,3000)
        if mode=='horizontal':e.red[0].x=4999
        if mode=='ground':e.red[0].z=-1;e.red[0].theta=-1
        if mode=='ceiling':e.red[0].z=-5999;e.red[0].theta=1
        if mode=='timeout':e.steps=999
    x,y=[e.step(np.zeros((4,3)),np.zeros((4,3))) for e in (a,b)]
    assert x[2:4]==y[2:4] and any(x[2:4]) and x[3]==(mode=='timeout')
    np.testing.assert_array_equal(states(a),states(b));np.testing.assert_array_equal(x[1],y[1])


@pytest.mark.parametrize('identity',[None,dict(environment_version='3.40',team_size=4),dict(environment_version='3.41',team_size=5)])
def test_wrong_mappo_identity_rejected(identity,tmp_path):
    from algorithm.mappo.runner import MAPPOTrainingRunner
    c=algorithm()
    if identity is None:c.pop('environment')
    else:c['environment']=identity
    with pytest.raises(RuntimeError,match='identity'):MAPPOTrainingRunner(config(),c,output_dir=tmp_path)


@pytest.mark.parametrize('v',['3.39','3.40',None])
@pytest.mark.parametrize('resume',[True,False])
def test_old_checkpoint_rejected_without_loading_weights(v,resume):
    from algorithm.common.checkpoint import validate_checkpoint_for_resume,validate_checkpoint_for_evaluation
    check=validate_checkpoint_for_resume if resume else validate_checkpoint_for_evaluation
    with pytest.raises(RuntimeError,match='environment_version'):
        check(dict(algorithm='MAPPO',extra=dict(environment_version=v)),config(),algorithm())


@pytest.mark.parametrize('field',['environment_config_sha256','algorithm_config_sha256'])
@pytest.mark.parametrize('resume',[True,False])
def test_fingerprints_mandatory(field,resume):
    from algorithm.common import checkpoint
    from algorithm.common.protocol import config_sha256
    c,a=config(),algorithm()
    s=dict(extra=dict(environment_config_sha256=config_sha256(c),algorithm_config_sha256=config_sha256(a)))
    s['extra'].pop(field)
    check=checkpoint.validate_checkpoint_for_resume if resume else checkpoint.validate_checkpoint_for_evaluation
    with patch.object(checkpoint,'_validate_common_checkpoint_contract'):
        with pytest.raises(RuntimeError,match=field):check(s,c,a)
