"""Controlled range experiment, real transition and MAPPO identity contracts."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import pytest
import yaml
from env.config import load_config, validate_config, environment_dimensions
from env.combat_env import MultiUAVCombatEnv
from env.combat_v339 import CombatEnvironmentV339
from env.combat_v340 import CombatEnvironmentV340
from env.v322_reward import distance_weights
from env.v337_reward import tactical_rewards_v337, tactical_pair_rewards_v337
from tools.audit_v339_attack_geometry import pair, states, PreFireObserver

ROOT=Path(__file__).resolve().parents[1]


def config(v='340'):return load_config(ROOT/f'configs/combat_environment_v{v}.yaml')


def env(v='340'):
    e=MultiUAVCombatEnv(config(v));e.reset(31);return e


def algorithm(v='340'):
    return yaml.safe_load((ROOT/f'configs/mappo_4v4_v{v}.yaml').read_text())


def one_pair(e,d=1500,headon=False):
    a,b=pair(d)
    if headon:b.psi=math.pi
    e.red=[a]+[a.copy() for _ in range(3)]
    e.blue=[b]+[b.copy() for _ in range(3)]
    for s in e.red[1:]+e.blue[1:]:s.alive=False


def test_exact_delta_class_and_config_immutability():
    a,b=config('339'),config()
    original=deepcopy(b)
    assert validate_config(b) is b and b==original
    b['environment_version']='3.39';b['weapon']['range_max']=1000.;b['reward']['tactical_distance_max']=3000.
    assert a==b
    assert set(CombatEnvironmentV340.__dict__)-{'__module__','__doc__'}=={'environment_version'}
    assert CombatEnvironmentV340.__bases__==(CombatEnvironmentV339,)
    for name in ('__init__','step','reset','_advance','initial_states','dense_combat_reward','_entry_attempts',
                 '_resolve_combat','_resolve_noncombat_losses','_outcome','_observations','attempt_hit_for_pair'):
        assert getattr(CombatEnvironmentV340,name) is getattr(CombatEnvironmentV339,name)
    assert type(env()) is CombatEnvironmentV340
    assert environment_dimensions(config())==(45,3,4)
    c,d=algorithm('339'),algorithm();d['environment']['environment_version']='3.39';assert c==d
    assert d['training']['total_sampled_steps']==3000000 and d['training']['device']=='cuda'


@pytest.mark.parametrize('section,key,value',[
    ('weapon','range_max',1000),('weapon','range_max',2000.00001),('weapon','range_max',float('nan')),
    ('reward','tactical_distance_max',3000),('reward','guide_distance_min',4000),
    ('weapon','range_min',1),('weapon','ata_max',.6),('weapon','ha_max',.6),('weapon','ammo_per_aircraft',7),
    ('weapon','effective_hit_distance',600),('weapon','azimuth_noise_scale',1),('weapon','elevation_noise_scale',1),
    ('reward','offense_reward_5deg',.2),('reward','threat_penalty_5deg',-.3),('reward','guide_reward',.002),
    ('fire_control','cooldown_seconds',2),('scenario','center_radius',4800),('simulation','dt',.2),
    ('arena','radius',6000),('observation','speed_scale',50),('blue_policy','mid_desired_speed',225),
    ('blue_policy','near_distance_max',2000),('blue_policy','far_distance_min',4000)])
def test_strict_config_rejects_drift_without_mutation(section,key,value):
    c=config();c[section][key]=value
    before=repr(c)
    with pytest.raises(ValueError):validate_config(c)
    assert repr(c)==before


@pytest.mark.parametrize('section,key,value',[('weapon','range_max',2000),('reward','tactical_distance_max',4000)])
def test_old_version_still_rejects_new_ranges(section,key,value):
    c=config('339');c[section][key]=value
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('d,old,new',[(0,True,True),(1000,True,True),(1000.000001,False,True),
    (1500,False,True),(1999.999999,False,True),(2000,False,True),(2000.000001,False,False)])
def test_weapon_inclusive_range(d,old,new):
    a,b=pair(d)
    assert env('339').weapon.qualifies(a,b,6)==old
    assert env().weapon.qualifies(a,b,6)==new
    assert not env().weapon.qualifies(a,b,0)


@pytest.mark.parametrize('ata,ha,ok',[(30,0,True),(-30,0,True),(0,30,True),(0,-30,True),
    (30.00001,0,False),(-30.00001,0,False),(0,30.00001,False),(0,-30.00001,False)])
def test_weapon_angles(ata,ha,ok):
    assert env().weapon.qualifies(*pair(1500,ata,ha),6)==ok


@pytest.mark.parametrize('d',[500,1000,1500,2000])
@pytest.mark.parametrize('ata,ha',[(0,0),(.2,.2),(.2,-.2),(.5,.5)])
def test_probability_and_random_draws_exact(d,ata,ha):
    a,b=env('339').weapon,env().weapon
    assert a.expected_hit_probability(d,ata,ha)==b.expected_hit_probability(d,ata,ha)
    x,y=np.random.default_rng(71),np.random.default_rng(71)
    assert [a.attempt_hit(x,d,ata,ha) for _ in range(100)]==[b.attempt_hit(y,d,ata,ha) for _ in range(100)]
    assert x.bit_generator.state==y.bit_generator.state


@pytest.mark.parametrize('d,w3,w4',[(1000,0,1),(1500,0,1),(2000,0,1),(2500,.25,.75),
                                  (3000,.5,.5),(3500,.75,.25),(4000,1,0),(4500,1,0)])
def test_actual_reward_and_distance_weights(d,w3,w4):
    c=config();a,b=pair(d)
    assert distance_weights(d,c['reward'],c['weapon'])==(w3,w4)
    actual=tactical_rewards_v337([a],[b],c['reward'],c['weapon'])
    assert actual['guide'][0]==pytest.approx(.001*w3)
    assert actual['tactical_offense'][0]==pytest.approx(.1*w4)
    reverse=tactical_rewards_v337([b],[a],c['reward'],c['weapon'])
    assert reverse['tactical_defense'][0]==pytest.approx(-.15*w4)
    old=config('339')
    assert distance_weights(3000,old['reward'],old['weapon'])==(1,0)


@pytest.mark.parametrize('axis',['ata','ha'])
@pytest.mark.parametrize('angle,coef',[(5,.1),(5.00001,.02),(15,.02),(15.00001,.01),(30,.01),(30.00001,0)])
def test_reward_tiers_and_reverse_aggregation(axis,angle,coef):
    c=config();a,b=pair(3000,**{axis:angle})
    dead=b.copy();dead.alive=False
    pairs=tactical_pair_rewards_v337([a,a.copy()],[b,dead],c['reward'],c['weapon'])
    actual=tactical_rewards_v337([a,a.copy()],[b,dead],c['reward'],c['weapon'])
    assert actual['tactical_offense'][0]==pytest.approx(coef*.5)
    np.testing.assert_array_equal(actual['tactical_offense'],pairs['tactical_offense'].max(1))
    np.testing.assert_array_equal(actual['tactical_defense'],pairs['tactical_defense'].min(0))
    np.testing.assert_array_equal(actual['guide'],pairs['guide'].max(1))


@pytest.mark.parametrize('aa',[30.00001,-30.00001,180])
def test_aa_gate(aa):
    c=config();a,b=pair(3000,aa=aa)
    assert tactical_rewards_v337([a],[b],c['reward'],c['weapon'])['tactical_offense'][0]==0


def test_real_step_simultaneous_kills_and_frozen_death():
    e=env();one_pair(e,1500,headon=True)
    frozen=states(e)[:,1:].copy()
    with patch.object(e,'attempt_hit_for_pair',return_value=True):
        obs,rewards,terminated,truncated,info=e.step(np.zeros((4,3)),np.zeros((4,3)))
    assert terminated and not truncated and obs.shape==(4,45)
    assert not e.red[0].alive and not e.blue[0].alive
    assert e.red_ammo[0]==e.blue_ammo[0]==5
    assert info['red_weapon_hits']==info['blue_weapon_hits']==1
    assert np.isfinite(rewards).all()
    np.testing.assert_array_equal(states(e)[:,1:],frozen)


def test_real_step_cooldown_ammo_exhaustion_and_reward():
    e=env();one_pair(e,1500)
    fire_steps=[]
    # Same heading/speed explicit commands hold the separation; live Blue never fires backward.
    with patch.object(e,'attempt_hit_for_pair',return_value=False):
        for _ in range(70):
            before=int(e.red_ammo[0])
            obs,reward,term,trunc,info=e.step(np.zeros((4,3)),np.zeros((4,3)))
            if e.red_ammo[0]<before:fire_steps.append(e.steps)
            assert not term and not trunc
            assert info['tactical_offense_rewards'][0]==.1
            assert info['guide_rewards'][0]==0
    assert e.red_ammo[0]==0 and e.blue_ammo[0]==6
    assert e.combat_counts['red']['fire_attempts']==6
    assert e.cooldown_steps==10
    assert fire_steps==[1,11,21,31,41,51]


@pytest.mark.parametrize('mode',['boundary','ground','ceiling','timeout'])
def test_normal_step_boundary_and_termination(mode):
    e=env();one_pair(e,3000)
    if mode=='boundary':e.red[0].x=4999
    if mode=='ground':e.red[0].z=-1;e.red[0].theta=-1
    if mode=='ceiling':e.red[0].z=-5999;e.red[0].theta=1
    if mode=='timeout':e.steps=999
    obs,reward,term,trunc,info=e.step(np.zeros((4,3)),np.zeros((4,3)))
    assert term or trunc
    assert trunc==(mode=='timeout') and obs.shape==(4,45) and np.isfinite(reward).all()
    if mode!='timeout':assert not e.red[0].alive


@pytest.mark.parametrize('seed',[0,1,19])
def test_motion_and_reset_match_before_combat(seed):
    a,b=env('339'),env();a.reset(seed);b.reset(seed)
    np.testing.assert_array_equal(states(a),states(b))
    rng=np.random.default_rng(21)
    for _ in range(50):
        actions=rng.uniform(-1,1,(4,3))
        x,y=a.step(actions),b.step(actions)
        np.testing.assert_array_equal(states(a),states(b))
        np.testing.assert_array_equal(x[0],y[0])
        assert a.rng.bit_generator.state==b.rng.bit_generator.state
        assert a.red_ammo.tolist()==b.red_ammo.tolist()==[6]*4


def test_observer_transparent_and_shots_recorded():
    a,b=env(),env()
    with PreFireObserver(b) as o:
        for _ in range(160):
            actions=np.zeros((4,3));before=states(b)
            x,y=a.step(actions),b.step(actions)
            o.finish(before,actions,np.full(4,-1),y)
            np.testing.assert_array_equal(states(a),states(b))
            np.testing.assert_array_equal(x[1],y[1])
            assert a.rng.bit_generator.state==b.rng.bit_generator.state
            if x[2] or x[3]:break
        assert o.calls==dict(dense=b.steps,resolve=b.steps)
        assert o.arrays()['attempt_targets'].max()>=0


@pytest.mark.parametrize('identity',[None,dict(environment_version='3.39',team_size=4),dict(environment_version='3.40',team_size=5)])
def test_mappo_identity_rejected_before_trainer(identity,tmp_path):
    from algorithm.mappo.runner import MAPPOTrainingRunner
    c=algorithm()
    if identity is None:c.pop('environment')
    else:c['environment']=identity
    with patch('algorithm.mappo.runner.MAPPOTrainer',side_effect=AssertionError('trainer called')):
        with pytest.raises(RuntimeError,match='identity'):
            MAPPOTrainingRunner(config(),c,output_dir=tmp_path)


def test_mappo_correct_config_reaches_unchanged_constructor(tmp_path):
    from algorithm.mappo.runner import MAPPOTrainingRunner
    with patch('algorithm.mappo.runner.MAPPOTrainer',side_effect=RuntimeError('identity passed')) as constructor:
        with pytest.raises(RuntimeError,match='identity passed'):
            MAPPOTrainingRunner(config(),algorithm(),output_dir=tmp_path)
    assert constructor.call_args.args[:4]==(45,3,4,256)
    assert constructor.call_args.kwargs['critic_hidden_dim']==256


@pytest.mark.parametrize('version',['3.38','3.39',None])
def test_old_checkpoint_identity_rejected(version):
    from algorithm.common.checkpoint import validate_checkpoint_environment
    with pytest.raises(RuntimeError,match='environment_version mismatch'):
        validate_checkpoint_environment({'extra':dict(environment_version=version)},config())


@pytest.mark.parametrize('resume',[False,True])
@pytest.mark.parametrize('field',['environment_config_sha256','algorithm_config_sha256'])
def test_v340_requires_checkpoint_fingerprints(resume,field):
    from algorithm.common import checkpoint
    from algorithm.common.protocol import config_sha256
    e,c=config(),algorithm()
    state=dict(extra=dict(environment_config_sha256=config_sha256(e),algorithm_config_sha256=config_sha256(c)))
    state['extra'].pop(field)
    check=checkpoint.validate_checkpoint_for_resume if resume else checkpoint.validate_checkpoint_for_evaluation
    # Test metadata only; never loads weights or evaluates a checkpoint on CPU.
    with patch.object(checkpoint,'_validate_common_checkpoint_contract'):
        with pytest.raises(RuntimeError,match=field):check(state,e,c)


def test_shared_evaluator_and_rollout_metrics_schema():
    from algorithm.common.evaluator import aggregate_combat_records
    from algorithm.mappo.runner import MAPPOTrainingRunner
    from types import SimpleNamespace
    def record(v):
        e=env(v);e.steps=999
        _,r,_,_,i=e.step(np.zeros((4,3)))
        return dict(episode_return=float(r.sum()),mean_agent_episode_return=float(r.mean()),**i)
    assert aggregate_combat_records([record('339')]).keys()==aggregate_combat_records([record('340')]).keys()
    runner=object.__new__(MAPPOTrainingRunner);runner.env_config=config()
    runner._begin_rollout_metrics()
    runner._accumulate_rollout_metrics(SimpleNamespace(infos=[record('340')],rewards=np.zeros((1,4))),[])
    for name in ('r3','r4','paper','safety','extended'):
        assert f'mean_{name}_reward' in runner._rollout_metrics['sums']
