"""Terminal PBRS isolation, physical pairing and authoritative timeout metrics."""
from pathlib import Path
import json
from types import SimpleNamespace
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.combat_v31 import CombatEnvironmentV31
from env.combat_v32 import CombatEnvironmentV32
from env.config import load_config,environment_dimensions
from env.models import AircraftState
from algorithm.common.metrics import episode_is_timeout
from algorithm.common.evaluator import aggregate_combat_records
from tools._capture_v32_reference import capture

ROOT=Path(__file__).resolve().parents[1]
def config(version='32'):return load_config(ROOT/f'configs/combat_environment_v{version}.yaml')
def env(version='32'):
    e=MultiUAVCombatEnv(config(version));e.reset(31);return e
def state(x,y=0,psi=0,alive=True):return AircraftState(x,y,-3000,225,0,psi,alive)
def timeout_env(version='32',red=1,blue=1):
    e=env(version);e.max_steps=1
    e.red=[state(-750,i*500,0,i<red) for i in range(5)]
    e.blue=[state(750,i*500,np.pi,i<blue) for i in range(5)]
    return e
def states(e):return np.asarray([[s.x,s.y,s.z,s.v,s.theta,s.psi,float(s.alive)] for s in e.red+e.blue])

def test_legacy_complete_terminal_trajectories_unchanged():
    assert capture()==json.loads((ROOT/'tests/v32_legacy_reference.json').read_text())

def test_config_and_shared_implementations():
    old,new=config('31'),config()
    assert new==dict(old,environment_version='3.2')
    assert environment_dimensions(new)==(66,3,5)
    assert type(env()) is CombatEnvironmentV32
    for method in ('reset','step','potential','_advance','_observations','_entry_attempts','_resolve_combat','_resolve_noncombat_losses','_outcome'):
        assert getattr(CombatEnvironmentV32,method) is getattr(CombatEnvironmentV31,method)
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        assert (ROOT/f'configs/{name}_5v5_v32.yaml').read_bytes()==(ROOT/f'configs/{name}_5v5_v31.yaml').read_bytes()
    assert MultiUAVCombatEnv().environment_version=='2.3'

@pytest.mark.parametrize('red,blue,reason',[(2,1,'red_win_timeout_survivors'),(1,2,'blue_win_timeout_survivors'),(1,1,'draw_timeout_equal_survivors')])
def test_timeout_nonzero_actual_potential_and_legacy_isolation(red,blue,reason):
    for version in ('31','32'):
        e=timeout_env(version,red,blue);phi,_=e.potential();assert np.any(phi!=0)
        _,reward,terminated,truncated,info=e.step(np.zeros((5,3)))
        assert not terminated and truncated and info['timeout'] and info['termination_reason']==reason
        assert np.any(info['phi_next_actual']!=0)
        np.testing.assert_array_equal(info['phi_next'],info['phi_next_actual'])
        assert info['mean_potential_next']==info['phi_next_actual'].mean()
        if version=='32':
            np.testing.assert_array_equal(info['phi_next_for_shaping'],np.zeros(5))
            np.testing.assert_array_equal(info['adv_rewards'],-phi*e.config['reward']['potential_scale'])
            assert not np.array_equal(info['adv_rewards'],e.config['reward']['potential_scale']*(.99*info['phi_next_actual']-phi))
        else:
            np.testing.assert_array_equal(info['phi_next_for_shaping'],info['phi_next_actual'])
            np.testing.assert_array_equal(info['adv_rewards'],e.config['reward']['potential_scale']*(.99*info['phi_next_actual']-phi))
        total=sum((info[f'{n}_rewards'] for n in ('event','outcome','adv','safe')),np.zeros(5))
        np.testing.assert_array_equal(reward,total.astype(np.float32))
        # Component-first versus agent-first summation can differ by float64 roundoff.
        np.testing.assert_allclose(sum(info[f'episode_{n}_total'] for n in ('event','outcome','adv','safe')),total.sum(),rtol=0,atol=1e-14)

def test_nonterminal_shaping_matches_v31():
    a,b=timeout_env('31'),timeout_env();a.max_steps=b.max_steps=10
    left=a.step(np.zeros((5,3)));right=b.step(np.zeros((5,3)))
    assert not right[2] and not right[3]
    np.testing.assert_array_equal(left[1],right[1])
    info=right[4]
    np.testing.assert_array_equal(info['phi_next_for_shaping'],info['phi_next_actual'])
    np.testing.assert_array_equal(info['adv_rewards'],.99*info['phi_next_actual']-info['phi_current'])

@pytest.mark.parametrize('outcome',['red_win_elimination','blue_win_elimination','draw_mutual_destruction'])
def test_elimination_shaping(outcome):
    e=env();e.red=[state(3500,alive=False) for _ in range(5)];e.blue=[state(3500,alive=False) for _ in range(5)]
    e.red[0]=state(3500 if outcome=='red_win_elimination' else 4999)
    e.blue[0]=state(3500 if outcome=='blue_win_elimination' else 4999)
    _,_,terminated,truncated,info=e.step(np.zeros((5,3)))
    assert terminated and not truncated and info['termination_reason']==outcome
    np.testing.assert_array_equal(info['phi_next_for_shaping'],np.zeros(5))
    np.testing.assert_array_equal(info['adv_rewards'],-info['phi_current'])

@pytest.mark.parametrize('seed',[1,31,32000000,33000000])
@pytest.mark.parametrize('mode',['ZERO','MIRROR'])
def test_entire_episode_physics_exactly_matches_v31(seed,mode):
    a,b=env('31'),env();left,_=a.reset(seed);right,_=b.reset(seed)
    for _ in range(1000):
        np.testing.assert_array_equal(states(a),states(b));np.testing.assert_array_equal(left,right)
        np.testing.assert_array_equal(a.red_ammo,b.red_ammo);np.testing.assert_array_equal(a.blue_ammo,b.blue_ammo)
        action=np.zeros((5,3)) if mode=='ZERO' else a.fixed_policy.team_actions(a.red,a.blue)
        # Explicit common Blue actions, so both environment and policy parity are exercised.
        blue=a.fixed_policy.team_actions(a.blue,a.red)
        other_blue=b.fixed_policy.team_actions(b.blue,b.red);np.testing.assert_array_equal(blue,other_blue)
        left,lr,lt,ltr,li=a.step(action,blue);right,rr,rt,rtr,ri=b.step(action,blue)
        assert (lt,ltr)==(rt,rtr)
        assert li['termination_reason']==ri['termination_reason'] and li['episode_length']==ri['episode_length']
        assert a.rng.bit_generator.state==b.rng.bit_generator.state
        for side in ('red','blue'):
            for metric in ('fire_attempts','weapon_hits','attack_kills','ammo_used','losses','boundary_exits','ground_losses','ceiling_losses'):
                assert li[f'{side}_{metric}']==ri[f'{side}_{metric}']
        for name in ('event','outcome','safe'):
            np.testing.assert_array_equal(li[f'{name}_rewards'],ri[f'{name}_rewards'])
        if not rt and not rtr:np.testing.assert_array_equal(lr,rr)
        else:
            np.testing.assert_array_equal(states(a),states(b));np.testing.assert_array_equal(left,right)
            assert np.isfinite(rr).all()
            break
    else:pytest.fail('episode failed to end')

@pytest.mark.parametrize('reason,expected',[
    ('red_failure_timeout',True),('red_win_timeout_survivors',True),('blue_win_timeout_survivors',True),
    ('draw_timeout_equal_survivors',True),('red_win_elimination',False),('blue_win_elimination',False),('draw_mutual_destruction',False)])
def test_timeout_compatibility_and_explicit_priority(reason,expected):
    assert episode_is_timeout({'termination_reason':reason}) is expected
    assert episode_is_timeout({'termination_reason':reason,'timeout':True}) is True
    assert episode_is_timeout({'termination_reason':reason,'timeout':False}) is False
    assert not episode_is_timeout({'termination_reason':'not_a_timeout_reason'})

@pytest.mark.parametrize('name',['mappo','rmappo','ea-mappo','stea-mappo'])
def test_true_truncated_episode_in_training_metrics_and_summary(tmp_path,name):
    from algorithm.mappo.runner import MAPPOTrainingRunner
    from algorithm.rmappo.runner import RMAPPOTrainingRunner
    from algorithm.ea_mappo.runner import EAMAPPOTrainingRunner
    from algorithm.stea_mappo.runner import STEAMAPPOTrainingRunner
    runner_class={'mappo':MAPPOTrainingRunner,'rmappo':RMAPPOTrainingRunner,
                  'ea-mappo':EAMAPPOTrainingRunner,'stea-mappo':STEAMAPPOTrainingRunner}[name]
    e=timeout_env();_,reward,_,truncated,info=e.step(np.zeros((5,3)));assert truncated
    row=dict(info,episode_return=float(reward.sum()),team_episode_return=float(reward.sum()),mean_agent_episode_return=float(reward.mean()))
    runner=object.__new__(runner_class)
    runner.trainer=SimpleNamespace(sampled_steps=1,vector_steps=1,ppo_update_count=0,actor_update_count=0,critic_update_count=0)
    runner.output_dir=tmp_path;runner.last_metrics={};runner.completed_records=[row]
    runner.best_evaluation=None;runner.evaluation_history=[];runner.startup_summary=lambda:{}
    runner._write_step_metrics(SimpleNamespace(rewards=reward[None,:],infos=[info]),[row])
    written=json.loads((tmp_path/'training_metrics.jsonl').read_text())
    assert written['timeout_rate']==1 and runner.summary()['timeout_rate']==1
    assert aggregate_combat_records([row])['timeout_rate']==1
