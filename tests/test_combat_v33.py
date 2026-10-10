"""Reward-only extension: causal loss cases, unchanged physics and old full info."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v32 import CombatEnvironmentV32
from env.combat_v33 import CombatEnvironmentV33
from env.config import load_config,validate_config,environment_dimensions
from env.models import AircraftState
from tools._capture_v33_reference import capture
from tools.audit_combat_v33_reward import paired_rollout
from algorithm.common.evaluator import aggregate_combat_records

ROOT=Path(__file__).resolve().parents[1]
def cfg(version='33'):return load_config(ROOT/f'configs/combat_environment_v{version}.yaml')
def state(x,y=0,z=-3000,theta=0,psi=0,alive=True):return AircraftState(x,y,z,225,theta,psi,alive)
class Hit:
    def random(self):return 0.
def combat():
    e=MultiUAVCombatEnv(cfg());e.reset(31)
    e.red=[state(-2500,500*i+1000) for i in range(5)];e.red[0]=state(-350,psi=np.pi)
    e.blue=[state(350,psi=np.pi,alive=i==0) for i in range(5)];e.rng=Hit()
    return e
def step(e):return e.step(np.zeros((5,3)),np.zeros((5,3)))[4]

def test_historical_v30_v31_v32_full_info_and_reward_bytes():
    assert capture()==json.loads((ROOT/'tests/v33_legacy_reference.json').read_text())

def test_configuration_and_shared_protocol():
    old,new=cfg('32'),cfg();expected=copy.deepcopy(old);expected['environment_version']='3.3'
    expected['reward'].update(team_casualty_penalty=-2.,win_reward=20.,lose_penalty=-20.)
    assert new==expected and environment_dimensions(new)==(66,3,5)
    assert MultiUAVCombatEnv().environment_version=='2.3'
    for method in ('step','reset','potential','shaping_next_potential','_observations','_advance','_outcome','_entry_attempts','_resolve_combat','_event_reward_components'):
        assert getattr(CombatEnvironmentV33,method) is getattr(CombatEnvironmentV32,method)
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        assert (ROOT/f'configs/{name}_5v5_v33.yaml').read_bytes()==(ROOT/f'configs/{name}_5v5_v32.yaml').read_bytes()

def test_one_combat_death_and_old_dead_slot():
    for dead in (None,3):
        e=combat()
        if dead is not None:e.red[dead].alive=False
        info=step(e);expected=np.full(5,-2.)
        if dead is not None:expected[dead]=0.
        np.testing.assert_array_equal(info['team_casualty_rewards'],expected)
        assert info['individual_event_rewards'][0]==-10 and info['event_rewards'][0]==-12
        assert info['red_step_attack_kills']==0 and info['blue_step_attack_kills']==1

@pytest.mark.parametrize('cause',['horizontal','ground','ceiling'])
def test_one_noncombat_loss(cause):
    e=combat()
    e.red[0]=(state(4999) if cause=='horizontal' else state(-350,z=-.1,theta=-np.pi/3)
              if cause=='ground' else state(-350,z=-5999,theta=np.pi/3))
    info=step(e)
    np.testing.assert_array_equal(info['team_casualty_rewards'],np.full(5,-2.))
    assert info['individual_event_rewards'][0]==-10 and info['event_rewards'][0]==-12
    assert info['red_losses']==1
    assert info['red_ceiling_losses']==int(cause=='ceiling')
    assert info['red_ground_losses']==int(cause=='ground')
    assert info['red_boundary_exits']==int(cause!='ground')

def test_two_losses_and_unique_indices():
    e=combat();e.red[1]=state(4999);info=step(e)
    np.testing.assert_array_equal(info['team_casualty_rewards'],np.full(5,-4.))
    np.testing.assert_array_equal(info['individual_event_rewards'][:2],[-10,-10])
    # Duplicate cause labels cannot count one aircraft twice, including ceiling/boundary overlap.
    _,diagnostics=e.event_reward_extension(np.zeros(5),np.ones(5),[0,0],[0],{0:[0]})
    np.testing.assert_array_equal(diagnostics['team_casualty'],np.full(5,-2.))

def test_prior_dead_slot_never_accumulates_later_casualties():
    e=combat();first=step(e);assert not e.red[0].alive
    prior=e.episode_reward_components['team_casualty'][0]
    e.red[1]=state(4999);second=step(e)
    assert second['team_casualty_rewards'][0]==0 and e.episode_reward_components['team_casualty'][0]==prior
    np.testing.assert_array_equal(second['team_casualty_rewards'][1:],np.full(4,-2.))

def test_kill_no_ally_bonus_and_shared_credit():
    for shared in (False,True):
        e=combat();e.red[0].psi=0.;e.blue[0].psi=0.
        if shared:e.red[1]=state(-400,100)
        info=step(e)
        np.testing.assert_array_equal(info['team_casualty_rewards'],np.zeros(5))
        np.testing.assert_array_equal(info['individual_event_rewards'],[5,5,0,0,0] if shared else [10,0,0,0,0])

def test_simultaneous_kill_and_own_death():
    e=combat();e.red[0].psi=0.;info=step(e)
    assert info['individual_event_rewards'][0]==0
    assert info['team_casualty_rewards'][0]==-2 and info['event_rewards'][0]==-2
    assert info['red_step_attack_kills']==info['blue_step_attack_kills']==1

@pytest.mark.parametrize('red,blue,value,reason',[(2,1,20,'red_win_timeout_survivors'),
    (1,2,-20,'blue_win_timeout_survivors'),(1,1,0,'draw_timeout_equal_survivors')])
def test_timeout_outcome_fixed_slots_and_terminal_pbrs(red,blue,value,reason):
    e=combat();e.max_steps=1
    e.red=[state(-750,i*500,alive=i<red) for i in range(5)]
    e.blue=[state(750,i*500,psi=np.pi,alive=i<blue) for i in range(5)]
    info=step(e);assert info['termination_reason']==reason
    np.testing.assert_array_equal(info['outcome_rewards'],np.full(5,value))
    assert np.any(info['phi_next_actual']!=0)
    np.testing.assert_array_equal(info['phi_next_for_shaping'],np.zeros(5))
    np.testing.assert_array_equal(info['adv_rewards'],-info['phi_current'])

@pytest.mark.parametrize('reason,value',[('red_win_elimination',20),('blue_win_elimination',-20),('draw_mutual_destruction',0)])
def test_elimination_outcome(reason,value):
    e=combat();e.red=[state(3500,alive=False) for _ in range(5)];e.blue=[state(3500,alive=False) for _ in range(5)]
    e.red[0]=state(3500 if reason=='red_win_elimination' else 4999)
    e.blue[0]=state(3500 if reason=='blue_win_elimination' else 4999)
    info=step(e);assert info['termination_reason']==reason
    np.testing.assert_array_equal(info['outcome_rewards'],np.full(5,value))
    np.testing.assert_array_equal(info['phi_next_for_shaping'],np.zeros(5))

def test_total_components_aliases_episode_diagnostics_and_aggregation():
    e=combat();obs,reward,_,_,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
    np.testing.assert_array_equal(info['event_rewards'],info['individual_event_rewards']+info['team_casualty_rewards'])
    np.testing.assert_array_equal(reward,sum((info[f'{n}_rewards'] for n in ('event','outcome','adv','safe')),np.zeros(5)).astype(np.float32))
    for alias,name in zip(('r1','r2','r3','r4'),('event','outcome','adv','safe')):
        np.testing.assert_array_equal(info[f'{alias}_rewards'],info[f'{name}_rewards'])
    assert not any(k.startswith('r5') for k in info)
    assert info['episode_individual_event_total']==-10 and info['episode_team_casualty_total']==-10
    result=aggregate_combat_records([dict(info,episode_return=float(reward.sum()),mean_agent_episode_return=float(reward.mean()))])
    assert result['average_episode_team_casualty_total']==-10 and result['average_episode_individual_event_total']==-10
    e.reset(31);assert e.episode_reward_components['team_casualty'].sum()==0

@pytest.mark.parametrize('seed',[1,31,32000000,33000000])
@pytest.mark.parametrize('mode',['ZERO','MIRROR'])
def test_complete_paired_reward_only_isolation(seed,mode):
    row=paired_rollout(seed,mode);assert row['physics_info_rng_mismatches']==0

@pytest.mark.parametrize('key,value',[('team_casualty_penalty',-1),('team_casualty_penalty',True),('win_reward',2),('lose_penalty',-2),('distance_weight',1)])
def test_frozen_reward_schema(key,value):
    c=cfg();c['reward'][key]=value
    with pytest.raises(ValueError):validate_config(c)

def test_only_v33_accepts_casualty_field():
    for version in ('30','31','32'):
        c=cfg(version);c['reward']['team_casualty_penalty']=-2.
        with pytest.raises(ValueError):validate_config(c)

@pytest.mark.parametrize('name',['mappo','rmappo','ea-mappo','stea-mappo'])
def test_new_diagnostics_in_training_writer_and_summary(tmp_path,name):
    from types import SimpleNamespace
    from algorithm.mappo.runner import MAPPOTrainingRunner
    from algorithm.rmappo.runner import RMAPPOTrainingRunner
    from algorithm.ea_mappo.runner import EAMAPPOTrainingRunner
    from algorithm.stea_mappo.runner import STEAMAPPOTrainingRunner
    cls={'mappo':MAPPOTrainingRunner,'rmappo':RMAPPOTrainingRunner,'ea-mappo':EAMAPPOTrainingRunner,'stea-mappo':STEAMAPPOTrainingRunner}[name]
    e=combat();_,reward,_,_,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
    row=dict(info,episode_return=float(reward.sum()),team_episode_return=float(reward.sum()),mean_agent_episode_return=float(reward.mean()))
    runner=object.__new__(cls);runner.env_config=cfg();runner.output_dir=tmp_path
    runner.trainer=SimpleNamespace(sampled_steps=1,vector_steps=1,ppo_update_count=0,actor_update_count=0,critic_update_count=0)
    runner.completed_records=[row];runner.best_evaluation=None;runner.evaluation_history=[];runner.last_metrics={};runner.startup_summary=lambda:{}
    runner._write_step_metrics(SimpleNamespace(rewards=reward[None,:],infos=[info]),[row])
    for record in (runner.summary(),json.loads((tmp_path/'training_metrics.jsonl').read_text())):
        assert record['average_episode_team_casualty_total']==-10 and record['average_episode_individual_event_total']==-10

@pytest.mark.parametrize('algorithm',['madsac','maddpg'])
def test_offpolicy_v33_cuda_dimensions_checkpoint_and_eval(tmp_path,algorithm):
    import torch
    from algorithm.common.protocol import config_sha256
    assert torch.cuda.is_available()
    e=cfg()
    c=yaml.safe_load((ROOT/f'configs/{"madsac" if algorithm=="madsac" else "maddpg_5v5"}.yaml').read_text())
    c['network'].update(observation_dim=66,action_dim=3,num_agents=5)
    n,t=c['network'],c['training'];extra=dict(environment_version='3.3',
        environment_config_sha256=config_sha256(e),algorithm_config_sha256=config_sha256(c),
        observation_dim=66,action_dim=3,num_agents=5,training_seed=t['seed'])
    path=tmp_path/f'{algorithm}.pt'
    if algorithm=='madsac':
        from algorithm.madsac.protocol import validate_madsac_config,validate_madsac_checkpoint
        from algorithm.madsac.factory import build_madsac_trainer
        from algorithm.madsac.evaluation import evaluate_madsac_episode
        validate_madsac_config(e,c);trainer=build_madsac_trainer(c,'cuda')
        extra['network_architecture']=dict(hidden_dim=n['actor_hidden_layers'][0],attention_heads=n['attention_heads'])
        trainer.save(path,extra);saved=torch.load(path,map_location='cpu',weights_only=False)
        validate_madsac_checkpoint(saved,e,c);trainer.load_for_evaluation(path)
        record=evaluate_madsac_episode(trainer,e,38001000,mode='deterministic')
        assert 'episode_team_casualty_total' in record and np.isfinite(record['episode_return'])
        saved['extra']['environment_version']='3.2'
        with pytest.raises(RuntimeError,match='environment_version'):validate_madsac_checkpoint(saved,e,c)
    else:
        from algorithm.maddpg.protocol import validate_config,validate_checkpoint
        from algorithm.maddpg.factory import build_maddpg_trainer
        from algorithm.maddpg.evaluation import evaluate
        validate_config(e,c);trainer=build_maddpg_trainer(c,'cuda')
        extra.update(gamma=t['gamma'],tau=t['tau'],training_num_envs=t['num_train_envs'],
            training_total_sampled_steps=t['total_sampled_steps'],training_smoke=False,
            network_architecture=trainer.network_architecture,parameter_counts=trainer.parameter_counts(),ou_noise_state=None)
        trainer.save(path,extra);saved=torch.load(path,map_location='cpu',weights_only=False)
        validate_checkpoint(saved,e,c);trainer.load(path,e,c)
        record=evaluate(trainer,e,[38001000]);assert 'average_episode_team_casualty_total' in record
        saved['extra']['environment_version']='3.2'
        with pytest.raises(RuntimeError,match='environment_version'):validate_checkpoint(saved,e,c)
