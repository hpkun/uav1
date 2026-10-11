"""Rollout logging granularity, mixed readers, resume and exact CUDA equivalence."""
import copy,csv,json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import yaml
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.rmappo.runner import RMAPPOTrainingRunner
from algorithm.stea_mappo.runner import STEAMAPPOTrainingRunner
from algorithm.ea_mappo.runner import EAMAPPOTrainingRunner
from algorithm.common.training_log import read_training_records,training_record_kind
from env.combat_env import MultiUAVCombatEnv

ROOT=Path(__file__).resolve().parents[1]
CLASSES=[MAPPOTrainingRunner,RMAPPOTrainingRunner,STEAMAPPOTrainingRunner,EAMAPPOTrainingRunner]
def bare(tmp_path,cls=MAPPOTrainingRunner):
    r=object.__new__(cls);r.output_dir=tmp_path;r.last_metrics=dict(actor_loss=1.,value_loss=2.,entropy=3.,approx_kl=.01,clip_fraction=.1,policy_log_std_mean_psi=-.5)
    r.trainer=SimpleNamespace(sampled_steps=15,ppo_update_count=1)
    r._begin_rollout_metrics();return r
def info():
    e=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v37.yaml');e.reset(31)
    return e.step(np.zeros((5,3)))[4]
def result(n,value):
    infos=[copy.deepcopy(info()) for _ in range(n)]
    for d in infos:
        for j in range(1,5):d[f'r{j}_rewards']=np.full(5,value*j)
        for side in ('red','blue'):
            d[f'{side}_fire_window_pairs']=value
            for event in ('fire_attempts','weapon_hits'):d[f'{side}_step_{event}']=value
    return SimpleNamespace(rewards=np.full((n,5),value),infos=infos)
def episode(value,win=True):
    d=info();d.update(team_episode_return=value,episode_return=value,mean_agent_episode_return=value/5,
        red_success=win,blue_win=not win,draw=False,episode_length=value)
    for j in range(1,5):d[f'episode_r{j}_total']=j*value
    return d

@pytest.mark.parametrize('cls',CLASSES)
def test_one_weighted_summary_after_whole_rollout(tmp_path,cls):
    r=bare(tmp_path,cls)
    r._accumulate_rollout_metrics(result(1,1),[episode(10)])
    r._accumulate_rollout_metrics(result(2,3),[episode(30,False)])
    assert not (tmp_path/'training_metrics.jsonl').exists()
    r._write_rollout_metrics()
    rows=list(read_training_records(tmp_path/'training_metrics.jsonl'));assert len(rows)==1
    d=rows[0];assert d['record_type']=='rollout_summary' and d['sampled_steps']==15 and d['rollout_update']==1
    assert d['rollout_vector_steps']==2 and d['rollout_sampled_steps']==3 and d['completed_episodes_in_rollout']==2
    assert d['mean_step_reward']==pytest.approx(7/3)
    for j in range(1,5):
        assert d[f'mean_r{j}_reward']==pytest.approx(j*7/3)
        assert d[f'episode_r{j}_total']==j*20
    assert d['red_step_weapon_hits']==pytest.approx(7/3)
    assert d['team_episode_return']==20 and d['mean_agent_episode_return']==4 and d['win_rate']==.5
    assert d['evaluation_episodes']==2 and d['episode_length']==20
    assert 'policy_log_std_mean_psi' not in d
    assert all(v is None or isinstance(v,(str,int,float,bool)) for v in d.values())

def test_no_episodes_null_and_theoretical_count(tmp_path):
    r=bare(tmp_path);r._accumulate_rollout_metrics(result(1,0),[]);d=r._finalize_rollout_metrics()
    for key in ('team_episode_return','win_rate','timeout_rate','episode_length','episode_r1_total'):assert d[key] is None
    assert d['completed_episodes_in_rollout']==0 and d['mean_step_reward']==0
    assert math.ceil(3000000/(16*256))==733

def test_accumulation_and_serialization_do_not_consume_rng(tmp_path):
    batch=result(2,1);completed=[episode(10)]
    before=(copy.deepcopy(np.random.get_state()),torch.get_rng_state().clone())
    r=bare(tmp_path);r._accumulate_rollout_metrics(batch,completed);r._write_rollout_metrics()
    equal(before,(np.random.get_state(),torch.get_rng_state()))

def test_mixed_legacy_compact_diagnostic_and_rollback(tmp_path):
    from tools.diagnose_stea_representation import analyze_logs
    from algorithm.train_mappo import prepare_resume_rollback
    updates=[dict(sampled_steps=s,actor_loss=float(s),approx_kl=s*.001) for s in (2,4,6)]
    completed=dict(evaluation_episodes=1,average_return=1.,win_rate=1.,average_red_loss=0.,average_blue_loss=5.,average_episode_length=10.)
    rows=[dict(sampled_steps=2,**completed),dict(record_type='ppo_update',**updates[0]),
          dict(record_type='rollout_summary',sampled_steps=4,**completed),
          dict(record_type='rollout_summary',sampled_steps=6,**completed)]
    for name,data in [('training_metrics',rows),('optimization_metrics',updates)]:
        (tmp_path/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data))
    (tmp_path/'evaluation_history.csv').write_text('sampled_steps,win_rate\n4,1\n6,0\n')
    assert [training_record_kind(r) for r in read_training_records(tmp_path/'training_metrics.jsonl')]==['legacy_step','ppo_update','rollout_summary','rollout_summary']
    assert analyze_logs(tmp_path,1)['stability']['updates']==3
    rows[1]['approx_kl']=-999.
    (tmp_path/'training_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(RuntimeError,match='duplicate log differs'):analyze_logs(tmp_path,1)
    rows[1]['approx_kl']=.002
    (tmp_path/'training_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    checkpoint=tmp_path/'checkpoint_4.pt';torch.save(dict(sampled_steps=4),checkpoint)
    prepare_resume_rollback(tmp_path,checkpoint,4)
    assert all(r['sampled_steps']<=4 for r in read_training_records(tmp_path/'training_metrics.jsonl'))
    assert [r['sampled_steps'] for r in read_training_records(tmp_path/'optimization_metrics.jsonl')]==[2,4]
    with (tmp_path/'evaluation_history.csv').open() as f:assert len(list(csv.DictReader(f)))==1
    # A wholly compact run has no duplicate requirement.
    compact=[r for r in read_training_records(tmp_path/'training_metrics.jsonl') if training_record_kind(r)!='ppo_update']
    (tmp_path/'training_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in compact))
    assert analyze_logs(tmp_path,1)['training_optimization_logs_consistent']

def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

@pytest.mark.parametrize('cls,stem',[(MAPPOTrainingRunner,'mappo'),(RMAPPOTrainingRunner,'rmappo'),(STEAMAPPOTrainingRunner,'stea_mappo')])
def test_compact_vs_noop_writer_exact_real_cuda_update(tmp_path,cls,stem):
    assert torch.cuda.is_available(),'CUDA is mandatory'
    env=yaml.safe_load((ROOT/'configs/combat_environment_v37.yaml').read_text())
    config=yaml.safe_load((ROOT/f'configs/{stem}_5v5_v37.yaml').read_text())
    snapshots=[]
    for logging in (True,False):
        runner=cls(env,config,num_envs=2,total_sampled_steps=16,device='cuda',seed=373,output_dir=tmp_path/str(logging),smoke=True)
        try:
            batch=runner.collect_rollout(8)
            runner.last_metrics=runner.trainer.update(batch)
            if not logging:runner._write_rollout_metrics=lambda:None
            runner._write_rollout_metrics()
            trainer=runner.trainer
            snapshots.append(dict(batch={k:getattr(batch,k) for k in batch.__dataclass_fields__},
                actor=copy.deepcopy(trainer.actor.state_dict()),critic=copy.deepcopy(trainer.critic.state_dict()),
                actor_optimizer=copy.deepcopy(trainer.actor_optimizer.state_dict()),critic_optimizer=copy.deepcopy(trainer.critic_optimizer.state_dict()),
                steps=trainer.sampled_steps,metrics=runner.last_metrics.copy(),
                numpy_rng=copy.deepcopy(np.random.get_state()),torch_rng=torch.get_rng_state().clone(),
                cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()],
                observations=runner.observations.copy()))
        finally:runner.vector.close()
    equal(snapshots[0],snapshots[1])
    assert len(list(read_training_records(tmp_path/'True/training_metrics.jsonl')))==1
    assert not (tmp_path/'False/training_metrics.jsonl').exists()

def test_new_format_checkpoint_resume_and_console_semantics(tmp_path):
    assert torch.cuda.is_available()
    from algorithm.train_mappo import prepare_resume_rollback
    env=yaml.safe_load((ROOT/'configs/combat_environment_v37.yaml').read_text())
    config=yaml.safe_load((ROOT/'configs/mappo_5v5_v37.yaml').read_text())
    first=MAPPOTrainingRunner(env,config,num_envs=2,total_sampled_steps=8,device='cuda',seed=373,output_dir=tmp_path,smoke=True)
    summary=first.run();assert summary['rollout_updates']==1
    checkpoint=tmp_path/'latest.pt'
    # Emulate records written after this checkpoint; rollback is tested on isolated data only.
    for filename in ('training_metrics.jsonl','optimization_metrics.jsonl'):
        with (tmp_path/filename).open('a') as f:f.write(json.dumps(dict(record_type='rollout_summary',sampled_steps=100))+'\n')
    (tmp_path/'evaluation_history.csv').write_text('sampled_steps,win_rate\n8,0\n100,1\n')
    prepare_resume_rollback(tmp_path,checkpoint,8)
    resumed=MAPPOTrainingRunner(env,config,num_envs=2,total_sampled_steps=16,device='cuda',seed=373,output_dir=tmp_path,smoke=True)
    resumed.resume(checkpoint);summary=resumed.run()
    training=list(read_training_records(tmp_path/'training_metrics.jsonl'))
    optimization=list(read_training_records(tmp_path/'optimization_metrics.jsonl'))
    assert [r['sampled_steps'] for r in training]==[8,16]
    assert len(training)==len(optimization)==summary['rollout_updates']==2
    assert all(r['record_type']=='rollout_summary' for r in training)
    # Logging still uses the most recent completed episodes, not the rollout accumulator.
    resumed.recent_episode_window=2;resumed.completed_records=[episode(99),episode(10),episode(30,False)]
    assert resumed.recent_episode_metrics()['return']==20 and resumed.recent_episode_metrics()['win']==.5
    assert '| return=20.00 | win=0.50' in resumed.train_log_line()
