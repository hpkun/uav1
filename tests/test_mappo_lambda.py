"""Lambda-only configuration, unchanged GAE/update math and passive diagnostics."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
from algorithm.mappo.trainer import compute_gae,raw_gae_diagnostics,RolloutBatch
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
from algorithm.common.protocol import config_sha256
from algorithm.common.evaluator import noncombat_losses,aggregate_combat_records
from algorithm.train_mappo import ensure_fresh_output_directory
from tools.diagnose_mappo_lambda import configs,time_to_event

def original_gae(rewards,values,next_values,dones,alive,next_alive,gamma,lam):
    # Frozen recurrence from the pre-diagnostics implementation, including operation order.
    advantages=torch.zeros_like(rewards);gae=torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        continuation=(1.0-dones[step].unsqueeze(-1))*next_alive[step]
        delta=rewards[step]+gamma*continuation*next_values[step]-values[step]
        gae=delta+gamma*lam*continuation*gae
        advantages[step]=gae*alive[step]
    return advantages,(advantages+values)*alive

@pytest.mark.parametrize('lam',[.95,.996195])
def test_gae_bitwise_reference_death_terminal_and_normalization(lam):
    rng=np.random.default_rng(8);shape=(64,2,4)
    r,v,n=(torch.tensor(rng.normal(size=shape),dtype=torch.float32) for _ in range(3))
    alive=torch.ones(shape);alive[22:32,0,1]=0;next_alive=alive.clone();next_alive[21,0,1]=0
    done=torch.zeros(shape[:2]);done[31,0]=1;done[-1]=1
    a,target=compute_gae(r,v,n,done,alive,next_alive,.99,lam)
    aa,tt=original_gae(r,v,n,done,alive,next_alive,.99,lam)
    assert torch.equal(a,aa) and torch.equal(target,tt)
    before=a.clone();metrics=raw_gae_diagnostics(a,target,v,alive)
    assert torch.equal(a,before) and all(np.isfinite(list(metrics.values())))
    live=a[alive>.5];normalized=(a-live.mean())/live.std(unbiased=False).clamp_min(1e-8)*alive
    expected=(aa-aa[alive>.5].mean())/aa[alive>.5].std(unbiased=False).clamp_min(1e-8)*alive
    assert torch.equal(normalized,expected)
    assert normalized[alive>.5].std(unbiased=False)==pytest.approx(1.,abs=1e-6)
    assert (a[alive==0]==0).all() and (target[alive==0]==0).all()

def test_five_second_trace_and_impulse():
    weight=(.99*.996195)**50
    assert weight==pytest.approx(.5,abs=1e-5)
    r=torch.zeros(51,1,1,dtype=torch.float64);r[-1]=10
    ones=torch.ones_like(r);zeros=torch.zeros_like(r);done=torch.zeros(51,1);done[-1]=1
    a,_=compute_gae(r,zeros,zeros,done,ones,ones,.99,.996195)
    assert a[0,0,0].item()==pytest.approx(10*weight,rel=1e-12)

def test_raw_diagnostics_no_gradient_rng_mutation_or_dead_contamination():
    a=torch.tensor([1.,-3.,10000.],requires_grad=True);target=a+2;value=a-2;alive=torch.tensor([1.,1.,0.])
    old=a.clone();rng=torch.get_rng_state().clone();m=raw_gae_diagnostics(a,target,value,alive)
    assert all(isinstance(v,float) for v in m.values())
    assert a.grad is None and torch.equal(a,old) and torch.equal(rng,torch.get_rng_state())
    assert m['raw_advantage_mean']==-1 and m['raw_advantage_std']==2
    assert m['raw_advantage_abs_mean']==2 and m['raw_advantage_abs_p50']==2
    assert m['return_target_mean']==1 and m['old_value_mean']==-3
    assert all(v==0 for v in raw_gae_diagnostics(a,target,value,alive*0).values())

def test_configuration_only_lambda_changes():
    _,a,b=configs();other=copy.deepcopy(b);other['training']['gae_lambda']=.95
    assert a==other and b['training']['gae_lambda']==.996195
    assert a['training']['total_sampled_steps']==b['training']['total_sampled_steps']==3000000

@pytest.mark.parametrize('lam',[.95,.996195])
def test_checkpoint_lambda_identity_and_mismatch_rejection(tmp_path,lam):
    assert torch.cuda.is_available(),'CUDA mandatory'
    env,a,b=configs();cfg=a if lam==.95 else b;wrong=b if lam==.95 else a
    trainer=build_mappo_trainer(cfg,'cuda')
    path=tmp_path/'checkpoint.pt';trainer.save(path,dict(environment_version='3.38',
        observation_dim=45,action_dim=3,num_agents=4,environment_config_sha256=config_sha256(env),
        algorithm_config_sha256=config_sha256(cfg)))
    state=torch.load(path,map_location='cuda',weights_only=False)
    assert state['extra']['training_gae_lambda']==lam
    for validator in [validate_checkpoint_for_evaluation,validate_checkpoint_for_resume]:
        validator(state,env,cfg)
        with pytest.raises(RuntimeError,match='training_gae_lambda|algorithm_config_sha256'):validator(state,env,wrong)
        inconsistent=copy.deepcopy(state);inconsistent['extra']['training_gae_lambda']=wrong['training']['gae_lambda']
        with pytest.raises(RuntimeError,match='training_gae_lambda'):validator(inconsistent,env,cfg)
    with pytest.raises(RuntimeError,match='training_gae_lambda'):build_mappo_trainer(wrong,'cuda').load(path)
    # Previous result checkpoints without the new additive field still use the config hash.
    state['extra'].pop('training_gae_lambda')
    with pytest.raises(RuntimeError,match='algorithm_config_sha256'):validate_checkpoint_for_evaluation(state,env,wrong)
    torch.save(state,path)
    with pytest.raises(RuntimeError,match='algorithm_config_sha256'):build_mappo_trainer(wrong,'cuda').load(path)

def test_diagnostics_do_not_change_update_parameters_or_existing_losses(monkeypatch):
    assert torch.cuda.is_available(),'CUDA mandatory'
    _,cfg,_=configs();cfg['training'].update(ppo_epochs=2,minibatch_size=32)
    a=build_mappo_trainer(cfg,'cuda');b=build_mappo_trainer(cfg,'cuda')
    rng=np.random.default_rng(15);obs=rng.normal(size=(32,2,4,45)).astype(np.float32)
    alive=np.ones((32,2,4),np.float32);act,raw,log=a.act(obs,alive,return_policy_data=True)
    batch=RolloutBatch(obs,act,raw,log,rng.normal(size=alive.shape).astype(np.float32),np.zeros((32,2),np.float32),alive,obs+.01,alive)
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
    am=a.update(batch)
    monkeypatch.setattr('algorithm.mappo.trainer.raw_gae_diagnostics',lambda *args:{})
    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng);bm=b.update(batch)
    for model in ['actor','critic']:
        assert all(torch.equal(x,y) for x,y in zip(getattr(a,model).parameters(),getattr(b,model).parameters()))
    assert all(am[k]==v for k,v in bm.items())

def test_fresh_outputs_cannot_overwrite_control_or_treatment(tmp_path):
    a,b=tmp_path/'control',tmp_path/'treatment'
    for directory in [a,b]:
        ensure_fresh_output_directory(directory);(directory/'run_config.json').write_text('protected')
        with pytest.raises(RuntimeError,match='non-empty'):ensure_fresh_output_directory(directory)
        assert (directory/'run_config.json').read_text()=='protected'
    assert a!=b

def test_ceiling_is_boundary_subtype_not_double_counted():
    from env.factory import make_combat_environment
    env,_,_=configs();e=make_combat_environment(env);e.reset(1)
    e.red[0].z=-6001;e.red[1].z=1;e.red[2].x=6000
    e._resolve_noncombat_losses()
    counts=e.combat_counts['red'];record={f'red_{k}':v for k,v in counts.items()}
    assert counts['boundary_exits']==2 and counts['ground_losses']==1 and counts['ceiling_losses']==1
    assert noncombat_losses(record,'red')==3
    _,r,_,_,info=e.step(np.zeros((4,3)))
    record=dict(episode_return=float(r.sum()),mean_agent_episode_return=float(r.mean()),**info)
    metrics=aggregate_combat_records([record])
    assert metrics['red_ceiling_losses']==1
    assert metrics['average_red_noncombat_losses']==3

def test_time_to_event_does_not_cross_terminal_death_or_reset():
    event=np.zeros((6,1,1),bool);event[4]=True
    dones=np.zeros((6,1));dones[2]=1;alive=np.ones((6,1,1))
    result=time_to_event(event,dones,alive)
    assert np.isinf(result[:3]).all() and result[3,0,0]==1 and result[4,0,0]==0
