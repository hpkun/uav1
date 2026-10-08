"""Focused checks for the new Gaussian protocol and epoch-level KL stopping."""
import copy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.mappo.trainer import RolloutBatch
from algorithm.stea_mappo.factory import build_stea_mappo_trainer
from algorithm.stea_mappo.trainer import RecurrentRolloutBatch
from algorithm.mappo.networks import SharedMAPPOActor
from algorithm.stea_mappo.networks import SpatioTemporalEntityAttentionActor
from algorithm.common.policy_protocol import validate_policy_protocol
from tools.evaluate_policy_modes import PolicyExecution

ROOT=Path(__file__).resolve().parents[1]

def config(name):
    return yaml.safe_load((ROOT/f'configs/{name}_5v5.yaml').read_text())

def trainer_for(name,zero_lr=False,target=.015):
    assert torch.cuda.is_available(),'CUDA mandatory'
    cfg=config(name);cfg['training'].update(ppo_epochs=4,minibatch_size=32,target_kl=target)
    if zero_lr:cfg['training'].update(actor_learning_rate=0.,critic_learning_rate=0.)
    return (build_mappo_trainer(cfg,'cuda') if name=='mappo' else build_stea_mappo_trainer(cfg,'cuda')),cfg

def rollout_for(trainer,recurrent,t=32,e=2):
    rng=np.random.default_rng(15);obs=rng.normal(size=(t,e,5,65)).astype(np.float32)
    alive=np.ones((t,e,5),np.float32);alive[8:11,0,1]=0
    starts=np.zeros((t,e),np.float32);starts[0]=1;starts[16,0]=1
    hidden=np.zeros((e,5,128),np.float32);actions=[];raw=[];logs=[];states=[]
    for step in range(t):
        if recurrent:
            states.append(hidden.copy());a,r,l,hidden=trainer.act(obs[step],alive[step],hidden,starts[step])
            if step+1<t:
                hidden*=alive[step+1,...,None];hidden[starts[step+1]>.5]=0
        else:a,r,l=trainer.act(obs[step],alive[step],return_policy_data=True)
        actions.append(a);raw.append(r);logs.append(l)
    dones=np.zeros((t,e),np.float32);dones[:-1]=starts[1:]
    values=dict(observations=obs,actions=np.stack(actions),raw_actions=np.stack(raw),old_log_probs=np.stack(logs),
                rewards=rng.normal(size=(t,e,5)).astype(np.float32),dones=dones,alive_masks=alive,
                next_observations=obs+.01,next_alive_masks=np.concatenate([alive[1:],alive[-1:]]))
    if recurrent:values.update(actor_hidden_states=np.stack(states),episode_starts=starts)
    return (RecurrentRolloutBatch if recurrent else RolloutBatch)(**values)

@pytest.mark.parametrize('name',['mappo','stea_mappo'])
def test_independent_std_initialization_broadcast_and_three_dimensions(name):
    trainer,_=trainer_for(name);actor=trainer.actor
    assert actor.log_std_parameter.shape==(3,) and not hasattr(actor,'log_std')
    assert torch.equal(actor.log_std_parameter,torch.full((3,),-.5,device='cuda'))
    torch.testing.assert_close(actor.mean.weight@actor.mean.weight.T,torch.eye(3,device='cuda')*.0001,atol=1e-8,rtol=1e-5)
    assert not actor.mean.bias.any()
    obs=torch.randn(2,5,65,device='cuda')
    def distribution(x):
        if name=='mappo':return actor.distribution(x)
        return actor.distribution_step(x,torch.randn(2,5,128,device='cuda'),torch.ones(2,5,device='cuda'),torch.zeros(2,device='cuda'))[0]
    first,second=distribution(obs),distribution(obs+1.)
    assert not torch.equal(first.mean,second.mean) and torch.equal(first.scale,second.scale)
    assert torch.equal(first.scale,actor.log_std_parameter.clamp(-5.,.5).exp().expand(2,5,3))
    with torch.no_grad():actor.log_std_parameter.copy_(torch.tensor([-1.,-.5,0.],device='cuda'))
    torch.testing.assert_close(distribution(obs).scale,torch.tensor([-1.,-.5,0.],device='cuda').exp().expand(2,5,3),rtol=0,atol=0)

@pytest.mark.parametrize('name',['mappo','stea_mappo'])
def test_update_gradients_and_ratio_audit(name):
    trainer,_=trainer_for(name);rollout=rollout_for(trainer,name!='mappo')
    metrics=trainer.update(rollout)
    for parameter in (trainer.actor.log_std_parameter,trainer.actor.mean.weight,trainer.actor.mean.bias):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert np.isfinite(list(metrics.values())).all()
    assert 1<=metrics['effective_ppo_epochs']<=4
    assert metrics['policy_sigma_heading']==pytest.approx(np.exp(metrics['policy_log_std_heading']))
    if name!='mappo':assert metrics['pre_update_ratio_max_abs_error']<1e-4

@pytest.mark.parametrize('name',['mappo','stea_mappo'])
@pytest.mark.parametrize('high_kl',[False,True])
def test_kl_stops_only_after_complete_epoch(name,high_kl,monkeypatch):
    trainer,_=trainer_for(name,zero_lr=True);rollout=rollout_for(trainer,name!='mappo')
    if high_kl:
        if name=='mappo':
            original=trainer.actor.evaluate_actions
            def shifted(*args,**kwargs):
                log,entropy=original(*args,**kwargs);return log+1.,entropy
            monkeypatch.setattr(trainer.actor,'evaluate_actions',shifted)
        else:
            original=trainer._evaluate
            def shifted(batch,entropy=True):
                values=original(batch,entropy)
                return (values[0]+1.,*values[1:]) if entropy else values
            monkeypatch.setattr(trainer,'_evaluate',shifted)
    metrics=trainer.update(rollout)
    assert metrics['effective_ppo_epochs']==(1 if high_kl else 4)
    assert metrics['kl_early_stop']==float(high_kl)
    assert trainer.actor_update_count==(2 if high_kl else 8)  # both minibatches in each epoch
    assert trainer.critic_update_count==trainer.actor_update_count
    assert (metrics['last_epoch_mean_kl']>.015)==high_kl
    assert not trainer._epoch_exceeds_target([{'approx_kl':.03},{'approx_kl':0.}])
    assert trainer._epoch_exceeds_target([{'approx_kl':.031},{'approx_kl':0.}])

@pytest.mark.parametrize('name',['mappo','stea_mappo'])
def test_old_new_protocol_rejection_and_metadata(name):
    trainer,cfg=trainer_for(name)
    old=copy.deepcopy(cfg)
    for field in ('policy_std_mode','log_std_init','mean_head_init_gain'):old['implementation'].pop(field)
    old['implementation']['log_std_max']=2.;old['training'].pop('target_kl')
    legacy=(build_mappo_trainer(old,'cuda') if name=='mappo' else build_stea_mappo_trainer(old,'cuda'))
    newstate=trainer.checkpoint_state();oldstate=legacy.checkpoint_state()
    for field in ('policy_std_mode','log_std_init','mean_head_init_gain','target_kl'):assert field in newstate['extra']
    validate_policy_protocol(newstate,cfg)
    for state,target in ((newstate,old),(oldstate,cfg)):
        with pytest.raises(RuntimeError,match='policy_std_mode'):validate_policy_protocol(state,target)
    for field in ('policy_std_mode','log_std_init','mean_head_init_gain','target_kl'):oldstate['extra'].pop(field)
    validate_policy_protocol(oldstate,old)
    if name!='mappo':
        assert 'policy_std_mode' in trainer.network_architecture
        assert 'policy_std_mode' not in legacy.network_architecture
    execution=PolicyExecution(trainer.actor,name!='mappo','cuda')
    try:
        execution.reset(5)
        _,_,dist,logstd=execution.decide(np.zeros((5,65),np.float32),np.ones(5,np.float32),'deterministic')
        torch.testing.assert_close(dist.scale,logstd.exp())
    finally:execution.close()

@pytest.mark.parametrize('actor_type',[SharedMAPPOActor,SpatioTemporalEntityAttentionActor])
def test_legacy_initialization_unchanged(actor_type):
    torch.manual_seed(31);implicit=actor_type()
    torch.manual_seed(31);explicit=actor_type(policy_std_mode='state_dependent')
    assert implicit.state_dict().keys()==explicit.state_dict().keys()
    assert all(torch.equal(value,explicit.state_dict()[name]) for name,value in implicit.state_dict().items())
