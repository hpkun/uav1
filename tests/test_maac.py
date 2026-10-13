"""Continuous paper MAAC protocol, Eq.(8) and replay contracts."""
from copy import deepcopy
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from torch import nn
from algorithm.maac.networks import GaussianLocalActor,CentralizedAttentionCritic
from algorithm.maac.trainer import MAACTrainer,score_function_loss
from algorithm.maac.replay import ReplayBatch,JointReplayBuffer
from algorithm.maac.protocol import validate_config_pair,validate_checkpoint
from algorithm.maac.protocol import validate_checkpoint_for_resume


def configs():
    return tuple(yaml.safe_load(Path(p).read_text()) for p in
        ('configs/combat_environment_v329.yaml','configs/maac_5v5_v329.yaml'))


def trainer(device='cpu',smoke=True):
    return MAACTrainer(*configs(),device=device,seed=4,smoke=smoke)


def batch(n=8):
    rng=np.random.default_rng(41)
    return ReplayBatch(rng.normal(size=(n,5,56)).astype('f'),rng.uniform(-1,1,(n,5,3)).astype('f'),
        rng.normal(size=(n,5)).astype('f'),rng.normal(size=(n,5,56)).astype('f'),
        np.zeros(n,'f'),np.ones((n,5),'f'),np.ones((n,5),'f'))


@pytest.mark.parametrize('shape',[(56,),(5,56),(3,5,56)])
def test_gaussian_actor_shape_range_logprob(shape):
    actor=GaussianLocalActor();o=torch.randn(shape);a,lp,u=actor.sample(o)
    assert a.shape==(*shape[:-1],3) and lp.shape==shape[:-1]
    assert torch.isfinite(lp).all() and (a.abs()<=1).all()
    assert not u.requires_grad and not a.requires_grad
    torch.testing.assert_close(lp,actor.log_prob(o,u))
    torch.testing.assert_close(actor(o),actor.sample(o,True)[0])


def test_shared_actor_identity_distribution_and_local_execution():
    t=trainer()
    for name in ('actors','target_actors','actor_optimizers'):assert not hasattr(t,name)
    authoritative=list(t.actor.parameters())
    optimized=[p for g in t.actor_optimizer.param_groups for p in g['params']]
    assert len(optimized)==len(authoritative) and set(map(id,optimized))==set(map(id,authoritative))
    assert len(set(map(id,optimized)))==len(optimized)
    o=torch.randn(2,5,56);o[:,4]=o[:,0]
    d=t.actor.distribution(o)
    torch.testing.assert_close(d.mean[:,0],d.mean[:,4],rtol=0,atol=0)
    torch.testing.assert_close(d.scale[:,0],d.scale[:,4],rtol=0,atol=0)
    torch.testing.assert_close(d.scale[:,0].log(),d.scale[:,4].log(),rtol=0,atol=0)
    a=t.act(o.numpy());other=o.numpy().copy();other[:,1:]=100
    np.testing.assert_array_equal(a[:,0],t.act(other)[:,0])
    assert not np.array_equal(a[:,0],a[:,1])
    assert not np.array_equal(a,t.act(o.numpy(),deterministic=False))


def test_tanh_logprob_extreme_raw_actions_finite():
    actor=GaussianLocalActor()
    lp=actor.log_prob(torch.zeros(4,56),torch.full((4,3),100.))
    assert torch.isfinite(lp).all()
    lp.sum().backward();assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)


@pytest.mark.parametrize('n',[1,4])
def test_critic_shapes_scaled_attention_excludes_self(n):
    c=CentralizedAttentionCritic()
    q,w=c(torch.randn(n,5,56),torch.randn(n,5,3),torch.ones(n,5),True)
    assert q.shape==(n,5,1) and w.shape==(n,4,5,5)
    assert (w.diagonal(dim1=-2,dim2=-1)==0).all()
    torch.testing.assert_close(w.sum(-1),torch.ones(n,4,5))


@pytest.mark.parametrize('bad',['obs','action','mask'])
def test_critic_invalid_shapes(bad):
    o=torch.zeros(2,5,56);a=torch.zeros(2,5,3);m=torch.ones(2,5)
    if bad=='obs':o=o[:,:,:52]
    elif bad=='action':a=a[:,:,:2]
    else:m=m[:,:4]
    with pytest.raises(ValueError):CentralizedAttentionCritic()(o,a,m)


def test_dead_mask_isolation_and_gradients():
    c=CentralizedAttentionCritic();o=torch.randn(3,5,56);a=torch.randn(3,5,3);m=torch.ones(3,5);m[:,2]=0
    q,w=c(o,a,m,True);other_o=o.clone();other_a=a.clone();other_o[:,2]=1e8;other_a[:,2]=-1e8
    torch.testing.assert_close(q,c(other_o,other_a,m),rtol=0,atol=0)
    assert (q[:,2]==0).all() and (w[:,:,:,2]==0).all() and (w[:,:,2,:]==0).all()
    a.requires_grad_();c(o,a,m).sum().backward();assert (a.grad[:,2]==0).all()


@pytest.mark.parametrize('living',[0,1])
def test_zero_context_no_other_survivor(living):
    c=CentralizedAttentionCritic();m=torch.zeros(2,5)
    if living:m[:,3]=1
    q,w=c(torch.randn(2,5,56),torch.randn(2,5,3),m,True)
    assert torch.isfinite(q).all() and torch.isfinite(w).all() and (w==0).all()
    if not living:assert (q==0).all()
    else:assert q[:,3].abs().sum()>0


def test_paper_shortcut_nonlinearity_and_agent_heads():
    c=CentralizedAttentionCritic()
    assert isinstance(c.value[-1],nn.LeakyReLU)
    assert len(c.encoders)==len(c.q_heads)==5 and c.q_heads[0][0].in_features==256
    seen=[];handle=c.q_heads[0].register_forward_pre_hook(lambda _,args:seen.append(args[0]))
    c(torch.randn(1,5,56),torch.randn(1,5,3),torch.ones(1,5));handle.remove()
    assert seen[0].shape==(1,256)


def test_target_initialization_freezing_and_polyak_online_weight():
    t=trainer()
    for online,target in [(t.actor,t.target_actor),(t.critic,t.target_critic)]:
        for a,b in zip(online.parameters(),target.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0);assert not b.requires_grad
    old=[p.clone() for p in t.target_critic.parameters()]
    with torch.no_grad():
        for p in t.critic.parameters():p.add_(1)
    t.update_targets()
    for a,b in zip(old,t.target_critic.parameters()):torch.testing.assert_close(b,a+.005)
    assert all(p.grad is None for p in t.target_critic.parameters())


def test_eq14_target_entropy_and_one_shared_target_actor():
    t=trainer();o=torch.randn(2,5,56);r=torch.randn(2,5);d=torch.tensor([1.,0.]);m=torch.ones(2,5);m[1,2]=0
    with patch.object(t.target_actor,'sample',wraps=t.target_actor.sample) as sampled:
        torch.manual_seed(91);y=t.td_targets(r,o,d,m)
        assert sampled.call_count==1 and sampled.call_args.args[0].shape==(2,5,56)
    with torch.no_grad():
        torch.manual_seed(91);a,logs,_=t.target_actor.sample(o);a=a*m[...,None]
        expected=r+t.gamma*(1-d[:,None])*m*(t.target_critic(o,a,m).squeeze(-1)-.01*logs)
    torch.testing.assert_close(y,expected)
    torch.testing.assert_close(y[0],r[0]);torch.testing.assert_close(y[1,2],r[1,2])
    assert not y.requires_grad


def test_td_uses_targets_not_online_parameters():
    t=trainer();o=torch.randn(2,5,56);r=torch.randn(2,5);d=torch.zeros(2);m=torch.ones(2,5)
    torch.manual_seed(2);a=t.td_targets(r,o,d,m)
    with torch.no_grad():
        for p in t.actor.parameters():p.add_(1)
        for p in t.critic.parameters():p.add_(1)
    torch.manual_seed(2);b=t.td_targets(r,o,d,m)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_eq8_coefficient_and_baseline_are_detached():
    logp=torch.tensor([-.2,-.5],requires_grad=True)
    q=torch.tensor([2.,3.],requires_grad=True);b=torch.tensor([.3,.7],requires_grad=True)
    loss=score_function_loss(logp,q,b,.01,torch.ones(2));loss.backward()
    torch.testing.assert_close(logp.grad,(.01*logp.detach()-(q.detach()-b.detach()))/2)
    assert q.grad is None and b.grad is None


@pytest.mark.parametrize('index',range(5))
def test_each_agent_score_updates_same_actor_without_action_path(index):
    t=trainer();o=torch.randn(3,5,56);raw=torch.randn(3,3,requires_grad=True)
    lp=t.actor.log_prob(o[:,index],raw)
    score_function_loss(lp,torch.ones(3),torch.zeros(3),.01,torch.ones(3)).backward()
    assert raw.grad is None
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in t.actor.parameters())
    assert all(torch.isfinite(p.grad).all() for p in t.actor.parameters() if p.grad is not None)
    assert all(p.grad is None for p in t.critic.parameters())


class LinearQ(nn.Module):
    def forward(self,o,a,m):
        return ((a[:,:,0]+a[:,:,0].sum(1,keepdim=True))*m).unsqueeze(-1)


class ZeroGaussian(nn.Module):
    def sample(self,o,deterministic=False):
        raw=torch.randn(*o.shape[:-1],3,device=o.device)
        return raw.tanh(),torch.zeros(o.shape[:-1],device=o.device),raw


def test_mc_baseline_fixed_others_independent_of_selected_own_action():
    t=trainer();t.critic=LinearQ()
    t.actor=ZeroGaussian()
    o=torch.zeros(2,5,56);a=torch.zeros(2,5,3);a[:,0,0]=.3;m=torch.ones(2,5)
    torch.manual_seed(91);b=t.counterfactual_baseline(o,a,m,2,samples=4096)
    assert not b.requires_grad
    # Compare with the exact finite-sample estimate for this linear toy Q,
    # rather than imposing a flaky tolerance on Monte Carlo sampling error.
    torch.manual_seed(91)
    expected=.3+2*torch.randn(4096,2,3)[:,:,0].tanh().mean(0)
    torch.testing.assert_close(b,expected)
    a[:,2]=.99
    torch.manual_seed(91);c=t.counterfactual_baseline(o,a,m,2,samples=4096)
    torch.testing.assert_close(b,c,rtol=0,atol=0)


def test_every_baseline_uses_same_authoritative_actor():
    t=trainer();o=torch.randn(2,5,56);a=torch.zeros(2,5,3);m=torch.ones(2,5)
    with patch.object(t.actor,'sample',wraps=t.actor.sample) as sample:
        for i in range(5):t.counterfactual_baseline(o,a,m,i)
    assert sample.call_count==5
    for i,call in enumerate(sample.call_args_list):
        torch.testing.assert_close(call.args[0],o[:,i][None].expand(t.baseline_samples,-1,-1))


@pytest.mark.parametrize('index',range(5))
def test_one_live_agent_has_finite_shared_update(index):
    t=trainer();b=batch();b.alive_masks.fill(0);b.alive_masks[:,index]=1
    before=[p.clone() for p in t.actor.parameters()]
    with patch.object(t.actor_optimizer,'step',wraps=t.actor_optimizer.step) as step:
        result=t.update_policies(b)
    assert step.call_count==1 and t.actor_updates==t.policy_updates==1
    assert np.isfinite(result['actor_grad_norm']) and result['actor_grad_norm']>0
    assert any(not torch.equal(p,q) for p,q in zip(before,t.actor.parameters()))


def test_all_contributions_precede_one_backward_and_optimizer_step():
    t=trainer();original=[p.clone() for p in t.actor.parameters()];grad_calls=[];samples=[]
    hook=next(t.actor.parameters()).register_hook(lambda g:grad_calls.append(g.clone()))
    sample_impl=t.actor.sample
    def observe_sample(*args,**kwargs):
        assert all(torch.equal(p,q) for p,q in zip(original,t.actor.parameters()))
        samples.append(args[0].shape)
        return sample_impl(*args,**kwargs)
    with patch.object(t.actor,'sample',side_effect=observe_sample),patch.object(t.actor_optimizer,'step',wraps=t.actor_optimizer.step) as step:
        t.update_policies(batch())
    hook.remove()
    assert samples[0]==(8,5,56) and len(samples)==6  # joint + five MC baselines
    assert len(grad_calls)==1 and step.call_count==1
    assert t.actor_updates==t.policy_updates==1


def test_shared_loss_contribution_order_has_no_sequential_update_artifact():
    a,b=trainer(),trainer();o=torch.randn(3,5,56);raw=torch.randn(3,5,3)
    q=torch.randn(3,5);base=torch.randn(3,5);alive=torch.ones(3,5)
    order=[4,2,0,3,1]
    loss_a=score_function_loss(a.actor.log_prob(o,raw),q,base,a.alpha,alive)
    lp=b.actor.log_prob(o,raw)
    loss_b=score_function_loss(lp[:,order],q[:,order],base[:,order],b.alpha,alive[:,order])
    loss_a.backward();loss_b.backward()
    for p,r in zip(a.actor.parameters(),b.actor.parameters()):torch.testing.assert_close(p.grad,r.grad,atol=1e-6,rtol=1e-5)
    a.actor_optimizer.step();b.actor_optimizer.step()
    for p,r in zip(a.actor.parameters(),b.actor.parameters()):torch.testing.assert_close(p,r,atol=1e-6,rtol=1e-5)


def test_update_actor_critic_finite_and_frozen_targets():
    t=trainer();actors=[p.clone() for p in t.actor.parameters()];critic=[p.clone() for p in t.critic.parameters()]
    result=t.update(batch())
    assert all(np.isfinite(v) for v in result.values())
    assert result['actor_grad_norm']>0 and result['critic_grad_norm']>0
    assert any(not torch.equal(a,b) for a,b in zip(actors,t.actor.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(critic,t.critic.parameters()))
    assert t.actor_updates==1 and t.policy_updates==t.critic_updates==t.target_updates==1
    assert all(p.grad is None for p in t.critic.parameters())
    assert all(p.requires_grad for p in t.critic.parameters())
    assert all(not p.requires_grad for p in t.target_actor.parameters())


def test_all_dead_skips_losses():
    t=trainer();b=batch();b.alive_masks.fill(0)
    assert t.update_critic(b)=={} and t.update_policies(b)=={}
    assert t.actor_updates==t.critic_updates==0


def test_replay_ring_wrap_and_lazy_allocation():
    replay=JointReplayBuffer(7,chunk_size=3)
    assert not replay.chunks
    for start,n in [(0,4),(4,8)]:
        b=batch(n);b.rewards[:]=np.arange(start,start+n)[:,None]
        replay.add_batch(**vars(b))
    assert len(replay)==7 and len(replay.chunks)==3
    sampled=replay.sample(7)
    assert sampled.observations.shape==(7,5,56) and sampled.actions.shape==(7,5,3)
    assert sorted(sampled.rewards[:,0].tolist())==list(range(5,12))
    assert len(replay)==7
    with pytest.raises(ValueError):replay.sample(8)


@pytest.mark.parametrize('field',['dones','actions'])
def test_replay_rejects_invalid_shapes(field):
    replay=JointReplayBuffer(100);b=batch()
    setattr(b,field,getattr(b,field)[:,:2] if field=='actions' else np.zeros((8,5)))
    with pytest.raises(ValueError):replay.add_batch(**vars(b))


def test_checkpoint_roundtrip_all_networks(tmp_path):
    t=trainer();t.update(batch());t.sampled_steps=16;path=tmp_path/'c.pt';t.save(path)
    other=trainer();other.load(path)
    obs=np.ones((2,5,56),'f');np.testing.assert_array_equal(t.act(obs),other.act(obs))
    for name in ('actor','critic','target_actor','target_critic'):
        for a,b in zip(getattr(t,name).parameters(),getattr(other,name).parameters()):torch.testing.assert_close(a,b)
    saved=torch.load(path,weights_only=False);validate_checkpoint(saved,*configs())
    assert saved['extra']['replay_buffer_in_checkpoint'] is False
    for k in ('actor','target_actor','actor_optimizer'):assert k in saved
    for k in ('actors','target_actors','actor_optimizers'):assert k not in saved
    arch=saved['extra']['network_architecture']
    assert arch['shared_actor'] is True and arch['independent_actors'] is False and arch['target_actors']==1


@pytest.mark.parametrize('field,value',[('environment_version','3.28'),('observation_dim',52),('num_agents',4),
    ('variant','self_inclusive_context_only_on_policy_deterministic_v1'),
    ('environment_config_sha256','bad'),('algorithm_config_sha256','bad')])
def test_checkpoint_strict_isolation(field,value,tmp_path):
    t=trainer();path=tmp_path/'c.pt';t.save(path);s=torch.load(path,weights_only=False);s['extra'][field]=value
    with pytest.raises(RuntimeError,match=field):validate_checkpoint(s,*configs())


@pytest.mark.parametrize('algorithm,version',[('MAAC',1),('MAAC',2),('MAPPO',3),('madsac',3)])
def test_reject_draft_and_other_algorithms(algorithm,version,tmp_path):
    t=trainer();path=tmp_path/'c.pt';t.save(path);s=torch.load(path,weights_only=False)
    s.update(algorithm=algorithm,implementation_version=version)
    with pytest.raises(RuntimeError,match='identity'):validate_checkpoint(s,*configs())


@pytest.mark.parametrize('version',[1,2])
def test_old_protocol_rejected_for_load_resume_and_recording(version,tmp_path):
    from tools.record_combat_episode import load_recording_policy
    t=trainer();path=tmp_path/'old.pt';t.save(path)
    saved=torch.load(path,weights_only=False);saved['implementation_version']=version;torch.save(saved,path)
    for validator in (validate_checkpoint,validate_checkpoint_for_resume):
        with pytest.raises(RuntimeError,match='v1/v2'):validator(saved,*configs())
    with pytest.raises(RuntimeError,match='v1/v2'):t.load(path)
    with pytest.raises(RuntimeError,match='v1/v2'):load_recording_policy(path,*configs(),'cuda')


def test_formal_smoke_isolation(tmp_path):
    t=trainer();path=tmp_path/'c.pt';t.save(path)
    with pytest.raises(RuntimeError,match='smoke/formal'):MAACTrainer(*configs(),device='cpu').load(path)


@pytest.mark.parametrize('version',['328','327'])
def test_environment_v329_only(version):
    ec,ac=configs();ec=yaml.safe_load(Path(f'configs/combat_environment_v{version}.yaml').read_text())
    with pytest.raises(RuntimeError,match='environment_version'):validate_config_pair(ec,ac)


def test_appendix_parameters_and_no_madsac_dependency():
    _,config=configs()
    expected=dict(lr_actor=.001,lr_critic=.001,gamma=.99,alpha=.01,tau=.005,
        num_train_envs=12,replay_capacity=1000000,minibatch_size=1024,sampled_steps_per_update=100,
        critic_updates_per_cycle=4,policy_updates_per_cycle=4)
    for k,v in expected.items():assert config['training'][k]==v
    assert config['network']['hidden_dim']==128 and config['network']['attention_heads']==4
    assert 'paper_appendix' not in config['metadata'] and 'original_maac_reference' in config['metadata']
    assert config['metadata']['madsac_paper_specific_modifications']['shared_actor_parameter_mechanism'] is True
    for file in Path('algorithm/maac').glob('*.py'):assert 'algorithm.madsac' not in file.read_text()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('smoke',[True,False])
def test_cuda_update_and_recording_load(tmp_path,smoke):
    from tools.record_combat_episode import load_recording_policy
    t=trainer('cuda',smoke);metrics=t.update(batch(8 if smoke else 1024));assert np.isfinite(list(metrics.values())).all()
    assert t.actor_updates==t.policy_updates==1
    if not smoke:assert t.baseline_samples==16 and t.critic.hidden_dim==128 and t.critic.heads==4
    path=tmp_path/'c.pt';t.save(path)
    loaded,s,_=load_recording_policy(path,*configs(),'cuda')
    assert loaded.device.type=='cuda' and s['implementation_version']==3
    np.testing.assert_array_equal(t.act(np.zeros((1,5,56),'f')),loaded.act(np.zeros((1,5,56),'f')))
