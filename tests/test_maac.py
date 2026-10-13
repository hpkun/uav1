"""Contracts for the explicitly requested MAAC-inspired engineering variant."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from algorithm.maac.networks import SharedLocalActor, CentralizedAttentionCritic
from algorithm.maac.trainer import MAACTrainer, RolloutBatch, own_action_joint
from algorithm.maac.protocol import validate_config_pair, validate_checkpoint


def configs():
    return tuple(yaml.safe_load(Path(p).read_text()) for p in
        ('configs/combat_environment_v329.yaml','configs/maac_5v5_v329.yaml'))


def trainer(device='cpu', smoke=True):
    return MAACTrainer(*configs(),device=device,seed=4,smoke=smoke)


def batch():
    rng=np.random.default_rng(41)
    return RolloutBatch(rng.normal(size=(4,2,5,56)).astype('f'),rng.uniform(-1,1,(4,2,5,3)).astype('f'),
        rng.normal(size=(4,2,5)).astype('f'),rng.normal(size=(4,2,5,56)).astype('f'),
        np.zeros((4,2),'f'),np.ones((4,2,5),'f'),np.ones((4,2,5),'f'))


@pytest.mark.parametrize('shape', [(56,), (5,56), (3,5,56)])
def test_shared_local_actor_shape_range_and_locality(shape):
    actor=SharedLocalActor();obs=torch.randn(shape);action=actor(obs)
    assert action.shape==(*shape[:-1],3)
    assert torch.isfinite(action).all() and (action.abs()<=1).all()
    if len(shape)==3:
        changed=obs.clone();changed[:,1:]=100
        torch.testing.assert_close(actor(changed)[:,0],action[:,0],rtol=0,atol=0)


@pytest.mark.parametrize('batch_size', [1,4])
def test_critic_shape_attention_includes_self(batch_size):
    critic=CentralizedAttentionCritic()
    q,attention=critic(torch.randn(batch_size,5,56),torch.randn(batch_size,5,3),torch.ones(batch_size,5),True)
    assert q.shape==(batch_size,5,1) and attention.shape==(batch_size,2,5,5)
    assert (attention.diagonal(dim1=-2,dim2=-1)>0).all()
    torch.testing.assert_close(attention.sum(-1),torch.ones(batch_size,2,5))


@pytest.mark.parametrize('invalid', ['obs','action','mask'])
def test_critic_rejects_invalid_joint_shapes(invalid):
    obs=torch.zeros(2,5,56);act=torch.zeros(2,5,3);mask=torch.ones(2,5)
    if invalid=='obs':obs=obs[:,:,:52]
    elif invalid=='action':act=act[:,:,:2]
    else:mask=mask[:,:4]
    with pytest.raises(ValueError):CentralizedAttentionCritic()(obs,act,mask)


def test_dead_agent_cannot_affect_living_q_or_gradient():
    critic=CentralizedAttentionCritic();obs=torch.randn(3,5,56);act=torch.randn(3,5,3)
    mask=torch.ones(3,5);mask[:,2]=0
    q,attention=critic(obs,act,mask,True)
    other_obs=obs.clone();other_act=act.clone();other_obs[:,2]=1e8;other_act[:,2]=-1e8
    torch.testing.assert_close(q,critic(other_obs,other_act,mask),rtol=0,atol=0)
    assert (q[:,2]==0).all() and (attention[:,:,:,2]==0).all() and (attention[:,:,2,:]==0).all()
    act.requires_grad_();critic(obs,act,mask).sum().backward()
    assert (act.grad[:,2]==0).all()


def test_all_dead_and_single_live_mask():
    c=CentralizedAttentionCritic();o=torch.randn(2,5,56);a=torch.randn(2,5,3);m=torch.zeros(2,5)
    q,w=c(o,a,m,True);assert torch.isfinite(q).all() and torch.isfinite(w).all()
    assert (q==0).all() and (w==0).all()
    m[:,3]=1;q,w=c(o,a,m,True)
    assert (w[:,:,3,3]==1).all()


def test_value_nonlinearity_and_context_only_head():
    c=CentralizedAttentionCritic()
    assert isinstance(c.value[-1],torch.nn.LeakyReLU)
    assert c.q_head[0].in_features==256  # not concat(e_i,context), which would require 512
    received=[]
    handle=c.q_head.register_forward_pre_hook(lambda _, args:received.append(args[0]))
    c(torch.randn(1,5,56),torch.randn(1,5,3),torch.ones(1,5));handle.remove()
    assert received[0].shape==(1,5,256)


@pytest.mark.parametrize('index', range(5))
def test_actor_own_action_gradient_only(index):
    c=CentralizedAttentionCritic(hidden_dim=32)
    actions=torch.randn(3,5,3,requires_grad=True)
    q=c(torch.randn(3,5,56),own_action_joint(actions,index),torch.ones(3,5))[:,index,0]
    q.sum().backward()
    assert actions.grad[:,index].abs().sum()>0
    for j in range(5):
        if j!=index:assert actions.grad[:,j].abs().sum()==0


def test_td_target_detached_terminal_and_dead_masks():
    t=trainer();o=torch.randn(2,5,56);r=torch.randn(2,5);d=torch.tensor([1.,0.]);m=torch.ones(2,5);m[1,2]=0
    y=t.td_targets(r,o,d,m)
    assert not y.requires_grad
    torch.testing.assert_close(y[0],r[0]);torch.testing.assert_close(y[1,2],r[1,2])
    with torch.no_grad():
        expected=r+t.gamma*(1-d[:,None])*m*t.critic(o,t.actor(o)*m[...,None],m).squeeze(-1)
    torch.testing.assert_close(y,expected)
    assert all(p.grad is None for p in t.actor.parameters())
    assert all(p.grad is None for p in t.critic.parameters())


def test_fresh_rollout_update_changes_actor_critic_finite_gradients():
    t=trainer();before_actor=[p.clone() for p in t.actor.parameters()];before_critic=[p.clone() for p in t.critic.parameters()]
    metrics=t.update(batch())
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics['actor_grad_norm']>0 and metrics['critic_grad_norm']>0
    assert any(not torch.equal(a,b) for a,b in zip(before_actor,t.actor.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(before_critic,t.critic.parameters()))
    assert all(p.requires_grad for p in t.critic.parameters())
    assert all(p.grad is None for p in t.critic.parameters())
    assert t.actor_updates==t.critic_updates==1


def test_no_targets_entropy_replay_or_madsac_dependency():
    t=trainer()
    for name in ('target_actor','target_critic','critic2','replay','alpha'):
        assert not hasattr(t,name)
    for file in Path('algorithm/maac').glob('*.py'):
        assert 'algorithm.madsac' not in file.read_text()


def test_dead_current_agents_do_not_contribute_loss():
    t=trainer();b=batch();b.alive_masks.fill(0)
    assert t.update(b)=={} and t.actor_updates==t.critic_updates==0


def test_save_load_and_algorithm_identity(tmp_path):
    t=trainer();t.update(batch());t.sampled_steps=16;path=tmp_path/'checkpoint.pt';t.save(path)
    other=trainer();other.load(path)
    obs=np.ones((2,5,56),'f')
    np.testing.assert_array_equal(t.act(obs),other.act(obs))
    for a,b in zip(t.critic.parameters(),other.critic.parameters()):torch.testing.assert_close(a,b)
    saved=torch.load(path,weights_only=False);ec,ac=configs();validate_checkpoint(saved,ec,ac)
    for algorithm in ('MAPPO','madsac'):
        altered=deepcopy(saved);altered['algorithm']=algorithm
        with pytest.raises(RuntimeError,match='identity'):validate_checkpoint(altered,ec,ac)


@pytest.mark.parametrize('field,value', [('environment_version','3.28'),('observation_dim',52),('num_agents',4),
    ('variant','original_maac'),('environment_config_sha256','bad'),('algorithm_config_sha256','bad')])
def test_checkpoint_strict_v329_isolation(field,value,tmp_path):
    t=trainer();path=tmp_path/'c.pt';t.save(path);s=torch.load(path,weights_only=False)
    s['extra'][field]=value
    with pytest.raises(RuntimeError,match=field):validate_checkpoint(s,*configs())


def test_formal_smoke_isolation(tmp_path):
    t=trainer();path=tmp_path/'s.pt';t.save(path)
    with pytest.raises(RuntimeError,match='smoke/formal'):MAACTrainer(*configs(),device='cpu').load(path)


@pytest.mark.parametrize('version', ['3.28','3.27','2.3'])
def test_reject_wrong_environment(version):
    ec,ac=configs();ec=yaml.safe_load(Path(f"configs/combat_environment_v{version.replace('.','')}.yaml").read_text()) if version!='2.3' else yaml.safe_load(Path('configs/combat_environment.yaml').read_text())
    with pytest.raises(RuntimeError,match='environment_version'):validate_config_pair(ec,ac)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_update_and_recording_loader(tmp_path):
    from tools.record_combat_episode import load_recording_policy
    t=trainer('cuda');metrics=t.update(batch());assert np.isfinite(list(metrics.values())).all()
    path=tmp_path/'cuda.pt';t.save(path)
    loaded,state,_=load_recording_policy(path,*configs(),'cuda')
    assert loaded.device.type=='cuda' and state['extra']['environment_version']=='3.29'
    np.testing.assert_array_equal(t.act(np.zeros((1,5,56),'f')),loaded.act(np.zeros((1,5,56),'f')))
