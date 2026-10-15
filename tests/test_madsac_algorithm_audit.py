"""CUDA checks of existing production math; no production code is replaced."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
import yaml
from algorithm.madsac.factory import build_madsac_trainer
from algorithm.madsac.replay_buffer import ReplayBatch
from algorithm.madsac.trainer import own_action_twin_q, critic_target, actor_objective, batch_mean_agent_sum
from algorithm.madsac.networks import SharedMADSACActor
from tools.audit_madsac_attention import collect
from tools.audit_madsac_training_schedule import theoretical_counts
from tools.madsac_audit_reference import offline_forward, attention_rows, summarize_rows

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def setup():
    assert torch.cuda.is_available(), 'CUDA mandatory for algorithm audit'
    torch.set_num_threads(1)
    c=yaml.safe_load((ROOT/'configs/madsac_4v4_v337.yaml').read_text())
    e=yaml.safe_load((ROOT/'configs/combat_environment_v337.yaml').read_text())
    tr=build_madsac_trainer(c,'cuda',seed=7)
    data=collect(e,tr,16,80000,random_policy=True)
    tensors=tuple(torch.tensor(x,device='cuda',dtype=torch.float32) for x in data)
    return tr,tensors,c


def test_offline_current_reconstructs_production_and_scale_factor(setup):
    tr,(obs,act,alive),_=setup
    for critic in (tr.critic1,tr.critic2):
        q,w=critic(obs,act,alive,True)
        qr,wr,z,_=offline_forward(critic,obs,act,alive)
        _,_,zs,_=offline_forward(critic,obs,act,alive,True)
        torch.testing.assert_close(q,qr);torch.testing.assert_close(w,wr)
        torch.testing.assert_close(z,zs*np.sqrt(128))


def test_dead_queries_and_keys_excluded_and_singleton_separate(setup):
    tr,(obs,act,alive),_=setup
    alive=alive.clone();alive[0]=0;alive[1]=0;alive[1,2]=1;alive[2,1]=0
    q,w,z,_=offline_forward(tr.critic1,obs,act,alive)
    rows=attention_rows(z,w,alive)
    assert all(r['item']!=0 and (r['item']!=1 or r['query']==2) for r in rows)
    assert not w[0].any() and not q[0].any() and not w[2,:,:,1].any()
    singleton=[r for r in rows if r['item']==1]
    assert len(singleton)==2 and all(r['max_attention_weight']==1 for r in singleton)
    assert all(r['normalized_attention_entropy'] is None for r in singleton)
    assert summarize_rows(rows)['single_alive_query_head_count']==2


def test_own_action_autograd_zero_peer_nonzero_own(setup):
    tr,(obs,act,alive),_=setup
    act=act.detach().requires_grad_(True)
    q=own_action_twin_q(tr.critic1,tr.critic2,obs,act,alive)
    for i in range(act.shape[1]):
        g=torch.autograd.grad(q[:,i].sum(),act,retain_graph=True)[0]
        assert g[:,i].abs().sum()>0
        peers=[j for j in range(act.shape[1]) if j!=i]
        assert torch.count_nonzero(g[:,peers])==0


def test_twin_independence_targets_equal_and_optimizers_independent(setup):
    tr,_,_=setup
    p1=list(tr.critic1.parameters());p2=list(tr.critic2.parameters())
    assert not {p.data_ptr() for p in p1}&{p.data_ptr() for p in p2}
    assert any(not torch.equal(a,b) for a,b in zip(p1,p2))
    for name in ('actor','critic1','critic2'):
        for online,target in zip(getattr(tr,name).parameters(),getattr(tr,'target_'+name).parameters()):
            assert online.data_ptr()!=target.data_ptr()
            torch.testing.assert_close(online,target,rtol=0,atol=0)
            assert not target.requires_grad
    assert tr.critic1_optimizer is not tr.critic2_optimizer
    assert {id(p) for group in tr.critic1_optimizer.param_groups for p in group['params']}=={id(p) for p in p1}
    assert {id(p) for group in tr.critic2_optimizer.param_groups for p in group['params']}=={id(p) for p in p2}


def test_actual_update_target_policy_and_delayed_polyak(setup):
    tr,(obs,act,alive),_=setup
    cpu=lambda x:x.detach().cpu().numpy()
    batch=ReplayBatch(cpu(obs),cpu(act),np.zeros(alive.shape,np.float32),cpu(obs),
                      np.zeros(len(obs),np.bool_),cpu(alive),cpu(alive))
    snapshots={name:deepcopy(getattr(tr,'target_'+name).state_dict()) for name in ('actor','critic1','critic2')}
    import algorithm.madsac.trainer as module
    original=module.polyak_update
    with patch.object(tr.target_actor,'sample',wraps=tr.target_actor.sample) as target_sample, \
         patch.object(tr.actor,'sample',wraps=tr.actor.sample) as actor_sample, \
         patch.object(module,'polyak_update',wraps=original) as polyak:
        m=tr.update(batch)
        assert target_sample.call_count==1 and actor_sample.call_count==0
        assert polyak.call_count==0 and m['actor_updated']==0
        for name in snapshots:
            for k,v in getattr(tr,'target_'+name).state_dict().items():
                torch.testing.assert_close(v,snapshots[name][k],rtol=0,atol=0)
        m=tr.update(batch)
        assert target_sample.call_count==2 and actor_sample.call_count==1
        assert polyak.call_count==3 and m['actor_updated']==1
        for name in snapshots:
            for k,v in getattr(tr,'target_'+name).state_dict().items():
                expected=(1-tr.tau)*snapshots[name][k]+tr.tau*getattr(tr,name).state_dict()[k]
                torch.testing.assert_close(v,expected)
    assert tr.critic_update_count==2 and tr.actor_update_count==1


def test_entropy_sign_min_q_terminal_mask_and_reduction(setup):
    _,_,_=setup
    tensor=lambda x:torch.tensor(x,device='cuda',dtype=torch.float32)
    rewards=tensor([[1,2],[3,4]]);dones=tensor([0,1]);alive=tensor([[1,0],[1,1]])
    q1=tensor([[5,7],[9,11]]);q2=tensor([[4,8],[10,8]]);lp=tensor([[-2,-3],[-4,-5]])
    target=critic_target(rewards,dones,alive,q1,q2,lp,.99,.1)
    expected=rewards+.99*(1-dones[:,None])*alive*(torch.minimum(q1,q2)-.1*lp)
    torch.testing.assert_close(target,expected)
    torch.testing.assert_close(actor_objective(lp,q1,q2,alive,.1),((.1*lp-torch.minimum(q1,q2))*alive).sum(1).mean())
    torch.testing.assert_close(batch_mean_agent_sum(q1,alive),tensor((5+9+11)/2))


def test_tanh_logprob_correction_and_reparameterization(setup):
    tr,(obs,_,_),_=setup
    distribution=tr.actor.distribution(obs)
    raw=torch.linspace(-2,2,distribution.mean.numel(),device='cuda').reshape_as(distribution.mean)
    expected=(distribution.log_prob(raw)-torch.log(1-torch.tanh(raw).square())).sum(-1)
    torch.testing.assert_close(SharedMADSACActor.squashed_log_prob(distribution,raw),expected)
    extreme=torch.full_like(raw,50.)
    assert torch.isfinite(SharedMADSACActor.squashed_log_prob(distribution,extreme)).all()
    actions,_,_=tr.actor.sample(obs)
    grads=torch.autograd.grad(actions.sum(),tr.actor.mean.weight)[0]
    assert grads.abs().sum()>0 and tr.alpha==.1


def test_schedule_boundaries_and_paired_optimizer_convention(setup):
    _,_,c=setup
    full=theoretical_counts(c)
    assert full['first_update_sampled_steps']==10008
    assert full['critic_update_count']==124584 and full['actor_update_count']==62292
    assert full['combined_critic_optimizer_steps']==249168
    smoke=theoretical_counts(c,24000)
    assert smoke['critic_update_count']==584 and smoke['actor_update_count']==292
    assert theoretical_counts(c,9984)['critic_update_count']==0
    assert theoretical_counts(c,10008)['critic_update_count']==1
