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
from tools.validate_madsac_v4 import expected_counts


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


def test_actual_update_target_policy_and_explicit_polyak(setup):
    tr,(obs,act,alive),_=setup
    cpu=lambda x:x.detach().cpu().numpy()
    batch=ReplayBatch(cpu(obs),cpu(act),np.zeros(alive.shape,np.float32),cpu(obs),
                      np.zeros(len(obs),np.bool_),cpu(alive),cpu(alive))
    import algorithm.madsac.trainer as module
    with patch.object(tr.target_actor,'sample',wraps=tr.target_actor.sample) as target_sample, \
         patch.object(tr.actor,'sample',wraps=tr.actor.sample) as actor_sample, \
         patch.object(module,'polyak_update',wraps=module.polyak_update) as polyak:
        tr.update_critics(batch)
        assert target_sample.call_count==1 and actor_sample.call_count==0 and polyak.call_count==0
        tr.update_actor(batch)
        assert target_sample.call_count==1 and actor_sample.call_count==1 and polyak.call_count==0
        tr.update_targets()
        assert polyak.call_count==3 and tr.target_update_count==1


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
    _,_,_=setup
    smoke=expected_counts(4800)
    assert smoke['first_update_sampled_steps']==1032
    assert smoke['critic_update_count']==128 and smoke['actor_update_count']==64
    assert smoke['critic_optimizer_steps']==256
    assert expected_counts(1008)['critic_update_count']==0
    assert expected_counts(1032)['critic_update_count']==4
