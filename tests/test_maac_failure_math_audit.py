"""Controlled CUDA math tests; do not load or alter historical models."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Normal,TransformedDistribution,TanhTransform
import yaml
from algorithm.maac.networks import GaussianLocalActor,CentralizedAttentionCritic
from algorithm.maac.trainer import MAACTrainer,score_function_loss
from algorithm.maac.replay import JointReplayBuffer


@pytest.fixture(autouse=True)
def cuda():
    assert torch.cuda.is_available(),'CUDA mandatory';torch.set_num_threads(1)


def trainer():
    return MAACTrainer(*[yaml.safe_load(Path(p).read_text()) for p in
        ('configs/combat_environment_v348.yaml','configs/maac_4v4_v348.yaml')],device='cuda',seed=11)


@pytest.mark.parametrize('alpha',[0.,.01,.5])
def test_exact_two_action_score_gradient_and_baseline_stop(alpha):
    theta=torch.tensor(.3,device='cuda',requires_grad=True)
    logits=torch.stack((theta,-theta));p=logits.softmax(0);lp=logits.log_softmax(0)
    q=torch.tensor([2.,-1.],device='cuda');base=(p*q).sum()
    loss=score_function_loss(lp,q,base,alpha,p.detach())
    g=torch.autograd.grad(loss,theta,retain_graph=True)[0]
    exact=-(p*q).sum()+alpha*(p*lp).sum()
    reference=torch.autograd.grad(exact,theta,retain_graph=True)[0]
    torch.testing.assert_close(g,reference,atol=1e-6,rtol=1e-6)
    assert g<0 # gradient descent increases probability of action with Q=2
    b=torch.tensor([3.,3.],device='cuda',requires_grad=True)
    score_function_loss(lp,q,b,alpha,p.detach()).backward()
    assert b.grad is None


def test_continuous_score_gradient_matches_quadrature():
    mu=torch.tensor(.2,device='cuda',requires_grad=True);sigma=.7
    x=torch.linspace(-7,7,40001,device='cuda');dx=x[1]-x[0]
    d=Normal(mu,sigma);weights=(d.log_prob(x).exp()*dx).detach()
    lp=GaussianLocalActor.squashed_log_prob(Normal(mu.expand(len(x),1),torch.full((len(x),1),sigma,device='cuda')),x[:,None])
    loss=score_function_loss(lp,x.tanh(),torch.zeros_like(x),0.,weights)
    gradient=torch.autograd.grad(loss,mu)[0]
    reference=-(weights*(1-x.tanh().square())).sum()/weights.sum()
    torch.testing.assert_close(gradient,reference,atol=1e-4,rtol=1e-4)


def test_squashed_log_probability_transform_and_fixed_raw_detach():
    mean=torch.tensor([[.2,-.3,.1]],device='cuda',requires_grad=True)
    scale=torch.tensor([[.4,.8,.6]],device='cuda');raw=torch.tensor([[.3,-.2,.7]],device='cuda',requires_grad=True)
    d=Normal(mean,scale)
    lp=GaussianLocalActor.squashed_log_prob(d,raw)
    expected=TransformedDistribution(d,[TanhTransform()]).log_prob(raw.tanh()).sum(-1)
    torch.testing.assert_close(lp,expected,atol=1e-6,rtol=1e-6)
    t=trainer();o=torch.randn(4,4,45,device='cuda');raw=torch.randn(4,4,3,device='cuda',requires_grad=True)
    t.actor.log_prob(o,raw).sum().backward();assert raw.grad is None
    assert torch.isfinite(t.actor.log_prob(o,torch.full_like(raw,100))).all()


@pytest.mark.parametrize('index',range(4))
def test_counterfactual_replaces_only_selected_agent(index):
    t=trainer();o=torch.randn(3,4,45,device='cuda');joint=torch.rand(3,4,3,device='cuda');mask=torch.ones(3,4,device='cuda')
    class Spy(nn.Module):
        def forward(self,obs,actions,alive):
            for j in range(4):
                if j!=index:torch.testing.assert_close(actions[:,j],joint[:,j].repeat(16,1),rtol=0,atol=0)
            return actions.sum(-1,keepdim=True)
    t.critic=Spy();value=t.counterfactual_baseline(o,joint,mask,index)
    assert value.shape==(3,) and not value.requires_grad


def test_dead_information_and_all_dead_attention_isolation():
    c=CentralizedAttentionCritic(45,3,4).cuda();o=torch.randn(5,4,45,device='cuda');a=torch.randn(5,4,3,device='cuda');m=torch.ones(5,4,device='cuda');m[:,2]=0
    q,w=c(o,a,m,True);o[:,2]=1e12;a[:,2]=-1e12
    torch.testing.assert_close(q,c(o,a,m),rtol=0,atol=0)
    assert not w[:,:,:,2].any() and not q[:,2].any()
    q,w=c(o,a,m*0,True);assert not q.any() and not w.any() and torch.isfinite(q).all()


def test_td_terminal_individual_death_and_soft_entropy_sign():
    t=trainer();reward=torch.randn(3,4,device='cuda');obs=torch.randn(3,4,45,device='cuda')
    alive=torch.ones(3,4,device='cuda');alive[1,2]=0;done=torch.tensor([1.,0,0],device='cuda')
    torch.manual_seed(78);target=t.td_targets(reward,obs,done,alive)
    torch.manual_seed(78)
    with torch.no_grad():
        act,lp,_=t.target_actor.sample(obs);q=t.target_critic(obs,act*alive[...,None],alive).squeeze(-1)
    expected=reward+t.gamma*(1-done[:,None])*alive*(q-t.alpha*lp)
    torch.testing.assert_close(target,expected,rtol=0,atol=0)
    torch.testing.assert_close(target[0],reward[0]);assert target[1,2]==reward[1,2]


def test_polyak_online_weight_and_one_shared_optimizer_copy():
    t=trainer();before=[p.clone() for p in t.target_actor.parameters()]
    with torch.no_grad():
        for p in t.actor.parameters():p.add_(1.)
    t.update_targets()
    for x,y in zip(before,t.target_actor.parameters()):torch.testing.assert_close(y,x+t.tau)
    p=[p for group in t.actor_optimizer.param_groups for p in group['params']]
    assert len(p)==len(set(map(id,p)))==len(list(t.actor.parameters()))


def test_shared_loss_equals_sum_of_live_agent_contributions():
    x=torch.tensor(.4,device='cuda',requires_grad=True);lp=torch.stack([x,2*x,3*x,4*x])
    mask=torch.tensor([1.,1,0,1],device='cuda');q=torch.tensor([1.,2,10000,4],device='cuda')
    loss=score_function_loss(lp,q,q*0,0.,mask)
    g=torch.autograd.grad(loss,x)[0]
    assert float(g)==pytest.approx(-7.,abs=1e-6) # -(1+4+16)/3


def test_replay_transition_alignment_and_wraparound():
    b=JointReplayBuffer(7,seed=1,chunk_size=3,observation_dim=45,num_agents=4,action_dim=3)
    for i in range(10):
        o=np.full((1,4,45),i);b.add_batch(o,np.zeros((1,4,3)),np.full((1,4),i),o+1,np.array([i%2]),np.ones((1,4)),np.ones((1,4)))
    d=b.sample(7);np.testing.assert_array_equal(d.next_observations,d.observations+1)
    np.testing.assert_array_equal(d.rewards[:,0],d.observations[:,0,0]);assert d.observations.min()==3 and d.observations.max()==9
