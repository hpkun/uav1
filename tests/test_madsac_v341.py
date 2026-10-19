"""Fresh v3.41 identities plus independent v5 math/schedule/replay checks."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch
from torch.distributions import Normal
import yaml
from algorithm.madsac.trainer import (own_action_twin_q,actor_objective,critic_target,
                                     SharedMADSACActor)
from algorithm.madsac.protocol import validate_madsac_config,validate_madsac_checkpoint
from algorithm.madsac.factory import build_madsac_trainer
from algorithm.madsac.runner import MADSACTrainingRunner
from algorithm.madsac.replay_buffer import JointReplayBuffer
from algorithm.common.vector_env import VectorStep

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cuda():
    assert torch.cuda.is_available(),'CUDA mandatory; no CPU fallback'
    torch.set_num_threads(1)


def config(v='341'):
    return tuple(yaml.safe_load((ROOT/f'configs/{name}_v{v}.yaml').read_text(encoding='utf-8'))
                 for name in ('combat_environment','madsac_4v4'))


def test_core_independent_closed_form_shared_actor_own_gradient():
    # Nonzero peer coefficients preserve centralized Q values, but Eq.(21)
    # permits only each own-action derivative. Closed form is independent of AD.
    dtype=torch.float64;device='cuda';B,N,D,A=2,4,3,2
    obs=torch.linspace(-.7,.9,B*N*D,device=device,dtype=dtype).reshape(B,N,D)
    w=torch.linspace(-.3,.4,D*A,device=device,dtype=dtype).reshape(D,A).requires_grad_()
    eps=torch.linspace(-1,1,B*N*A,device=device,dtype=dtype).reshape(B,N,A)
    alive=torch.tensor([[1,1,0,1],[1,0,1,1]],device=device,dtype=dtype)
    coeff=torch.arange(1,N*N+1,device=device,dtype=dtype).reshape(N,N)/10
    mean=obs@w;dist=Normal(mean,torch.ones_like(mean)*.7);raw=mean+.7*eps
    action=raw.tanh()*alive[...,None];lp=SharedMADSACActor.squashed_log_prob(dist,raw)*alive
    def q1(o,a,m):return (a.square().sum(-1)@coeff.T)*m
    def q2(o,a,m):return q1(o,a,m)+1
    q=own_action_twin_q(q1,q2,obs,action,alive)
    torch.testing.assert_close(q,q1(obs,action,alive))
    loss=actor_objective(lp,q,q,alive,.1)
    actual=torch.autograd.grad(loss,w,retain_graph=True)[0]
    a=raw.detach().tanh()
    derivative=(.2*a-2*coeff.diagonal()[None,:,None]*a*(1-a.square()))*alive[...,None]
    expected=torch.einsum('bnd,bna->da',obs,derivative)/B
    torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
    wrong=actor_objective(lp,q1(obs,action,alive),q2(obs,action,alive),alive,.1)
    assert (torch.autograd.grad(wrong,w)[0]-expected).abs().max()>1e-4


def test_core_independent_terminal_and_next_death_target():
    t=lambda x:torch.tensor(x,device='cuda',dtype=torch.float64)
    target=critic_target(t([[1,2,3,4],[5,6,7,8]]),t([0,1]),t([[1,0,1,1],[1,1,1,1]]),
        t([[10,9,8,7],[1,2,3,4]]),t([[9,10,6,8],[4,3,2,1]]),t([[-2,-3,-4,-5],[-1,-1,-1,-1]]),.99,.1)
    expected=t([[1+.99*9.2,2,3+.99*6.4,4+.99*7.5],[5,6,7,8]])
    torch.testing.assert_close(target,expected)


def test_exact_configuration_delta_and_budget_provenance():
    e,c=config();_,old=config('339');mapped=deepcopy(c)
    mapped['environment']['environment_version']='3.39'
    mapped['training']['total_sampled_steps']=old['training']['total_sampled_steps']
    for key in ('formal_total_sampled_steps','formal_budget_source'):
        mapped['metadata']['project_implementation_choice'][key]=old['metadata']['project_implementation_choice'][key]
    assert mapped==old and c['implementation_version']==5
    assert c['training']['total_sampled_steps']==3000000
    assert c['metadata']['paper_reported']==old['metadata']['paper_reported']
    assert c['metadata']['project_implementation_choice']['formal_total_sampled_steps']==3000000
    assert c['implementation']['evaluation_mode']=='stochastic'
    validate_madsac_config(e,c)


@pytest.mark.parametrize('section,key,value',[
    ('environment','environment_version','3.39'),('environment','team_size',5),
    ('network','observation_dim',46),('network','action_dim',4),('network','num_agents',5),
    ('network','actor_hidden_layers',[128,128]),('network','critic_hidden_layers',[128,128]),
    ('network','attention_heads',4)])
def test_wrong_identity_and_structure_rejected(section,key,value):
    e,c=config();c[section][key]=value
    with pytest.raises(RuntimeError):validate_madsac_config(e,c)


def test_v341_networks_actor_sampling_masks_and_target_initialization():
    e,c=config();t=build_madsac_trainer(c,'cuda',seed=3)
    x=torch.randn(3,4,45,device='cuda');alive=torch.tensor([[1,1,0,1]]*3,device='cuda')
    a,raw,lp=t.actor.sample(x,t.policy_generator)
    assert a.shape==(3,4,3) and lp.shape==(3,4) and raw.requires_grad
    deterministic=t.act(x,alive,deterministic=True)
    assert np.isfinite(deterministic).all() and np.count_nonzero(deterministic[:,2])==0
    for q in (t.critic1,t.critic2):
        output,w=q(x,a,alive,True)
        assert output.shape==(3,4) and w.shape==(3,2,4,4)
        assert torch.count_nonzero(w[:,:,:,2])==torch.count_nonzero(w[:,:,2,:])==0
        assert (w[:,:,0,0]>0).all()
    assert not {p.data_ptr() for p in t.critic1.parameters()}&{p.data_ptr() for p in t.critic2.parameters()}
    for name in ('actor','critic1','critic2'):
        for online,target in zip(getattr(t,name).parameters(),getattr(t,'target_'+name).parameters()):
            assert torch.equal(online,target) and online.data_ptr()!=target.data_ptr() and not target.requires_grad


def test_v341_true_terminal_observation_and_joint_replay():
    r=object.__new__(MADSACTrainingRunner);r.replay=JointReplayBuffer(4,4,45,3)
    x=np.zeros((1,4,45),np.float32);terminal=np.full_like(x,7);reset=np.full_like(x,99)
    result=VectorStep(reset,terminal,np.ones((1,4)),np.array([True]),np.array([False]),[{}],np.ones((1,4)),np.zeros((1,4)))
    r.store_vector_step(x,np.zeros((1,4,3)),result)
    b=r.replay.sample(1)
    assert b.observations.shape==(1,4,45) and b.actions.shape==(1,4,3)
    assert (b.next_observations==7).all() and b.dones[0] and not b.next_alive_masks.any()


@pytest.mark.parametrize('fault',['old_version','env_hash','algo_hash','seed','heads','delay','v3','v4'])
def test_checkpoint_guards_seed3(fault,tmp_path):
    e,c=config();from algorithm.madsac.critic_contract import CRITIC_ARCHITECTURE,schedule_contract
    from algorithm.common.protocol import config_sha256
    t=build_madsac_trainer(c,'cuda',seed=3)
    arch=dict(CRITIC_ARCHITECTURE,hidden_dim=256,attention_heads=2,attention_head_dim=128,
              shared_actor=True,twin_independent_critics=True,target_actor=True,target_critics=2,
              observation_dim=45,action_dim=3,num_agents=4)
    extra=dict(environment_version='3.41',environment_config_sha256=config_sha256(e),
        algorithm_config_sha256=config_sha256(c),training_seed=3,observation_dim=45,action_dim=3,num_agents=4,
        network_architecture=arch,formal_exact_resume_supported=False,replay_buffer_in_checkpoint=False,
        implementation_version=5,training_schedule=schedule_contract(100,4,2))
    s=t.checkpoint_state(extra);validate_madsac_checkpoint(s,e,c,expected_training_seed=3)
    if fault=='old_version':
        old_e,old_c=config('339')
        with pytest.raises(RuntimeError):validate_madsac_checkpoint(s,old_e,old_c,expected_training_seed=3)
        s['extra']['environment_version']='3.39'
        # A relabeled new checkpoint still cannot match historical config hashes.
        with pytest.raises(RuntimeError,match='sha256'):validate_madsac_checkpoint(s,old_e,old_c,expected_training_seed=3)
    elif fault=='env_hash':s['extra']['environment_config_sha256']='invalid'
    elif fault=='algo_hash':s['extra']['algorithm_config_sha256']='invalid'
    elif fault=='seed':s['extra']['training_seed']=0
    elif fault=='heads':s['extra']['network_architecture']['attention_heads']=4
    elif fault=='delay':s['training_schedule']['policy_delay']=3
    else:s['implementation_version']=int(fault[1:])
    with pytest.raises(RuntimeError):validate_madsac_checkpoint(s,e,c,expected_training_seed=3)
    with pytest.raises(RuntimeError,match='exact resume unsupported'):t.resume('unused')
