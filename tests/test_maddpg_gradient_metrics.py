"""MADDPG reports and clips gradients only for critics with active samples."""
import numpy as np
import pytest
import torch

from algorithm.maddpg import MADDPG
from algorithm.maddpg_config import MADDPGConfig
from env import make_environment
from experiment.interception import read_config


def _policy_and_batch(active_count):
    path=('configs/interception_one_vs_one_jia_velocity_relative.yaml' if active_count==1
          else 'configs/interception_ideal_jia_low_level_obs24_no_heading.yaml')
    config,_,_=read_config(path)
    contract=make_environment(config).contract
    parameters=MADDPGConfig(hidden_dim=16,batch_size=4,buffer_capacity=8,learning_starts=4,
                            actor_max_grad_norm=1e-6,critic_max_grad_norm=1e-6)
    policy=MADDPG(parameters,device='cpu',seed=7,environment_config=config,
                  environment_contract=contract)
    mask=np.zeros((4,5),np.float32)
    mask[:,:active_count]=1
    batch=dict(observations=np.ones((4,5,24),np.float32),
               next_observations=np.ones((4,5,24),np.float32),
               states=np.ones((4,75),np.float32),next_states=np.ones((4,75),np.float32),
               actions=np.zeros((4,5,2),np.float32),rewards=np.ones((4,5),np.float32),
               dones=np.zeros(4,np.float32),active_masks=mask,next_active_masks=mask.copy())
    return policy,batch


def test_one_vs_one_only_active_actor_and_critic_are_counted():
    policy,batch=_policy_and_batch(1)
    before=[parameter.detach().clone() for parameter in policy.critics[1].parameters()]
    metrics=policy.update(batch)
    assert metrics['active_critic_count']==1
    assert policy.actor_optimizer_steps==[1,0,0,0,0]
    assert metrics['actor_grad_norm']>0 and metrics['critic_grad_norm']>0
    assert metrics['actor_clip_rate']==1 and metrics['critic_clip_rate']==1
    for old,new in zip(before,policy.critics[1].parameters()):
        torch.testing.assert_close(old,new)
    assert metrics['critic_grad_norm_mean_agent']==pytest.approx(metrics['critic_grad_norm'])


def test_five_vs_five_critics_are_clipped_independently():
    policy,batch=_policy_and_batch(5)
    metrics=policy.update(batch)
    assert metrics['active_critic_count']==5
    assert policy.actor_optimizer_steps==[1]*5
    assert metrics['critic_clip_rate']==1
    assert metrics['critic_grad_norm']>policy.config.critic_max_grad_norm
    for critic in policy.critics:
        norm=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in critic.parameters()
                                                   if p.grad is not None]))
        assert float(norm)<=policy.config.critic_max_grad_norm+1e-7


def test_retired_defender_does_not_bootstrap_from_next_state():
    first,batch=_policy_and_batch(1)
    second,other=_policy_and_batch(1)
    batch['next_active_masks'][:]=0
    other['next_active_masks'][:]=0
    other['next_states'][:]=100
    other['next_observations'][:]=-100
    first_metrics=first.update(batch)
    second_metrics=second.update(other)
    assert first_metrics['td_abs']==pytest.approx(second_metrics['td_abs'])
    for left,right in zip(first.critics[0].parameters(),second.critics[0].parameters()):
        torch.testing.assert_close(left,right)


def test_retired_critic_is_not_moved_by_adam_momentum():
    policy,batch=_policy_and_batch(5)
    policy.update(batch)
    before=[parameter.detach().clone() for parameter in policy.critics[1].parameters()]
    batch['active_masks'][:,1:]=0
    batch['next_active_masks'][:,1:]=0
    assert policy.update(batch)['active_critic_count']==1
    for old,new in zip(before,policy.critics[1].parameters()):
        torch.testing.assert_close(old,new)
