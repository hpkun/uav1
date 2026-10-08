"""Small CUDA checks for the independent deployment-mode diagnostic."""
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from torch.distributions import Normal
from algorithm.mappo.networks import SharedMAPPOActor
from algorithm.stea_mappo.networks import SpatioTemporalEntityAttentionActor
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.common.protocol import config_sha256
from env.config import load_config
from tools import evaluate_policy_modes as diagnostic

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def device():
    assert torch.cuda.is_available(), 'CUDA mandatory for policy diagnostic checks'
    return 'cuda'

@pytest.mark.parametrize('recurrent',[False,True])
def test_actual_distribution_actions_sigma_and_seed(device,recurrent):
    torch.manual_seed(31)
    actor=(SpatioTemporalEntityAttentionActor() if recurrent else SharedMAPPOActor(65)).to(device)
    with torch.no_grad():
        actor.log_std.weight.zero_();actor.log_std.bias.copy_(torch.tensor([-9.,0.,9.],device=device))
    execution=diagnostic.PolicyExecution(actor,recurrent,device)
    observation=np.zeros((5,65),np.float32);alive=np.ones(5,np.float32)
    try:
        execution.reset(5)
        actions,raw,dist,log_std=execution.decide(observation,alive,'deterministic')
        assert torch.equal(actions,torch.tanh(dist.mean)) and torch.equal(raw,dist.mean)
        assert torch.equal(dist.scale,log_std.exp())
        assert torch.equal(log_std,torch.tensor([-5.,0.,2.],device=device).expand(5,3))
        samples=[]
        for seed in (120,120,121):
            execution.reset(5)
            with diagnostic.policy_rng(device,seed):
                samples.append(execution.decide(observation,alive,'stochastic')[1])
        assert torch.equal(samples[0],samples[1])
        assert not torch.equal(samples[0],samples[2])
    finally:execution.close()
    assert not actor.log_std._forward_hooks

def test_dead_agents_excluded_from_variance_statistics(device):
    sigma=torch.tensor([[1.,2.,3.],[100.,100.,100.]],device=device)
    distribution=Normal(torch.zeros_like(sigma),sigma)
    stats=diagnostic.VarianceStatistics()
    stats.add(distribution,sigma.log(),distribution.mean,torch.tensor([1.,0.],device=device))
    result=stats.summarize()
    for dim,name in enumerate(diagnostic.ACTION_DIMENSIONS):
        assert result[name]['live_decisions']==1
        assert result[name]['mean_sigma']==float(dim+1)
        assert result[name]['mean_absolute_sampled_raw_action_minus_mu']==0

def test_stea_episode_and_death_reset_match_actor(device):
    actor=SpatioTemporalEntityAttentionActor().to(device)
    execution=diagnostic.PolicyExecution(actor,True,device)
    observation=np.ones((5,65),np.float32);alive=np.ones(5,np.float32)
    try:
        execution.reset(5);execution.hidden.fill_(17.)
        first=execution.decide(observation,alive,'deterministic')[0]
        execution.reset(5)
        second=execution.decide(observation,alive,'deterministic')[0]
        assert torch.equal(first,second)
        dead=alive.copy();dead[1]=0
        execution.after_transition(dead,False)
        assert not execution.hidden[1].any()
        actions=execution.decide(observation,dead,'deterministic')[0]
        assert not actions[1].any() and not execution.hidden[1].any()
        execution.after_transition(dead,True)
        assert not execution.hidden.any()
        execution.reset(5)
        assert torch.equal(first,execution.decide(observation,alive,'deterministic')[0])
    finally:execution.close()


class TwoStepEnvironment:
    """Only exercises the evaluator loop; no real combat audit or rollout."""
    team_size=5
    def reset(self,seed):
        self.steps=0;self.red_alive_mask=np.ones(5,np.float32)
        self.observation=np.random.default_rng(seed).normal(size=(5,65)).astype(np.float32)
        return self.observation,{}
    def step(self,actions):
        self.steps+=1;self.red_alive_mask[1]=0
        info=dict(red_success=False,blue_win=False,draw=False,termination_reason='red_failure_timeout',
                  episode_length=self.steps,red_losses=1,blue_losses=0,red_survivors=4,blue_survivors=5)
        for side in ('red','blue'):
            for key in ('fire_attempts','weapon_hits','attack_kills','boundary_exits','ground_losses'):info[f'{side}_{key}']=0
            for key in ('fire_window','attempt','hit','kill'):info[f'{side}_first_{key}_step']=None
        for key in ('r1','r2','r3','r4'):info[f'episode_{key}_total']=0.
        return self.observation,actions.sum(-1),self.steps==2,False,info

@pytest.mark.parametrize('recurrent',[False,True])
def test_modes_repeated_results_and_separate_episode_rng(device,monkeypatch,recurrent):
    monkeypatch.setattr(diagnostic,'make_combat_environment',lambda cfg:TwoStepEnvironment())
    actor=(SpatioTemporalEntityAttentionActor() if recurrent else SharedMAPPOActor(65)).to(device)
    for mode in ('deterministic','stochastic'):
        first=diagnostic.evaluate_mode(actor,recurrent,{},[50000000,50000001],device,mode,700)
        torch.rand(97,device=device)  # unrelated startup RNG must not matter
        second=diagnostic.evaluate_mode(actor,recurrent,{},[50000000,50000001],device,mode,700)
        assert first==second
        changed=diagnostic.evaluate_mode(actor,recurrent,{},[50000000,50000001],device,mode,701)
        if mode=='deterministic':assert first==changed
        else:assert first['episodes'][0]['executed_action_sha256']!=changed['episodes'][0]['executed_action_sha256']
        assert first['policy_statistics']['heading']['live_decisions']==18
        assert [r['environment_seed'] for r in first['episodes']]==[50000000,50000001]

@pytest.mark.parametrize('corruption',['algorithm_hash','environment_hash','implementation','cross_algorithm'])
def test_checkpoint_contract_is_strict(device,tmp_path,corruption):
    env=load_config(ROOT/'configs/combat_environment_v25.yaml')
    cfg=yaml.safe_load((ROOT/'configs/mappo_5v5.yaml').read_text())
    trainer=build_mappo_trainer(cfg,device)
    extra=dict(environment_version='2.5',observation_dim=65,action_dim=3,num_agents=5,
               environment_config_sha256=config_sha256(env),algorithm_config_sha256=config_sha256(cfg),
               training_seed=31,training_total_sampled_steps=512,training_num_envs=2,effective_hidden_dim=256,
               training_gamma=.99,training_smoke=False)
    state=trainer.checkpoint_state(extra)
    algorithm='mappo'
    if corruption=='algorithm_hash':extra['algorithm_config_sha256']='wrong'
    if corruption=='environment_hash':extra['environment_config_sha256']='wrong'
    if corruption=='implementation':state['mappo_impl_version']=-1
    if corruption=='cross_algorithm':
        algorithm='stea-mappo';cfg=yaml.safe_load((ROOT/'configs/stea_mappo_5v5.yaml').read_text())
    checkpoint=tmp_path/'checkpoint.pt';torch.save(state,checkpoint)
    with pytest.raises(RuntimeError):diagnostic.load_policy(algorithm,checkpoint,env,cfg,device)
