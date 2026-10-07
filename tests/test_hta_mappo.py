from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modules import ManagerTransitionBatch,compute_smdp_gae,discounted_macro_reward,smdp_boundary_masks


def configs(hidden=16):
    plain=load_config("configs/diag_mappo_learnability_common_3m.yaml")
    hta=load_config("configs/dev_hta_mappo_v1_3m.yaml")
    for cfg in (plain,hta):
        cfg["network"]["actor_hidden_layers"]=[hidden,hidden]
        cfg["network"]["critic_hidden_layers"]=[hidden,hidden]
    return plain,hta


def trainers(seed=5301,hidden=16):
    plain_cfg,hta_cfg=configs(hidden)
    for cfg in (plain_cfg,hta_cfg):cfg["training"]["seed"]=seed
    return (build_modular_mappo_trainer(plain_cfg,"cpu",hidden,1000),
            build_modular_mappo_trainer(hta_cfg,"cpu",hidden,1000))


def test_initial_worker_and_tactical_critic_are_plain_equivalent():
    plain,hta=trainers()
    for name,value in plain.actor.state_dict().items():
        assert torch.equal(value,hta.actor.state_dict()[name])
    for name,value in plain.critic.state_dict().items():
        assert torch.equal(value,hta.critic.state_dict()[name])
    assert all(torch.count_nonzero(p)==0 for head in hta.actor.option_mean_residuals for p in head.parameters())
    assert torch.count_nonzero(hta.critic.mission_context_projection)==0
    obs=torch.randn(2,4,52);alive=torch.ones(2,4)
    plain_dist,_=plain.actor.distribution_step(obs,alive_mask=alive)
    for option in range(4):
        ids=torch.full((2,4),option)
        dist,_=hta.actor.distribution_step(obs,alive_mask=alive,option_ids=ids)
        assert torch.equal(dist.mean,plain_dist.mean)
        assert torch.equal(dist.stddev,plain_dist.stddev)
        pv,_=plain.critic.forward_step(obs,alive)
        hv,_=hta.critic.forward_step(obs,alive,torch.nn.functional.one_hot(ids,4).float())
        assert torch.equal(hv,pv)


def test_manager_uniform_and_sampling_does_not_advance_worker_rng():
    plain,hta=trainers(seed=71)
    obs=np.zeros((2,4,52),np.float32);alive=np.ones((2,4),np.float32)
    probs=hta.manager_actor.probabilities(torch.as_tensor(obs))
    assert torch.allclose(probs,torch.full_like(probs,.25))
    before=torch.get_rng_state().clone();hta.manager_act(obs,alive,False);after=torch.get_rng_state()
    assert torch.equal(before,after)
    state=torch.get_rng_state().clone();torch.set_rng_state(state);pa,pr,pl,_=plain.act(obs,alive,False,True)
    torch.set_rng_state(state);ha,hr,hl,_=hta.act(obs,alive,False,True,option_ids=np.full((2,4),3))
    assert np.array_equal(pa,ha) and np.array_equal(pr,hr) and np.array_equal(pl,hl)


def test_option_residual_and_tactical_projection_are_controllable():
    _,hta=trainers(seed=9)
    obs=torch.randn(2,4,52);alive=torch.ones(2,4);z0=torch.zeros(2,4,dtype=torch.long);z2=torch.full((2,4),2)
    d0,_=hta.actor.distribution_step(obs,alive_mask=alive,option_ids=z0)
    with torch.no_grad():hta.actor.option_mean_residuals[2].bias.fill_(.2)
    d2,_=hta.actor.distribution_step(obs,alive_mask=alive,option_ids=z2)
    d0_after,_=hta.actor.distribution_step(obs,alive_mask=alive,option_ids=z0)
    assert torch.equal(d0.mean,d0_after.mean) and not torch.equal(d0.mean,d2.mean)
    c0=torch.nn.functional.one_hot(z0,4).float();c2=torch.nn.functional.one_hot(z2,4).float()
    v0,_=hta.critic.forward_step(obs,alive,c0);v2,_=hta.critic.forward_step(obs,alive,c2)
    assert torch.equal(v0,v2)
    with torch.no_grad():hta.critic.mission_context_projection[:,2].fill_(.5)
    v2_changed,_=hta.critic.forward_step(obs,alive,c2)
    assert not torch.equal(v0,v2_changed)


def test_discounted_macro_reward_and_exact_smdp_gae_masks():
    assert np.isclose(discounted_macro_reward(np.asarray([1.,2.,3.]),.9),1+.9*2+.9**2*3)
    reward=np.asarray([[1.],[2.],[3.]],np.float32);value=np.asarray([[.5],[.6],[.7]],np.float32)
    next_value=np.asarray([[.6],[.7],[.8]],np.float32);duration=np.asarray([2,3,1])
    bootstrap=np.ones((3,1),np.float32);trace=np.asarray([[1.],[0.],[0.]],np.float32)
    advantage,returns,delta=compute_smdp_gae(reward,value,next_value,duration,bootstrap,trace,.9,.95)
    expected_delta=reward+np.power(.9,duration)[:,None]*next_value-value
    expected=np.empty_like(delta);expected[2]=expected_delta[2];expected[1]=expected_delta[1]
    expected[0]=expected_delta[0]+.9**2*.95*expected[1]
    assert np.allclose(delta,expected_delta) and np.allclose(advantage,expected) and np.allclose(returns,expected+value)
    # Rollout truncation bootstraps but trace=0; terminal does neither.
    trunc_adv,_,_=compute_smdp_gae([[1.]],[[.5]],[[2.]],[3],[[1.]],[[0.]],.9,.95)
    terminal_adv,_,_=compute_smdp_gae([[1.]],[[.5]],[[2.]],[3],[[0.]],[[0.]],.9,.95)
    assert np.isclose(trunc_adv[0,0],1+.9**3*2-.5)
    assert np.isclose(terminal_adv[0,0],.5)
    alive=np.asarray([1,0,1,1],np.float32)
    wave_bootstrap,wave_trace=smdp_boundary_masks(alive,episode_terminal=False,rollout_truncation=False)
    rollout_bootstrap,rollout_trace=smdp_boundary_masks(alive,episode_terminal=False,rollout_truncation=True)
    terminal_bootstrap,terminal_trace=smdp_boundary_masks(alive,episode_terminal=True,rollout_truncation=False)
    assert np.array_equal(wave_bootstrap,alive) and np.array_equal(wave_trace,alive)
    assert np.array_equal(rollout_bootstrap,alive) and not rollout_trace.any()
    assert not terminal_bootstrap.any() and not terminal_trace.any()


def test_manager_and_worker_optimizer_steps_are_strictly_separate():
    _,hta=trainers(seed=13);obs=torch.randn(2,4,52);alive=torch.ones(2,4);options=torch.zeros(2,4,dtype=torch.long)
    worker_before={k:v.clone() for k,v in hta.actor.state_dict().items()}
    manager_loss=-hta.manager_actor.distribution(obs).log_prob(options).mean()
    hta.manager_actor_optimizer.zero_grad();manager_loss.backward();hta.manager_actor_optimizer.step()
    assert all(torch.equal(value,hta.actor.state_dict()[key]) for key,value in worker_before.items())
    manager_before={k:v.clone() for k,v in hta.manager_actor.state_dict().items()}
    worker_loss=hta.actor.distribution(obs,option_ids=options).mean.sum()
    hta.actor_optimizer.zero_grad();worker_loss.backward();hta.actor_optimizer.step()
    assert all(torch.equal(value,hta.manager_actor.state_dict()[key]) for key,value in manager_before.items())


def synthetic_batch(trainer,T=4,E=2):
    rng=np.random.default_rng(2);A=4
    obs=rng.normal(size=(T,E,A,52)).astype(np.float32);nobs=rng.normal(size=(T,E,A,52)).astype(np.float32)
    alive=np.ones((T,E,A),np.float32);options=rng.integers(0,4,size=(T,E,A),dtype=np.int64)
    next_options=np.roll(options,-1,axis=0);actions=[];raw=[];logs=[]
    for t in range(T):
        a,r,l,_=trainer.act(obs[t],alive[t],False,True,option_ids=options[t]);actions.append(a);raw.append(r);logs.append(l)
    rewards=rng.normal(scale=.1,size=(T,E,A)).astype(np.float32);zero_context=np.zeros((T,E,0),np.float32)
    manager=ManagerTransitionBatch(
        observations=np.asarray([obs[0,0],obs[0,1],obs[2,0],obs[2,1]]),alive_masks=np.ones((4,A),np.float32),
        options=np.asarray([options[0,0],options[0,1],options[2,0],options[2,1]]),old_log_probs=np.full((4,A),-np.log(4),np.float32),
        rewards=rng.normal(scale=.2,size=(4,A)).astype(np.float32),next_observations=np.asarray([nobs[1,0],nobs[1,1],nobs[3,0],nobs[3,1]]),
        next_alive_masks=np.ones((4,A),np.float32),durations=np.asarray([2,2,2,2]),bootstrap_masks=np.ones((4,A),np.float32),
        trace_masks=np.asarray([[1]*A,[1]*A,[0]*A,[0]*A],np.float32),env_ids=np.asarray([0,1,0,1]),
        end_reasons=np.asarray(["periodic","wave_transition","rollout_truncation","rollout_truncation"]))
    return ModularRolloutBatch(obs,np.asarray(actions),np.asarray(raw),np.asarray(logs),rewards,rewards.copy(),
        np.zeros((T,E),np.float32),alive,nobs,alive.copy(),np.ones((T,E),np.int64),np.full((T,E),3,np.int64),
        zero_context,zero_context.copy(),hta_options=options,hta_next_options=next_options,hta_manager_transitions=manager)


def test_active_update_is_finite_and_manager_worker_parameters_are_separate():
    _,hta=trainers(seed=17);batch=synthetic_batch(hta)
    worker_before={k:v.clone() for k,v in hta.actor.state_dict().items()};manager_before={k:v.clone() for k,v in hta.manager_actor.state_dict().items()}
    metrics=hta.update(batch)
    assert all(np.isfinite(value) for value in metrics.values())
    assert hta.actor_update_count>0 and hta.critic_update_count>0 and hta.manager_actor_update_count>0 and hta.manager_critic_update_count>0
    assert any(not torch.equal(worker_before[k],v) for k,v in hta.actor.state_dict().items())
    assert any(not torch.equal(manager_before[k],v) for k,v in hta.manager_actor.state_dict().items())
    assert not set(map(id,hta.actor.parameters())) & set(map(id,hta.manager_actor.parameters()))


def test_checkpoint_roundtrip_restores_manager_rng_and_outputs(tmp_path:Path):
    _,hta=trainers(seed=22);hta.update(synthetic_batch(hta));path=tmp_path/"hta.pt";hta.save(path)
    _,restored=trainers(seed=22);restored.load(path)
    obs=torch.randn(2,4,52);alive=torch.ones(2,4);options=torch.full((2,4),2)
    assert torch.equal(hta.manager_actor.logits(obs),restored.manager_actor.logits(obs))
    assert torch.equal(hta.actor.distribution(obs,option_ids=options).mean,restored.actor.distribution(obs,option_ids=options).mean)
    context=torch.nn.functional.one_hot(options,4).float()
    assert torch.equal(hta.critic(obs,alive,context=context),restored.critic(obs,alive,context=context))
    assert torch.equal(hta.manager_critic(obs,alive),restored.manager_critic(obs,alive))
    assert hta.hta_rng.bit_generator.state==restored.hta_rng.bit_generator.state
    assert torch.equal(hta.hta_manager_generator.get_state(),restored.hta_manager_generator.get_state())
