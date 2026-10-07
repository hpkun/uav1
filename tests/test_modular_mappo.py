import numpy as np
import pytest
import torch
from algorithm.mappo.trainer import MAPPOTrainer,RolloutBatch
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer,aggregate_update_rows,stable_ratio_terms
from algorithm.modular_mappo.buffer import ModularRolloutBatch,contiguous_chunks,recurrent_batch_plan,recurrent_alive_mean

def data(seed=4):
 r=np.random.default_rng(seed);T,E,A,F=4,2,4,52;obs=r.normal(size=(T,E,A,F)).astype("f");actions=np.tanh(r.normal(size=(T,E,A,3))).astype("f");raw=np.arctanh(np.clip(actions,-.999,.999));alive=np.ones((T,E,A),"f");dones=np.zeros((T,E),"f");rew=r.normal(size=(T,E,A)).astype("f")
 base=dict(observations=obs,actions=actions,raw_actions=raw,old_log_probs=np.zeros((T,E,A),"f"),rewards=rew,dones=dones,alive_masks=alive,next_observations=obs.copy(),next_alive_masks=alive.copy())
 return base
def test_all_off_network_initialization_and_actions_equal_baseline():
 b=MAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=8,seed=7);m=ModularMAPPOTrainer(hidden_dim=32,ppo_epochs=1,minibatch_size=8,seed=7,modules_config={});assert all(torch.equal(v,m.actor.state_dict()[k]) for k,v in b.actor.state_dict().items());assert all(torch.equal(v,m.critic.state_dict()[k]) for k,v in b.critic.state_dict().items());x=np.random.default_rng(1).normal(size=(2,4,52)).astype("f");assert np.allclose(b.act(x,deterministic=True),m.act(x,deterministic=True)[0])
def test_context_critic_only_actor_invariant():
 m=ModularMAPPOTrainer(hidden_dim=16,modules_config={"wave_context":{"enabled":True,"context_target":"critic_only"}});x=np.ones((1,4,52),"f");a1=m.act(x,np.ones((1,4)),True,context=m.context_numpy([1],[3]))[0];a3=m.act(x,np.ones((1,4)),True,context=m.context_numpy([3],[3]))[0];assert np.array_equal(a1,a3)
def test_recurrent_history_changes_output_and_chunks_contiguous():
 m=ModularMAPPOTrainer(hidden_dim=16,modules_config={"recurrent_memory":{"enabled":True,"hidden_dim":8,"sequence_length":2}});x=np.ones((1,4,52),"f");alive=np.ones((1,4),"f");h,_=m.initial_hidden(1);a1,h1=m.act(x,alive,True,hidden=h,episode_mask=np.ones(1));a2,_=m.act(x,alive,True,hidden=h1,episode_mask=np.ones(1));assert not np.allclose(a1,a2);assert contiguous_chunks(5,1,2)==[(0,0,2),(0,2,4),(0,4,5)]
def test_recurrent_optimizer_count_matches_baseline_scale():
 plan=recurrent_batch_plan(256,24,32,512,10);assert plan=={"sequence_chunks":192,"sequences_per_minibatch":16,"recurrent_minibatches_per_epoch":12,"optimizer_steps":120};assert int(np.ceil(256*24/512))==12
def test_recurrent_alive_mean_uses_all_agent_time_samples():
 values=torch.tensor([[[1.,2.,3.,4.]],[[10.,99.,99.,99.]]]);alive=torch.tensor([[[1.,1.,1.,1.]],[[1.,0.,0.,0.]]]);valid=torch.ones(2,1);assert torch.isclose(recurrent_alive_mean(values,alive,valid),torch.tensor(4.0))
def test_recurrent_update_groups_sequences_and_keeps_partial_chunk():
 rng=np.random.default_rng(8);T,E,A,F=5,4,4,52;obs=rng.normal(size=(T,E,A,F)).astype("f");actions=np.tanh(rng.normal(size=(T,E,A,3))).astype("f");raw=np.arctanh(np.clip(actions,-.999,.999));alive=np.ones((T,E,A),"f");rewards=rng.normal(size=(T,E,A)).astype("f");zeros=np.zeros((T,E),"f");hidden=np.zeros((T,E,A,8),"f");ctx=np.zeros((T,E,0),"f")
 rollout=ModularRolloutBatch(obs,actions,raw,np.zeros((T,E,A),"f"),rewards,rewards.copy(),zeros,alive,obs.copy(),alive.copy(),np.ones((T,E),int),np.ones((T,E),int),ctx,ctx,hidden,hidden,np.ones((T,E),"f"));trainer=ModularMAPPOTrainer(hidden_dim=16,ppo_epochs=1,minibatch_size=4,modules_config={"recurrent_memory":{"enabled":True,"hidden_dim":8,"sequence_length":2}});metrics=trainer.update(rollout);assert metrics["sequence_chunks"]==12 and metrics["sequences_per_minibatch"]==2 and metrics["recurrent_minibatches_per_epoch"]==6 and metrics["actor_optimizer_steps_this_update"]==6
def test_modular_update_checkpoint_protocol(tmp_path):
 d=data();m=ModularMAPPOTrainer(hidden_dim=16,ppo_epochs=1,minibatch_size=8,modules_config={});ctx=np.zeros((4,2,0),"f");r=ModularRolloutBatch(**d,raw_environment_rewards=d["rewards"].copy(),wave_indices=np.ones((4,2),int),total_waves=np.ones((4,2),int),contexts=ctx,next_contexts=ctx,episode_masks=np.ones((4,2),"f"));metrics=m.update(r);assert all(np.isfinite(list(metrics.values())));p=tmp_path/"x.pt";m.save(p);m.load(p);bad=ModularMAPPOTrainer(hidden_dim=16,modules_config={"popart":{"enabled":True}});import pytest
 with pytest.raises(RuntimeError):bad.load(p)

def test_all_off_update_is_numerically_baseline_equivalent():
 d=data(9);b=MAPPOTrainer(hidden_dim=16,ppo_epochs=1,minibatch_size=8,seed=3);m=ModularMAPPOTrainer(hidden_dim=16,ppo_epochs=1,minibatch_size=8,seed=3,modules_config={});obs=torch.as_tensor(d["observations"])
 with torch.no_grad():
  dist=b.actor.distribution(obs);d["old_log_probs"]=b.actor._squashed_log_prob(dist,torch.as_tensor(d["raw_actions"]),torch.as_tensor(d["actions"])).numpy()
 rb=RolloutBatch(**d);ctx=np.zeros((4,2,0),"f");rm=ModularRolloutBatch(**d,raw_environment_rewards=d["rewards"].copy(),wave_indices=np.ones((4,2),int),total_waves=np.ones((4,2),int),contexts=ctx,next_contexts=ctx,episode_masks=np.ones((4,2),"f"));state=torch.random.get_rng_state();bm=b.update(rb);torch.random.set_rng_state(state);mm=m.update(rm)
 for key in ("approx_kl","clip_fraction","ratio_mean","ratio_std","ratio_p1","ratio_p50","ratio_p99","ratio_min","ratio_max"):assert mm[key]==pytest.approx(bm[key],abs=1e-6,rel=1e-6)
 for p,q in zip(b.actor.parameters(),m.actor.parameters()):assert torch.allclose(p,q,atol=2e-6,rtol=2e-6)
 for p,q in zip(b.critic.parameters(),m.critic.parameters()):assert torch.allclose(p,q,atol=2e-6,rtol=2e-6)

def test_extreme_negative_log_ratio_underflow_keeps_kl_finite():
 new=torch.tensor([-120.0],dtype=torch.float32);old=torch.zeros_like(new);log_ratio,ratio=stable_ratio_terms(new,old)
 trainer=ModularMAPPOTrainer(hidden_dim=16,modules_config={});zero=torch.zeros(())
 row=trainer._row((zero,zero,zero,zero,ratio,log_ratio,new,old,None,None,0.0),torch.ones(1),torch.tensor(0.),torch.tensor(0.))
 metrics=aggregate_update_rows([row],.2)
 assert ratio.item()==0.0 and np.isfinite(metrics["approx_kl"])
 assert metrics["approx_kl"]==pytest.approx(119.0) and metrics["ratio_underflow_count"]==1 and metrics["ratio_underflow_fraction"]==1

def test_large_finite_positive_log_ratio_remains_finite():
 new=torch.tensor([80.0],dtype=torch.float32);log_ratio,ratio=stable_ratio_terms(new,torch.zeros_like(new))
 metrics=aggregate_update_rows([{"_valid_count":1,"_ratio_values":ratio.numpy(),"_log_ratio_values":log_ratio.numpy()}],.2)
 assert torch.isfinite(ratio).all() and np.isfinite(metrics["approx_kl"])
 assert metrics["log_ratio_max"]==80.0 and metrics["ratio_max"]>1e34

def test_mixed_ratio_samples_have_exact_global_underflow_and_extrema():
 new=torch.tensor([0.0,-120.0,1.0,-2.0],dtype=torch.float32);log_ratio,ratio=stable_ratio_terms(new,torch.zeros_like(new))
 rows=[{"entropy":2.0,"approx_kl":0.0,"clip_fraction":0.0,"_valid_count":1,"_ratio_values":ratio[:1].numpy(),"_log_ratio_values":log_ratio[:1].numpy()},
       {"entropy":4.0,"approx_kl":0.0,"clip_fraction":0.0,"_valid_count":3,"_ratio_values":ratio[1:].numpy(),"_log_ratio_values":log_ratio[1:].numpy()}]
 metrics=aggregate_update_rows(rows,.2);expected=((ratio.double()-1)-log_ratio.double()).mean().item()
 assert metrics["approx_kl"]==pytest.approx(expected) and metrics["entropy"]==pytest.approx(3.5)
 assert metrics["ratio_underflow_count"]==1 and metrics["ratio_underflow_fraction"]==pytest.approx(.25)
 assert metrics["ratio_min"]==0 and metrics["ratio_max"]==pytest.approx(np.exp(1.0))
 assert metrics["log_ratio_min"]==-120 and metrics["log_ratio_max"]==1 and metrics["max_abs_log_ratio"]==120
 assert metrics["ratio_p50"]==pytest.approx(np.quantile(ratio.numpy().astype(np.float64),.5))

def test_nonfinite_log_prob_and_log_ratio_are_rejected():
 with pytest.raises(FloatingPointError,match="new_log_prob"):stable_ratio_terms(torch.tensor([float("nan")]),torch.zeros(1))
 maximum=torch.finfo(torch.float32).max
 with pytest.raises(FloatingPointError,match="log_ratio"):stable_ratio_terms(torch.tensor([-maximum]),torch.tensor([maximum]))
