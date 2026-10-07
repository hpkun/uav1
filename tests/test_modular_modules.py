import copy
import numpy as np
import torch
from algorithm.modules import WaveContextModule,RecurrentMemoryModule,PopArtValueNormalizer,MultiWaveRewardAdapter,WaveBalancingModule,CurriculumController,PolicyAnchorRegularizer,WarmStartInitializer
from algorithm.modular_mappo.networks import ModularMAPPOActor
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer

def test_m1_wave_context_distinct():
 m=WaveContextModule({"enabled":True,"max_waves":3});x=m.encode_numpy([1,2,3],[3,3,3]);assert len({tuple(v) for v in x})==3
def test_m2_hidden_lifecycle_wave_does_not_reset():
 m=RecurrentMemoryModule({"enabled":True,"hidden_dim":8,"sequence_length":4});h=np.ones((2,4,8),np.float32);m.reset_for_episode(h,np.array([True,False]));assert not h[0].any() and h[1].all()
def test_m3_popart_preserves_unnormalized_output():
 p=PopArtValueNormalizer({"enabled":True,"beta":.9});layer=torch.nn.Linear(3,1);x=torch.randn(20,3);before=p.denormalize_values(layer(x)).detach();p.update(torch.randn(100)*10+20,layer);after=p.denormalize_values(layer(x)).detach();assert torch.allclose(before,after,atol=2e-5)
def test_m4_reward_raw_unchanged_and_clear_only():
 m=MultiWaveRewardAdapter({"enabled":True,"mode":"wave_clear_bonus","bonuses":{"wave1":1}});raw=np.zeros((2,4),np.float32);snapshot=raw.copy();train,metrics=m.adapt(raw,[{"wave_cleared_this_step":False,"wave_index":1},{"wave_cleared_this_step":True,"waves_cleared":1,"red_alive_mask":[1,1,0,0]}]);assert np.array_equal(raw,snapshot) and train[0].sum()==0 and train[1].sum()==2 and metrics["reward_bonus_total"]==2
def test_m4_uses_pre_transition_wave_for_clear_and_scale():
 raw=np.ones((1,4),np.float32);info=[{"wave_cleared_this_step":True,"waves_cleared":1,"wave_index":2,"red_alive_mask":[1,1,1,1]}]
 bonus=MultiWaveRewardAdapter({"enabled":True,"mode":"wave_clear_bonus","bonuses":{"wave1":2,"wave2":20}});training,_=bonus.adapt(raw,info,np.array([1]));assert np.all(training==3)
 scale=MultiWaveRewardAdapter({"enabled":True,"mode":"round_scaled","round_scales":{"wave1":3,"wave2":30}});training,_=scale.adapt(raw,info,np.array([1]));assert np.all(training==3)
def test_m5_inverse_frequency_and_direct_degeneracy():
 m=WaveBalancingModule({"enabled":True,"max_weight":3});w,met=m.compute_numpy(np.array([1]*1000+[2]*300+[3]*50));assert met["weight_wave_3"]>met["weight_wave_2"]>met["weight_wave_1"];w,_=m.compute_numpy(np.ones(25));assert np.all(w==1)
def test_m5_alive_agent_basis_cap_and_mean_preservation():
 wave=np.array([1,1,2,2,2]);alive=np.array([[1,1,1,1],[1,1,1,1],[1,0,0,0],[1,0,0,0],[1,0,0,0]],np.float32)
 m=WaveBalancingModule({"enabled":True,"frequency_basis":"alive_agent","max_weight":2});weights,metrics=m.compute_numpy(wave,alive);assert metrics["alive_agent_samples_wave_1"]==8 and metrics["alive_agent_samples_wave_2"]==3
 assert metrics["weight_wave_2"]>metrics["weight_wave_1"] and max(metrics["weight_wave_1"],metrics["weight_wave_2"])<=2 and abs(metrics["effective_wave_weight_mean"]-1)<1e-10
def test_m7_curriculum_does_not_mutate_source():
 m=CurriculumController({"enabled":True,"stage1_end":10,"stage2_end":20});source={"persistent_waves":{"total_waves":3}};original=copy.deepcopy(source);assert [m.runtime_config(source,s)["persistent_waves"]["total_waves"] for s in (0,10,20)]==[1,2,3] and source==original
def test_m7_disabled_curriculum_is_exact_noop_at_all_steps():
 m=CurriculumController({"enabled":False,"stage1_end":500_000,"stage2_end":1_000_000})
 source={"simulation":{"max_steps":1000},"persistent_waves":{"total_waves":1},"reward":{"kill_reward":10}}
 original=copy.deepcopy(source)
 for step in (0,500_000,1_000_000,3_000_000,9_999_999):
  effective=m.runtime_config(source,step)
  assert effective==source and effective is not source and effective["persistent_waves"] is not source["persistent_waves"]
  assert m.stage(step)==(0,0)
 assert source==original
def test_m8_anchor_zero_then_positive_and_frozen():
 torch.manual_seed(0);cur=ModularMAPPOActor(5,2,8);ref=copy.deepcopy(cur);a=PolicyAnchorRegularizer({"enabled":True,"coefficient":.1});a.attach(ref);x=torch.randn(4,5);loss,m=a.loss(cur.distribution(x),ref.distribution(x),0,torch.ones(4));assert m["anchor_kl"]<1e-7 and all(not p.requires_grad for p in ref.parameters());cur.mean.bias.data.add_(.2);_,m=a.loss(cur.distribution(x),ref.distribution(x),0,torch.ones(4));assert m["anchor_kl"]>0

def test_m6_warm_start_exact_actor_behavior(tmp_path):
 source=ModularMAPPOTrainer(hidden_dim=16,seed=11);target=ModularMAPPOTrainer(hidden_dim=16,seed=12,modules_config={"warm_start":{"enabled":True,"mode":"actor_only"}});path=tmp_path/"source.pt";source.save(path);report=WarmStartInitializer({"enabled":True,"mode":"actor_only"}).initialize(target,str(path));x=np.random.default_rng(2).normal(size=(2,4,52)).astype("f");a=source.act(x,np.ones((2,4)),True)[0];b=target.act(x,np.ones((2,4)),True)[0];assert np.array_equal(a,b) and report["actor"]["not_loaded"]==[]
