from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.networks import ModularMAPPOActor
from algorithm.train_modular_mappo import load_config

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"outputs/formal_eawb/mappo_seed3101/latest.pt"

def observations(*shape,device="cpu"):
 x=torch.randn(*shape,52,device=device)
 x[...,13]=1;x[...,20]=0;x[...,27]=1;x[...,33]=1;x[...,39]=0;x[...,45]=1;x[...,51]=0
 return x

def actor():
 return ModularMAPPOActor(entity_attention_config={"enabled":True,"mode":"frozen_base_mean_residual","entity_dim":32,"attention_heads":2,"max_mean_correction":.25})

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def assert_optimizer_state_equal(left,right):
 assert left["param_groups"]==right["param_groups"]
 assert left["state"].keys()==right["state"].keys()
 for key in left["state"]:
  assert left["state"][key].keys()==right["state"][key].keys()
  for field,value in left["state"][key].items():
   actual=right["state"][key][field]
   assert torch.equal(value.cpu(),actual.cpu()) if torch.is_tensor(value) else value==actual

def synthetic_rollout(trainer):
 rng=np.random.default_rng(901);t,e,a,f=3,2,4,52
 obs=rng.normal(size=(t,e,a,f)).astype("f");alive=np.ones((t,e,a),"f")
 for index in (13,27,33,45):obs[...,index]=1
 for index in (20,39,51):obs[...,index]=0
 raw=rng.normal(size=(t,e,a,3)).astype("f");actions=np.tanh(raw).astype("f")
 with torch.no_grad():
  dist=trainer.actor.distribution(torch.as_tensor(obs,device=trainer.device));old=trainer.actor._squashed_log_prob(dist,torch.as_tensor(raw,device=trainer.device),torch.as_tensor(actions,device=trainer.device)).cpu().numpy()
 rewards=rng.normal(size=(t,e,a)).astype("f");ctx=np.zeros((t,e,0),"f")
 return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),np.zeros((t,e),"f"),alive,obs.copy(),alive.copy(),np.ones((t,e),int),np.full((t,e),3),ctx,ctx,episode_masks=np.ones((t,e),"f"))

def test_fbmr_formula_zero_init_bound_and_logstd_isolation():
 torch.manual_seed(3);baseline=ModularMAPPOActor();torch.manual_seed(3);model=actor();obs=observations(5,4)
 base=baseline.distribution(obs);full,diag=model.distribution(obs,return_attention=True)
 assert torch.equal(base.mean,full.mean) and torch.equal(base.stddev,full.stddev)
 assert torch.count_nonzero(diag["entity_delta_mu"])==0
 assert torch.count_nonzero(model.entity_mean_adapter.weight)==0 and torch.count_nonzero(model.entity_mean_adapter.bias)==0
 assert all(not p.requires_grad for _,p in model.frozen_baseline_named_parameters())
 assert all(p.requires_grad for p in model.trainable_policy_parameters())
 with torch.no_grad():model.entity_mean_adapter.weight.fill_(1e6);model.entity_mean_adapter.bias.fill_(1e6)
 full,diag=model.distribution(obs,return_attention=True)
 assert torch.all(diag["entity_delta_mu"].abs()<=.25)
 base_logstd=model.log_std(model.backbone(obs)).clamp(model.log_std_min,model.log_std_max)
 assert torch.equal(full.scale,base_logstd.exp())

@pytest.mark.skipif(not torch.cuda.is_available() or not SOURCE.exists(),reason="CUDA/source checkpoint unavailable")
def test_source_exact_policy_branch_freeze_update_and_round_trip(tmp_path):
 config=load_config(ROOT/"configs/dev_fbmr_mean_residual_1200k.yaml");source_state=torch.load(SOURCE,map_location="cuda",weights_only=False);before_hash=digest(SOURCE)
 trainer=build_modular_mappo_trainer(config,"cuda");extra=trainer.load_fbmr_branch(SOURCE,before_hash,restore_rng=True)
 baseline=ModularMAPPOActor().cuda();baseline.load_state_dict(source_state["actor"]);baseline.eval();obs=observations(12,4,device="cuda")
 with torch.no_grad():base=baseline.distribution(obs);full,diag=trainer.actor.distribution(obs,return_attention=True)
 assert torch.equal(base.mean,full.mean) and torch.equal(base.stddev,full.stddev) and torch.equal(torch.tanh(base.mean),torch.tanh(full.mean))
 assert torch.count_nonzero(diag["entity_delta_mu"])==0 and extra["training_seed"]==3101
 assert not trainer.actor_optimizer.state and trainer.fbmr_branch_metadata["actor_optimizer_restore"] is False
 assert trainer.fbmr_branch_metadata["critic_optimizer_restore"] is True and trainer.fbmr_branch_metadata["source_rng_restored"] is True
 assert_optimizer_state_equal(source_state["critic_optimizer"],trainer.critic_optimizer.state_dict())
 optimized={id(p) for g in trainer.actor_optimizer.param_groups for p in g["params"]};assert optimized=={id(p) for p in trainer.actor.trainable_policy_parameters()}
 frozen_before={n:p.detach().clone() for n,p in trainer.actor.frozen_baseline_named_parameters()};critic_before={n:p.detach().clone() for n,p in trainer.critic.named_parameters()}
 batch=synthetic_rollout(trainer);first=trainer.update(batch)
 assert np.isfinite(list(first.values())).all() and torch.count_nonzero(trainer.actor.entity_mean_adapter.weight)>0
 for key in ("entity_delta_mu_abs_mean","entity_delta_mu_rms","entity_delta_mu_abs_max","entity_delta_mu_heading_abs_mean","entity_delta_mu_pitch_abs_mean","entity_delta_mu_speed_abs_mean","entity_delta_mu_bound_fraction","base_mean_abs_mean","final_mean_abs_mean","base_std_mean","final_std_mean","entity_delta_logstd_abs_max","frozen_base_parameter_drift_max","frozen_mean_head_parameter_drift_max","frozen_logstd_head_parameter_drift_max"):
  assert key in first and np.isfinite(first[key])
 assert first["entity_delta_logstd_abs_max"]==0 and first["base_std_mean"]==first["final_std_mean"]
 trainer.update(synthetic_rollout(trainer))
 assert all(torch.equal(frozen_before[n],p) for n,p in trainer.actor.frozen_baseline_named_parameters())
 assert any(not torch.equal(critic_before[n],p) for n,p in trainer.critic.named_parameters())
 entity_grads=[p.grad for n,p in trainer.actor.named_parameters() if n.startswith(("self_encoder","ally_encoder","enemy_encoder","ally_attention","enemy_attention","entity_fusion"))]
 assert any(g is not None and torch.isfinite(g).all() and torch.count_nonzero(g)>0 for g in entity_grads)
 with torch.no_grad():after=trainer.actor.distribution(obs);base_after=trainer.actor.log_std(trainer.actor.backbone(obs)).clamp(trainer.actor.log_std_min,trainer.actor.log_std_max).exp()
 assert torch.equal(after.stddev,base_after) and trainer.frozen_actor_drift_metrics()["frozen_base_parameter_drift_max"]==0
 checkpoint=tmp_path/"fbmr.pt";trainer.save(checkpoint,{"network_architecture":{},"training_seed":3101})
 restored=build_modular_mappo_trainer(config,"cuda");restored.load(checkpoint)
 with torch.no_grad():actual=restored.actor.distribution(obs)
 assert torch.equal(after.mean,actual.mean) and torch.equal(after.stddev,actual.stddev)
 assert digest(SOURCE)==before_hash

@pytest.mark.skipif(not torch.cuda.is_available() or not SOURCE.exists(),reason="CUDA/source checkpoint unavailable")
def test_fbmr_construction_does_not_shift_restored_source_rng_stream():
 source_config=torch.load(SOURCE,map_location="cpu",weights_only=False)["extra"]["algorithm_config"]
 control=build_modular_mappo_trainer(source_config,"cuda");control.load(SOURCE,strict_protocol=False,restore_rng=True)
 expected_torch=torch.rand(8,device="cuda");expected_permutation=control.rng.random(8)
 fbmr=build_modular_mappo_trainer(load_config(ROOT/"configs/dev_fbmr_mean_residual_1200k.yaml"),"cuda")
 fbmr.load_fbmr_branch(SOURCE,digest(SOURCE),restore_rng=True)
 assert torch.equal(torch.rand(8,device="cuda"),expected_torch)
 assert np.array_equal(fbmr.rng.random(8),expected_permutation)

def test_old_modes_and_disabled_topologies_remain_compatible():
 obs=observations(2,4)
 for mode in (None,"replacement","residual","gated_residual"):
  cfg={"enabled":mode is not None,"entity_dim":32,"attention_heads":2}
  if mode is not None:cfg["mode"]=mode
  if mode=="gated_residual":cfg["initial_gate"]=.05
  model=ModularMAPPOActor(entity_attention_config=cfg);dist=model.distribution(obs)
  assert torch.isfinite(dist.mean).all() and torch.isfinite(dist.stddev).all()

def test_fbmr_invalid_bound_rejected():
 for value in (0,-.1,1.01):
  with pytest.raises(ValueError,match="max_mean_correction"):
   ModularMAPPOActor(entity_attention_config={"enabled":True,"mode":"frozen_base_mean_residual","max_mean_correction":value})
