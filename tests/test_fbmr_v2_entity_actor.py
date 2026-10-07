from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.networks import ModularMAPPOActor
from algorithm.modular_mappo.protocol import checkpoint_architecture,validate_fbmr_v1_v2_only_bound_diff,validate_fbmr_v2_stage2_branch
from algorithm.train_modular_mappo import load_config

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"outputs/formal_eawb/mappo_seed3101/latest.pt"
V1=ROOT/"configs/dev_fbmr_mean_residual_1200k.yaml"
V2=ROOT/"configs/dev_fbmr_v2_dual_bound_1200k.yaml"

def obs(*shape,device="cpu"):
 x=torch.randn(*shape,52,device=device)
 for i in (13,27,33,45):x[...,i]=1
 for i in (20,39,51):x[...,i]=0
 return x

def actor(mode="frozen_base_dual_bounded_mean_residual"):
 cfg={"enabled":True,"mode":mode,"entity_dim":32,"attention_heads":2}
 if mode=="frozen_base_mean_residual":cfg["max_mean_correction"]=.25
 else:cfg.update(alpha_abs=.25,alpha_rel=.25)
 return ModularMAPPOActor(entity_attention_config=cfg)

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def synthetic_rollout(trainer):
 rng=np.random.default_rng(902);t,e,a,f=3,2,4,52
 observations=rng.normal(size=(t,e,a,f)).astype("f");alive=np.ones((t,e,a),"f")
 for i in (13,27,33,45):observations[...,i]=1
 for i in (20,39,51):observations[...,i]=0
 raw=rng.normal(size=(t,e,a,3)).astype("f");actions=np.tanh(raw).astype("f")
 with torch.no_grad():
  dist=trainer.actor.distribution(torch.as_tensor(observations,device=trainer.device))
  old=trainer.actor._squashed_log_prob(dist,torch.as_tensor(raw,device=trainer.device),torch.as_tensor(actions,device=trainer.device)).cpu().numpy()
 rewards=rng.normal(size=(t,e,a)).astype("f");ctx=np.zeros((t,e,0),"f")
 return ModularRolloutBatch(observations,actions,raw,old,rewards,rewards.copy(),np.zeros((t,e),"f"),alive,observations.copy(),alive.copy(),np.ones((t,e),int),np.full((t,e),3),ctx,ctx,episode_masks=np.ones((t,e),"f"))

def test_v1_formula_unchanged_and_v2_has_identical_parameter_set():
 torch.manual_seed(19);v1=actor("frozen_base_mean_residual");v2=actor();x=obs(9,4)
 with torch.no_grad():
  v1.entity_mean_adapter.weight.normal_();v1.entity_mean_adapter.bias.normal_()
  _,d=v1.distribution(x,return_attention=True)
  expected=.25*torch.tanh(d["entity_raw_delta_mu"])
 assert torch.equal(d["entity_delta_mu"],expected)
 assert {n for n,p in v1.named_parameters() if p.requires_grad}=={n for n,p in v2.named_parameters() if p.requires_grad}
 assert sum(p.numel() for p in v1.trainable_policy_parameters())==sum(p.numel() for p in v2.trainable_policy_parameters())

def test_dual_bound_across_base_std_and_kl_limit():
 model=actor();x=obs(7,4).requires_grad_(True)
 with torch.no_grad():
  model.log_std.weight.zero_();model.entity_mean_adapter.weight.zero_();model.entity_mean_adapter.bias.fill_(1e6)
  for sigma in (.01,.05,.2,.5,1.,2.,5.):
   model.log_std.bias.fill_(float(np.log(sigma)))
   dist,d=model.distribution(x,return_attention=True);expected=min(.25,.25*sigma)
   assert d["entity_dual_scale"].requires_grad is False
   assert torch.allclose(d["entity_dual_scale"],torch.full_like(d["entity_dual_scale"],expected),rtol=1e-5,atol=1e-7)
   delta=d["entity_delta_mu"]
   assert float(delta.abs().max())<=.25+1e-7
   assert float((delta.abs()/dist.stddev).max())<=.25+1e-6
   assert float((.5*(delta/dist.stddev).square().sum(-1)).max())<=.09375+1e-6
   assert torch.equal(dist.stddev,model.log_std(model.backbone(x)).clamp(model.log_std_min,model.log_std_max).exp())

def test_invalid_dual_config_and_v1_field_leak_rejected():
 for key in ("alpha_abs","alpha_rel"):
  for value in (0.,-.1,1.01):
   cfg={"enabled":True,"mode":"frozen_base_dual_bounded_mean_residual","entity_dim":32,"attention_heads":2,"alpha_abs":.25,"alpha_rel":.25};cfg[key]=value
   with pytest.raises(ValueError,match=key):ModularMAPPOActor(entity_attention_config=cfg)
 with pytest.raises(ValueError,match="does not accept"):
  ModularMAPPOActor(entity_attention_config={"enabled":True,"mode":"frozen_base_dual_bounded_mean_residual","max_mean_correction":.25})

@pytest.mark.skipif(not torch.cuda.is_available() or not SOURCE.exists(),reason="CUDA/source checkpoint unavailable")
def test_v2_source_equivalence_freeze_update_diagnostics_and_roundtrip(tmp_path):
 config=load_config(V2);source=torch.load(SOURCE,map_location="cuda",weights_only=False);before=digest(SOURCE)
 trainer=build_modular_mappo_trainer(config,"cuda");extra=trainer.load_fbmr_branch(SOURCE,before,restore_rng=True)
 baseline=ModularMAPPOActor().cuda();baseline.load_state_dict(source["actor"]);x=obs(12,4,device="cuda")
 with torch.no_grad():base=baseline.distribution(x);full,d=trainer.actor.distribution(x,return_attention=True)
 assert torch.equal(base.mean,full.mean) and torch.equal(base.stddev,full.stddev) and torch.equal(torch.tanh(base.mean),torch.tanh(full.mean))
 assert torch.count_nonzero(d["entity_delta_mu"])==0
 assert d["entity_dual_scale"].requires_grad is False
 assert extra["training_seed"]==3101 and not trainer.actor_optimizer.state
 assert trainer.fbmr_branch_metadata["critic_optimizer_restored"] and trainer.fbmr_branch_metadata["RNG_restored_from_source"]
 frozen={n:p.detach().clone() for n,p in trainer.actor.frozen_baseline_named_parameters()};critic={n:p.detach().clone() for n,p in trainer.critic.named_parameters()}
 first=trainer.update(synthetic_rollout(trainer));trainer.update(synthetic_rollout(trainer))
 required=("entity_dual_scale_mean","entity_dual_scale_min","entity_dual_scale_max","entity_delta_mu_over_sigma_abs_mean","entity_delta_mu_over_sigma_abs_max","entity_source_relative_kl_mean","entity_source_relative_kl_max","relative_scale_active_fraction","dual_bound_effective_fraction","relative_saturation_fraction")
 assert all(k in first and np.isfinite(first[k]) for k in required)
 assert first["entity_delta_mu_over_sigma_abs_max"]<=.25+1e-6 and first["entity_source_relative_kl_max"]<=.09375+1e-6
 assert all(torch.equal(frozen[n],p) for n,p in trainer.actor.frozen_baseline_named_parameters())
 assert any(not torch.equal(critic[n],p) for n,p in trainer.critic.named_parameters())
 assert trainer.frozen_actor_drift_metrics()["frozen_base_parameter_drift_max"]==0 and first["entity_delta_logstd_abs_max"]==0
 entity_grads=[p.grad for n,p in trainer.actor.named_parameters() if n.startswith(("self_encoder","ally_encoder","enemy_encoder","ally_attention","enemy_attention","entity_fusion"))]
 assert any(g is not None and torch.isfinite(g).all() and torch.count_nonzero(g)>0 for g in entity_grads)
 path=tmp_path/"v2.pt";trainer.save(path,{"network_architecture":checkpoint_architecture(trainer),"training_seed":3101})
 state=torch.load(path,map_location="cpu",weights_only=False);assert state["development_feature_versions"]["fbmr_dual_bound"]==1
 restored=build_modular_mappo_trainer(config,"cuda");restored.load(path)
 with torch.no_grad():again=restored.actor.distribution(x)
 assert torch.equal(trainer.actor.distribution(x).mean,again.mean) and digest(SOURCE)==before

@pytest.mark.skipif(not torch.cuda.is_available() or not SOURCE.exists(),reason="CUDA/source checkpoint unavailable")
def test_v2_protocol_source_hash_and_only_bound_difference():
 import yaml
 env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"));state=torch.load(SOURCE,map_location="cuda",weights_only=False)
 assert validate_fbmr_v1_v2_only_bound_diff(load_config(V1),load_config(V2))
 result=validate_fbmr_v2_stage2_branch(state,env,load_config(V2),{"training_seed":3101,"training_num_envs":24,"training_smoke":False})
 assert result["intervention"]=="frozen_base_dual_bounded_mean_residual" and result["alpha_abs"]==result["alpha_rel"]==.25
 assert digest(SOURCE)=="6884448dae6c14a2b37c25b7a438f50dfa6eb84267528ee3d4de693cd0c0d6d0"
