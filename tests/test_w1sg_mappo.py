from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from algorithm.modules.wave1_sensitivity_gating import (Wave1SensitivityGatingModule,
 inverse_sqrt_gate,normalize_importance)
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_w1sg_branch
from algorithm.train_modular_mappo import load_config

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"

def digest(value):
 h=hashlib.sha256()
 for name,tensor in value.items():h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def config():return load_config(ROOT/"configs/dev_w1sg_current_actor_300k.yaml")

def test_importance_normalization_gate_and_norm_restoration():
 values=[torch.tensor([1.,3.]),torch.tensor([2.])];normalized,mean=normalize_importance(values)
 assert mean==pytest.approx(2.);assert torch.cat(normalized).mean()==pytest.approx(1.)
 assert torch.allclose(inverse_sqrt_gate(torch.tensor([0.,3.])),torch.tensor([1.,.5]))
 module=Wave1SensitivityGatingModule(config()["modules"]["wave1_sensitivity_gating"])
 p1=torch.nn.Parameter(torch.ones(2));p2=torch.nn.Parameter(torch.ones(1));p1.grad=torch.tensor([3.,4.]);p2.grad=torch.tensor([2.])
 module._normalized={"backbone.weight":torch.tensor([0.,3.]),"mean.weight":torch.tensor([1.])};raw=torch.sqrt(torch.tensor(29.)).item()
 metrics=module.gate_actor_gradients([("backbone.weight",p1),("mean.weight",p2)])
 protected=torch.sqrt(p1.grad.square().sum()+p2.grad.square().sum()).item()
 assert protected==pytest.approx(raw,rel=1e-6);assert metrics["w1sg_norm_preservation_relative_error"]<=1e-5
 assert metrics["w1sg_raw_protected_gradient_cosine"]<1.

def test_zero_importance_zero_gradient_and_alignment():
 normalized,mean=normalize_importance([torch.zeros(3)]);assert mean==0 and torch.equal(normalized[0],torch.zeros(3))
 module=Wave1SensitivityGatingModule(config()["modules"]["wave1_sensitivity_gating"]);p=torch.nn.Parameter(torch.ones(2));p.grad=torch.zeros(2);module._normalized={"backbone.weight":torch.ones(2)}
 metrics=module.gate_actor_gradients([("backbone.weight",p)]);assert metrics["w1sg_actor_grad_norm_after_renorm"]==0
 p.grad=torch.ones(2);module._normalized={"wrong":torch.ones(2)}
 with pytest.raises(RuntimeError,match="alignment"):module.gate_actor_gradients([("backbone.weight",p)])

def test_no_w1_fallback_and_sensitivity_does_not_mutate_model_optimizer_or_rng():
 trainer=build_modular_mappo_trainer(config(),"cpu",256,1_805_280);module=trainer.wave1_sensitivity_gating
 obs=torch.zeros(2,1,4,52);raw=torch.zeros(2,1,4,3);alive=torch.ones(2,1,4);waves=torch.full((2,1),2);adv=torch.ones(2,1,4)
 with torch.no_grad():
  dist,_=trainer.actor.distribution_step(obs[:,0],None,None,None,alive[:,0]);old=trainer.actor._squashed_log_prob(dist,raw[:,0],torch.zeros_like(raw[:,0]))[:,None]
 before_actor=digest(trainer.actor.state_dict());before_opt=deepcopy(trainer.actor_optimizer.state_dict());before_rng=torch.get_rng_state().clone()
 metrics=module.compute_sensitivity(trainer.actor,obs,raw,old,alive,waves,adv,.2)
 assert metrics["w1sg_active"]==0 and metrics["w1sg_identity_fallback_count"]==1
 assert digest(trainer.actor.state_dict())==before_actor and trainer.actor_optimizer.state_dict()==before_opt and torch.equal(torch.get_rng_state(),before_rng)

def test_exact_mode_rejections_and_protocol():
 base=config()
 for key,value in (("critic_replay",True),("replay_source","recent_uniform"),("priority_enabled",True),("bridge_enabled",True)):
  bad=deepcopy(base);bad["modules"]["persistent_wave_trajectory_replay"][key]=value
  with pytest.raises((ValueError,RuntimeError)):build_modular_mappo_trainer(bad,"cpu",256,1_805_280)
 if SOURCE.is_file():
  state=torch.load(SOURCE,map_location="cpu",weights_only=False);env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text())
  result=validate_w1sg_branch(state,env,base,{"training_seed":5301,"training_num_envs":24,"training_smoke":False});assert result["intervention"]=="w1sg_current_actor"

def test_checkpoint_counters_roundtrip_without_stale_importance():
 cfg=config()["modules"]["wave1_sensitivity_gating"];module=Wave1SensitivityGatingModule(cfg);module.active_update_count=2;module.identity_fallback_count=1;module.replay_gradient_step_count=3;module._normalized={"stale":torch.ones(1)}
 restored=Wave1SensitivityGatingModule(cfg);restored.load_state_dict(module.state_dict())
 assert (restored.active_update_count,restored.identity_fallback_count,restored.replay_gradient_step_count)==(2,1,3)
 assert restored._normalized=={} and restored._importance=={}
