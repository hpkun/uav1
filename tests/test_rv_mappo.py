from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import torch
import pytest

from algorithm.modules import ReferenceVarianceModule
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_rv_branch, validate_rv_config_pair
from algorithm.modular_mappo.trainer import stable_ratio_terms
from algorithm.train_modular_mappo import load_config
from tools.analyze_rv_fixed10_300k import (METHODS,actor_state_sha256,
 validate_rv_frozen_actor_state,validate_rv_optimization_rows,
 validate_control_optimization_rows,validate_formal_run_artifacts)

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"

def configs():
 return (load_config(ROOT/"configs/dev_rv_fixed10_control_300k.yaml"),load_config(ROOT/"configs/dev_rv_mappo_v1_300k.yaml"))
def trainer(enabled=True,device="cpu"):
 return build_modular_mappo_trainer(configs()[1 if enabled else 0],device,256,1_805_280)

def test_exact_module_protocol_and_invalid_values():
 ReferenceVarianceModule({"enabled":True,"mode":"frozen_source_state_dependent_variance","source_sampled_steps":1505280,"freeze_current_log_std_head":True,"behavior_uses_reference_mean":False})
 with pytest.raises(ValueError):ReferenceVarianceModule({"enabled":True,"mode":"other"})

def test_config_pair_and_exact_modules():
 c,r=configs();assert validate_rv_config_pair(c,r)
 assert sorted(k for k,v in c["modules"].items() if isinstance(v,dict) and v.get("enabled"))==["actor_gradient_clipping","actor_lr_decay"]
 assert sorted(k for k,v in r["modules"].items() if isinstance(v,dict) and v.get("enabled"))==["actor_gradient_clipping","actor_lr_decay","reference_variance"]

def test_pair_rejects_unregistered_difference():
 c,r=configs();r["training"]["entropy_coefficient"]=.02
 with pytest.raises(RuntimeError):validate_rv_config_pair(c,r)

def test_branch_validator_both_arms():
 state=torch.load(SOURCE,map_location="cpu",weights_only=False);env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 for cfg in configs():assert validate_rv_branch(state,env,cfg,{"training_seed":5301,"training_num_envs":24,"training_smoke":False})["target_sampled_steps"]==1_805_280

def test_incompatible_dawe_rejected():
 cfg=configs()[1];cfg["modules"]["deployment_aligned_wave_exploration"]={"enabled":True,"mode":"fixed_wave_std_multiplier","wave1_multiplier":.25,"wave2_multiplier":.25,"wave3_multiplier":1.}
 with pytest.raises(ValueError):build_modular_mappo_trainer(cfg,"cpu",256,1_805_280)

def test_control_has_no_reference_and_identity_distribution():
 t=trainer(False);obs=torch.randn(3,4,52);alive=torch.ones(3,4);base,_=t.actor.distribution_step(obs,None,None,None,alive)
 assert t.reference_variance_actor is None and t._behavior_actor_distribution(base,obs,alive,torch.tensor([1,2,3])) is base

def test_branch_creates_frozen_reference_and_preserves_optimizer_membership():
 t=trainer(True);t.load(SOURCE,strict_protocol=False,restore_rng=False)
 assert t.reference_variance_actor_sha256()==t._module_state_sha256(t.actor)
 assert not t.reference_variance_actor.training and all(not p.requires_grad for p in t.reference_variance_actor.parameters())
 params={id(p) for g in t.actor_optimizer.param_groups for p in g["params"]}
 assert all(not p.requires_grad and id(p) in params for p in t.actor.log_std.parameters())

def test_behavior_uses_current_mean_reference_scale_and_ignores_wave():
 t=trainer(True);t.load(SOURCE,strict_protocol=False,restore_rng=False);obs=torch.randn(3,4,52);alive=torch.ones(3,4)
 current,_=t.actor.distribution_step(obs,None,None,None,alive);reference,_=t.reference_variance_actor.distribution_step(obs,None,None,None,alive)
 behavior=t._behavior_actor_distribution(current,obs,alive,torch.tensor([1,2,3]))
 assert torch.equal(behavior.loc,current.loc) and torch.equal(behavior.scale,reference.scale)
 assert torch.equal(behavior.scale,t._behavior_actor_distribution(current,obs,alive,torch.tensor([3,1,2])).scale)

def test_same_rng_control_rv_initial_stochastic_and_deterministic_parity():
 c=trainer(False);r=trainer(True);c.load(SOURCE,strict_protocol=False,restore_rng=False);r.load(SOURCE,strict_protocol=False,restore_rng=False)
 obs=torch.randn(3,4,52).numpy();alive=torch.ones(3,4).numpy();waves=torch.tensor([1,2,3]).numpy()
 state=torch.get_rng_state().clone();torch.set_rng_state(state);ca=c.act(obs,alive,False,True,wave_indices=waves);torch.set_rng_state(state);ra=r.act(obs,alive,False,True,wave_indices=waves)
 for i in range(3):assert (ca[i]==ra[i]).all()
 assert (c.act(obs,alive,True,True,wave_indices=waves)[0]==r.act(obs,alive,True,True,wave_indices=waves)[0]).all()

def test_loss_ratio_identity_and_gradient_semantics():
 t=trainer(True);t.load(SOURCE,strict_protocol=False,restore_rng=False);obs=torch.randn(3,4,52);alive=torch.ones(3,4);waves=torch.tensor([1,2,3])
 base,_=t.actor.distribution_step(obs,None,None,None,alive);dist=t._behavior_actor_distribution(base,obs,alive,waves);raw=dist.rsample();act=torch.tanh(raw);old=t.actor._squashed_log_prob(dist,raw,act);ones=torch.ones(3,4);zeros=torch.zeros(3,4)
 losses=t._loss_step(obs,act,raw,old,alive,ones,zeros,zeros,ones,torch.zeros(3,0),wave_indices=waves);_,ratio=stable_ratio_terms(losses[6],old);assert torch.allclose(ratio,torch.ones_like(ratio),atol=1e-6)
 t.actor_optimizer.zero_grad();(losses[0]-.01*losses[2]).backward()
 assert all(p.grad is None for p in t.actor.log_std.parameters()) and all(p.grad is None for p in t.reference_variance_actor.parameters())
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in list(t.actor.backbone.parameters())+list(t.actor.mean.parameters()))

def test_self_contained_checkpoint_and_missing_reference_fail_closed(tmp_path):
 t=trainer(True);t.load(SOURCE,strict_protocol=False,restore_rng=False);path=tmp_path/"rv.pt";t.save(path)
 restored=trainer(True);restored.load(path,strict_protocol=True,restore_rng=False);assert restored.reference_variance_actor_sha256()==t.reference_variance_actor_sha256()
 state=torch.load(path,map_location="cpu",weights_only=False);state["reference_variance_actor_state"]=None;bad=tmp_path/"bad.pt";torch.save(state,bad)
 with pytest.raises(RuntimeError):trainer(True).load(bad,strict_protocol=True,restore_rng=False)

def test_frozen_logstd_does_not_update():
 t=trainer(True);t.load(SOURCE,strict_protocol=False,restore_rng=False);before=[p.detach().clone() for p in t.actor.log_std.parameters()]
 t.actor_optimizer.zero_grad();sum(p.sum() for p in t.actor.mean.parameters()).backward();t.actor_optimizer.step()
 assert all(torch.equal(a,b) for a,b in zip(before,t.actor.log_std.parameters()))

def test_44m_and_45m_protocol_and_formal_dirs_absent():
 for cfg in configs():
  assert cfg["development_protocol"]["validation"]=={"seed_start":44000000,"seed_end":44000049,"episodes":50,"deterministic":True,"common_scenarios":True,"is_holdout":False}
  assert cfg["development_protocol"]["reserved_future_final_test"]=={"seed_start":45000000,"seed_end":45000199,"executed":False}
 assert not any((ROOT/f"outputs/dev_rv_{kind}_seed{seed}_300k").exists() for seed in (5301,5302,5303) for kind in ("control","v1"))

def test_runner_defers_reference_creation_until_after_rng_restore():
 text=(ROOT/"algorithm/modular_mappo/runner.py").read_text()
 assert text.index("self.trainer.restore_rng_state(state)") < text.index("self.trainer.finalize_reference_variance_branch()")

def test_analyzer_method_names_match_resolved_configs():
 control,rv=configs();assert METHODS=={"Control":"rv_fixed10_control","RV":"rv_mappo_v1"}
 assert METHODS["Control"]==control["development_method"] and METHODS["RV"]==rv["development_method"]

def synthetic_rv_endpoint():
 actor={"backbone.weight":torch.tensor([[1.,2.]]),"mean.weight":torch.tensor([[3.]]),
        "log_std.weight":torch.tensor([[4.,5.]]),"log_std.bias":torch.tensor([6.])}
 source={"actor":deepcopy(actor)};sha=actor_state_sha256(actor)
 endpoint={"actor":deepcopy(actor),"reference_variance_actor_state":deepcopy(actor),
  "reference_variance_actor_sha256":sha,"reference_variance_source_checkpoint_sha256":"parent",
  "reference_variance_source_sampled_steps":1_505_280,"reference_variance_source_training_seed":5301,
  "reference_variance_state":{"version":1}}
 return endpoint,source

def test_rv_endpoint_integrity_passes():
 endpoint,source=synthetic_rv_endpoint();result=validate_rv_frozen_actor_state(endpoint,source,"parent",5301,"final.pt","parent")
 assert all(result.values())

@pytest.mark.parametrize("label,name",[("final.pt","log_std.weight"),("final.pt","log_std.bias"),("latest.pt","log_std.weight")])
def test_rv_endpoint_logstd_mutation_fails(label,name):
 endpoint,source=synthetic_rv_endpoint();endpoint["actor"][name].reshape(-1)[0]+=1
 with pytest.raises(RuntimeError,match=name.replace(".",r"\.")):validate_rv_frozen_actor_state(endpoint,source,"parent",5301,label,"parent")

def test_rv_reference_sha_mismatch_fails():
 endpoint,source=synthetic_rv_endpoint();endpoint["reference_variance_actor_sha256"]="bad"
 with pytest.raises(RuntimeError,match="reference actor/source SHA mismatch"):validate_rv_frozen_actor_state(endpoint,source,"parent",5301,"final.pt","parent")

def test_rv_reference_source_checkpoint_sha_mismatch_fails():
 endpoint,source=synthetic_rv_endpoint();endpoint["reference_variance_source_checkpoint_sha256"]="bad"
 with pytest.raises(RuntimeError,match="source checkpoint SHA mismatch"):validate_rv_frozen_actor_state(endpoint,source,"parent",5301,"final.pt","parent")

def test_rv_reference_source_seed_mismatch_fails():
 endpoint,source=synthetic_rv_endpoint();endpoint["reference_variance_source_training_seed"]=5302
 with pytest.raises(RuntimeError,match="source training seed mismatch"):validate_rv_frozen_actor_state(endpoint,source,"parent",5301,"final.pt","parent")

def valid_rv_row():
 row={"rv_enabled":1.,"rv_log_std_head_requires_grad":0.,"rv_log_std_head_optimizer_membership":1.,
      "rv_log_std_head_grad_norm":0.,"rv_reference_any_grad_present":0.,"rv_reference_mutation_detected":0.}
 for wave in (1,2,3):row[f"rv_wave{wave}_alive_sample_count"]=4.;row[f"rv_wave{wave}_behavior_vs_reference_std_max_abs_error"]=0.
 return row

def test_rv_optimization_row_passes_and_empty_wave_null_allowed():
 row=valid_rv_row();row["rv_wave3_alive_sample_count"]=0.;row["rv_wave3_behavior_vs_reference_std_max_abs_error"]=None
 assert validate_rv_optimization_rows([row],"RV")["rv_training_diagnostics_pass"]

@pytest.mark.parametrize("key,value",[("rv_enabled",0.),("rv_log_std_head_requires_grad",1.),
 ("rv_log_std_head_optimizer_membership",0.),("rv_log_std_head_grad_norm",.1),
 ("rv_reference_any_grad_present",1.),("rv_reference_mutation_detected",1.)])
def test_rv_optimization_exact_semantics_fail_closed(key,value):
 row=valid_rv_row();row[key]=value
 with pytest.raises(RuntimeError,match=key):validate_rv_optimization_rows([row],"RV")

def test_rv_enabled_missing_fails_closed():
 row=valid_rv_row();row.pop("rv_enabled")
 with pytest.raises(RuntimeError,match="missing rv_enabled"):validate_rv_optimization_rows([row],"RV")

def test_rv_behavior_reference_sigma_error_fails():
 row=valid_rv_row();row["rv_wave2_behavior_vs_reference_std_max_abs_error"]=1.01e-7
 with pytest.raises(RuntimeError,match="sigma error"):validate_rv_optimization_rows([row],"RV")

def test_control_rv_enabled_zero_passes_and_nonzero_fails():
 validate_control_optimization_rows([{"rv_enabled":0.}],"Control")
 with pytest.raises(RuntimeError):validate_control_optimization_rows([{"rv_enabled":1.}],"Control")

def test_screening_gate_constants_and_formal_protocol_checks_unchanged():
 text=(ROOT/"tools/analyze_rv_fixed10_300k.py").read_text()
 for token in ('> -0.50','> -0.25','>= -0.05','["wins"] >= 2','"PROMISING"','"NOT_SUPPORTED"','"SAFETY_FAIL"','5860','45_000_000'):
  assert token in text

def test_analyzer_distinguishes_not_started_and_incomplete_runs(tmp_path):
 missing=tmp_path/"not_started"
 with pytest.raises(RuntimeError,match="has not been started or migrated"):validate_formal_run_artifacts(missing)
 missing.mkdir();(missing/"run_config.json").write_text("{}")
 with pytest.raises(RuntimeError,match="formal run is incomplete"):validate_formal_run_artifacts(missing)
