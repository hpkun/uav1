from __future__ import annotations
import copy,json
from dataclasses import fields
from pathlib import Path
import numpy as np
import pytest
import torch
from algorithm.modules import DeploymentAlignedWaveExplorationModule
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_dawe_branch,validate_dawe_config_pair
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer,stable_ratio_terms
from algorithm.train_modular_mappo import load_config
from algorithm.common.protocol import config_sha256
from tools.analyze_dawe_fixed10_300k import (
    evaluation_interval, intervals_intersect, validate_checkpoint,
    validate_evaluation_history, validate_optimization_metrics, validate_run_config,
)
from tools.preflight_dawe_fixed10_300k import validate_evaluation_protocol

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"
def configs():return (load_config(ROOT/"configs/dev_dawe_fixed10_control_300k.yaml"),load_config(ROOT/"configs/dev_dawe_fixed10_v1_300k.yaml"))
def module(enabled=True):return DeploymentAlignedWaveExplorationModule({"enabled":enabled,"mode":"fixed_wave_std_multiplier","wave1_multiplier":.25,"wave2_multiplier":.25,"wave3_multiplier":1.})
def trainer(enabled=True,device="cpu"):
 c=configs()[1 if enabled else 0];return build_modular_mappo_trainer(c,device,16,1_805_280)

def test_disabled_distribution_is_object_identity():
 base=torch.distributions.Normal(torch.zeros(2,4,3),torch.ones(2,4,3));assert module(False).effective_distribution(base,None) is base
def test_exact_v1_config_is_accepted():assert module(True).multipliers==(.25,.25,1.)
def test_enabled_extra_config_rejected():
 cfg={"enabled":True,"mode":"fixed_wave_std_multiplier","wave1_multiplier":.25,"wave2_multiplier":.25,"wave3_multiplier":1.,"schedule":True}
 with pytest.raises(ValueError):DeploymentAlignedWaveExplorationModule(cfg)
def test_nonpositive_multiplier_rejected():
 with pytest.raises(ValueError):DeploymentAlignedWaveExplorationModule({"enabled":False,"wave1_multiplier":0})
def test_invalid_wave_rejected():
 with pytest.raises(ValueError):module().multiplier(0)
def test_batch_wave_broadcast():
 base=torch.distributions.Normal(torch.zeros(3,4,3),torch.ones(3,4,3));effective=module().effective_distribution(base,torch.tensor([1,2,3]));assert effective.scale.shape==(3,4,3)
def test_wave1_std_quarter():
 base=torch.distributions.Normal(torch.zeros(1,4,3),torch.full((1,4,3),2.));assert torch.equal(module().effective_distribution(base,torch.tensor([1])).scale,torch.full((1,4,3),.5))
def test_wave2_std_quarter():
 base=torch.distributions.Normal(torch.zeros(1,4,3),torch.full((1,4,3),2.));assert torch.equal(module().effective_distribution(base,torch.tensor([2])).scale,torch.full((1,4,3),.5))
def test_wave3_std_identity():
 base=torch.distributions.Normal(torch.zeros(1,4,3),torch.full((1,4,3),2.));assert torch.equal(module().effective_distribution(base,torch.tensor([3])).scale,base.scale)
def test_wave_batch_mismatch_rejected():
 base=torch.distributions.Normal(torch.zeros(2,4,3),torch.ones(2,4,3))
 with pytest.raises(ValueError):module().effective_distribution(base,torch.tensor([1]))
def test_deterministic_action_parity():
 c,d=trainer(False),trainer(True);d.actor.load_state_dict(c.actor.state_dict());obs=np.ones((3,4,52),"f");alive=np.ones((3,4),"f");waves=np.array([1,2,3]);assert np.array_equal(c.act(obs,alive,True,True,wave_indices=waves)[0],d.act(obs,alive,True,True,wave_indices=waves)[0])
def test_same_rng_residual_scaling_cpu():
 c,d=trainer(False),trainer(True);d.actor.load_state_dict(c.actor.state_dict());obs=np.ones((3,4,52),"f");alive=np.ones((3,4),"f");waves=np.array([1,2,3]);state=torch.get_rng_state();torch.set_rng_state(state);cr=c.act(obs,alive,False,True,wave_indices=waves)[1];torch.set_rng_state(state);dr=d.act(obs,alive,False,True,wave_indices=waves)[1]
 with torch.no_grad():base,_=d.actor.distribution_step(torch.tensor(obs),None,None,None,torch.tensor(alive));mu=base.loc.numpy()
 assert np.allclose(dr[:2]-mu[:2],.25*(cr[:2]-mu[:2]),atol=1e-6);assert np.allclose(dr[2],cr[2],atol=1e-6)
def test_rollout_update_logprob_identity_all_waves():
 d=trainer(True);obs=torch.randn(3,4,52);alive=torch.ones(3,4);base,_=d.actor.distribution_step(obs,None,None,None,alive);eff=d._effective_actor_distribution(base,torch.tensor([1,2,3]));raw=eff.rsample();action=torch.tanh(raw);old=d.actor._squashed_log_prob(eff,raw,action);new=d.actor._squashed_log_prob(d._effective_actor_distribution(base,torch.tensor([1,2,3])),raw,action);_,ratio=stable_ratio_terms(new,old);assert torch.allclose(new,old) and torch.allclose(ratio,torch.ones_like(ratio))
def test_base_distribution_would_break_scaled_logprob_identity():
 d=trainer(True);obs=torch.randn(1,4,52);base,_=d.actor.distribution_step(obs,None,None,None,torch.ones(1,4));eff=d._effective_actor_distribution(base,torch.tensor([1]));raw=eff.rsample();act=torch.tanh(raw);assert not torch.allclose(d.actor._squashed_log_prob(eff,raw,act),d.actor._squashed_log_prob(base,raw,act))
def test_squashed_jacobian_uses_existing_actor_implementation():
 source=Path(__import__("algorithm.modular_mappo.trainer",fromlist=["x"]).__file__).read_text();assert "self.actor._squashed_log_prob(dist,raw,act)" in source
def test_effective_distribution_construction_does_not_consume_rng():
 base=torch.distributions.Normal(torch.zeros(3,4,3),torch.ones(3,4,3));state=torch.get_rng_state().clone();module().effective_distribution(base,torch.tensor([1,2,3]));assert torch.equal(state,torch.get_rng_state())
def test_rollout_schema_has_no_dawe_field():assert not any("dawe" in item.name for item in fields(ModularRolloutBatch))
def test_runner_uses_pre_wave_for_action():
 source=(ROOT/"algorithm/modular_mappo/runner.py").read_text();assert "wave_indices=pre_wave" in source
def test_loss_step_accepts_wave_indices():
 import inspect;assert "wave_indices" in inspect.signature(ModularMAPPOTrainer._loss_step).parameters
def test_training_cli_dispatches_both_dawe_interventions():
 source=(ROOT/"algorithm/train_modular_mappo.py").read_text();assert 'validate_dawe_branch if intervention in {"dawe_fixed10_control","deployment_aligned_wave_exploration"}' in source
def test_control_enabled_modules_exact():assert sorted(k for k,v in configs()[0]["modules"].items() if v.get("enabled"))==["actor_gradient_clipping","actor_lr_decay"]
def test_dawe_enabled_modules_exact():assert sorted(k for k,v in configs()[1]["modules"].items() if v.get("enabled"))==["actor_gradient_clipping","actor_lr_decay","deployment_aligned_wave_exploration"]
def test_config_pair_has_only_registered_differences():assert validate_dawe_config_pair(*configs())
def test_config_pair_unregistered_difference_rejected():
 c,d=configs();d["training"]["entropy_coefficient"]=.02
 with pytest.raises(RuntimeError):validate_dawe_config_pair(c,d)
def test_branch_validator_both_arms():
 state=torch.load(SOURCE,map_location="cpu",weights_only=False);env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 for cfg in configs():assert validate_dawe_branch(state,env,cfg,{"training_seed":5301,"training_num_envs":24,"training_smoke":False})["target_sampled_steps"]==1_805_280
def test_branch_validator_rejects_wrong_source_step():
 state=torch.load(SOURCE,map_location="cpu",weights_only=False);state["sampled_steps"]-=1;env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 with pytest.raises(RuntimeError):validate_dawe_branch(state,env,configs()[1],{"training_seed":5301,"training_num_envs":24,"training_smoke":False})
def test_branch_requires_optimizer_and_rng_restore():
 c=configs()[0];c["development_branch"]["rng_restore"]=False;state=torch.load(SOURCE,map_location="cpu",weights_only=False);env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 with pytest.raises(RuntimeError):validate_dawe_branch(state,env,c,{"training_seed":5301,"training_num_envs":24,"training_smoke":False})
def test_45m_is_declared_untouched():
 manifest=json.loads((ROOT/"experiments/dawe_fixed10_300k_manifest.json").read_text());assert manifest["reserved_future_final_test"]=={"seed_start":45000000,"seed_end":45000199,"executed":False}
def test_expected_update_arithmetic_is_preregistered():
 source=(ROOT/"tools/preflight_dawe_fixed10_300k.py").read_text();assert '"expected_ppo_updates": 49' in source and '"expected_actor_optimizer_steps": 5860' in source
def test_incompatible_module_rejected():
 c=configs()[1];c["modules"]["wave_specific_mean_heads"]={"enabled":True}
 with pytest.raises(ValueError):build_modular_mappo_trainer(c,"cpu",16,1_805_280)
def test_checkpoint_roundtrip_is_self_describing(tmp_path):
 d=trainer(True);path=tmp_path/"dawe.pt";d.save(path);state=torch.load(path,map_location="cpu",weights_only=False);assert state["deployment_aligned_wave_exploration_state"]["version"]==1;trainer(True).load(path,strict_protocol=True,restore_rng=False)
@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")
def test_cuda_same_rng_scaling_and_deterministic_parity():
 c,d=trainer(False,"cuda"),trainer(True,"cuda");d.actor.load_state_dict(c.actor.state_dict());obs=np.ones((3,4,52),"f");alive=np.ones((3,4),"f");waves=np.array([1,2,3]);assert np.array_equal(c.act(obs,alive,True,True,wave_indices=waves)[0],d.act(obs,alive,True,True,wave_indices=waves)[0]);state=torch.cuda.get_rng_state();torch.cuda.set_rng_state(state);cr=c.act(obs,alive,False,True,wave_indices=waves)[1];torch.cuda.set_rng_state(state);dr=d.act(obs,alive,False,True,wave_indices=waves)[1]
 with torch.no_grad():base,_=d.actor.distribution_step(torch.tensor(obs,device="cuda"),None,None,None,torch.tensor(alive,device="cuda"));mu=base.loc.cpu().numpy()
 assert np.allclose(dr[:2]-mu[:2],.25*(cr[:2]-mu[:2]),atol=2e-5) and np.allclose(dr[2],cr[2],atol=2e-5)


def test_loss_step_ratio_identity_all_waves():
 d=trainer(True);count=3;obs=torch.randn(count,4,52);alive=torch.ones(count,4);waves=torch.tensor([1,2,3])
 base,_=d.actor.distribution_step(obs,None,None,None,alive);effective=d._effective_actor_distribution(base,waves)
 raw=effective.rsample();action=torch.tanh(raw);oldlog=d.actor._squashed_log_prob(effective,raw,action)
 zeros=torch.zeros(count,4);ones=torch.ones(count,4)
 losses=d._loss_step(obs,action,raw,oldlog,alive,ones,zeros,zeros,ones,zeros,wave_indices=waves)
 ratio,newlog,returned_old=losses[4],losses[6],losses[7]
 assert torch.allclose(newlog,returned_old,atol=1e-6,rtol=1e-6)
 for wave in (1,2,3):
  selected=waves==wave;assert torch.allclose(ratio[selected],torch.ones_like(ratio[selected]),atol=1e-6,rtol=1e-6)


def test_control_disabled_distribution_and_rng_identity():
 d=trainer(False);obs=np.ones((3,4,52),dtype=np.float32);alive=np.ones((3,4),dtype=np.float32);waves=np.array([1,2,3])
 state=torch.get_rng_state().clone();torch.set_rng_state(state);actual=d.act(obs,alive,False,True,wave_indices=waves);actual_after=torch.get_rng_state().clone()
 torch.set_rng_state(state)
 with torch.no_grad():
  base,_=d.actor.distribution_step(torch.tensor(obs),None,None,None,torch.tensor(alive));raw=base.rsample();action=torch.tanh(raw);log=d.actor._squashed_log_prob(base,raw,action)
 expected_after=torch.get_rng_state().clone()
 assert np.array_equal(actual[0],action.numpy()) and np.array_equal(actual[1],raw.numpy()) and np.array_equal(actual[2],log.numpy())
 assert torch.equal(actual_after,expected_after)
 direct=torch.tanh(base.mean).numpy();assert np.array_equal(d.act(obs,alive,True,True,wave_indices=waves)[0],direct)


def test_control_disabled_loss_step_wave_indices_are_identity():
 d=trainer(False);count=3;obs=torch.randn(count,4,52);alive=torch.ones(count,4);waves=torch.tensor([1,2,3])
 base,_=d.actor.distribution_step(obs,None,None,None,alive);raw=base.rsample();action=torch.tanh(raw);old=d.actor._squashed_log_prob(base,raw,action)
 zeros=torch.zeros(count,4);ones=torch.ones(count,4);args=(obs,action,raw,old,alive,ones,zeros,zeros,ones,zeros)
 state=torch.get_rng_state().clone();torch.set_rng_state(state);with_waves=d._loss_step(*args,wave_indices=waves)
 torch.set_rng_state(state);without_waves=d._loss_step(*args,wave_indices=None)
 for index in (0,1,2,4,5,6,7):assert torch.equal(with_waves[index],without_waves[index])


def test_preflight_rejects_evaluation_seed_change():
 control,treatment=configs();treatment["implementation"]["evaluation_seed_base"]+=1
 with pytest.raises(RuntimeError):validate_evaluation_protocol(control,treatment)


def test_preflight_rejects_evaluation_episode_change():
 control,treatment=configs();control["training"]["evaluation_episodes"]=49
 with pytest.raises(RuntimeError):validate_evaluation_protocol(control,treatment)


def test_preflight_rejects_reserved_45m_executed():
 control,treatment=configs();treatment["development_protocol"]["reserved_future_final_test"]["executed"]=True
 with pytest.raises(RuntimeError):validate_evaluation_protocol(control,treatment)


def test_resolved_environment_and_algorithm_hashes_are_distinct_and_stable():
 env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml");control,treatment=configs()
 assert config_sha256(env)==config_sha256(load_config(ROOT/"configs/persistent_wave_v2_environment.yaml"))
 assert config_sha256(control)==config_sha256(configs()[0]);assert config_sha256(control)!=config_sha256(treatment)


def test_launcher_contains_all_four_fail_closed_locks():
 text=(ROOT/"tools/run_dawe_fixed10_300k.sh").read_text()
 for token in ("LOCK_RUNTIME_SOURCE_SHA","LOCK_ENVIRONMENT_CONFIG_SHA","LOCK_CONTROL_CONFIG_SHA","LOCK_TREATMENT_CONFIG_SHA","check_all_locks"):
  assert token in text
 assert "config_sha256(load_config" in text and "--print-only" in text


def synthetic_run(method="Control"):
 cfg=configs()[0 if method=="Control" else 1];env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 return {"algorithm":"modular_mappo","seed":5301,"total_sampled_steps":1_805_280,"num_envs":24,"smoke":False,"device":"cuda","environment_variant":"persistent_wave_v2","environment_config_sha256":config_sha256(env),"algorithm_config_sha256":config_sha256(cfg),"development_method":"dawe_fixed10_control" if method=="Control" else "dawe_fixed10_v1","module_config_sha256":config_sha256(cfg["modules"]),"modular_mappo_impl_version":2,"baseline_mappo_impl_version":2,"enabled_modules":["actor_gradient_clipping","actor_lr_decay"]+([] if method=="Control" else ["deployment_aligned_wave_exploration"]),"network_architecture":torch.load(SOURCE,map_location="cpu",weights_only=False)["extra"]["network_architecture"],"runtime_source_manifest_sha256":"abc"}


def test_analyzer_rejects_run_and_module_hash_errors():
 run=synthetic_run();env_hash=run["environment_config_sha256"];algorithm_hash=run["algorithm_config_sha256"];module_hash=run["module_config_sha256"]
 validate_run_config(run,"Control",5301,env_hash,algorithm_hash,module_hash)
 bad=copy.deepcopy(run);bad["algorithm_config_sha256"]="bad"
 with pytest.raises(RuntimeError):validate_run_config(bad,"Control",5301,env_hash,algorithm_hash,module_hash)
 bad=copy.deepcopy(run);bad["module_config_sha256"]="bad"
 with pytest.raises(RuntimeError):validate_run_config(bad,"Control",5301,env_hash,algorithm_hash,module_hash)


def test_analyzer_rejects_checkpoint_endpoint_and_update_deltas():
 source=torch.load(SOURCE,map_location="cpu",weights_only=False);run=synthetic_run();state=copy.deepcopy(source);state.update({"sampled_steps":1_805_280,"ppo_updates":source["ppo_updates"]+49,"actor_updates":source["actor_updates"]+5860,"critic_updates":source["critic_updates"]+5860,"enabled_modules":run["enabled_modules"],"module_config_sha256":run["module_config_sha256"]});state["extra"]=copy.deepcopy(source["extra"]);state["extra"].update({"training_seed":5301,"training_total_sampled_steps":1_805_280,"training_num_envs":24,"training_smoke":False,"environment_variant":"persistent_wave_v2","environment_config_sha256":run["environment_config_sha256"],"algorithm_config_sha256":run["algorithm_config_sha256"],"runtime_source_manifest_sha256":"abc"})
 validate_checkpoint(state,run,5301,source,"final.pt")
 for key in ("sampled_steps","ppo_updates","actor_updates","critic_updates"):
  bad=copy.deepcopy(state);bad[key]-=1
  with pytest.raises(RuntimeError):validate_checkpoint(bad,run,5301,source,"final.pt")


def test_analyzer_rejects_optimization_counts():
 rows=[{"ppo_epochs_executed":10,"actor_optimizer_steps_this_update":120,"critic_optimizer_steps_this_update":120} for _ in range(48)]+[{"ppo_epochs_executed":10,"actor_optimizer_steps_this_update":100,"critic_optimizer_steps_this_update":100}]
 assert validate_optimization_metrics(rows)==(5860,5860)
 for key in ("actor_optimizer_steps_this_update","critic_optimizer_steps_this_update"):
  bad=copy.deepcopy(rows);bad[0][key]-=1
  with pytest.raises(RuntimeError):validate_optimization_metrics(bad)
 with pytest.raises(RuntimeError):validate_optimization_metrics(rows[:-1])


def test_evaluation_interval_fallback_and_reserved_intersection():
 assert evaluation_interval({"evaluation_seed_base":"44000000","evaluation_episodes":"50"})==(44_000_000,44_000_049,50)
 assert intervals_intersect((44_999_999,45_000_000),(45_000_000,45_000_199))
 endpoint={"sampled_steps":str(1_805_280),"evaluation_seed_base":"44000000","evaluation_episodes":"50"}
 assert validate_evaluation_history([endpoint]) is endpoint
 bad=copy.deepcopy(endpoint);bad.update({"evaluation_seed_base":"44999999","evaluation_episodes":"2"})
 with pytest.raises(RuntimeError):validate_evaluation_history([bad])
 bad=copy.deepcopy(endpoint);bad["evaluation_episodes"]="49"
 with pytest.raises(RuntimeError):validate_evaluation_history([bad])


def test_dawe_multiplier_and_screening_gate_remain_registered():
 assert module(True).multipliers==(.25,.25,1.)
 text=(ROOT/"tools/analyze_dawe_fixed10_300k.py").read_text()
 for token in ('> -0.50','> -0.25','>= -0.05','"wins"] >= 2','"PROMISING"','"NOT_SUPPORTED"','"SAFETY_FAIL"'):
  assert token in text
