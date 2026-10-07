from __future__ import annotations
from copy import deepcopy
import hashlib
from pathlib import Path
import pytest,torch,yaml

from tools.preflight_mappo_late_lr_continuation import (
 CONTROL,TREATMENT,SOURCE,SOURCE_STEP,RUNS,build_seeded_trainer,load_yaml,
 nested_equal,sha256,validate_configs,
)
from tools.analyze_mappo_late_lr_continuation import (
 matched_configs,retention,validate_evaluations,
)

ROOT=Path(__file__).resolve().parents[1]

def configs():return load_yaml(CONTROL),load_yaml(TREATMENT)

def test_configs_are_matched_except_actor_lr_and_identity():
 control,treatment=configs();validate_configs(control,treatment);assert matched_configs(control,treatment)
 assert control["development_method"]!=treatment["development_method"]

def test_learning_rates_are_the_only_training_difference():
 control,treatment=configs();assert control["training"]["actor_learning_rate"]==pytest.approx(3e-4);assert treatment["training"]["actor_learning_rate"]==pytest.approx(1e-4)
 assert control["training"]["critic_learning_rate"]==treatment["training"]["critic_learning_rate"]==pytest.approx(3e-4)

def test_source_checkpoint_identity_and_common_sha():
 assert SOURCE.is_file() and sha256(SOURCE)==hashlib.sha256(SOURCE.read_bytes()).hexdigest()
 state=torch.load(SOURCE,map_location="cpu",weights_only=False);extra=state["extra"]
 assert state["algorithm"]=="MAPPO" and state["critic_type"]=="attention" and state["sampled_steps"]==SOURCE_STEP
 assert (extra["training_seed"],extra["environment_variant"],extra["observation_dim"],extra["action_dim"],extra["num_agents"])==(5303,"persistent_wave_v2",52,3,4)

def test_source_checkpoint_rng_limit_is_explicit():
 state=torch.load(SOURCE,map_location="cpu",weights_only=False)
 assert all(key not in state for key in ("rng_state","python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","trainer_permutation_rng_state"))

def test_branch_pre_intervention_parameter_optimizer_and_counter_parity():
 control,treatment=configs();c=build_seeded_trainer(control,"cpu",5303);t=build_seeded_trainer(treatment,"cpu",5303);c.load(SOURCE);t.load(SOURCE)
 assert nested_equal(c.actor.state_dict(),t.actor.state_dict());assert nested_equal(c.critic.state_dict(),t.critic.state_dict())
 assert nested_equal(c.actor_optimizer.state_dict(),t.actor_optimizer.state_dict());assert nested_equal(c.critic_optimizer.state_dict(),t.critic_optimizer.state_dict())
 assert (c.sampled_steps,c.vector_steps,c.ppo_update_count,c.actor_update_count,c.critic_update_count)==(t.sampled_steps,t.vector_steps,t.ppo_update_count,t.actor_update_count,t.critic_update_count)

def test_treatment_intervention_changes_only_actor_optimizer_lr():
 control,treatment=configs();c=build_seeded_trainer(control,"cpu",5303);t=build_seeded_trainer(treatment,"cpu",5303);c.load(SOURCE);t.load(SOURCE)
 actor=deepcopy(t.actor.state_dict());critic=deepcopy(t.critic.state_dict());critic_opt=deepcopy(t.critic_optimizer.state_dict())
 for group in t.actor_optimizer.param_groups:group["lr"]=1e-4
 assert nested_equal(actor,t.actor.state_dict()) and nested_equal(critic,t.critic.state_dict()) and nested_equal(critic_opt,t.critic_optimizer.state_dict())
 assert {g["lr"] for g in c.actor_optimizer.param_groups}=={3e-4};assert {g["lr"] for g in t.actor_optimizer.param_groups}=={1e-4}

def test_fixed_training_protocol_and_dimensions():
 for cfg in configs():
  t=cfg["training"];n=cfg["network"]
  assert (t["total_sampled_steps"],t["num_train_envs"],t["rollout_steps"],t["ppo_epochs"],t["minibatch_size"])==(1_500_000,24,256,10,512)
  assert (t["gamma"],t["gae_lambda"],t["clip_ratio"],t["entropy_coefficient"],t["value_loss_coefficient"],t["max_grad_norm"])==(.999,.95,.2,.01,.5,.5)
  assert (n["observation_dim"],n["action_dim"],n["num_agents"],n["critic_type"])==(52,3,4,"attention")

def test_no_wave_context_gru_popart_or_reward_override():
 forbidden={"modules","wave_context","wave_index","mission_context","popart","recurrent_memory","gru","reward"}
 for cfg in configs():assert forbidden.isdisjoint(cfg) and forbidden.isdisjoint(cfg["network"])

def test_evaluation_range_is_47m_and_not_44_45_46m():
 for cfg in configs():
  base=cfg["implementation"]["evaluation_seed_base"];count=cfg["training"]["evaluation_episodes"]
  assert (base,base+count-1,count)==(47_000_000,47_000_049,50);assert base not in (44_000_000,45_000_000,46_000_000)

def eval_row(step=1_500_000,w1=.8,w2=.5,w3=.2,aw=1.5):
 return {"sampled_steps":step,"evaluation_seed_base":47_000_000,"evaluation_seed_end":47_000_049,"evaluation_episodes":50,"clear_wave_1_probability":w1,"clear_wave_2_probability":w2,"clear_wave_3_probability":w3,"average_waves_cleared":aw}

def test_analyzer_requires_exact_endpoint_and_common_scenarios():
 validate_evaluations("Control",[eval_row(1_100_000),eval_row()])
 with pytest.raises(RuntimeError,match="exactly one"):validate_evaluations("Control",[eval_row(1_400_000)])
 bad=eval_row();bad["evaluation_seed_end"]=47_000_048
 with pytest.raises(RuntimeError,match="protocol mismatch"):validate_evaluations("Treatment",[bad])

@pytest.mark.parametrize("field",["evaluation_seed_base","evaluation_seed_end","evaluation_episodes"])
def test_analyzer_rejects_missing_evaluation_provenance(field):
 row=eval_row();row.pop(field)
 with pytest.raises(RuntimeError,match="missing provenance"):validate_evaluations("Control",[row])

def test_retention_metrics_are_descriptive_and_correct():
 rows=[eval_row(1_100_000,w3=.1,aw=1.1),eval_row(1_400_000,w3=.5,aw=2.0),eval_row(w3=.25,aw=1.5)]
 value=retention(rows);assert value["best_pre_endpoint_W3"]==.5 and value["best_pre_endpoint_AW"]==2.0
 assert value["W3_retention"]==.5 and value["AW_retention"]==.75 and value["W3_drop"]==-.25 and value["AW_drop"]==-.5

def test_retention_zero_denominator_is_none():
 rows=[eval_row(1_100_000,w3=0,aw=0),eval_row(w3=0,aw=0)];value=retention(rows)
 assert value["W3_retention"] is None and value["AW_retention"] is None

def test_formal_output_paths_are_static_distinct_and_shared_by_protocol_tools():
 assert len(RUNS)==2 and RUNS[0]!=RUNS[1]
 assert RUNS[0].name=="mappo_attn_lr3e4_cont_seed5303_1p5m"
 assert RUNS[1].name=="mappo_attn_lr1e4_cont_seed5303_1p5m"
 analyzer=(ROOT/"tools/analyze_mappo_late_lr_continuation.py").read_text(encoding="utf-8")
 preflight=(ROOT/"tools/preflight_mappo_late_lr_continuation.py").read_text(encoding="utf-8")
 for path in RUNS:
  assert path.name in analyzer and path.name in preflight

def test_method_document_discloses_non_bitwise_rng_continuation():
 text=(ROOT/"docs/mappo_late_lr_continuation.md").read_text(encoding="utf-8")
 assert "does **not** claim" in text and "RNG state" in text and "47,000,000--47,000,049" in text
