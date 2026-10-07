#!/usr/bin/env python3
"""Fail-closed preflight for matched late-training MAPPO LR continuation."""
from __future__ import annotations
import argparse,hashlib,json,sys
from copy import deepcopy
from pathlib import Path
from typing import Any
import torch,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.common.checkpoint import validate_checkpoint_for_resume
from algorithm.common.protocol import config_sha256,runtime_source_manifest
from tools.preflight_mappo_critic_baseline_1p5m import build_seeded_trainer

ENV=ROOT/"configs/persistent_wave_v2_environment.yaml"
SOURCE_RUN=ROOT/"outputs/mappo_attention_seed5303_1p5m"
SOURCE=SOURCE_RUN/"checkpoint_1001472.pt"
SOURCE_CONFIG=SOURCE_RUN/"algorithm_config.yaml"
CONTROL=ROOT/"configs/mappo_attn_lr_control_3e4_cont_1p5m.yaml"
TREATMENT=ROOT/"configs/mappo_attn_lr_treatment_1e4_cont_1p5m.yaml"
RUNS=(ROOT/"outputs/mappo_attn_lr3e4_cont_seed5303_1p5m",ROOT/"outputs/mappo_attn_lr1e4_cont_seed5303_1p5m")
OUTPUT=ROOT/"outputs/mappo_late_lr_continuation_preflight.json"
SOURCE_STEP=1_001_472

def load_yaml(path:Path)->dict:return yaml.safe_load(path.read_text(encoding="utf-8"))
def sha256(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def nested_equal(left:Any,right:Any)->bool:
 if torch.is_tensor(left) or torch.is_tensor(right):return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left,right)
 if isinstance(left,dict) or isinstance(right,dict):return isinstance(left,dict) and isinstance(right,dict) and set(left)==set(right) and all(nested_equal(left[k],right[k]) for k in left)
 if isinstance(left,(list,tuple)) or isinstance(right,(list,tuple)):return type(left) is type(right) and len(left)==len(right) and all(nested_equal(a,b) for a,b in zip(left,right))
 return left==right
def comparable(config:dict)->dict:
 value=deepcopy(config);value.pop("development_method",None);value["training"].pop("actor_learning_rate",None);return value
def validate_configs(control:dict,treatment:dict)->None:
 if comparable(control)!=comparable(treatment):raise RuntimeError("continuation configs differ beyond Actor LR and development_method")
 expected={"seed":5303,"total_sampled_steps":1_500_000,"num_train_envs":24,"rollout_steps":256,"ppo_epochs":10,"minibatch_size":512,"gamma":.999,"gae_lambda":.95,"clip_ratio":.2,"value_loss_coefficient":.5,"entropy_coefficient":.01,"max_grad_norm":.5,"critic_learning_rate":.0003,"evaluation_episodes":50,"evaluation_interval_sampled_steps":100_000,"device":"cuda"}
 for name,cfg,method,actor_lr in (("Control",control,"mappo_attn_lr_control_3e4",.0003),("Treatment",treatment,"mappo_attn_lr_treatment_1e4",.0001)):
  if cfg.get("algorithm")!="MAPPO" or cfg.get("development_method")!=method:raise RuntimeError(f"{name} method identity mismatch")
  bad={k:(cfg["training"].get(k),v) for k,v in expected.items() if cfg["training"].get(k)!=v}
  if bad or cfg["training"].get("actor_learning_rate")!=actor_lr:raise RuntimeError(f"{name} training protocol mismatch: {bad}")
  n=cfg["network"]
  if (n.get("observation_dim"),n.get("action_dim"),n.get("num_agents"),n.get("critic_type"),n.get("attention_heads"))!=(52,3,4,"attention",2):raise RuntimeError(f"{name} network mismatch")
  forbidden=("modules","wave_context","wave_index","mission_context","popart","recurrent_memory","gru")
  if any(key in cfg or key in n for key in forbidden):raise RuntimeError(f"{name} contains forbidden research module")
  base=cfg["implementation"].get("evaluation_seed_base")
  if base!=47_000_000:raise RuntimeError(f"{name} evaluation seed mismatch")

def load_source(device:str="cuda"):
 if not SOURCE.is_file():raise FileNotFoundError(SOURCE)
 env=load_yaml(ENV);source_cfg=load_yaml(SOURCE_CONFIG)
 state=torch.load(SOURCE,map_location=device,weights_only=False)
 validate_checkpoint_for_resume(state,env,source_cfg)
 extra=state.get("extra",{})
 expected={"algorithm":"MAPPO","critic_type":"attention","sampled_steps":SOURCE_STEP}
 bad={k:(state.get(k),v) for k,v in expected.items() if state.get(k)!=v}
 extra_expected={"training_seed":5303,"environment_variant":"persistent_wave_v2","observation_dim":52,"action_dim":3,"num_agents":4,"training_num_envs":24,"environment_config_sha256":config_sha256(env),"algorithm_config_sha256":config_sha256(source_cfg)}
 bad_extra={k:(extra.get(k),v) for k,v in extra_expected.items() if extra.get(k)!=v}
 if bad or bad_extra:raise RuntimeError(f"source checkpoint identity mismatch: state={bad}, extra={bad_extra}")
 if len(extra.get("episode_indices",[]))!=24:raise RuntimeError("source checkpoint episode indices missing")
 return env,source_cfg,state

def build_report()->dict:
 if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 control,treatment=load_yaml(CONTROL),load_yaml(TREATMENT);validate_configs(control,treatment)
 env,source_cfg,state=load_source("cuda")
 if load_yaml(SOURCE_RUN/"env_config.yaml")!=env:raise RuntimeError("source environment snapshot differs from frozen environment")
 run=json.loads((SOURCE_RUN/"run_config.json").read_text(encoding="utf-8"))
 if run.get("environment_config_sha256")!=config_sha256(env) or run.get("algorithm_config_sha256")!=config_sha256(source_cfg):raise RuntimeError("source run_config SHA mismatch")
 existing=[str(path) for path in RUNS if path.exists()]
 if existing:raise RuntimeError(f"formal continuation output directories already exist: {existing}")
 c=build_seeded_trainer(control,"cuda",5303);t=build_seeded_trainer(treatment,"cuda",5303)
 c.load(SOURCE);t.load(SOURCE)
 actor_before=nested_equal(c.actor.state_dict(),t.actor.state_dict());critic_before=nested_equal(c.critic.state_dict(),t.critic.state_dict())
 actor_opt_before=nested_equal(c.actor_optimizer.state_dict(),t.actor_optimizer.state_dict());critic_opt_before=nested_equal(c.critic_optimizer.state_dict(),t.critic_optimizer.state_dict())
 counters_before=all(getattr(c,key)==getattr(t,key) for key in ("sampled_steps","vector_steps","ppo_update_count","actor_update_count","critic_update_count"))
 if not all((actor_before,critic_before,actor_opt_before,critic_opt_before,counters_before)):raise RuntimeError("pre-intervention branch parity failed")
 actor_snapshot=deepcopy(t.actor.state_dict());critic_snapshot=deepcopy(t.critic.state_dict());critic_opt_snapshot=deepcopy(t.critic_optimizer.state_dict())
 for group in t.actor_optimizer.param_groups:group["lr"]=.0001
 post_parameters=nested_equal(actor_snapshot,t.actor.state_dict()) and nested_equal(critic_snapshot,t.critic.state_dict())
 post_critic_optimizer=nested_equal(critic_opt_snapshot,t.critic_optimizer.state_dict())
 control_lr={float(g["lr"]) for g in c.actor_optimizer.param_groups};treatment_lr={float(g["lr"]) for g in t.actor_optimizer.param_groups}
 if not post_parameters or not post_critic_optimizer or control_lr!={.0003} or treatment_lr!={.0001}:raise RuntimeError("post-intervention parity failed")
 rng_fields=("rng_state","python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","trainer_permutation_rng_state")
 manifest=runtime_source_manifest(ROOT)
 return {"status":"READY_FOR_MAPPO_LATE_LR_CONTINUATION","cuda":torch.cuda.get_device_name(0),"source_checkpoint":str(SOURCE.relative_to(ROOT)),"source_checkpoint_sha256":sha256(SOURCE),"source_sampled_steps":SOURCE_STEP,"source_identity":{"algorithm":"MAPPO","critic_type":"attention","seed":5303,"environment_variant":"persistent_wave_v2","observation_dim":52,"action_dim":3,"num_agents":4},"source_environment_config_sha256":config_sha256(env),"source_algorithm_config_sha256":config_sha256(source_cfg),"control_config_sha256":config_sha256(control),"treatment_config_sha256":config_sha256(treatment),"runtime_source_manifest_sha256":manifest["runtime_source_manifest_sha256"],"pre_intervention_parity":{"actor_state_bitwise_equal":actor_before,"critic_state_bitwise_equal":critic_before,"actor_optimizer_state_equal":actor_opt_before,"critic_optimizer_state_equal":critic_opt_before,"training_counters_equal":counters_before},"post_intervention_parity":{"parameter_tensors_unchanged":post_parameters,"critic_optimizer_unchanged":post_critic_optimizer,"only_allowed_difference":"actor_optimizer.param_groups[*].lr","control_actor_lr":.0003,"treatment_actor_lr":.0001,"control_critic_lr":.0003,"treatment_critic_lr":.0003},"rng_continuation":{"checkpoint_rng_fields_present":{key:key in state for key in rng_fields},"full_rng_restoration_available":False,"bitwise_continuation_from_original_run_claimed":False,"matched_branch_initialization_seed":5303,"episode_indices_restored":True,"common_action_noise_trajectory_claimed":False},"evaluation_seed_range":[47_000_000,47_000_049],"uses_44m":False,"uses_45m":False,"uses_46m":False,"formal_outputs_absent":"2/2"}

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--print-only",action="store_true");parser.add_argument("--output",type=Path,default=OUTPUT);args=parser.parse_args();report=build_report()
 if not args.print_only:
  path=args.output if args.output.is_absolute() else ROOT/args.output
  if path.exists():raise FileExistsError(path)
  path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2),encoding="utf-8")
 print(json.dumps(report,indent=2));print(report["status"])
if __name__=="__main__":main()
