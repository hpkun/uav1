#!/usr/bin/env python3
"""Strict preflight for the matched Actor-only gradient clipping screen."""
from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.protocol import validate_actor_grad_clip_branch,validate_actor_grad_clip_config_pair
SEEDS=(5301,5302,5303);SOURCE=lambda s:ROOT/f"outputs/diag_mappo_learnability/l3_seed{s}/checkpoint_1505280.pt";HIST=lambda s:ROOT/f"outputs/dev_pwtr_plain_seed{s}_300k"
OUTPUTS=lambda s:(ROOT/f"outputs/dev_actor_clip05_seed{s}_300k",ROOT/f"outputs/dev_actor_clip10_seed{s}_300k")
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def main():
 env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml");control=load_config(ROOT/"configs/dev_actor_grad_clip_05_control_300k.yaml");treatment=load_config(ROOT/"configs/dev_actor_grad_clip_10_300k.yaml");validate_actor_grad_clip_config_pair(control,treatment)
 expected={"total_sampled_steps":1_805_280,"num_train_envs":24,"rollout_steps":256,"gamma":.999,"gae_lambda":.95,"clip_ratio":.2,"entropy_coefficient":.01,"value_loss_coefficient":.5,"max_grad_norm":.5,"ppo_epochs":10,"minibatch_size":512}
 for name,config,limit in (("control",control,.5),("treatment",treatment,1.)):
  enabled=sorted(k for k,v in config["modules"].items() if isinstance(v,dict) and v.get("enabled",False))
  if enabled!=["actor_gradient_clipping","actor_lr_decay"]:raise RuntimeError(f"{name} enabled modules mismatch")
  for key,value in expected.items():
   if config["training"].get(key)!=value:raise RuntimeError(f"{name} training.{key} mismatch")
  module=config["modules"]["actor_gradient_clipping"]
  if float(module["actor_max_grad_norm"])!=limit or float(module["critic_max_grad_norm"])!=.5 or not module["critic_max_grad_norm_unchanged"]:raise RuntimeError(f"{name} clip config mismatch")
 if env["environment_variant"]!="persistent_wave_v2" or env["persistent_waves"]["total_waves"]!=3 or env["simulation"]["max_steps"]!=3000 or env["scenario"]["team_size"]!=4 or control["network"]["observation_dim"]!=52 or control["network"]["action_dim"]!=3:raise RuntimeError("environment/network mismatch")
 sources=[]
 for seed in SEEDS:
  source=SOURCE(seed);state=torch.load(source,map_location="cpu",weights_only=False);rng=state.get("rng_state",{})
  if int(state["sampled_steps"])!=1_505_280 or int(state["extra"]["training_seed"])!=seed or state["enabled_modules"]!=["actor_lr_decay"]:raise RuntimeError(f"source identity mismatch: {seed}")
  if not state["actor_optimizer"]["state"] or not state["critic_optimizer"]["state"]:raise RuntimeError(f"source optimizer incomplete: {seed}")
  required=("python_random_state","numpy_random_state","torch_cpu_rng_state","torch_cuda_rng_state_all","trainer_permutation_rng_state")
  if any(key not in rng for key in required):raise RuntimeError(f"source RNG incomplete: {seed}")
  if float(state["actor_optimizer"]["param_groups"][0]["lr"])!=1e-4:raise RuntimeError(f"source Actor LR mismatch: {seed}")
  runtime={"training_seed":seed,"training_num_envs":24,"training_smoke":False};validate_actor_grad_clip_branch(state,env,control,runtime);validate_actor_grad_clip_branch(state,env,treatment,runtime)
  if not HIST(seed).is_dir():raise RuntimeError(f"historical Plain reference missing: {seed}")
  rows=list(csv.DictReader((HIST(seed)/"evaluation_history.csv").open(newline="",encoding="utf-8")))
  if not any(int(r["sampled_steps"])==1_805_280 and int(r["evaluation_seed_base"])==44_000_000 and int(r["evaluation_seed_end"])==44_000_049 for r in rows):raise RuntimeError(f"historical reference endpoint mismatch: {seed}")
  sources.append({"seed":seed,"sha256":sha(source),"source":"PASS","historical_plain":"PASS"})
 existing=[str(path) for seed in SEEDS for path in OUTPUTS(seed) if path.exists()]
 if existing:raise RuntimeError(f"formal output directories already exist: {existing}")
 if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 report={"status":"READY_FOR_ACTOR_GRAD_CLIP_300K_SCREEN","cuda":torch.cuda.get_device_name(0),"sources":sources,"config_pair_single_variable":"PASS","actor_limits":{"control":.5,"treatment":1.},"critic_limit_both":.5,"training_max_grad_norm_both":.5,"formal_outputs_absent":"6/6","uses_45m":False}
 print(json.dumps(report,indent=2));print(report["status"])
if __name__=="__main__":main()
