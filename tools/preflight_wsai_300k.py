#!/usr/bin/env python3
"""Strict, read-only preflight for the three WSAI 300k continuations."""
from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.protocol import validate_wsai_branch
from algorithm.modules.wave_specific_actor_isolation import EXPECTED_WSAI_CONFIG

SEEDS=(5301,5302,5303)
SOURCE=lambda seed:ROOT/f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt"
FORMAL=lambda seed:ROOT/f"outputs/dev_wsai_seed{seed}_300k"
PLAIN=lambda seed:ROOT/f"outputs/dev_pwtr_plain_seed{seed}_300k"
STRAT=lambda seed:ROOT/f"outputs/dev_pwtr_stratified_seed{seed}_300k"

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
 return h.hexdigest()

def check_reference(path,seed,method,source_sha):
 if not path.is_dir():raise RuntimeError(f"missing reference: {path}")
 run=json.loads((path/"run_config.json").read_text());branch=json.loads((path/"branch_from.json").read_text())
 rows=list(csv.DictReader((path/"evaluation_history.csv").open(newline="",encoding="utf-8")));last=rows[-1]
 if (int(run["seed"]),run["development_method"],int(run["total_sampled_steps"]))!=(seed,method,1_805_280):raise RuntimeError(f"reference identity mismatch: {path}")
 if (int(last["sampled_steps"]),int(last["evaluation_episodes"]),int(last["evaluation_seed_base"]),int(last["evaluation_seed_end"]))!=(1_805_280,50,44_000_000,44_000_049):raise RuntimeError(f"reference endpoint mismatch: {path}")
 if branch["parent_checkpoint_sha256"]!=source_sha or int(branch["source_training_seed"])!=seed:raise RuntimeError(f"reference parent mismatch: {path}")
 if not all((path/name).is_file() for name in ("latest.pt","final.pt","run_summary.json")):raise RuntimeError(f"reference incomplete: {path}")
 return run

def main():
 config=load_config(ROOT/"configs/dev_wsai_300k.yaml");env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 enabled=sorted(k for k,v in config["modules"].items() if isinstance(v,dict) and v.get("enabled",False))
 if enabled!=["actor_lr_decay","wave_specific_actor_isolation"]:raise RuntimeError(f"enabled modules mismatch: {enabled}")
 if config["modules"]["wave_specific_actor_isolation"]!=EXPECTED_WSAI_CONFIG:raise RuntimeError("WSAI config mismatch")
 training=config["training"]
 expected={"total_sampled_steps":1_805_280,"num_train_envs":24,"rollout_steps":256,"gamma":.999,"gae_lambda":.95,"clip_ratio":.2,"entropy_coefficient":.01,"value_loss_coefficient":.5,"max_grad_norm":.5,"ppo_epochs":10,"minibatch_size":512}
 for key,value in expected.items():
  if training.get(key)!=value:raise RuntimeError(f"training.{key} mismatch: {training.get(key)}")
 if env["environment_variant"]!="persistent_wave_v2" or env["persistent_waves"]["total_waves"]!=3 or env["simulation"]["max_steps"]!=3000:raise RuntimeError("environment identity mismatch")
 if env["scenario"]["team_size"]!=4 or config["network"]["observation_dim"]!=52 or config["network"]["action_dim"]!=3:raise RuntimeError("agent/network dimensions mismatch")
 results=[];reference_manifest=None
 for seed in SEEDS:
  source=SOURCE(seed)
  if not source.is_file():raise RuntimeError(f"missing source: {source}")
  digest=sha(source);state=torch.load(source,map_location="cpu",weights_only=False)
  if int(state["sampled_steps"])!=1_505_280 or int(state["extra"]["training_seed"])!=seed:raise RuntimeError(f"source identity mismatch: {seed}")
  if state["enabled_modules"]!=["actor_lr_decay"]:raise RuntimeError(f"source modules mismatch: {seed}")
  if float(state["actor_optimizer"]["param_groups"][0]["lr"])!=1e-4 or not state["actor_optimizer"]["state"] or not state["critic_optimizer"]["state"]:raise RuntimeError(f"source optimizer mismatch: {seed}")
  validate_wsai_branch(state,env,config,{"training_seed":seed,"training_num_envs":24,"training_smoke":False})
  plain=check_reference(PLAIN(seed),seed,"pwtr_plain_matched_control",digest)
  check_reference(STRAT(seed),seed,"pwtr_stratified",digest)
  if reference_manifest is None:reference_manifest={row["path"]:row["sha256"] for row in plain["runtime_source_manifest_files"]}
  results.append({"seed":seed,"source_sha256":digest,"plain_reference":"PASS","stratified_reference":"PASS"})
 for relative in ("env/combat_env.py","env/persistent_env.py","algorithm/modular_mappo/networks.py"):
  if reference_manifest.get(relative)!=sha(ROOT/relative):raise RuntimeError(f"frozen source SHA mismatch: {relative}")
 existing=[str(FORMAL(seed)) for seed in SEEDS if FORMAL(seed).exists()]
 if existing:raise RuntimeError(f"formal output directories already exist: {existing}")
 if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 report={"status":"READY_FOR_WSAI_300K_SCREEN","cuda":torch.cuda.get_device_name(0),"enabled_modules":enabled,
  "sources":results,"formal_outputs_absent":"3/3","frozen_source_sha":"PASS","uses_45m":False}
 print(json.dumps(report,indent=2));print("READY_FOR_WSAI_300K_SCREEN")

if __name__=="__main__":main()
