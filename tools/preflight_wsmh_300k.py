#!/usr/bin/env python3
"""Strict, read-only preflight for the three WSMH 300k continuations."""
from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.protocol import validate_wsmh_branch
from algorithm.modules.wave_specific_mean_heads import EXPECTED_WSMH_CONFIG
SEEDS=(5301,5302,5303);SOURCE=lambda s:ROOT/f"outputs/diag_mappo_learnability/l3_seed{s}/checkpoint_1505280.pt"
FORMAL=lambda s:ROOT/f"outputs/dev_wsmh_seed{s}_300k";PLAIN=lambda s:ROOT/f"outputs/dev_pwtr_plain_seed{s}_300k"
STRAT=lambda s:ROOT/f"outputs/dev_pwtr_stratified_seed{s}_300k";WSAI=lambda s:ROOT/f"outputs/dev_wsai_seed{s}_300k"
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
 return run
def main():
 config=load_config(ROOT/"configs/dev_wsmh_300k.yaml");env=load_config(ROOT/"configs/persistent_wave_v2_environment.yaml")
 enabled=sorted(k for k,v in config["modules"].items() if isinstance(v,dict) and v.get("enabled",False))
 if enabled!=["actor_lr_decay","wave_specific_mean_heads"]:raise RuntimeError(f"enabled modules mismatch: {enabled}")
 if config["modules"]["wave_specific_mean_heads"]!=EXPECTED_WSMH_CONFIG:raise RuntimeError("WSMH config mismatch")
 expected={"total_sampled_steps":1_805_280,"num_train_envs":24,"rollout_steps":256,"gamma":.999,"gae_lambda":.95,"clip_ratio":.2,"entropy_coefficient":.01,"value_loss_coefficient":.5,"max_grad_norm":.5,"ppo_epochs":10,"minibatch_size":512}
 for key,value in expected.items():
  if config["training"].get(key)!=value:raise RuntimeError(f"training.{key} mismatch")
 if env["environment_variant"]!="persistent_wave_v2" or env["persistent_waves"]["total_waves"]!=3 or env["simulation"]["max_steps"]!=3000 or env["scenario"]["team_size"]!=4:raise RuntimeError("environment mismatch")
 if config["network"]["observation_dim"]!=52 or config["network"]["action_dim"]!=3:raise RuntimeError("network dimensions mismatch")
 results=[];manifest=None
 for seed in SEEDS:
  source=SOURCE(seed);digest=sha(source);state=torch.load(source,map_location="cpu",weights_only=False)
  if int(state["sampled_steps"])!=1_505_280 or state["enabled_modules"]!=["actor_lr_decay"]:raise RuntimeError(f"source mismatch: {seed}")
  if not state["actor_optimizer"]["state"] or not state["critic_optimizer"]["state"]:raise RuntimeError(f"source optimizer missing: {seed}")
  validate_wsmh_branch(state,env,config,{"training_seed":seed,"training_num_envs":24,"training_smoke":False})
  plain=check_reference(PLAIN(seed),seed,"pwtr_plain_matched_control",digest);check_reference(STRAT(seed),seed,"pwtr_stratified",digest);check_reference(WSAI(seed),seed,"wsai_mappo",digest)
  if manifest is None:manifest={row["path"]:row["sha256"] for row in plain["runtime_source_manifest_files"]}
  results.append({"seed":seed,"source_sha256":digest,"plain":"PASS","stratified":"PASS","wsai":"PASS"})
 for relative in ("env/combat_env.py","env/persistent_env.py"):
  if manifest.get(relative)!=sha(ROOT/relative):raise RuntimeError(f"frozen environment SHA mismatch: {relative}")
 existing=[str(FORMAL(seed)) for seed in SEEDS if FORMAL(seed).exists()]
 if existing:raise RuntimeError(f"formal output directories already exist: {existing}")
 if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 print(json.dumps({"status":"READY_FOR_WSMH_300K_SCREEN","cuda":torch.cuda.get_device_name(0),"enabled_modules":enabled,"sources":results,"formal_outputs_absent":"3/3","environment_sha":"PASS","uses_45m":False},indent=2));print("READY_FOR_WSMH_300K_SCREEN")
if __name__=="__main__":main()
