"""Strict preflight for paired 900k->1.2M MAPPO/FBMR Stage-2 branches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_fbmr_stage2_branch

MANIFEST=ROOT/"experiments/fbmr_stage2_development_manifest.json"
ENV_PATH=ROOT/"configs/persistent_wave_v2_environment.yaml"
SEEDS=(3101,3102,3103)
VALIDATION=(34_000_000,34_000_019)
RESERVED=(33_000_000,33_000_199)
FORBIDDEN_RANGES=((29_000_000,29_000_019),(30_000_000,30_000_199),(31_000_000,31_000_019),(32_000_000,32_000_019))
OFF_MODULES=("wave_context","recurrent_memory","popart","multi_wave_reward","wave_balancing","warm_start","curriculum","policy_anchor","advantage_priority","ppo_stabilization")

def file_sha256(path):
 digest=hashlib.sha256()
 with Path(path).open("rb") as stream:
  for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
 return digest.hexdigest()

def _assert_34m_unused_outside_protocol():
 outputs=ROOT/"outputs"
 if not outputs.exists():return
 target=outputs/"dev_fbmr_stage2"
 for path in outputs.rglob("*"):
  if not path.is_file() or target in path.parents or path.suffix.lower() not in {".json",".jsonl",".csv",".yaml",".yml",".log",".txt"}:continue
  try:text=path.read_text(encoding="utf-8",errors="ignore")
  except OSError:continue
  occupied=(re.search(r'"(?:evaluation_seed_base|seed_start)"\s*:\s*34000000\b',text) or
            re.search(r'"(?:evaluation_seed_end|seed_end)"\s*:\s*34000019\b',text) or
            re.search(r'(?:evaluation_seed_base|seed_start)\s*:\s*34000000\b',text))
  if occupied:raise RuntimeError(f"34M development validation range already appears in existing output record: {path}")

def validate(check_outputs=True,check_cuda=True,check_existing_34m=True):
 if check_cuda and not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for FBMR checkpoint preflight")
 manifest=json.loads(MANIFEST.read_text(encoding="utf-8"));env=yaml.safe_load(ENV_PATH.read_text(encoding="utf-8"))
 if manifest.get("protocol_role")!="development_only":raise RuntimeError("FBMR protocol must be development_only")
 if manifest.get("source_training_seeds")!=list(SEEDS) or manifest.get("methods")!=["MAPPO Continuation","FBMR-EA"]:raise RuntimeError("FBMR methods/source lineages mismatch")
 if len(manifest.get("runs",[]))!=6:raise RuntimeError("FBMR matrix must contain two methods x three sources")
 expected={(m,s) for m in manifest["methods"] for s in SEEDS};actual={(r["method"],int(r["source_training_seed"])) for r in manifest["runs"]}
 if actual!=expected:raise RuntimeError("FBMR paired lineage matrix mismatch")
 if (manifest["source_sampled_steps"],manifest["additional_sampled_steps"],manifest["target_total_sampled_steps"])!=(900000,300000,1200000):raise RuntimeError("FBMR Stage-2 budget mismatch")
 val=manifest["validation"];reserved=manifest["reserved_untouched_future_final_test"]
 if (val["seed_start"],val["seed_end"],val["episodes"])!=(*VALIDATION,20):raise RuntimeError("validation must be exactly 34M..34M+19")
 if (reserved["seed_start"],reserved["seed_end"],reserved["episodes"],reserved["executed"])!=(*RESERVED,200,False):raise RuntimeError("33M final range is not reserved and untouched")
 used=set(range(VALIDATION[0],VALIDATION[1]+1))
 if any(used & set(range(a,b+1)) for a,b in (*FORBIDDEN_RANGES,RESERVED)):raise RuntimeError("FBMR validation range overlaps forbidden/reserved ranges")
 if check_existing_34m:_assert_34m_unused_outside_protocol()
 sources={int(x["source_training_seed"]):x for x in manifest["source_checkpoints"]};hashes=[];validations=[]
 for seed in SEEDS:
  record=sources.get(seed);path=ROOT/record["path"]
  if path.name!="latest.pt" or "best_eval" in str(path):raise RuntimeError("FBMR source must be latest.pt, never best_eval.pt")
  if not path.is_file():raise FileNotFoundError(path)
  digest=file_sha256(path);hashes.append(digest)
  if digest!=record["sha256"]:raise RuntimeError(f"source checkpoint hash mismatch for seed {seed}")
  state=torch.load(path,map_location="cuda" if torch.cuda.is_available() else "cpu",weights_only=False)
  if int(state.get("extra",{}).get("training_seed",-1))!=seed:raise RuntimeError("source seed mismatch")
  for method,config_name in (("MAPPO Continuation","configs/dev_fbmr_mappo_continuation_1200k.yaml"),("FBMR-EA","configs/dev_fbmr_mean_residual_1200k.yaml")):
   config=load_config(ROOT/config_name)
   validations.append(validate_fbmr_stage2_branch(state,env,config,{"training_seed":seed,"training_num_envs":24,"training_smoke":False}))
 if len(set(hashes))!=3:raise RuntimeError("source hashes must be three distinct recorded checkpoints")
 configs=[load_config(ROOT/"configs/dev_fbmr_mappo_continuation_1200k.yaml"),load_config(ROOT/"configs/dev_fbmr_mean_residual_1200k.yaml")]
 for config in configs:
  t=config["training"]
  expected={"critic_learning_rate":3e-4,"gamma":.999,"gae_lambda":.95,"clip_ratio":.2,"entropy_coefficient":.01,"value_loss_coefficient":.5,"max_grad_norm":.5,"rollout_steps":256,"ppo_epochs":10,"minibatch_size":512,"num_train_envs":24,"total_sampled_steps":1200000,"evaluation_episodes":20,"evaluation_interval_sampled_steps":100000}
  if any(t.get(k)!=v for k,v in expected.items()):raise RuntimeError("Stage-2 common training config mismatch")
  if any(config["modules"].get(name,{}).get("enabled",False) for name in OFF_MODULES):raise RuntimeError("forbidden Stage-2 module enabled")
 fbmr=configs[1];entity=fbmr["modules"]["entity_attention"]
 if entity!={"enabled":True,"entity_dim":32,"attention_heads":2,"mode":"frozen_base_mean_residual","max_mean_correction":.25}:raise RuntimeError("FBMR entity config mismatch")
 trainer=build_modular_mappo_trainer(fbmr,"cuda" if torch.cuda.is_available() else "cpu")
 frozen={name for name,p in trainer.actor.named_parameters() if not p.requires_grad};expected_frozen={name for name,_ in trainer.actor.frozen_baseline_named_parameters()}
 if frozen!=expected_frozen or not frozen:raise RuntimeError("FBMR baseline actor freeze mismatch")
 optimized={id(p) for group in trainer.actor_optimizer.param_groups for p in group["params"]}
 if optimized!={id(p) for p in trainer.actor.trainable_policy_parameters()}:raise RuntimeError("FBMR actor optimizer parameter set mismatch")
 outputs=[ROOT/r["output_dir"] for r in manifest["runs"]];reference=ROOT/manifest["source_reference_evaluations"]["output_dir"]
 if len(outputs)!=len(set(outputs)):raise RuntimeError("FBMR output dirs are not unique")
 if check_outputs:
  occupied=[str(p) for p in [reference,*outputs] if p.exists()]
  if occupied:raise FileExistsError("FBMR outputs must be fresh: "+", ".join(occupied))
 return {"status":"READY_FOR_FBMR_STAGE2_DEVELOPMENT","source_checkpoints":3,"source_hashes":hashes,"planned_source_reference_evaluations":3,"planned_runs":6,"paired_source_training_seeds":list(SEEDS),"source_sampled_steps":900000,"additional_sampled_steps":300000,"target_sampled_steps":1200000,"validation_seed_range":list(VALIDATION),"reserved_33m":list(RESERVED),"reserved_33m_executed":False,"actor_effective_lr":1e-4,"critic_lr":3e-4,"fbmr_max_mean_correction":.25,"fbmr_base_frozen":True,"fbmr_log_std_isolated":True,"outputs_checked":check_outputs,"cuda_checked":check_cuda}

def main():
 p=argparse.ArgumentParser();p.add_argument("--skip-output-check",action="store_true");p.add_argument("--skip-existing-34m-check",action="store_true");args=p.parse_args()
 print(json.dumps(validate(not args.skip_output_check,True,not args.skip_existing_34m_check),indent=2))

if __name__=="__main__":main()
