"""Strict preflight for the final three FBMR-V2 900k->1.2M branches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_fbmr_v2_stage2_branch,validate_fbmr_v1_v2_only_bound_diff

MANIFEST=ROOT/"experiments/fbmr_v2_dual_bound_development_manifest.json"
ENV_PATH=ROOT/"configs/persistent_wave_v2_environment.yaml"
V1_CONFIG=ROOT/"configs/dev_fbmr_mean_residual_1200k.yaml"
V2_CONFIG=ROOT/"configs/dev_fbmr_v2_dual_bound_1200k.yaml"
LAUNCHER=ROOT/"tools/run_fbmr_v2_development.sh"
SEEDS=(3101,3102,3103)
EXPECTED_HASHES={3101:"6884448dae6c14a2b37c25b7a438f50dfa6eb84267528ee3d4de693cd0c0d6d0",3102:"4c26422b273b5e0802cad820711efedc3f54811cafaa3cfa7fd292bf0bb0e098",3103:"076bb4f1065d6da23bdf2cfdf8db951bacd93878f452c56a582fec4019ea091f"}
OFF=("wave_context","recurrent_memory","popart","multi_wave_reward","wave_balancing","warm_start","curriculum","policy_anchor","advantage_priority","ppo_stabilization")

def sha256(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()

def validate(check_outputs=True,check_cuda=True):
 if check_cuda and not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for FBMR V2 preflight")
 manifest=json.loads(MANIFEST.read_text(encoding="utf-8"));env=yaml.safe_load(ENV_PATH.read_text(encoding="utf-8"))
 v1,v2=load_config(V1_CONFIG),load_config(V2_CONFIG)
 if manifest.get("protocol_role")!="development_only" or manifest.get("research_status")!="final_entity_actor_architecture_development_cycle":raise RuntimeError("V2 manifest role/status mismatch")
 if manifest.get("source_training_seeds")!=list(SEEDS) or manifest.get("v2_runs")!=3 or len(manifest.get("runs",[]))!=3:raise RuntimeError("V2 must contain exactly three source-lineage runs")
 if not manifest.get("existing_comparators_are_not_rerun"):raise RuntimeError("existing comparator rerun prohibition missing")
 if (manifest["source_sampled_steps"],manifest["additional_sampled_steps"],manifest["target_total_sampled_steps"])!=(900000,300000,1200000):raise RuntimeError("V2 budget mismatch")
 val=manifest["validation"];reserved=manifest["reserved_untouched_future_final_test"]
 if (val["seed_start"],val["seed_end"],val["episodes"])!=(34000000,34000019,20) or not val["seed_range_reused_from_fbmr_v1_development"] or not val["validation_already_exposed_in_prior_development"] or val["is_holdout"]:raise RuntimeError("34M reused-development semantics mismatch")
 if (reserved["seed_start"],reserved["seed_end"],reserved["episodes"],reserved["executed"])!=(33000000,33000199,200,False):raise RuntimeError("33M is not reserved untouched")
 validate_fbmr_v1_v2_only_bound_diff(v1,v2)
 entity=v2["modules"]["entity_attention"]
 expected_entity={"enabled":True,"mode":"frozen_base_dual_bounded_mean_residual","entity_dim":32,"attention_heads":2,"alpha_abs":.25,"alpha_rel":.25}
 if entity!=expected_entity or "max_mean_correction" in entity:raise RuntimeError("V2 entity config is not exact")
 t=v2["training"]
 expected_training={"actor_learning_rate":1e-4,"critic_learning_rate":3e-4,"gamma":.999,"gae_lambda":.95,"clip_ratio":.2,"value_loss_coefficient":.5,"entropy_coefficient":.01,"max_grad_norm":.5,"rollout_steps":256,"ppo_epochs":10,"minibatch_size":512,"num_train_envs":24,"total_sampled_steps":1200000,"evaluation_episodes":20,"evaluation_interval_sampled_steps":100000}
 if any(t.get(k)!=v for k,v in expected_training.items()):raise RuntimeError("V2 resolved training protocol mismatch")
 if any(v2["modules"].get(name,{}).get("enabled",False) for name in OFF) or v2["modules"]["actor_lr_decay"].get("enabled",False):raise RuntimeError("forbidden V2 module enabled")
 v1_trainer=build_modular_mappo_trainer(v1,"cuda" if torch.cuda.is_available() else "cpu")
 v2_trainer=build_modular_mappo_trainer(v2,"cuda" if torch.cuda.is_available() else "cpu")
 v1_trainable={n for n,p in v1_trainer.actor.named_parameters() if p.requires_grad};v2_trainable={n for n,p in v2_trainer.actor.named_parameters() if p.requires_grad}
 if v1_trainable!=v2_trainable or sum(p.numel() for p in v1_trainer.actor.trainable_policy_parameters())!=sum(p.numel() for p in v2_trainer.actor.trainable_policy_parameters()):raise RuntimeError("V1/V2 trainable actor sets differ")
 expected_frozen={n for n,_ in v2_trainer.actor.frozen_baseline_named_parameters()}
 if expected_frozen!={n for n,p in v2_trainer.actor.named_parameters() if not p.requires_grad}:raise RuntimeError("V2 baseline freeze mismatch")
 records={int(x["source_training_seed"]):x for x in manifest["source_checkpoints"]};hashes={}
 for seed in SEEDS:
  record=records[seed];path=ROOT/record["path"]
  if path.name!="latest.pt" or not path.is_file():raise FileNotFoundError(path)
  digest=sha256(path);hashes[seed]=digest
  if digest!=EXPECTED_HASHES[seed] or digest!=record["sha256"]:raise RuntimeError(f"source SHA256 mismatch: {seed}")
  state=torch.load(path,map_location="cuda" if torch.cuda.is_available() else "cpu",weights_only=False)
  validate_fbmr_v2_stage2_branch(state,env,v2,{"training_seed":seed,"training_num_envs":24,"training_smoke":False})
 launcher=LAUNCHER.read_text(encoding="utf-8")
 if "33000000" in launcher or "mappo_cont_seed" in launcher or "fbmr_seed" in launcher:raise RuntimeError("launcher touches reserved seeds or reruns comparators")
 if any(f"run_v2_branch {s}" not in launcher for s in SEEDS) or "34000000" not in launcher:raise RuntimeError("launcher V2 matrix/validation mismatch")
 outputs=[ROOT/r["output_dir"] for r in manifest["runs"]]
 if len(outputs)!=len(set(outputs)):raise RuntimeError("V2 output dirs are not unique")
 if check_outputs:
  occupied=[str(p) for p in outputs if p.exists()]
  if occupied:raise FileExistsError("V2 outputs must be fresh: "+", ".join(occupied))
 return {"status":"READY_FOR_FBMR_V2_DEVELOPMENT","device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unchecked","source_hashes":hashes,"source_sampled_steps":900000,"additional_sampled_steps":300000,"target_sampled_steps":1200000,"planned_runs":3,"comparators_rerun":False,"actor_lr":1e-4,"critic_lr":3e-4,"entity_mode":entity["mode"],"alpha_abs":.25,"alpha_rel":.25,"base_actor_frozen":True,"log_std_isolated":True,"v1_v2_only_bound_diff":True,"v1_v2_trainable_parameter_count":sum(p.numel() for p in v2_trainer.actor.trainable_policy_parameters()),"validation_seed_range":[34000000,34000019],"validation_seed_range_reused_from_fbmr_v1_development":True,"reserved_33m":[33000000,33000199],"reserved_33m_executed":False,"outputs_checked":check_outputs}

def main():
 p=argparse.ArgumentParser();p.add_argument("--skip-output-check",action="store_true");args=p.parse_args()
 print(json.dumps(validate(not args.skip_output_check,True),indent=2))

if __name__=="__main__":main()
