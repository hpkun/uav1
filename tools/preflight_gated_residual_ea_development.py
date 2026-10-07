"""Static, development-only preflight for the 4x3 GREA architecture screen."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config

MANIFEST=ROOT/"experiments/gated_residual_ea_development_manifest.json"
TRAINING_SEEDS=(4101,4102,4103)
METHODS=("MAPPO","Full EA","Residual EA","Gated Residual EA")
VALIDATION=(32_000_000,32_000_019)
RESERVED=(33_000_000,33_000_199)
OFF_MODULES=("wave_balancing","wave_context","recurrent_memory","popart","multi_wave_reward",
             "warm_start","curriculum","policy_anchor","advantage_priority","ppo_stabilization","actor_lr_decay")
EXPECTED_ENTITY={
 "MAPPO":{"enabled":False,"entity_dim":32,"attention_heads":2},
 "Full EA":{"enabled":True,"mode":"replacement","entity_dim":32,"attention_heads":2},
 "Residual EA":{"enabled":True,"mode":"residual","entity_dim":32,"attention_heads":2},
 "Gated Residual EA":{"enabled":True,"mode":"gated_residual","entity_dim":32,"attention_heads":2,"initial_gate":.05},
}
EXPECTED_TRAINING={"actor_learning_rate":3e-4,"critic_learning_rate":3e-4,"gamma":.999,"gae_lambda":.95,
 "clip_ratio":.2,"value_loss_coefficient":.5,"entropy_coefficient":.01,"max_grad_norm":.5,
 "rollout_steps":256,"ppo_epochs":10,"minibatch_size":512,"num_train_envs":24,
 "total_sampled_steps":400_000,"evaluation_episodes":20,"evaluation_interval_sampled_steps":100_000,"device":"cuda"}

def _normalized(config):
 value=deepcopy(config);value["modules"]["entity_attention"]="METHOD_SPECIFIC_ENTITY_BLOCK";value["training"]["seed"]="RUNTIME_TRAINING_SEED"
 return value

def validate(check_outputs=True):
 manifest=json.loads(MANIFEST.read_text(encoding="utf-8"));runs=manifest["runs"]
 if manifest.get("protocol_role")!="development_only":raise RuntimeError("protocol must be development_only")
 if len(runs)!=12:raise RuntimeError("development matrix must contain exactly 12 runs")
 pairs={(run["method"],int(run["training_seed"])) for run in runs}
 expected_pairs={(method,seed) for method in METHODS for seed in TRAINING_SEEDS}
 if pairs!=expected_pairs or len(pairs)!=len(runs):raise RuntimeError("matrix must be exactly four methods x three unique per-method seeds")
 outputs=[run["output_dir"] for run in runs]
 if len(outputs)!=len(set(outputs)):raise RuntimeError("run output directories are not unique")
 if manifest.get("training_seeds")!=list(TRAINING_SEEDS) or manifest.get("budget_sampled_steps")!=400_000:raise RuntimeError("manifest seed/budget mismatch")
 validation=manifest["validation"];reserved=manifest["reserved_untouched_future_final_test"]
 if (validation["seed_start"],validation["seed_end"],validation["episodes"])!=(*VALIDATION,20):raise RuntimeError("validation must be exactly 32M..32M+19")
 if (reserved["seed_start"],reserved["seed_end"],reserved["episodes"],reserved["executed"])!=(*RESERVED,200,False):raise RuntimeError("future final test must be untouched 33M..33M+199")
 validation_set=set(range(VALIDATION[0],VALIDATION[1]+1));reserved_set=set(range(RESERVED[0],RESERVED[1]+1))
 forbidden=set(range(29_000_000,29_000_020))|set(range(30_000_000,30_000_200))|set(range(31_000_000,31_000_020))
 if validation_set&reserved_set or (validation_set|reserved_set)&forbidden:raise RuntimeError("development/final/legacy seed ranges overlap")
 configs={}
 env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
 if env.get("environment_variant")!="persistent_wave_v2":raise RuntimeError("environment is not persistent_wave_v2")
 for run in runs:
  config=load_config(ROOT/run["config_path"]);configs.setdefault(run["method"],config)
  if config!=configs[run["method"]]:raise RuntimeError(f"{run['method']} resolves inconsistently")
  training=config["training"]
  for key,expected in EXPECTED_TRAINING.items():
   if training.get(key)!=expected:raise RuntimeError(f"{run['method']} training.{key} mismatch")
  if any(bool(config["modules"].get(name,{}).get("enabled",False)) for name in OFF_MODULES):raise RuntimeError(f"{run['method']} has a forbidden module enabled")
  if config["modules"].get("entity_attention")!=EXPECTED_ENTITY[run["method"]]:raise RuntimeError(f"{run['method']} entity block mismatch")
  protocol=config.get("development_protocol",{})
  if protocol.get("role")!="development_only" or protocol.get("checkpoint_selection",{}).get("primary")!="latest_at_budget":raise RuntimeError("development checkpoint protocol mismatch")
  if config["implementation"].get("evaluation_seed_base")!=VALIDATION[0]:raise RuntimeError("evaluation seed base is not 32M")
  output=ROOT/run["output_dir"]
  if check_outputs and output.exists() and any(output.iterdir()):raise FileExistsError(f"development output contains results: {output}")
 normalized=[_normalized(configs[name]) for name in METHODS]
 if any(value!=normalized[0] for value in normalized[1:]):raise RuntimeError("method configs differ outside entity_attention/training seed")
 return {"status":"READY_FOR_GATED_RESIDUAL_EA_DEVELOPMENT","protocol_role":"development_only","planned_runs":12,
  "methods":list(METHODS),"training_seeds":list(TRAINING_SEEDS),"budget_sampled_steps":400_000,
  "validation_seed_range":list(VALIDATION),"reserved_untouched_future_final_test":list(RESERVED),
  "other_modules":"all_off","primary_checkpoint":"latest.pt@400000","outputs_checked":bool(check_outputs)}

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--skip-output-check",action="store_true");args=parser.parse_args()
 print(json.dumps(validate(not args.skip_output_check),indent=2))

if __name__=="__main__":main()

