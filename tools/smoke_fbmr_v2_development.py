"""One 96-transition CUDA smoke for FBMR V2 using non-33M/non-34M scenarios."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modular_mappo.protocol import validate_fbmr_v2_stage2_branch
from algorithm.train_modular_mappo import load_config

SOURCE=ROOT/"outputs/formal_eawb/mappo_seed3101/latest.pt"
TEST_EVAL_BASE=12_350_000

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
 if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for FBMR V2 smoke")
 env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"));formal=load_config(ROOT/"configs/dev_fbmr_v2_dual_bound_1200k.yaml")
 state=torch.load(SOURCE,map_location="cuda",weights_only=False);source_hash=digest(SOURCE)
 validate_fbmr_v2_stage2_branch(state,env,formal,{"training_seed":3101,"training_num_envs":24,"training_smoke":False})
 config=deepcopy(formal);config["training"].update(total_sampled_steps=900096,rollout_steps=4,ppo_epochs=1,evaluation_episodes=2)
 config["implementation"].update(evaluation_seed_base=TEST_EVAL_BASE,checkpoint_interval_sampled_steps=9_999_999)
 config["runtime_logging"]["console_interval_sampled_steps"]=9_999_999
 with tempfile.TemporaryDirectory(prefix="fbmr_v2_smoke_") as temp:
  output=Path(temp)/"v2"
  runner=ModularMAPPOTrainingRunner(env,config,24,900096,"cuda",3101,output,False,resume_mode=True,branch_provenance={"intervention":"frozen_base_dual_bounded_mean_residual","smoke_only":True})
  runner.branch_from(SOURCE,"frozen_base_dual_bounded_mean_residual",source_hash)
  actor_optimizer_reset=not bool(runner.trainer.actor_optimizer.state)
  frozen={n:p.detach().clone() for n,p in runner.trainer.actor.frozen_baseline_named_parameters()}
  summary=runner.run();latest_path=output/"latest.pt";latest=torch.load(latest_path,map_location="cuda",weights_only=False)
  restored=ModularMAPPOTrainingRunner(env,config,24,900096,"cuda",3101,Path(temp)/"restored",False,resume_mode=True)
  restored.trainer.load(latest_path)
  architecture=latest["extra"]["network_architecture"]
  checks={"sampled_steps":int(summary["sampled_steps"]),"added_transitions":int(summary["sampled_steps"])-900000,
          "checkpoint_round_trip":restored.trainer.sampled_steps==summary["sampled_steps"],"actor_optimizer_reset":actor_optimizer_reset,
          "critic_optimizer_restored":runner.trainer.fbmr_branch_metadata["critic_optimizer_restored"],"rng_restored":runner.trainer.rng_restore_metadata["rng_state_restored"],
          "frozen_exact":all(torch.equal(frozen[n],p) for n,p in runner.trainer.actor.frozen_baseline_named_parameters()),
          "frozen_drift_max":runner.trainer.frozen_actor_drift_metrics()["frozen_base_parameter_drift_max"],
          "delta_finite":bool(torch.isfinite(runner.trainer.actor.entity_mean_adapter.weight).all()),
          "dual_bound_enabled":architecture["dual_bound_enabled"],"alpha_abs":architecture["alpha_abs"],"alpha_rel":architecture["alpha_rel"],
          "log_std_source":architecture["log_std_source"],"source_checkpoint_unchanged":digest(SOURCE)==source_hash}
 if checks["added_transitions"]!=96 or not all((checks["checkpoint_round_trip"],checks["actor_optimizer_reset"],checks["critic_optimizer_restored"],checks["rng_restored"],checks["frozen_exact"],checks["delta_finite"],checks["dual_bound_enabled"],checks["source_checkpoint_unchanged"])):raise RuntimeError("FBMR V2 smoke invariant failed")
 print(json.dumps({"status":"FBMR_V2_SMOKE_PASS","device":torch.cuda.get_device_name(0),"test_evaluation_seed_range":[TEST_EVAL_BASE,TEST_EVAL_BASE+1],"reserved_33m_used":False,"development_34m_used":False,"checks":checks},indent=2))

if __name__=="__main__":main()
