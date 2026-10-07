"""One 96-transition-per-branch CUDA smoke using non-reserved test scenarios."""
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
from algorithm.modular_mappo.protocol import validate_fbmr_stage2_branch
from algorithm.train_modular_mappo import load_config

SOURCE=ROOT/"outputs/formal_eawb/mappo_seed3101/latest.pt"
TEST_EVAL_BASE=12_340_000

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def smoke_config(path):
 config=deepcopy(load_config(ROOT/path));config["training"].update(total_sampled_steps=900096,rollout_steps=4,ppo_epochs=1,evaluation_episodes=2)
 config["implementation"].update(evaluation_seed_base=TEST_EVAL_BASE,checkpoint_interval_sampled_steps=9_999_999)
 config["runtime_logging"]["console_interval_sampled_steps"]=9_999_999
 return config

def main():
 if not torch.cuda.is_available():raise RuntimeError("CUDA is mandatory for FBMR Stage-2 smoke")
 env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"));state=torch.load(SOURCE,map_location="cuda",weights_only=False);source_hash=digest(SOURCE)
 # Validate the untouched formal configs before applying smoke-only runtime reductions.
 for path in ("configs/dev_fbmr_mappo_continuation_1200k.yaml","configs/dev_fbmr_mean_residual_1200k.yaml"):
  validate_fbmr_stage2_branch(state,env,load_config(ROOT/path),{"training_seed":3101,"training_num_envs":24,"training_smoke":False})
 results={}
 with tempfile.TemporaryDirectory(prefix="fbmr_stage2_smoke_") as temp:
  for name,path,intervention in (("mappo_cont","configs/dev_fbmr_mappo_continuation_1200k.yaml","mappo_continuation"),("fbmr","configs/dev_fbmr_mean_residual_1200k.yaml","frozen_base_mean_residual")):
   config=smoke_config(path);output=Path(temp)/name
   runner=ModularMAPPOTrainingRunner(env,config,24,900096,"cuda",3101,output,False,resume_mode=True,branch_provenance={"intervention":intervention,"smoke_only":True})
   runner.branch_from(SOURCE,intervention,source_hash)
   actor_opt_restored=bool(runner.trainer.actor_optimizer.state)
   frozen_before={n:p.detach().clone() for n,p in runner.trainer.actor.frozen_baseline_named_parameters()}
   summary=runner.run();latest=torch.load(output/"latest.pt",map_location="cuda",weights_only=False)
   result={"sampled_steps":summary["sampled_steps"],"source_sampled_steps":900000,"added_transitions":summary["sampled_steps"]-900000,"latest_round_trip":int(latest["sampled_steps"])==summary["sampled_steps"],"actor_optimizer_restored":actor_opt_restored,"critic_optimizer_state_nonempty":bool(runner.trainer.critic_optimizer.state),"rng_restored":runner.trainer.rng_restore_metadata["rng_state_restored"]}
   if intervention=="frozen_base_mean_residual":
    result.update({"frozen_exact":all(torch.equal(frozen_before[n],p) for n,p in runner.trainer.actor.frozen_baseline_named_parameters()),"frozen_drift_max":runner.trainer.frozen_actor_drift_metrics()["frozen_base_parameter_drift_max"],"delta_mu_finite":torch.isfinite(runner.trainer.actor.entity_mean_adapter.weight).all().item(),"delta_mu_learned":bool(torch.count_nonzero(runner.trainer.actor.entity_mean_adapter.weight)),"log_std_source":latest["extra"]["log_std_source"],"actor_optimizer_reset":not actor_opt_restored})
   results[name]=result
 if digest(SOURCE)!=source_hash:raise RuntimeError("source checkpoint changed during smoke")
 if results["mappo_cont"]["actor_optimizer_restored"] is not True:raise RuntimeError("control actor optimizer was not restored")
 if not results["fbmr"].get("frozen_exact") or not results["fbmr"].get("delta_mu_learned"):raise RuntimeError("FBMR smoke invariant failed")
 print(json.dumps({"status":"FBMR_STAGE2_SMOKE_PASS","device":torch.cuda.get_device_name(0),"test_evaluation_seed_range":[TEST_EVAL_BASE,TEST_EVAL_BASE+1],"reserved_33m_used":False,"development_34m_used":False,"source_checkpoint_unchanged":True,"branches":results},indent=2))

if __name__=="__main__":main()
