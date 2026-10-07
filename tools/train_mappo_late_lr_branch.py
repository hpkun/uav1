#!/usr/bin/env python3
"""Run one audited MAPPO late-LR continuation branch from a common source."""
from __future__ import annotations
import argparse,hashlib,json,sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
import numpy as np,torch,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.common.protocol import config_sha256,runtime_source_manifest
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.train_mappo import TeeOutput,write_yaml_snapshot
from tools.preflight_mappo_late_lr_continuation import SOURCE_STEP,load_source,load_yaml,sha256,validate_configs,CONTROL,TREATMENT

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--algorithm-config",required=True);parser.add_argument("--output-dir",required=True);parser.add_argument("--source-checkpoint",default="outputs/mappo_attention_seed5303_1p5m/checkpoint_1001472.pt");args=parser.parse_args()
 if not torch.cuda.is_available():raise RuntimeError("CUDA mandatory")
 config_path=(ROOT/args.algorithm_config).resolve();output=(ROOT/args.output_dir).resolve();source=(ROOT/args.source_checkpoint).resolve()
 if source!=(ROOT/"outputs/mappo_attention_seed5303_1p5m/checkpoint_1001472.pt").resolve():raise RuntimeError("unexpected source checkpoint")
 if output.exists():raise RuntimeError(f"refusing existing formal output directory: {output}")
 config=load_yaml(config_path);control,treatment=load_yaml(CONTROL),load_yaml(TREATMENT);validate_configs(control,treatment)
 method=config.get("development_method")
 expected={control["development_method"]:control,treatment["development_method"]:treatment}
 if method not in expected or config!=expected[method]:raise RuntimeError("unapproved continuation config")
 env,source_config,state=load_source("cuda")
 output.mkdir(parents=True);write_yaml_snapshot(output/"env_config.yaml",env);write_yaml_snapshot(output/"algorithm_config.yaml",config)
 manifest=runtime_source_manifest(ROOT);source_digest=sha256(source)
 runner=MAPPOTrainingRunner(env,config,24,1_500_000,"cuda",5303,output,False)
 extra=runner.trainer.load(source)
 if runner.trainer.sampled_steps!=SOURCE_STEP:raise RuntimeError("source step changed during load")
 for group in runner.trainer.actor_optimizer.param_groups:group["lr"]=float(config["training"]["actor_learning_rate"])
 for group in runner.trainer.critic_optimizer.param_groups:group["lr"]=float(config["training"]["critic_learning_rate"])
 previous=np.asarray(extra["episode_indices"],dtype=np.int64)
 if previous.shape!=(24,):raise RuntimeError("source episode-index shape mismatch")
 runner.vector.episode_indices=previous+1;runner.observations=runner.vector.reset();runner.alive_masks=runner.vector.current_alive_masks.copy()
 runner.evaluation_history=[];runner.best_evaluation=None
 runner.next_console_log=(SOURCE_STEP//runner.console_interval+1)*runner.console_interval
 runner.next_evaluation=(SOURCE_STEP//runner.evaluation_interval+1)*runner.evaluation_interval
 runner.next_checkpoint=(SOURCE_STEP//runner.checkpoint_interval+1)*runner.checkpoint_interval
 branch={"development_method":method,"parent_checkpoint":str(source.relative_to(ROOT)),"parent_checkpoint_sha256":source_digest,"source_sampled_steps":SOURCE_STEP,"source_training_seed":5303,"target_sampled_steps":1_500_000,"environment_config_sha256":config_sha256(env),"source_algorithm_config_sha256":config_sha256(source_config),"branch_algorithm_config_sha256":config_sha256(config),"runtime_source_manifest_sha256":manifest["runtime_source_manifest_sha256"],"restored":{"actor":True,"critic":True,"actor_optimizer":True,"critic_optimizer":True,"ppo_counters":True,"sampled_and_vector_steps":True,"episode_indices":True,"rng_state":False},"bitwise_continuation_from_original_run_claimed":False,"common_action_noise_trajectory_claimed":False,"actor_learning_rate":float(config["training"]["actor_learning_rate"]),"critic_learning_rate":float(config["training"]["critic_learning_rate"]),"evaluation_seed_base":47_000_000,"evaluation_seed_end":47_000_049}
 (output/"branch_from.json").write_text(json.dumps(branch,indent=2),encoding="utf-8")
 startup=runner.startup_summary();run_config={"device":"cuda","seed":5303,"num_envs":24,"total_sampled_steps":1_500_000,"source_sampled_steps":SOURCE_STEP,"smoke":False,"environment_config_path":str((ROOT/"configs/persistent_wave_v2_environment.yaml").resolve()),"algorithm_config_path":str(config_path),"environment_variant":"persistent_wave_v2","environment_version":str(env["environment_version"]),"algorithm":"MAPPO","development_method":method,"critic_type":"attention","actor_learning_rate":float(config["training"]["actor_learning_rate"]),"critic_learning_rate":float(config["training"]["critic_learning_rate"]),"actor_parameter_count":startup["actor_parameter_count"],"critic_parameter_count":startup["critic_parameter_count"],"total_parameter_count":startup["total_parameter_count"],"output_dir":str(output),"resume_checkpoint":str(source),"parent_checkpoint_sha256":source_digest,"environment_config_sha256":config_sha256(env),"algorithm_config_sha256":config_sha256(config),"runtime_source_manifest_sha256":manifest["runtime_source_manifest_sha256"],"evaluation_seed_base":47_000_000,"evaluation_seed_end":47_000_049,"evaluation_episodes":50}
 (output/"run_config.json").write_text(json.dumps(run_config,indent=2),encoding="utf-8")
 with (output/"train.log").open("a",encoding="utf-8") as log_stream,redirect_stdout(TeeOutput(sys.stdout,log_stream)):
  print(runner.start_log_line(),flush=True);print(f"[BRANCH] source_steps={SOURCE_STEP} | method={method} | actor_lr={config['training']['actor_learning_rate']} | rng_restore=unavailable",flush=True)
  summary=runner.run();runner.save_checkpoint(output/"final.pt")
  summary.update({"development_method":method,"source_sampled_steps":SOURCE_STEP,"parent_checkpoint_sha256":source_digest,"actor_learning_rate":float(config["training"]["actor_learning_rate"]),"critic_learning_rate":float(config["training"]["critic_learning_rate"])})
  (output/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");print(runner.done_log_line(summary),flush=True)

if __name__=="__main__":main()
