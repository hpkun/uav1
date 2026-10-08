"""CUDA-only formal RMAPPO entry, with isolated immutable run snapshots."""
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import json
import torch
import yaml
from algorithm.train_mappo import (TeeOutput, resolved, resolve_run_paths,
    validate_resume_config_snapshots, write_yaml_snapshot, load_run_config,
    resolve_runtime_settings, reject_stale_resume_checkpoint, prepare_resume_rollback)
from algorithm.common.protocol import config_sha256
from algorithm.rmappo.runner import RMAPPOTrainingRunner
from algorithm.rmappo.protocol import validate_checkpoint, require_cuda


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",choices=["cuda"])
    parser.add_argument("--seed",type=int)
    parser.add_argument("--total-sampled-steps",type=int)
    parser.add_argument("--num-envs",type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--env-config",default="configs/combat_environment_v25.yaml")
    parser.add_argument("--algorithm-config",default="configs/rmappo_5v5.yaml")
    parser.add_argument("--smoke",action="store_true",default=None)
    args = parser.parse_args()
    env_path,algorithm_path = resolved(args.env_config),resolved(args.algorithm_config)
    env_config,config = (yaml.safe_load(path.read_text(encoding="utf-8")) for path in (env_path,algorithm_path))
    require_cuda(args.device or config["training"]["device"])
    if args.output_dir is None and args.resume is None:
        args.output_dir = f"outputs/rmappo_v25_5v5_seed{config['training']['seed'] if args.seed is None else args.seed}"
    if args.smoke and args.total_sampled_steps is None and args.resume is None:
        args.total_sampled_steps = 512
    output,checkpoint = resolve_run_paths(args.output_dir,args.resume,
        config["training"]["seed"] if args.seed is None else args.seed)
    state,stored = None,None
    if checkpoint is not None:
        reject_stale_resume_checkpoint(output,checkpoint)
        validate_resume_config_snapshots(output,env_config,config)
        state = torch.load(checkpoint,map_location="cpu",weights_only=False)
        validate_checkpoint(state,env_config,config)
        stored = load_run_config(output)
    runtime = resolve_runtime_settings(config,seed=args.seed,num_envs=args.num_envs,
        total_sampled_steps=args.total_sampled_steps,device=args.device,smoke=args.smoke,
        run_config=stored,checkpoint_state=state)
    runner = RMAPPOTrainingRunner(env_config,config,runtime["num_envs"],runtime["total_sampled_steps"],
        runtime["device"],runtime["seed"],output,runtime["smoke"])
    try:
        startup = runner.startup_summary()
        if checkpoint is None:
            write_yaml_snapshot(output/"env_config.yaml",env_config)
            write_yaml_snapshot(output/"algorithm_config.yaml",config)
            run_config = {**startup,"num_envs":runner.num_envs,"smoke":runner.smoke,
                "environment_config_path":str(env_path),"algorithm_config_path":str(algorithm_path),
                "environment_config_sha256":config_sha256(env_config),
                "algorithm_config_sha256":config_sha256(config),"output_dir":str(output.resolve()),
                "training_gamma":runner.trainer.gamma,"resume_checkpoint":None}
            (output/"run_config.json").write_text(json.dumps(run_config,indent=2),encoding="utf-8")
        with (output/"train.log").open("a",encoding="utf-8") as log:
            with redirect_stdout(TeeOutput(sys.stdout,log)):
                print(runner.start_log_line(),flush=True)
                if checkpoint is not None:
                    runner.resume(checkpoint)
                    rollback = prepare_resume_rollback(output,checkpoint,int(state["sampled_steps"]))
                    record = {"timestamp":datetime.now().astimezone().isoformat(),
                        "checkpoint":str(checkpoint),"checkpoint_sampled_steps":int(state["sampled_steps"]),
                        "seed":runner.seed,"num_envs":runner.num_envs,
                        "total_sampled_steps":runner.total_sampled_steps,
                        "extended_training_target":runtime["extended_training_target"],
                        "recurrent_state_reset":True,**rollback}
                    with (output/"resume_history.jsonl").open("a",encoding="utf-8") as stream:
                        stream.write(json.dumps(record)+"\n")
                summary = runner.run()
                (output/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
                print(runner.done_log_line(summary),flush=True)
    finally:
        runner.vector.close()


if __name__ == "__main__":
    main()
