#!/usr/bin/env python3
"""Tiny CUDA rollout/update/checkpoint smoke for both critic baselines."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.common.evaluator import evaluate
OUT=ROOT/"outputs/mappo_critic_baseline_cuda_smoke"
def load(p):return yaml.safe_load((ROOT/p).read_text(encoding="utf-8"))
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,default=OUT);args=parser.parse_args();output=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
 if output.exists():raise FileExistsError(output)
 if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 env=load("configs/persistent_wave_v2_environment.yaml");env["simulation"]["max_steps"]=8;reports={}
 for name in ("mlp","attention"):
  cfg=load(f"configs/mappo_{name}_baseline_1p5m.yaml");path=output/name
  runner=MAPPOTrainingRunner(env,cfg,num_envs=1,total_sampled_steps=8,device="cuda",seed=5303,output_dir=path,smoke=True)
  summary=runner.run();state=torch.load(path/"latest.pt",map_location="cuda",weights_only=False)
  restored=MAPPOTrainingRunner(env,cfg,num_envs=1,total_sampled_steps=8,device="cuda",seed=5303,output_dir=path/"restore",smoke=True)
  try:restored.trainer.load(path/"latest.pt")
  finally:restored.vector.close()
  evaluation=evaluate(runner.trainer,env,[88_000_000]);metrics=summary["last_update_metrics"];reports[name]={"critic_type":summary["critic_type"],"finite":all(__import__('math').isfinite(float(v)) for v in metrics.values()),"actor_grad_norm":metrics["actor_grad_norm"],"critic_grad_norm":metrics["critic_grad_norm"],"sampled_steps":summary["sampled_steps"],"checkpoint_critic_type":state["critic_type"],"tiny_evaluation_finite":all(__import__('math').isfinite(float(v)) for v in evaluation.values())}
 report={"status":"MAPPO_CRITIC_BASELINE_CUDA_SMOKE_PASS","device":torch.cuda.get_device_name(0),"arms":reports,"training_environment_steps":16,"tiny_evaluation_max_steps":16,"smoke_evaluation_seed":88_000_000,"formal_training":False,"uses_44m":False,"uses_45m":False}
 if not all(v["finite"] and v["tiny_evaluation_finite"] and v["actor_grad_norm"]>0 and v["critic_grad_norm"]>0 and v["critic_type"]==k and v["checkpoint_critic_type"]==k for k,v in reports.items()):raise RuntimeError(report)
 output.mkdir(parents=True,exist_ok=True);(output/"smoke_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
