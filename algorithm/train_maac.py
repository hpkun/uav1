"""MAAC-inspired on-policy v3.29 training; only explicit CUDA execution."""
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from algorithm.maac.protocol import validate_config_pair
from algorithm.maac.runner import MAACTrainingRunner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-config',default='configs/combat_environment_v329.yaml')
    p.add_argument('--algorithm-config',default='configs/maac_5v5_v329.yaml')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',choices=['cuda'],default='cuda')
    p.add_argument('--seed',type=int);p.add_argument('--num-envs',type=int)
    p.add_argument('--total-sampled-steps',type=int);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; unavailable')
    resolve=lambda x:Path(x) if Path(x).is_absolute() else ROOT/x
    ec=yaml.safe_load(resolve(args.env_config).read_text());ac=yaml.safe_load(resolve(args.algorithm_config).read_text())
    validate_config_pair(ec,ac);out=resolve(args.output_dir)
    if out.exists() and any(out.iterdir()):raise RuntimeError('MAAC output directory must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('env_config.yaml',ec),('algorithm_config.yaml',ac)]:
        (out/name).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
    runner=MAACTrainingRunner(ec,ac,out,args.num_envs,args.total_sampled_steps,args.seed,args.device,args.smoke)
    (out/'run_config.json').write_text(json.dumps(dict(seed=runner.seed,num_envs=runner.num_envs,
        total_sampled_steps=runner.total_steps,device='cuda',smoke=args.smoke,
        formal_exact_resume_supported=False),indent=2))
    summary=runner.run();(out/'run_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(f"[DONE] MAAC | steps={summary['sampled_steps']} | saved={out}",flush=True)


if __name__=='__main__':main()
