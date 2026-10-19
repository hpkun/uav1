"""Independent ACIRL training CLI; smoke preserves all formal network widths."""
import argparse
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.acirl.runner import ACIRLTrainingRunner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-config',default='configs/combat_environment_v339.yaml')
    p.add_argument('--algorithm-config',default='configs/acirl_4v4_v339.yaml')
    p.add_argument('--output-dir',required=True);p.add_argument('--device',choices=('cuda',),default='cuda')
    p.add_argument('--seed',type=int);p.add_argument('--num-envs',type=int);p.add_argument('--total-sampled-steps',type=int)
    p.add_argument('--smoke',action='store_true');p.add_argument('--warm-start')
    args=p.parse_args()
    def resolved(path):return Path(path) if Path(path).is_absolute() else ROOT/path
    env=yaml.safe_load(resolved(args.env_config).read_text(encoding='utf-8'))
    config=yaml.safe_load(resolved(args.algorithm_config).read_text(encoding='utf-8'))
    runner=ACIRLTrainingRunner(env,config,resolved(args.output_dir),args.device,args.seed,args.num_envs,args.total_sampled_steps,args.smoke,
        resolved(args.warm_start) if args.warm_start else None)
    runner.run()


if __name__=='__main__':main()
