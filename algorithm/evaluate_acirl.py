"""Stateful ACIRL evaluation using strict training snapshots and isolated RNG."""
import argparse
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.acirl.evaluation import evaluate_checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--env-config');p.add_argument('--algorithm-config')
    p.add_argument('--device',default='cuda',choices=('cuda',));p.add_argument('--mode',choices=('deterministic','stochastic'),default='deterministic')
    p.add_argument('--seed-base',type=int,default=10000000);p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seeds',nargs='+',type=int);p.add_argument('--holdout',action='store_true');p.add_argument('--output')
    a=p.parse_args()
    def resolved(path):return Path(path) if Path(path).is_absolute() else ROOT/path
    checkpoint=resolved(a.checkpoint)
    env=yaml.safe_load(resolved(a.env_config or checkpoint.parent/'env_config.yaml').read_text(encoding='utf-8'))
    config=yaml.safe_load(resolved(a.algorithm_config or checkpoint.parent/'algorithm_config.yaml').read_text(encoding='utf-8'))
    seeds=a.seeds if a.seeds else range(a.seed_base,a.seed_base+a.episodes)
    result=evaluate_checkpoint(checkpoint,env,config,a.device,seeds,a.mode,a.holdout)
    text=json.dumps(result,indent=2,allow_nan=False)
    if a.output:
        output=resolved(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
