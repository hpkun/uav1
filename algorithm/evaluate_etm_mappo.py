"""Strict deterministic evaluation of standalone navigation checkpoints."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse
import json
import yaml
from algorithm.etm_mappo.checkpoint import load_checkpoint
from algorithm.etm_mappo.evaluation import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--algorithm-config')
    p.add_argument('--seed-base',type=int,default=20000000)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--scene',choices=('simple','random'))
    p.add_argument('--device',choices=('cuda',),default='cuda')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.episodes<=0:raise ValueError('episodes must be positive')
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    config=yaml.safe_load(Path(args.algorithm_config).read_text(encoding='utf-8')) if args.algorithm_config else None
    trainer,state=load_checkpoint(args.checkpoint,args.device,config)
    result=dict(algorithm=state['algorithm'],implementation_version=state['implementation_version'],
        variant=state['network_architecture']['variant'],checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sampled_steps=state['sampled_steps'],config_sha256=state['config_sha256'],
        **evaluate(trainer,range(args.seed_base,args.seed_base+args.episodes),args.scene))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

