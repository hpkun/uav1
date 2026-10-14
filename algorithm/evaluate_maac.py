"""Strict CUDA deterministic evaluation of the v3.29 continuous MAAC adaptation."""
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from algorithm.maac.trainer import MAACTrainer
from algorithm.maac.protocol import validate_checkpoint
from algorithm.common.evaluator import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--env-config',default='configs/combat_environment_v329.yaml')
    p.add_argument('--algorithm-config',default='configs/maac_5v5_v329.yaml')
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed-base',type=int,default=10000000)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory')
    if args.episodes<=0:raise ValueError('episodes must be positive')
    ec=yaml.safe_load(Path(args.env_config).read_text());ac=yaml.safe_load(Path(args.algorithm_config).read_text())
    state=torch.load(args.checkpoint,map_location='cuda',weights_only=False);validate_checkpoint(state,ec,ac)
    trainer=MAACTrainer(ec,ac,'cuda',state['training_seed'],state['extra']['training_smoke']);trainer.load(args.checkpoint)
    row=evaluate(trainer,ec,range(args.seed_base,args.seed_base+args.episodes))
    row.update(algorithm='MAAC',variant=ac['implementation']['variant'],environment_version=ec['environment_version'],
        evaluation_mode='deterministic',fire_rate=row['red_attempt_episode_rate'],kill_rate=row['red_kill_episode_rate'])
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(row,indent=2,allow_nan=False))
    print(json.dumps(row,indent=2))


if __name__=='__main__':main()
