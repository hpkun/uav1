"""Construct the formal CUDA MAPPO and 16 reset workers, then close.

No rollout, update, evaluation, checkpoint loading or model saving is performed.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from env.config import load_config
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.common.protocol import config_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v340_preflight/readiness')
    args=p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory for formal MAPPO readiness; no CPU fallback')
    env=load_config(ROOT/'configs/combat_environment_v340.yaml')
    algorithm=yaml.safe_load((ROOT/'configs/mappo_4v4_v340.yaml').read_text())
    runner=MAPPOTrainingRunner(env,algorithm,device='cuda',seed=1,output_dir=args.output)
    try:
        summary=runner.startup_summary()
        assert summary['num_envs_M']==16 and summary['total_sampled_steps']==3000000
        assert runner.observations.shape==(16,4,45) and np.isfinite(runner.observations).all()
        assert runner.trainer.sampled_steps==0
        assert all('CombatEnvironmentV340' in s for s in runner.vector.worker_environment_classes)
        assert next(runner.trainer.actor.parameters()).device.type=='cuda'
        assert next(runner.trainer.critic.parameters()).device.type=='cuda'
        summary.update(cuda_available=True,cuda_name=torch.cuda.get_device_name(),
            sampled_steps=0,optimizer_updates=0,checkpoint_loaded=False,training_started=False,
            worker_classes=runner.vector.worker_environment_classes,
            worker_fixed_policy_classes=runner.vector.worker_fixed_policy_classes,
            observation_shape=list(runner.observations.shape),
            environment_config_sha256=config_sha256(env),algorithm_config_sha256=config_sha256(algorithm))
        target=args.output/'readiness.json'
        target.write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2),flush=True)
    finally:runner.vector.close()


if __name__=='__main__':main()
