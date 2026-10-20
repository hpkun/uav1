"""Standalone CUDA training for cooperative navigation; no combat runtime imports."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime
import json
import yaml
from algorithm.etm_mappo.config import runtime_config
from algorithm.etm_mappo.runner import ETMTrainingRunner


class Tee:
    def __init__(self,terminal,log):self.terminal,self.log=terminal,log
    def write(self,value):self.terminal.write(value);self.log.write(value);self.log.flush();return len(value)
    def flush(self):self.terminal.flush();self.log.flush()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algorithm-config','--config',dest='config',default=str(ROOT/'configs/etm_mappo.yaml'))
    p.add_argument('--env-config',help='Optional standalone navigation environment YAML')
    p.add_argument('--seed',type=int)
    p.add_argument('--num-envs',type=int)
    p.add_argument('--total-sampled-steps',type=int)
    p.add_argument('--variant',choices=('B0','B1','B2','B3'))
    p.add_argument('--device',choices=('cuda',),default='cuda')
    p.add_argument('--output-dir')
    p.add_argument('--reset-memory-each-step',action='store_true',default=None)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',help='Unsupported: checkpoints support evaluation reload, not exact training resume')
    args=p.parse_args()
    if args.resume:raise RuntimeError('exact training resume is unsupported; evaluation reload is supported')
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if args.env_config:
        config['environment']=yaml.safe_load(Path(args.env_config).read_text(encoding='utf-8'))
    budget=args.total_sampled_steps if args.total_sampled_steps is not None else (512 if args.smoke else None)
    if args.smoke and budget>16384:raise ValueError('smoke is bounded to 16384 joint transitions')
    config=runtime_config(config,seed=args.seed,num_envs=args.num_envs,total_sampled_steps=budget,
        variant=args.variant,device=args.device,memory_reset_each_step=args.reset_memory_each_step)
    output=Path(args.output_dir) if args.output_dir else ROOT/'outputs'/f'etm_{datetime.now():%Y%m%d_%H%M%S_%f}_seed{config["training"]["seed"]}'
    runner=ETMTrainingRunner(config,output)
    with (output/'train.log').open('a',encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
            print(json.dumps(runner.run_config),flush=True)
            summary=runner.run()
            print(json.dumps({key:summary[key] for key in ('sampled_steps','actor_optimizer_steps','critic_optimizer_steps','return_change')}),flush=True)


if __name__=='__main__':main()

