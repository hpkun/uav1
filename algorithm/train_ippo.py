"""Fresh PS-IPPO v3.41 training; smoke retains full configured networks."""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
import json
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))
import yaml
from algorithm.train_mappo import resolved, default_output_dir, ensure_fresh_output_directory, TeeOutput, write_yaml_snapshot
from algorithm.ippo.runner import IPPOTrainingRunner
from algorithm.ippo.protocol import require_cuda, validate_algorithm_config
from algorithm.common.protocol import config_sha256

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-config',default='configs/combat_environment_v341.yaml')
    parser.add_argument('--algorithm-config',default='configs/ippo_4v4_v341.yaml')
    parser.add_argument('--output-dir')
    parser.add_argument('--device',choices=('cuda',),default='cuda')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--num-envs',type=int)
    parser.add_argument('--total-sampled-steps',type=int)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    env=yaml.safe_load(resolved(args.env_config).read_text(encoding='utf-8'))
    config=yaml.safe_load(resolved(args.algorithm_config).read_text(encoding='utf-8'))
    validate_algorithm_config(config,env)
    require_cuda(args.device)
    seed=config['training']['seed'] if args.seed is None else args.seed
    num_envs=config['training']['num_train_envs'] if args.num_envs is None else args.num_envs
    budget=args.total_sampled_steps
    if budget is None:
        budget=4096 if args.smoke else config['training']['total_sampled_steps']
    if args.smoke and budget > 16384:
        raise ValueError('smoke is bounded to at most 16384 environment transitions')
    output=resolved(args.output_dir) if args.output_dir else default_output_dir(seed)
    ensure_fresh_output_directory(output)
    write_yaml_snapshot(output/'env_config.yaml',env)
    write_yaml_snapshot(output/'algorithm_config.yaml',config)
    with (output/'train.log').open('a',encoding='utf-8') as stream:
        with redirect_stdout(TeeOutput(sys.stdout,stream)), redirect_stderr(TeeOutput(sys.stderr,stream)):
            runner=IPPOTrainingRunner(env,config,num_envs,budget,args.device,seed,output,args.smoke)
            run=dict(runner.startup_summary(),seed=seed,num_envs=num_envs,
                smoke=args.smoke,environment_config_sha256=config_sha256(env),
                algorithm_config_sha256=config_sha256(config),
                environment_config_path=str(resolved(args.env_config)),
                algorithm_config_path=str(resolved(args.algorithm_config)),
                evaluation_seed_base=runner.evaluation_seeds[0],
                bit_exact_training_resume=False)
            (output/'run_config.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
            print(runner.start_log_line(),flush=True)
            summary=runner.run()
            (output/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            print(runner.done_log_line(summary),flush=True)

if __name__=='__main__':
    main()
