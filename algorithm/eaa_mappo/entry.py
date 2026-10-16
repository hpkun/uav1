"""Fresh-run CLI shared by EA/EAA and the native MAPPO entry's CA dispatch."""
from contextlib import redirect_stdout
import json
from pathlib import Path
import yaml
from algorithm.train_mappo import resolved, default_output_dir, ensure_fresh_output_directory, TeeOutput
from .runner import EAAMAPPOTrainingRunner


def run_variant(args):
    if getattr(args,'resume',None):raise RuntimeError('EAA family v1 training resume is not implemented')
    env = yaml.safe_load(resolved(args.env_config).read_text(encoding='utf-8'))
    config = yaml.safe_load(resolved(args.algorithm_config).read_text(encoding='utf-8'))
    smoke = bool(args.smoke)
    seed = config['training']['seed'] if args.seed is None else args.seed
    output = resolved(args.output_dir) if args.output_dir else default_output_dir(seed)
    ensure_fresh_output_directory(output)
    runner = EAAMAPPOTrainingRunner(env,config,
        args.num_envs if args.num_envs is not None else (2 if smoke else None),
        args.total_sampled_steps if args.total_sampled_steps is not None else (512 if smoke else None),
        args.device,seed,output,smoke)
    try:
        for name,value in [('env_config.yaml',env),('algorithm_config.yaml',config)]:
            (output/name).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
        (output/'run_config.json').write_text(json.dumps(runner.startup_summary(),indent=2),encoding='utf-8')
        with (output/'train.log').open('a',encoding='utf-8') as log:
            import sys
            with redirect_stdout(TeeOutput(sys.stdout,log)):
                print(runner.start_log_line(),flush=True)
                summary = runner.run()
                (output/'run_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
                print(runner.done_log_line(summary).replace('algorithm=MAPPO',f"algorithm={config['algorithm']}",1),flush=True)
    finally:
        runner.vector.close()
