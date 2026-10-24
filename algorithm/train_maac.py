"""Continuous off-policy MAAC training for registered versions; CUDA only."""
import argparse
import json
from pathlib import Path
import sys
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from algorithm.maac.protocol import validate_config_pair
from algorithm.maac.protocol import IMPLEMENTATION_VERSION
from algorithm.maac.protocol import validate_checkpoint_for_resume
from algorithm.maac.runner import MAACTrainingRunner
from algorithm.maac.metrics import DWELL_VERSIONS,EVENT_SEMANTICS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-config',default='configs/combat_environment_v329.yaml')
    p.add_argument('--algorithm-config',default='configs/maac_5v5_v329.yaml')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--resume',help='Validate identity, then reject unsupported exact resume')
    p.add_argument('--device',choices=['cuda'],default='cuda')
    p.add_argument('--seed',type=int);p.add_argument('--num-envs',type=int)
    p.add_argument('--total-sampled-steps',type=int);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; unavailable')
    resolve=lambda x:Path(x) if Path(x).is_absolute() else ROOT/x
    ec=yaml.safe_load(resolve(args.env_config).read_text());ac=yaml.safe_load(resolve(args.algorithm_config).read_text())
    validate_config_pair(ec,ac)
    if args.resume:
        state=torch.load(resolve(args.resume),map_location='cuda',weights_only=False)
        validate_checkpoint_for_resume(state,ec,ac)
    out=resolve(args.output_dir)
    if out.exists() and any(out.iterdir()):raise RuntimeError('MAAC output directory must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('env_config.yaml',ec),('algorithm_config.yaml',ac)]:
        (out/name).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
    runner=MAACTrainingRunner(ec,ac,out,args.num_envs,args.total_sampled_steps,args.seed,args.device,args.smoke)
    mechanism={}
    if ec['environment_version']in ('3.48','3.49'):
        from env.v348_metrics import mechanism_metrics
        mechanism=mechanism_metrics(ec['environment_version'])
    (out/'run_config.json').write_text(json.dumps(dict(seed=runner.seed,num_envs=runner.num_envs,
        total_sampled_steps=runner.total_steps,device='cuda',smoke=args.smoke,
        implementation_version=IMPLEMENTATION_VERSION,variant=ac['implementation']['variant'],
        effective_hidden_dim=32 if args.smoke else ac['network']['hidden_dim'],
        effective_minibatch_size=runner.batch_size,effective_learning_starts=runner.learning_starts,
        effective_sampled_steps_per_update=runner.update_interval,
        effective_baseline_samples=runner.trainer.baseline_samples,
        effective_replay_capacity=runner.replay.capacity,
        **(dict(event_semantics=EVENT_SEMANTICS,legacy_fire_metrics_applicable=False)
           if ec['environment_version'] in DWELL_VERSIONS else {}),
        **mechanism,formal_exact_resume_supported=False,replay_buffer_in_checkpoint=False),indent=2))
    summary=runner.run();(out/'run_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(f"[DONE] MAAC | steps={summary['sampled_steps']} | saved={out}",flush=True)


if __name__=='__main__':main()
