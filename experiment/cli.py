"""Interception preflight, training, evaluation and reporting commands."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import yaml

from env import make_environment

from .interception import (evaluate_policy,evaluation_interval_for_budget,load_run,read_config,require_matching_attack_mode,
                           summarize,train,write_json)
from .prioritized_replay import ReplayConfig


def parser():
    root=argparse.ArgumentParser(description=__doc__)
    commands=root.add_subparsers(dest='command',required=True)
    for command in ('preflight','train'):
        p=commands.add_parser(command)
        p.add_argument('--env-config',type=Path,required=True)
        p.add_argument('--algorithm-config',type=Path,required=True)
        p.add_argument('--algorithm',choices=('mappo','maddpg','dual_critic'),required=True)
        p.add_argument('--attack-mode',choices=('direct_rush','split_attack','jia_straight'),required=True)
        p.add_argument('--total-sampled-steps',type=int,required=True)
        p.add_argument('--num-envs',type=int,default=1)
        p.add_argument('--seed',type=int,default=7)
        p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        p.add_argument('--eval-interval',type=int,default=None,
                       help='Sampled steps between evaluations; default is 5%% of the training budget')
        p.add_argument('--eval-episodes',type=int,default=10)
        p.add_argument('--log-every-steps',type=int,default=10000)
        p.add_argument('--output-dir',type=Path,required=True)
    p=commands.add_parser('evaluate')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--label',choices=('Best','Last'),default='Best')
    p.add_argument('--attack-mode',choices=('direct_rush','split_attack','jia_straight'))
    p.add_argument('--episodes-per-mode',type=int,default=200)
    p.add_argument('--seed',type=int,default=900000)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--output-dir',type=Path,required=True)
    p=commands.add_parser('report')
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--evaluation',type=Path)
    p.add_argument('--plot',type=Path)
    return root


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command in ('preflight','train'):
        config,parameters,rollout=read_config(args.env_config,args.algorithm_config)
        config=replace(config,attack_mode=args.attack_mode)
        data=yaml.safe_load(args.algorithm_config.read_text(encoding='utf-8'))
        if data['algorithm']!=args.algorithm:raise ValueError('Algorithm config mismatch')
        replay_config=ReplayConfig.from_mapping(data['training'].get('replay'))
        if args.algorithm!='maddpg' and replay_config.mode!='uniform':
            raise ValueError('Prioritized replay supports MADDPG only')
        if (replay_config.mode!='uniform' and not
                0<round(parameters.batch_size*replay_config.prioritized_fraction)<parameters.batch_size):
            raise ValueError('MADDPG replay batch needs both sampling components')
        if args.total_sampled_steps<1 or args.num_envs<1 or args.total_sampled_steps%args.num_envs:
            raise ValueError('Sampled steps must be positive and divisible by num-envs')
        if (args.eval_interval is not None and args.eval_interval<0 or
                args.eval_episodes<1 or args.seed<0 or args.log_every_steps<1):
            raise ValueError('Invalid evaluation interval, count, log cadence or seed')
        eval_interval=(evaluation_interval_for_budget(args.total_sampled_steps,args.num_envs)
                       if args.eval_interval is None else args.eval_interval)
        if args.output_dir.exists():raise FileExistsError(args.output_dir)
        if args.device=='cuda':
            import torch
            if not torch.cuda.is_available():raise ValueError('CUDA is unavailable')
        plan=dict(status='supported',environment_contract=make_environment(config).contract,
                  algorithm=args.algorithm,attack_mode=args.attack_mode,sampled_steps=args.total_sampled_steps,
                  num_envs=args.num_envs,rollout_length=rollout,log_every_steps=args.log_every_steps,
                  eval_interval=eval_interval,eval_episodes=args.eval_episodes,
                  output=str(args.output_dir.resolve()))
        if replay_config.mode!='uniform':plan['replay']=vars(replay_config)
        if args.command=='preflight':
            print(json.dumps(plan,ensure_ascii=False,indent=2))
            return plan
        return train(config,parameters,rollout,args.algorithm,steps=args.total_sampled_steps,num_envs=args.num_envs,
                     seed=args.seed,device=args.device,attack_mode=args.attack_mode,output=args.output_dir,
                     eval_interval=eval_interval,eval_episodes=args.eval_episodes,
                     log_every_steps=args.log_every_steps,replay_config=replay_config)
    if args.command=='evaluate':
        if args.output_dir.exists():raise FileExistsError(args.output_dir)
        if args.episodes_per_mode<1 or args.seed<0:raise ValueError('Invalid evaluation count or seed')
        config,policy,name,model,settings=load_run(args.checkpoint,args.label,args.device)
        requested=args.attack_mode or settings['attack_mode']
        require_matching_attack_mode(settings,requested)
        rows={requested:evaluate_policy(config,policy,requested,args.episodes_per_mode,args.seed)}
        args.output_dir.mkdir(parents=True)
        result=dict(schema='InterceptionEvaluation/1',algorithm=name,checkpoint=str(model.resolve()),
                    attack_modes={mode:summarize(value) for mode,value in rows.items()},
                    seed_start=args.seed,episodes_per_mode=args.episodes_per_mode)
        write_json(args.output_dir/'evaluation.json',result)
        with (args.output_dir/'episodes.jsonl').open('x',encoding='utf-8') as file:
            for mode,episodes in rows.items():
                for episode in episodes:file.write(json.dumps(episode,allow_nan=False)+'\n')
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return result
    run_settings=json.loads((args.run/'settings.json').read_text(encoding='utf-8'))
    mode=run_settings.get('attack_mode')
    if mode not in ('direct_rush','split_attack','jia_straight'):
        raise ValueError('Run training attack mode is unsupported')
    summary=json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
    if args.evaluation:
        evaluation=args.evaluation/'evaluation.json' if args.evaluation.is_dir() else args.evaluation
        evaluation_modes=json.loads(evaluation.read_text(encoding='utf-8'))['attack_modes']
        if set(evaluation_modes)!={mode}:
            raise ValueError('Evaluation mode differs from run training mode')
        summary['evaluation']=evaluation_modes
    if args.plot:
        if args.plot.exists():raise FileExistsError(args.plot)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        points=[json.loads(line) for line in (args.run/'evaluations.jsonl').read_text(encoding='utf-8').splitlines()]
        fig,axis=plt.subplots(figsize=(8,4.5))
        x=[row['step'] for row in points if mode in row['modes']]
        y=[row['modes'][mode]['all_capture_rate'] for row in points if mode in row['modes']]
        if x:axis.plot(x,y,marker='o',label=mode)
        axis.set(xlabel='Sampled steps',ylabel='All-capture rate',ylim=(0,1),title='Interception validation')
        axis.grid(True,alpha=.3)
        if points:axis.legend()
        fig.tight_layout()
        fig.savefig(args.plot)
        plt.close(fig)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':main()
