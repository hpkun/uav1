"""Record a verified interception checkpoint and render an offline HTML replay."""
import argparse
from dataclasses import replace
import gzip
import json
from pathlib import Path

import numpy as np

from env import make_environment
from experiment.interception import digest,load_run,require_matching_attack_mode


def collect(config,policy,mode,seed):
    if mode!=config.attack_mode:
        raise ValueError('Replay mode differs from checkpoint training mode')
    env=make_environment(replace(config,attack_mode=mode))
    obs,info=env.reset(seed=seed)
    frames=[]
    events=[]

    def frame(step,actions=None,physical=None):
        return dict(step=step,time_s=step*config.dt,positions=env.positions.tolist(),
                    velocities=env.velocities.tolist(),attacker_alive=env.attacker_alive.tolist(),
                    defender_alive=env.defender_alive.tolist(),
                    headings=(env.headings.tolist() if hasattr(env,'headings') else None),
                    assigned_target_ids=(env.assignment.targets.tolist() if env.assignment is not None else None),
                    defender_actions=None if actions is None else np.asarray(actions).tolist(),
                    physical_commands=None if physical is None else np.asarray(physical).tolist())

    frames.append(frame(0))
    while True:
        actions=policy.policy_actions(obs[None],info['active_masks'][None],deterministic=True)[0]
        obs,_,terminated,truncated,info=env.step(actions)
        frames.append(frame(env.step_count,actions,info.get('physical_commands')))
        events.extend(info['events'])
        if terminated or truncated:
            break
    trace=dict(schema=('InterceptionReplay/3' if env.contract['version'] in ('interception_13','interception_14') else
                       'InterceptionReplay/2' if mode=='radial_straight' else 'InterceptionReplay/1'),mode=mode,seed=seed,
               config=env.contract['config'],frames=frames,events=events,summary=info['episode_summary'])
    env.close()
    return trace


def validate_trace(trace):
    frames=trace['frames'];events=trace['events'];summary=trace['summary']
    pairs=trace['config'].get('active_pairs') if trace['config'].get('scenario')=='radial_scale' else None
    attackers=defenders=pairs if pairs is not None else 1 if trace['config'].get('scenario')=='one_vs_one' else 5
    if len(frames)!=summary['steps']+1 or [row['step'] for row in frames]!=list(range(len(frames))):
        raise ValueError('Replay frame count or order mismatch')
    if any(not 1<=event['step']<=summary['steps'] for event in events):
        raise ValueError('Replay event lies outside the episode')
    if sum(event['type']=='capture' for event in events)!=summary['captured_count']:
        raise ValueError('Replay capture events disagree with episode')
    if sum(event['type']=='penetration' for event in events)!=summary['penetrated_count']:
        raise ValueError('Replay penetration events disagree with episode')
    if summary['outcome']=='penetrated' and not summary['penetrated_count']:
        raise ValueError('Penetration outcome has no event')
    if summary['outcome']=='all_captured' and summary['captured_count']!=attackers:
        raise ValueError('All-capture outcome has incomplete captures')
    if trace['config'].get('episode_protocol') in ('resolve_all','resolve_all_one_penalty','resolve_all_one_penalty_trade') and summary['captured_count']+summary['penetrated_count']!=attackers:
        raise ValueError('Resolved episode does not account for all attackers')
    if trace['config'].get('episode_protocol')=='resolve_all_one_penalty_trade':
        if any(len(row['defender_alive'])!=5 for row in frames):
            raise ValueError('Replay defender status is incomplete')
        if len({event['defender'] for event in events if event['type']=='capture'})!=summary['captured_count']:
            raise ValueError('A defender captured more than one attacker')
        expected_retired = summary['captured_count'] + (summary['defender_oob_count']
                            if trace['config'].get('defender_oob_retire') else 0)
        if defenders-sum(frames[-1]['defender_alive'])!=expected_retired:
            raise ValueError('Retired defenders disagree with captures')
    return trace


def html(trace,stride):
    keep=set(range(0,len(trace['frames']),stride))|{0,len(trace['frames'])-1}
    keep.update(event['step'] for event in trace['events'])
    display=[trace['frames'][index] for index in sorted(keep)]
    data=json.dumps(dict(frames=display,events=trace['events'],summary=trace['summary'],
                         config=trace['config'],mode=trace['mode'],seed=trace['seed']),
                    ensure_ascii=False,separators=(',',':'))
    data=data.replace('</','<\\/')
    page=Path(__file__).with_name('replay_template.html').read_text(encoding='utf-8')
    return page.replace('__DATA__',data)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--label',default=None,
                   help='Checkpoint catalog label (e.g. Best, Last, step_100000); defaults to Best for a run directory or auto-resolved for a .pt file')
    p.add_argument('--attack-mode',choices=('direct_rush','split_attack','jia_straight','radial_straight'),required=True)
    p.add_argument('--seed',type=int,default=900001)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--frame-stride',type=int,default=5)
    p.add_argument('--html',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    if args.frame_stride<1 or args.output.exists():raise ValueError('Invalid stride or existing output')
    config,policy,name,model,settings=load_run(args.checkpoint,args.label,args.device)
    require_matching_attack_mode(settings,args.attack_mode)
    trace=validate_trace(collect(config,policy,args.attack_mode,args.seed))
    trace['algorithm']=name
    trace['checkpoint_sha256']=digest(model)
    args.output.mkdir(parents=True)
    with gzip.open(args.output/'trace.json.gz','wt',encoding='utf-8') as file:
        json.dump(trace,file,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    (args.output/'replay.html').write_text(html(trace,args.frame_stride),encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(trace['summary'],indent=2)+'\n',encoding='utf-8')
    print(json.dumps(trace['summary'],indent=2))
    return trace


if __name__=='__main__':main()
