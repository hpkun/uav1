"""Frozen v3.38 policy cross-environment intervention, never checkpoint migration."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def source_policy(job):
    import torch
    from tools.audit_red_speed import load_policy
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory; no CPU fallback')
    trainer, source, provenance = load_policy(job)  # Official loader with SOURCE snapshots.
    if source['environment_version'] != '3.38':
        raise ValueError('source policy must be a validated v3.38 checkpoint')
    for value in vars(trainer).values():
        if isinstance(value, torch.nn.Module):
            value.eval()
            value.requires_grad_(False)
    return trainer, source, provenance


def weight_hash(trainer):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(vars(trainer).items()):
        if isinstance(value, torch.nn.Module):
            for key, tensor in sorted(value.state_dict().items()):
                digest.update((name + key).encode())
                digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def provenance(source, target, loaded):
    from algorithm.common.protocol import config_sha256
    return dict(source_checkpoint_environment_version='3.38',
        source_environment_config_sha256=config_sha256(source),
        source_algorithm_config_sha256=loaded['algorithm_config_sha256'],
        target_environment_version=target['environment_version'],
        target_environment_config_sha256=config_sha256(target),
        cross_environment_inference=True, intervention='frozen v3.38 policy cross-environment intervention',
        checkpoint_sha256=loaded['checkpoint_sha256'], deterministic=True, device='cuda')


def episode_steps(config, seed):
    import numpy as np
    from tools.combat_visualization import RecordingCombatEnv, states_array
    from tools.analyze_red_speed import overshoot_proxies
    env = RecordingCombatEnv(config)
    obs, _ = env.reset(seed)
    frames = {k: [] for k in ('red_kinematics','blue_kinematics','red_alive','blue_alive')}
    targets, speed, distance = [], [], []
    def frame():
        for side in ('red', 'blue'):
            frames[side+'_kinematics'].append(states_array(getattr(env, side)))
            frames[side+'_alive'].append(getattr(env, side+'_alive_mask').copy())
    frame()
    total = 0.
    while True:
        for own in env.blue:
            if not own.alive:
                continue
            speed.append(own.v)
            target = env.fixed_policy.nearest_target_index(own, env.red)
            if target is not None:
                distance.append(float(np.linalg.norm(own.as_array()[:3]-env.red[target].as_array()[:3])))
        actions = yield obs, env.red_alive_mask.copy()
        if not np.isfinite(actions).all():
            raise FloatingPointError('non-finite frozen inference actions')
        obs, reward, terminated, truncated, info = env.step(actions)
        total += float(np.sum(reward))
        targets.append(info['blue_pursuit_target_indices'].copy())
        frame()
        if terminated or truncated:
            break
    trace = {k: np.asarray(v) for k, v in frames.items()}
    trace.update(blue_pursuit_target_indices=np.asarray(targets), dt=env.dt)
    speed, distance = np.asarray(speed), np.asarray(distance)
    row = dict(episode_seed=int(seed), red_win=int(info['red_win']), episode_return=total,
        episode_length=env.steps, overshoot=len(overshoot_proxies(trace)),
        target_switches=int(info['blue_target_switches']), termination_reason=info['termination_reason'])
    for side in ('red','blue'):
        for key in ('survivors','losses','fire_attempts','weapon_hits','attack_kills',
                    'boundary_exits','ground_losses','ceiling_losses','ammo_used'):
            row[side+'_'+key] = int(info[side+'_'+key])
    for name, values in [('blue_speed',speed),('pursuit_distance',distance)]:
        row[name+'_count'] = int(len(values))
        row[name+'_mean'] = float(values.mean()) if len(values) else None
        for q in (0,5,25,50,75,95,100):
            row[name+f'_p{q}'] = float(np.percentile(values,q)) if len(values) else None
    for label, mask in [('near',distance<=1000),('mid',(distance>1000)&(distance<=3000)),('far',distance>3000)]:
        row['blue_'+label+'_agent_steps'] = int(mask.sum())
        row['blue_'+label+'_occupancy'] = float(mask.mean()) if len(mask) else 0.
    edges = [150,175,200,210,220,225,230,240,250,275,300.00001]
    row['blue_speed_histogram'] = np.histogram(speed,edges)[0].tolist()
    return row


def episode(trainer, config, seed):
    iterator = episode_steps(config,seed)
    request = next(iterator)
    while True:
        obs,alive=request
        action=trainer.act(obs[None],alive[None],deterministic=True)[0]
        try:
            request=iterator.send(action)
        except StopIteration as finished:
            return finished.value


def episode_batch(trainer, config, seeds):
    """Batch only frozen CUDA inference; every environment still steps normally."""
    import numpy as np
    iterators=[episode_steps(config,seed) for seed in seeds]
    active={i:next(iterator) for i,iterator in enumerate(iterators)}
    result={}
    while active:
        indices=list(active)
        observations=np.stack([active[i][0] for i in indices])
        alive=np.stack([active[i][1] for i in indices])
        actions=trainer.act(observations,alive,deterministic=True)
        for i,action in zip(indices,actions):
            try:
                active[i]=iterators[i].send(action)
            except StopIteration as finished:
                result[i]=finished.value
                del active[i]
    return [result[i] for i in range(len(seeds))]


def environment_audit_worker(connection, config, seeds):
    """CPU simulation only; frozen policy inference remains on the parent CUDA device."""
    import numpy as np
    import traceback
    try:
        iterators=[episode_steps(config,seed) for seed in seeds]
        active={i:next(iterator) for i,iterator in enumerate(iterators)}
        completed=[]
        while active:
            indices=list(active)
            connection.send(('ready', (np.stack([active[i][0] for i in indices]),
                                      np.stack([active[i][1] for i in indices]),completed)))
            completed=[]
            actions=connection.recv()
            for i,action in zip(indices,actions):
                try:
                    active[i]=iterators[i].send(action)
                except StopIteration as finished:
                    completed.append(finished.value)
                    del active[i]
        connection.send(('done',completed))
    except BaseException:
        connection.send(('error',traceback.format_exc()))
    finally:
        connection.close()


def parallel_episode_batch(trainer, config, seeds, workers):
    """One CUDA policy, multiple canonical CPU environment collectors (no Torch workers)."""
    import multiprocessing as mp
    import numpy as np
    seeds=list(seeds)
    context=mp.get_context('spawn')
    connections=[];processes=[];result=[]
    try:
        for subset in np.array_split(seeds,min(workers,len(seeds))):
            parent,child=context.Pipe()
            process=context.Process(target=environment_audit_worker,args=(child,config,subset.tolist()))
            process.start();child.close()
            connections.append(parent);processes.append(process)
        active=list(connections)
        while active:
            requests=[]
            for connection in active:
                status,payload=connection.recv()
                if status=='error':
                    raise RuntimeError(payload)
                if status=='done':
                    result.extend(payload)
                else:
                    obs,alive,completed=payload
                    result.extend(completed)
                    requests.append((connection,obs,alive))
            active=[row[0] for row in requests]
            if requests:
                actions=trainer.act(np.concatenate([r[1] for r in requests]),
                    np.concatenate([r[2] for r in requests]),deterministic=True)
                offset=0
                for connection,obs,_ in requests:
                    connection.send(actions[offset:offset+len(obs)])
                    offset+=len(obs)
        return sorted(result,key=lambda r:r['episode_seed'])
    finally:
        for connection in connections:
            connection.close()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate();process.join()


def run_policy(job, output, count, seed_base, env_workers=0):
    import torch
    import yaml
    from tools.combat_visualization import checkpoint_sha256
    torch.set_num_threads(1)
    trainer, source, loaded = source_policy(job)
    before = weight_hash(trainer)
    output = Path(output)
    rows = []
    for version in ('338','339'):
        target = yaml.safe_load((ROOT/f'configs/combat_environment_v{version}.yaml').read_text(encoding='utf-8'))
        meta = provenance(source,target,loaded)
        path = output/f"{job['run_id']}_v{version}.json"
        episodes = []
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['provenance'] != meta or saved['seeds'] != list(range(seed_base,seed_base+count)):
                raise RuntimeError('existing result provenance/seed mismatch')
            episodes=saved['episodes']
            if [r['episode_seed'] for r in episodes] != list(range(seed_base,seed_base+len(episodes))):
                raise RuntimeError('partial result seed order mismatch')
            if len(episodes)==count:
                rows.extend(episodes)
                continue
        start = time.monotonic()
        with torch.inference_mode():
            batch_size=count if env_workers else 20
            for start_index in range(len(episodes),count,batch_size):
                seeds=range(seed_base+start_index,seed_base+min(count,start_index+batch_size))
                batch=(parallel_episode_batch(trainer,target,seeds,env_workers) if env_workers
                       else episode_batch(trainer,target,seeds))
                for row in batch:
                    row.update(algorithm=job['algorithm'],training_seed=job['training_seed'],target_environment=target['environment_version'])
                episodes.extend(batch)
                if weight_hash(trainer) != before:
                    raise RuntimeError('frozen weights changed')
                path.write_text(json.dumps(dict(provenance=meta,seeds=list(range(seed_base,seed_base+count)),
                    weights_before=before,weights_after=weight_hash(trainer),episodes=episodes),
                    indent=2,allow_nan=False),encoding='utf-8')
                print(f"{job['run_id']} -> {target['environment_version']}: {len(episodes)}/{count}, {time.monotonic()-start:.0f}s",flush=True)
        if weight_hash(trainer) != before or checkpoint_sha256(job['checkpoint']) != job['checkpoint_sha256']:
            raise RuntimeError('frozen weights/checkpoint changed')
        path.write_text(json.dumps(dict(provenance=meta,seeds=list(range(seed_base,seed_base+count)),
            weights_before=before,weights_after=weight_hash(trainer),episodes=episodes),indent=2,allow_nan=False),encoding='utf-8')
        rows.extend(episodes)
    return rows


def paired_delta(a,b, rng, resamples=2000, clusters=None):
    import numpy as np
    delta = np.asarray(b,float)-np.asarray(a,float)
    units=delta
    if clusters is not None:
        _,inverse=np.unique(clusters,return_inverse=True)
        counts=np.bincount(inverse)
        if not np.all(counts==counts[0]):
            raise ValueError('scenario-cluster bootstrap requires balanced frozen-policy coverage')
        units=np.bincount(inverse,weights=delta)/counts
    bootstrap = units[rng.integers(0,len(units),(resamples,len(units)))].mean(1)
    result=dict(delta=float(delta.mean()),ci95=np.percentile(bootstrap,[2.5,97.5]).tolist(),episodes=len(delta))
    if clusters is not None:
        result['bootstrap_scenario_units']=len(units)
    return result


def summarize(rows, output):
    import numpy as np
    import pandas as pd
    frame = pd.DataFrame(rows)
    frame.drop(columns=['blue_speed_histogram']).to_csv(output/'episodes.csv',index=False)
    groups = [('policy',['algorithm','training_seed']),('algorithm',['algorithm']),('all',[])]
    summary, deltas = [], []
    rng = np.random.default_rng(339)
    for level, keys in groups:
        grouped = frame.groupby(keys) if keys else [('all',frame)]
        for group, values in grouped:
            label = dict(level=level,group=str(group))
            if keys:
                identity=group if isinstance(group,tuple) else (group,)
                label.update({key:(value.item() if hasattr(value,'item') else value)
                              for key,value in zip(keys,identity)})
            for version, part in values.groupby('target_environment'):
                result = dict(**label,target_environment=version,episodes=len(part),red_wins=int(part.red_win.sum()))
                wins,n=int(part.red_win.sum()),len(part)
                p=wins/n; z=1.959963984540054
                center=(p+z*z/(2*n))/(1+z*z/n)
                half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
                if level=='policy':
                    result['red_win_wilson_ci95']=[center-half,center+half]
                result['bootstrap_scenario_units']=part.episode_seed.nunique()
                result['mean_ci95']={}
                for name in part.select_dtypes(include='number').columns:
                    if name not in ('episode_seed','training_seed'):
                        result[name] = float(part[name].mean())
                        if not name.endswith('_count'):
                            # Shared scenario seeds are kept together across frozen policies.
                            sample_values=part.groupby('episode_seed')[name].mean().dropna().to_numpy(float)
                            boot=sample_values[rng.integers(0,len(sample_values),(2000,len(sample_values)))].mean(1)
                            result['mean_ci95'][name]=np.percentile(boot,[2.5,97.5]).tolist()
                result['median_return'] = float(part.episode_return.median())
                ordered=part.sort_values(['episode_seed','algorithm','training_seed'])
                returns=ordered.episode_return.to_numpy(float).reshape(part.episode_seed.nunique(),-1)
                result['median_return_ci95']=np.percentile(np.median(
                    returns[rng.integers(0,len(returns),(2000,len(returns)))],axis=(1,2)),[2.5,97.5]).tolist()
                result['blue_speed_histogram'] = np.stack(part.blue_speed_histogram).sum(0).tolist()
                summary.append(result)
            a = values[values.target_environment=='3.38'].sort_values(['algorithm','training_seed','episode_seed'])
            b = values[values.target_environment=='3.39'].sort_values(['algorithm','training_seed','episode_seed'])
            if not np.array_equal(a[['algorithm','training_seed','episode_seed']].values,b[['algorithm','training_seed','episode_seed']].values):
                raise ValueError('unpaired interventions')
            deltas.append(dict(**label,metrics={name:paired_delta(a[name],b[name],rng,clusters=a.episode_seed) for name in
                ('red_win','episode_return','episode_length','blue_attack_kills','red_attack_kills','overshoot')}))
    (output/'summary.json').write_text(json.dumps(dict(summary=summary,paired_deltas=deltas,
        delta_definition='v3.39 minus v3.38',ci='paired episode/scenario bootstrap, 2000 draws; shared scenario seeds clustered across fixed policies; no timestep independence',
        occupancy='distance tier of live Blue current nearest living Red; pre-transition',
        speed_histogram_edges=[150,175,200,210,220,225,230,240,250,275,300.00001]),indent=2,allow_nan=False),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',default='outputs/v339_frozen_blue_comparison')
    parser.add_argument('--episodes',type=int,default=100)
    parser.add_argument('--seed-base',type=int,default=30001000)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--environment-workers',type=int,default=0,
        help='Serialize CUDA policies, parallelize CPU environments to avoid Windows Torch paging pressure')
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory; no CPU fallback')
    if min(args.episodes,args.workers)<1 or args.environment_workers<0 or args.seed_base < 30001000:
        parser.error('positive counts and fresh seed range >=30001000 required')
    output = ROOT/args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    from tools.audit_red_speed import inventory
    jobs = [j for j in inventory(output) if j['status']=='available' and j['kind']=='best']
    if not jobs:
        raise RuntimeError('no completed formal v3.38 best checkpoints')
    print(f'{len(jobs)} frozen policies; {args.episodes} episodes x 2 environments each',flush=True)
    rows=[]
    if args.environment_workers:
        for job in jobs:
            rows.extend(run_policy(job,str(output),args.episodes,args.seed_base,args.environment_workers))
            print(f'completed {len(rows)} intervention episodes',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_policy,j,str(output),args.episodes,args.seed_base) for j in jobs]
            for future in as_completed(futures):
                rows.extend(future.result())
                print(f'completed {len(rows)} intervention episodes',flush=True)
    summarize(rows,output)


if __name__=='__main__':
    main()
