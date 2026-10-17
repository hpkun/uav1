"""Read-only CUDA speed audit. Reuses official loaders and canonical recorder."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

ALGORITHMS = {'MAPPO': 'mappo', 'MAAC': 'maac', 'EA-MAPPO': 'ea_mappo', 'EAA-MAPPO': 'eaa_mappo'}


def inventory(output):
    import torch
    import yaml
    from algorithm.common.protocol import config_sha256
    from tools.combat_visualization import checkpoint_sha256
    jobs = []
    for algorithm, prefix in ALGORITHMS.items():
        for seed in (1, 2, 3):
            folder = ROOT/'outputs'/f'{prefix}_v338_seed{seed}_3m'
            files = ['run_summary.json', 'env_config.yaml', 'algorithm_config.yaml', 'best_eval.pt']
            if not all((folder/f).exists() for f in files):
                jobs.append(dict(algorithm=algorithm, training_seed=seed, status='unavailable')); continue
            env = yaml.safe_load((folder/'env_config.yaml').read_text(encoding='utf-8'))
            evaluation = folder/'evaluation_history.csv'
            if evaluation.exists():
                with evaluation.open(encoding='utf-8-sig') as f: rows = list(csv.DictReader(f))
            else:
                rows = [json.loads(x) for x in (folder/'evaluation_metrics.jsonl').read_text().splitlines() if x.strip()]
            completed = max(int(r['sampled_steps']) for r in rows) >= 3_000_000
            if not completed: raise RuntimeError(f'incomplete formal training: {folder}')
            for kind in (['best', 'final'] if seed == 3 else ['best']):
                checkpoint = folder/('best_eval.pt' if kind=='best' else 'final.pt')
                if kind=='final' and not checkpoint.exists():
                    # PPO runners name their completed endpoint checkpoint by step.
                    checkpoint = folder/'checkpoint_3000000.pt'
                if not checkpoint.exists():
                    jobs.append(dict(algorithm=algorithm,training_seed=seed,kind=kind,status='unavailable')); continue
                state = torch.load(checkpoint,map_location='cpu',weights_only=False)
                if kind=='final' and int(state['sampled_steps']) != 3_000_000:
                    raise RuntimeError('final comparison requires completed 3M checkpoint')
                extra = state.get('extra', {})
                if extra.get('training_smoke', False): raise RuntimeError('smoke checkpoint is not formal evidence')
                jobs.append(dict(algorithm=algorithm,training_seed=seed,kind=kind,status='available',
                    run_id=f'{prefix}_seed{seed}_{kind}',folder=str(folder),checkpoint=str(checkpoint),
                    checkpoint_sha256=checkpoint_sha256(checkpoint),sampled_steps=int(state['sampled_steps']),
                    checkpoint_family=state.get('checkpoint_family'),environment_config_sha256=config_sha256(env),
                    environment_version=str(env['environment_version']),training_complete_steps=3_000_000))
    available = [r for r in jobs if r['status']=='available']
    if len({r['environment_config_sha256'] for r in available})!=1: raise RuntimeError('environment drift across runs')
    if any(r['environment_version']!='3.38' for r in available): raise RuntimeError('v3.38 required')
    (output/'inventory.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
    return jobs


def load_policy(job):
    import yaml
    import torch
    from tools.record_combat_episode import load_recording_policy
    from tools.combat_visualization import checkpoint_sha256
    from algorithm.common.protocol import config_sha256
    folder = Path(job['folder'])
    ec = yaml.safe_load((folder/'env_config.yaml').read_text(encoding='utf-8'))
    ac = yaml.safe_load((folder/'algorithm_config.yaml').read_text(encoding='utf-8'))
    if job.get('checkpoint_family') == 'eaa_mappo_v338':
        # The existing replay dispatcher predates this explicit checkpoint family.
        # Use its official evaluation loader, including its own strict validator.
        from algorithm.eaa_mappo.evaluation import load_checkpoint
        trainer = load_checkpoint(job['checkpoint'],ac,ec,'cuda')
        provenance = dict(algorithm=job['algorithm'],checkpoint_path=job['checkpoint'],
            checkpoint_sha256=job['checkpoint_sha256'],checkpoint_sampled_steps=job['sampled_steps'],
            training_seed=job['training_seed'],environment_config_sha256=config_sha256(ec),
            algorithm_config_sha256=config_sha256(ac),observation_dim=45,action_dim=3,num_agents=4)
    else:
        trainer, _, provenance = load_recording_policy(job['checkpoint'],ec,ac,'cuda')
    if checkpoint_sha256(job['checkpoint']) != job['checkpoint_sha256']:
        raise RuntimeError('checkpoint changed during audit')
    return trainer, ec, provenance


def run_job(job, output, count, seed_base):
    import torch
    from tools.record_combat_episode import record
    if not torch.cuda.is_available(): raise RuntimeError('CUDA mandatory; no CPU fallback')
    torch.set_num_threads(1)
    trainer, ec, provenance = load_policy(job)
    directory = Path(output)/'traces'/job['run_id']
    directory.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    rows = []
    for i in range(count):
        seed = seed_base+i
        target = directory/f'episode_{seed}'
        meta = target/'metadata.json'
        if meta.exists():
            m = json.loads(meta.read_text())
            if m.get('checkpoint_sha256') != job['checkpoint_sha256']: raise RuntimeError('resume provenance mismatch')
            from tools.combat_trace import read_trace
            read_trace(target/'episode_trace.npz',m)
        else:
            m = record(trainer,ec,seed,target,provenance)
        rows.append(dict(run_id=job['run_id'],algorithm=job['algorithm'],training_seed=job['training_seed'],
            kind=job['kind'],episode_seed=seed,episode_steps=m['episode_steps'],episode_return=m['episode_return'],
            red_win=m['red_win'],blue_win=m['blue_win'],draw=m['draw'],termination_reason=m['termination_reason'],
            red_survivors=m['red_survivors'],blue_survivors=m['blue_survivors'],
            red_fire_attempts=m['red_fire_attempts'],blue_fire_attempts=m['blue_fire_attempts']))
        if (i+1)%10==0 or i==0:
            print(f"{job['run_id']}: {i+1}/{count} episodes, {time.monotonic()-started:.1f}s",flush=True)
    (directory/'episode_summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    return dict(run_id=job['run_id'],episodes=len(rows),seconds=time.monotonic()-started)


def protected_hashes():
    paths = [p for d in ('env','algorithm','configs') for p in (ROOT/d).rglob('*')
             if p.is_file() and p.suffix in ('.py','.yaml','.yml')]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',default='outputs/red_speed_audit_v338_20261017')
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--seed-base',type=int,default=20001000)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--inventory-only',action='store_true')
    p.add_argument('--run-id',help='Comma-separated checkpoint run IDs, for partial/resumed collection')
    a = p.parse_args()
    if a.episodes<1 or a.workers<1: p.error('positive episodes/workers required')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA mandatory for checkpoint inventory/evaluation; no CPU fallback')
    output = ROOT/a.output_dir; output.mkdir(parents=True,exist_ok=True)
    before = protected_hashes()
    if not (output/'protected_before.json').exists():
        (output/'protected_before.json').write_text(json.dumps(before,indent=2))
    jobs = inventory(output)
    requested=set(a.run_id.split(',')) if a.run_id else None
    selected = [j for j in jobs if j['status']=='available' and (requested is None or j['run_id'] in requested)]
    print(f'{len(selected)} checkpoints; CUDA; seeds {a.seed_base}..{a.seed_base+a.episodes-1}',flush=True)
    if a.inventory_only:return
    (output/'protocol.json').write_text(json.dumps(dict(episodes_per_checkpoint=a.episodes,
        evaluation_seeds=list(range(a.seed_base,a.seed_base+a.episodes)),deterministic=True,
        device='cuda',workers=a.workers,environment_version='3.38',
        primary='12 best checkpoints (4 algorithms x 3 training seeds)',
        supplementary='4 final checkpoints (training seed3)',
        notes='No training, no environment/policy/config edits; canonical recorder only.'),indent=2))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(run_job,j,str(output),a.episodes,a.seed_base) for j in selected]
        completed = []
        for f in as_completed(futures):
            completed.append(f.result())
            (output/'completed_jobs.json').write_text(json.dumps(completed,indent=2))
    after = protected_hashes()
    changed = sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))
    (output/'protected_verification.json').write_text(json.dumps(dict(changed_files=changed),indent=2))
    if changed: raise RuntimeError(f'protected source files changed concurrently: {changed}')


if __name__=='__main__': main()
