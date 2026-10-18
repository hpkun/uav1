"""Paired CPU fixed-policy preflight; real lethal step(), no training/models.

Uses the previous audit's transparent pre-fire observer and identical policies.
Trace arrays retain motion, actions, firing geometry, ammo, hits and deaths.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from tools.audit_v339_attack_geometry import (
    PreFireObserver, states, red_actions, vector_geometry, write_csv, POLICIES)


def first(mask, dt=.1):
    indices = np.flatnonzero(np.asarray(mask).reshape(len(mask), -1).any(axis=1))
    return float((indices[0] + 1) * dt) if len(indices) else None


def episode(version, policy, seed, output):
    cfg = load_config(ROOT / f'configs/combat_environment_v{version}.yaml')
    e = MultiUAVCombatEnv(cfg)
    e.reset(seed)
    with PreFireObserver(e) as observer:
        while True:
            actions, targets = red_actions(e, policy)
            before = states(e)
            result = e.step(actions)
            observer.finish(before, actions, targets, result)
            if result[2] or result[3]:
                break
        assert observer.calls == dict(dense=e.steps, resolve=e.steps)
        data = observer.arrays()
    assert e.steps <= 1000 and np.isfinite(data['post']).all()
    assert data['done'][-1].any() and not data['done'][:-1].any()
    key = f'v{version}_{policy}_s{seed}'
    path = Path(output) / 'trajectory_data' / (key + '.npz')
    np.savez_compressed(path, **data)
    info = result[4]
    row = dict(episode_id=key, version='3.'+version[1:], policy=policy, seed=seed,
               duration_s=e.steps*e.dt, red_win=bool(info['red_win']),
               blue_win=bool(info['blue_win']), draw=bool(info['draw']),
               termination_reason=info['termination_reason'], timeout=bool(result[3]),
               first_fire_s=first(data['attempt_targets'] >= 0),
               first_hit_s=first(data['hit']), first_kill_s=first(data['credited_kills']),
               trace=str(path.relative_to(ROOT)), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    aircraft, shots = [], []
    for side, label in enumerate(('red','blue')):
        g = vector_geometry(data['pre'], side)
        attempts = data['attempt_targets'][:, side]
        chosen = np.maximum(attempts, 0)
        distances = np.take_along_axis(g['distance'], chosen[...,None], axis=2)[...,0]
        firing = attempts >= 0
        hit = data['hit'][:,side]
        assert np.array_equal(firing.sum(axis=0), 6-data['ammo_post'][-1,side])
        row[label+'_shots'] = int(firing.sum())
        row[label+'_hits'] = int(hit.sum())
        row[label+'_kills'] = int((data['death_cause'][:,1-side] == 4).sum())
        row[label+'_long_shots'] = int((firing & (distances > 1000.)).sum())
        row[label+'_ammo_left'] = int(data['ammo_post'][-1,side].sum())
        for cause, code in (('ground',1),('horizontal',2),('ceiling',3)):
            row[label+'_'+cause+'_losses'] = int((data['death_cause'][:,side] == code).sum())
        # Agent-specific first entry within 1 km of any living opponent.
        entered = np.maximum.accumulate((g['live'] & (g['distance']<=1000.)).any(axis=2),axis=0)
        row[label+'_shots_before_first_1km_entry'] = int((firing & ~entered).sum())
        for i in range(4):
            aircraft.append(dict(episode_id=key,version=row['version'],policy=policy,seed=seed,
                side=label,aircraft=i,shots=int(firing[:,i].sum()),hits=int(hit[:,i].sum()),
                ammo_left=int(data['ammo_post'][-1,side,i]),
                alive_final=bool(data['post'][-1,side,i,6]),
                shots_before_first_1km_entry=int((firing[:,i]&~entered[:,i]).sum()),
                first_1km_entry_s=first(entered[:,i]),
                last_shot_s=float((np.flatnonzero(firing[:,i])[-1]+1)*.1) if firing[:,i].any() else None))
        for t,i in zip(*np.where(firing)):
            j = int(attempts[t,i])
            shots.append(dict(episode_id=key,side=label,aircraft=int(i),target=j,time_s=(int(t)+1)*.1,
                distance_m=float(distances[t,i]),ata_rad=float(g['ata'][t,i,j]),ha_rad=float(g['ha'][t,i,j]),
                hit=bool(hit[t,i]),before_first_1km_entry=bool(not entered[t,i]),
                theoretical_hit_probability=e.weapon.expected_hit_probability(distances[t,i],g['ata'][t,i,j],g['ha'][t,i,j])))
    for k,name in enumerate(('r3','r4_offense','r4_defense')):
        values = data['dense'][:,k]
        row[name+'_sum'] = float(values.sum())
        row[name+'_agent_steps'] = int((values!=0).sum())
        row[name+'_episode'] = bool((values!=0).any())
    positive = data['dense'][:,1] > 0
    armed = data['ammo_pre'][:,0] > 0
    row['first_positive_r4_s'] = first(positive)
    row['armed_positive_r4_agent_steps'] = int((positive&armed).sum())
    row['armed_positive_r4_episode'] = bool((positive&armed).any())
    row['first_armed_positive_r4_s'] = first(positive&armed)
    row['red_live_agent_steps'] = int(((data['pre'][:,0,:,6]>0)&(data['pre'][:,1,:,6]>0).any(axis=1)[:,None]).sum())
    return row, aircraft, shots


def summarize(rows):
    summaries=[]
    for version in ('3.39','3.40'):
        for policy in POLICIES:
            group=[r for r in rows if r['version']==version and r['policy']==policy]
            n=len(group)
            s=dict(version=version,policy=policy,episodes=n)
            for field in ('duration_s','red_shots','blue_shots','red_ammo_left','blue_ammo_left',
                          'red_kills','blue_kills','r3_sum','r4_offense_sum','r4_defense_sum'):
                s[field+'_mean']=float(np.mean([r[field] for r in group]))
            for field in ('first_fire_s','first_hit_s','first_kill_s','first_positive_r4_s','first_armed_positive_r4_s'):
                values=[r[field] for r in group if r[field] is not None]
                s[field+'_median']=float(np.median(values)) if values else None
                s[field+'_observed_n']=len(values)
            for field in ('red_win','blue_win','draw','timeout','r4_offense_episode','r4_defense_episode','armed_positive_r4_episode'):
                s[field+'_count']=sum(r[field] for r in group)
            for side in ('red','blue'):
                for field in ('shots','hits','long_shots','shots_before_first_1km_entry','horizontal_losses','ground_losses','ceiling_losses'):
                    s[side+'_'+field+'_total']=sum(r[side+'_'+field] for r in group)
                count=s[side+'_shots_total']
                s[side+'_long_shot_fraction']=s[side+'_long_shots_total']/count if count else None
                s[side+'_hit_rate']=s[side+'_hits_total']/count if count else None
                s[side+'_pre_1km_shot_fraction']=s[side+'_shots_before_first_1km_entry_total']/count if count else None
            den=sum(r['red_live_agent_steps'] for r in group)
            for name in ('r3','r4_offense','r4_defense'):
                s[name+'_agent_step_fraction']=sum(r[name+'_agent_steps'] for r in group)/den if den else 0.
            s['armed_positive_r4_agent_steps']=sum(r['armed_positive_r4_agent_steps'] for r in group)
            summaries.append(s)
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,default=20)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/v340_preflight')
    args=parser.parse_args()
    if args.seeds<1 or args.workers<1:parser.error('seeds/workers must be positive')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'trajectory_data').mkdir(exist_ok=True)
    rows=[];aircraft=[];shots=[];start=time.perf_counter()
    specs=[(v,p,s,str(out)) for v in ('339','340') for p in POLICIES for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(episode,*spec) for spec in specs]
        for i,f in enumerate(as_completed(jobs),1):
            row,planes,events=f.result();rows.append(row);aircraft.extend(planes);shots.extend(events)
            print(f'{i}/{len(specs)} {row["episode_id"]} {row["duration_s"]:.1f}s',flush=True)
    rows.sort(key=lambda r:r['episode_id'])
    write_csv(out/'episodes.csv',rows);write_csv(out/'aircraft.csv',aircraft);write_csv(out/'shots.csv',shots)
    summary=summarize(rows);write_csv(out/'comparison.csv',summary)
    paired=[]
    for policy in POLICIES:
        for seed in range(args.seeds):
            a,b=[next(r for r in rows if r['version']==v and r['policy']==policy and r['seed']==seed) for v in ('3.39','3.40')]
            # Same reset state; do not demand RNG/trajectory equality after firing.
            with np.load(ROOT/a['trace']) as x,np.load(ROOT/b['trace']) as y:
                np.testing.assert_array_equal(x['start'][0],y['start'][0])
            p=dict(policy=policy,seed=seed,initial_states_exact=True)
            for name in ('duration_s','first_fire_s','first_hit_s','first_kill_s','red_shots','blue_shots','red_kills','blue_kills','r4_offense_sum'):
                p[name+'_delta']=b[name]-a[name] if a[name] is not None and b[name] is not None else None
            paired.append(p)
    write_csv(out/'paired_deltas.csv',paired)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ('env','algorithm','configs') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')}
    (out/'summary.json').write_text(json.dumps(dict(episodes=len(rows),seeds=list(range(args.seeds)),
        policies=POLICIES,device='cpu',training_started=False,lethal=True,elapsed_s=time.perf_counter()-start,
        summaries=summary,source_sha256=sources,
        definitions=dict(long_shot='actual selected target distance > 1000m at pre-fire sample',
          pre_1km='shooter has never entered <=1000m of any living opponent at a pre-fire sample, including current step',
          exposure='positive offense / negative defense counted separately on living pre-fire Red recipients',
          armed_exposure='recipient Red ammo_pre>0, no cooldown or weapon-range requirement',
          rates='pooled shots/hits and living Red agent-steps; sums are per-episode mean, first times conditional medians',
          trace='start/pre/post [step,side,aircraft,x/y/z/v/theta/psi/alive]; dense [step,R3/offense/defense,Red]; cooldown in steps; death 0none 1ground 2horizontal 3ceiling 4weapon')),
        indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
