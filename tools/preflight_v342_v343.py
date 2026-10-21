"""Repeatable CPU fixed-controller research audit; no training/checkpoint loading.

Same reset seeds pair initial states only. Event results may branch trajectories.
Observers forward each authoritative hook once and never touch RNG or controls.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from env.geometry import engagement_geometry
from env.v36_geometry import angle_within
from tools.audit_v339_attack_geometry import red_actions, POLICIES, states


def write_csv(path, rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def quantiles(values):
    return dict(zip(('min','p10','p50','p90','max'),map(float,np.quantile(values,[0,.1,.5,.9,1])))) if values else None


def episode(version, policy, seed):
    c=load_config(ROOT/f'configs/combat_environment_v{version}.yaml')
    e=MultiUAVCombatEnv(c);e.reset(seed)
    initial=hashlib.sha256(states(e).tobytes()).hexdigest()
    row=dict(version=e.environment_version,policy=policy,seed=seed,initial_sha256=initial)
    alphas=[];range_alphas=[]; disagreements=0; ranged_pairs=0; all_pairs=0
    old_only=0;new_only=0;first_geometry=dict(red=None,blue=None)
    attempts=[];sums=np.zeros((4,4));total=np.zeros(4);nonzero=np.zeros(2,dtype=int)
    live_samples=0;offense=0.;defense=0.;positive_r3=0.;negative_r3=0.
    discounted_r3=np.zeros(4);phi_initial=e.potential()[0]
    trajectory=hashlib.sha256()
    dense_original=e.dense_combat_reward;entry_original=e._entry_attempts
    dense_calls=0;entry_calls=0
    def dense_observer():
        nonlocal disagreements,ranged_pairs,all_pairs,old_only,new_only,dense_calls
        dense_calls+=1
        for side,team,targets in (('red',e.red,e.blue),('blue',e.blue,e.red)):
            for a in team:
                for b in targets:
                    if not a.alive or not b.alive:continue
                    g=engagement_geometry(a,b);alphas.append(np.rad2deg(g.off_boresight));all_pairs+=1
                    if not c['weapon']['range_min']<=g.distance<=c['weapon']['range_max']:continue
                    range_alphas.append(np.rad2deg(g.off_boresight));ranged_pairs+=1
                    old=bool(angle_within(abs(g.ata),np.pi/6) and angle_within(abs(g.ha),np.pi/6))
                    new=bool(angle_within(g.off_boresight,np.pi/6))
                    disagreements+=old!=new;old_only+=old and not new;new_only+=new and not old
                    if (old if version=='341' else new) and first_geometry[side] is None:
                        first_geometry[side]=e.steps*e.dt
        return dense_original()
    def entry_observer(attackers,targets,fire_states,side):
        nonlocal entry_calls
        entry_calls+=1
        result=entry_original(attackers,targets,fire_states,side)
        for i,j,hit in result:
            g=engagement_geometry(attackers[i],targets[j])
            az,el=(g.ata,g.ha) if version=='341' else (g.boresight_azimuth_error,g.boresight_elevation_error)
            attempts.append(dict(version=e.environment_version,policy=policy,seed=seed,
                side=side,step=e.steps,time_s=e.steps*e.dt,attacker=i,target=j,hit=bool(hit),
                distance_m=g.distance,ata_deg=np.rad2deg(g.ata),ha_deg=np.rad2deg(g.ha),
                alpha_deg=np.rad2deg(g.off_boresight),body_azimuth_deg=np.rad2deg(g.boresight_azimuth_error),
                body_elevation_deg=np.rad2deg(g.boresight_elevation_error),
                theoretical_probability=e.weapon.expected_hit_probability(g.distance,az,el)))
        return result
    e.dense_combat_reward=dense_observer;e._entry_attempts=entry_observer
    while True:
        live=e.red_alive_mask.astype(bool);live_samples+=int(live.sum())
        action,_=red_actions(e,policy)
        obs,r,term,trunc,info=e.step(action)
        assert np.isfinite(obs).all() and np.isfinite(r).all()
        components=np.stack([info[f'r{k}_rewards'] for k in range(1,5)])
        np.testing.assert_array_equal(r,components.sum(axis=0))
        sums+=components;total+=r
        nonzero+=np.asarray([np.count_nonzero(info[f'r{k}_rewards'][live]) for k in (3,4)])
        offense+=float(info['tactical_offense_rewards'].sum());defense+=float(info['tactical_defense_rewards'].sum())
        positive_r3+=float(np.maximum(info['r3_rewards'],0).sum());negative_r3+=float(np.minimum(info['r3_rewards'],0).sum())
        discounted_r3+=.99**(e.steps-1)*info['r3_rewards']
        trajectory.update(states(e).tobytes());trajectory.update(e.red_ammo.tobytes());trajectory.update(e.blue_ammo.tobytes())
        if term or trunc:break
    assert dense_calls==e.steps and entry_calls==2*e.steps
    for k in range(4):np.testing.assert_array_equal(sums[k],info[f'episode_r{k+1}_rewards'])
    np.testing.assert_allclose(total,sums.sum(axis=0),atol=2e-12,rtol=0)
    if version=='343':np.testing.assert_allclose(discounted_r3,-phi_initial,atol=3e-15,rtol=0)
    row.update(duration_s=e.steps*e.dt,red_win=bool(info['red_win']),blue_win=bool(info['blue_win']),draw=bool(info['draw']),
        terminated=bool(term),truncated=bool(trunc),termination_reason=info['termination_reason'],
        trajectory_sha256=trajectory.hexdigest(),red_losses=int(info['red_losses']),blue_losses=int(info['blue_losses']),
        red_first_geometry_s=first_geometry['red'],blue_first_geometry_s=first_geometry['blue'],
        live_pair_samples=all_pairs,ranged_pair_samples=ranged_pairs,disagreement_pairs=disagreements,
        old_only_pairs=old_only,new_only_pairs=new_only,
        eligibility_difference_fraction=disagreements/max(1,ranged_pairs),
        alpha_distribution_deg=json.dumps(quantiles(alphas)),range_alpha_distribution_deg=json.dumps(quantiles(range_alphas)),
        live_red_agent_steps=live_samples,r3_nonzero_agent_steps=int(nonzero[0]),r4_nonzero_agent_steps=int(nonzero[1]),
        r3_nonzero_fraction=float(nonzero[0]/max(1,live_samples)),r4_nonzero_fraction=float(nonzero[1]/max(1,live_samples)),
        r1_sum=float(sums[0].sum()),r2_sum=float(sums[1].sum()),r3_sum=float(sums[2].sum()),r4_sum=float(sums[3].sum()),
        offense_sum=offense,defense_sum=defense,offense_fraction=offense/max(1e-15,offense+abs(defense)),
        defense_fraction=abs(defense)/max(1e-15,offense+abs(defense)),r3_positive_sum=positive_r3,r3_negative_sum=negative_r3,
        total_reward=float(total.sum()),ledger_max_abs_error=float(np.max(np.abs(total-sums.sum(axis=0)))),
        discounted_r3_sum=float(discounted_r3.sum()),phi_initial_sum=float(phi_initial.sum()))
    for side in ('red','blue'):
        events=[x for x in attempts if x['side']==side]
        row[side+'_attempts']=len(events);row[side+'_hits']=sum(x['hit'] for x in events)
        row[side+'_kills']=int(info[side+'_attack_kills'])
        row[side+'_event_alpha_deg']=json.dumps(quantiles([x['alpha_deg'] for x in events]))
    # Explicit screening rule, not a theorem about learned-policy exploits.
    row['dense_exceeds_two_kill_equivalents_per_agent']=bool(np.any(sums[2]+sums[3]>20.))
    row['dense_far_above_event_magnitude']=bool(sums[2:].sum()>2*max(10.,np.abs(sums[:2]).sum()))
    row['reward_anomaly']=row['dense_exceeds_two_kill_equivalents_per_agent'] or row['dense_far_above_event_magnitude']
    return row,attempts, np.histogram(alphas,bins=np.arange(0,181,10))[0].tolist()


def summarize(rows):
    result=[]
    for version in ('3.41','3.42','3.43'):
        for policy in POLICIES:
            group=[r for r in rows if r['version']==version and r['policy']==policy]
            s=dict(version=version,policy=policy,episodes=len(group),
                   red_wins=sum(r['red_win'] for r in group),blue_wins=sum(r['blue_win'] for r in group),draws=sum(r['draw'] for r in group))
            for key in ('duration_s','red_losses','blue_losses','red_attempts','blue_attempts','red_hits','blue_hits',
                        'r1_sum','r2_sum','r3_sum','r4_sum','offense_sum','defense_sum','r3_positive_sum','r3_negative_sum'):
                s[key+'_mean']=float(np.mean([r[key] for r in group]))
            for side in ('red','blue'):
                values=[r[side+'_first_geometry_s'] for r in group if r[side+'_first_geometry_s'] is not None]
                s[side+'_first_geometry_s_mean']=float(np.mean(values)) if values else None
                s[side+'_geometry_observed_episodes']=len(values)
            den=sum(r['live_red_agent_steps'] for r in group)
            for k in (3,4):s[f'r{k}_nonzero_fraction']=sum(r[f'r{k}_nonzero_agent_steps'] for r in group)/max(1,den)
            s['eligibility_difference_fraction']=sum(r['disagreement_pairs'] for r in group)/max(1,sum(r['ranged_pair_samples'] for r in group))
            s['reward_anomaly_episodes']=sum(r['reward_anomaly'] for r in group)
            result.append(s)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=20);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if min(args.seeds,args.workers)<1:p.error('seeds and workers must be positive')
    out=args.output
    if out.exists() and any(out.iterdir()):raise FileExistsError('audit output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    jobs=[(v,p,s) for v in ('341','342','343') for p in POLICIES for s in range(args.seeds)]
    rows=[];events=[];histograms=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(episode,*job) for job in jobs]
        for i,f in enumerate(as_completed(futures),1):
            row,shots,hist=f.result();rows.append(row);events.extend(shots)
            histograms.append(dict(version=row['version'],policy=row['policy'],seed=row['seed'],counts=hist))
            if i%10==0 or i==len(jobs):print(f'{i}/{len(jobs)} episodes completed',flush=True)
    rows.sort(key=lambda r:(r['version'],r['policy'],r['seed']))
    events.sort(key=lambda r:(r['version'],r['policy'],r['seed'],r['step'],r['side'],r['attacker']))
    paired=[]
    for policy in POLICIES:
        for seed in range(args.seeds):
            a,b,c=[next(r for r in rows if r['version']==v and r['policy']==policy and r['seed']==seed) for v in ('3.41','3.42','3.43')]
            assert a['initial_sha256']==b['initial_sha256']==c['initial_sha256']
            assert b['trajectory_sha256']==c['trajectory_sha256'], 'reward/observation changed fixed-controller physical trajectory'
            paired.append(dict(policy=policy,seed=seed,initial_states_equal=True,v342_v343_physical_trajectory_equal=True,
                v341_v342_trajectory_equal=a['trajectory_sha256']==b['trajectory_sha256'],duration_delta_342_341=b['duration_s']-a['duration_s']))
    summary=summarize(rows)
    write_csv(out/'episodes.csv',rows);write_csv(out/'events.csv',events);write_csv(out/'comparison.csv',summary);write_csv(out/'paired_initialization.csv',paired)
    payload=dict(protocol='same initial states; full trajectories can diverge after different events',
        controller_source='tools.audit_v339_attack_geometry.red_actions (unchanged)',seeds=list(range(args.seeds)),
        formal_training_started=False,math_and_fixed_controller_device='cpu',summary=summary,
        alpha_histogram_edges_deg=list(range(0,181,10)),alpha_histograms=histograms,
        ledger_max_abs_error=max(r['ledger_max_abs_error'] for r in rows),
        reward_anomaly_episodes=sum(r['reward_anomaly'] for r in rows),all_initial_states_paired=True,
        v342_v343_fixed_controller_trajectories_equal=True)
    (out/'summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    print('version policy    W/L/D   seconds red/blue losses  R3 mean    R4 mean   R3/R4 nonzero',flush=True)
    for s in summary:
        print(f"{s['version']:7} {s['policy']:8} {s['red_wins']:2}/{s['blue_wins']:2}/{s['draws']:2} "
              f"{s['duration_s_mean']:7.2f} {s['red_losses_mean']:.2f}/{s['blue_losses_mean']:.2f} "
              f"{s['r3_sum_mean']:9.4f} {s['r4_sum_mean']:9.4f} "
              f"{s['r3_nonzero_fraction']:.3%}/{s['r4_nonzero_fraction']:.3%}",flush=True)
    print(f"ledger maximum error={payload['ledger_max_abs_error']:.3g}; anomaly episodes={payload['reward_anomaly_episodes']}",flush=True)


if __name__=='__main__':main()
