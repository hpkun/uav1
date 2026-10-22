"""CPU fixed-controller comparison of the isolated probability/dwell experiments.

No model loading or training. Initial states pair; subsequent events may branch
trajectories. Geometry is observed post-motion and before simultaneous settlement.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,csv,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from env.geometry import engagement_geometry
from env.v344_dwell import DwellState
from tools.audit_v339_attack_geometry import red_actions,POLICIES,states
from tools.preflight_v342_v343 import write_csv

DWELL_METRICS=('dwell_started','dwell_interrupted','dwell_target_switches','dwell_completed',
               'dwell_neutralizations','dwell_active_agent_steps','dwell_eligible_pair_steps',
               'mean_dwell_duration_steps','max_dwell_duration_steps')


def episode(version,policy,seed):
    e=MultiUAVCombatEnv(load_config(ROOT/f'configs/combat_environment_v{version}.yaml'))
    e.reset(seed);initial=hashlib.sha256(states(e).tobytes()).hexdigest()
    trajectory=hashlib.sha256();first=dict(red=None,blue=None)
    counts={s:dict(live_pairs=0,eligible_pairs=0) for s in ('red','blue')}
    events=[];sums=np.zeros((4,4));total=np.zeros(4);nz=np.zeros(4,dtype=int);samples=0
    phi=e.potential()[0];discounted=np.zeros(4);alpha=[]
    dense_original=e.dense_combat_reward;entry_original=e._entry_attempts
    calls=0
    def dense():
        nonlocal calls
        calls+=1
        for side,team,targets in (('red',e.red,e.blue),('blue',e.blue,e.red)):
            # Unit resource only removes the baseline ammo mask for geometry exposure.
            valid,distance=e.weapon.eligibility(team,targets,np.ones(4,dtype=int))
            live=np.asarray([s.alive for s in team])[:,None]&np.asarray([s.alive for s in targets])[None,:]
            counts[side]['live_pairs']+=int(live.sum());counts[side]['eligible_pairs']+=int(valid.sum())
            if valid.any() and first[side] is None:first[side]=e.steps*e.dt
            for i,j in zip(*np.where(valid)):
                alpha.append(float(np.rad2deg(engagement_geometry(team[i],targets[j]).off_boresight)))
        return dense_original()
    def entry(team,targets,fire_states,side):
        result=entry_original(team,targets,fire_states,side)
        for i,j,hit in result:
            g=engagement_geometry(team[i],targets[j])
            events.append(dict(version=e.environment_version,policy=policy,seed=seed,side=side,
                step=e.steps,time_s=e.steps*e.dt,attacker=i,target=j,
                type='probability_attempt' if version=='343' else 'dwell_complete',
                probability_hit=bool(hit) if version=='343' else None,
                duration_steps=None if version=='343' else int(e.dwell[side].sampled_counts[i]),
                distance_m=g.distance,alpha_deg=float(np.rad2deg(g.off_boresight)),
                attacker_x=team[i].x,attacker_y=team[i].y,attacker_altitude=team[i].altitude,
                target_x=targets[j].x,target_y=targets[j].y,target_altitude=targets[j].altitude))
        return result
    e.dense_combat_reward=dense;e._entry_attempts=entry
    while True:
        live=e.red_alive_mask.astype(bool);samples+=int(live.sum())
        actions,_=red_actions(e,policy)
        obs,r,term,trunc,info=e.step(actions)
        assert obs.shape==(4,49) and np.isfinite(obs).all() and np.isfinite(r).all()
        components=np.stack([info[f'r{k}_rewards'] for k in range(1,5)])
        np.testing.assert_array_equal(r,components.sum(axis=0))
        sums+=components;total+=r;nz+=np.count_nonzero(components[:,live],axis=1)
        discounted+=.99**(e.steps-1)*components[2]
        trajectory.update(states(e).tobytes())
        if term or trunc:break
    np.testing.assert_allclose(discounted,-phi,atol=4e-15,rtol=0)
    np.testing.assert_allclose(total,sums.sum(axis=0),atol=3e-12,rtol=0)
    assert calls==e.steps
    row=dict(version=e.environment_version,policy=policy,seed=seed,initial_sha256=initial,
        trajectory_sha256=trajectory.hexdigest(),event_semantics=info.get('event_semantics','probability_hit'),
        red_win=bool(info['red_win']),blue_win=bool(info['blue_win']),draw=bool(info['draw']),
        duration_s=e.steps*e.dt,timeout=bool(trunc),red_losses=int(info['red_losses']),blue_losses=int(info['blue_losses']),
        total_reward=float(total.sum()),live_red_agent_steps=samples,observation_dim=49,observation_finite=True,
        ledger_max_abs_error=float(np.max(np.abs(total-sums.sum(axis=0)))),
        eligible_alpha_p50_deg=float(np.median(alpha)) if alpha else None,
        eligible_alpha_p90_deg=float(np.quantile(alpha,.9)) if alpha else None)
    for k in range(4):
        np.testing.assert_array_equal(sums[k],info[f'episode_r{k+1}_rewards'])
        row[f'r{k+1}_sum']=float(sums[k].sum());row[f'r{k+1}_nonzero_samples']=int(nz[k])
    for side in ('red','blue'):
        row.update({side+'_'+k:v for k,v in counts[side].items()})
        row[side+'_exposure_fraction']=counts[side]['eligible_pairs']/max(1,counts[side]['live_pairs'])
        row[side+'_first_geometry_s']=first[side]
        row[side+'_first_dwell_complete_s']=None if version=='343' or info[side+'_first_dwell_complete_step'] is None else info[side+'_first_dwell_complete_step']*e.dt
        row[side+'_first_dwell_start_step']=info.get(side+'_first_dwell_start_step')
        row[side+'_first_dwell_complete_step']=info.get(side+'_first_dwell_complete_step')
        for name in ('attack_kills','fire_attempts','weapon_hits',*DWELL_METRICS):row[side+'_'+name]=info.get(side+'_'+name)
        if version!='343':
            assert row[side+'_attack_kills']==row[side+'_dwell_neutralizations']
            assert row[side+'_fire_attempts']==row[side+'_weapon_hits']==info[side+'_ammo_used']==0
    return row,events


def fixed_trajectory_probe():
    """Exogenous pair traces; first event latches, no lethal trajectory branching."""
    from env.models import AircraftState
    result=[]
    patterns={'aligned_4':[0]*4,'aligned_5':[0]*5,'35deg_8':[35]*8,'45deg_8':[45]*8,
        'outside_8':[46]*8,'break_reenter':[0,0,46,0,0,0,0,0]}
    for name,angles in patterns.items():
        for version in ('343','344','345'):
            e=MultiUAVCombatEnv(load_config(ROOT/f'configs/combat_environment_v{version}.yaml'))
            a=AircraftState(0,0,-3000,225,0,0);s=DwellState(1,5);rng=np.random.default_rng(20261022)
            resource=6;cooldown=0;first=None;exposure=0;normal_draws=0
            for step,degrees in enumerate(angles,1):
                angle=np.deg2rad(degrees);b=AircraftState(1500*np.cos(angle),1500*np.sin(angle),-3000,225,0,np.pi)
                legal=e.weapon.qualifies(a,b,1);exposure+=legal
                if first is not None:continue
                if version=='343':
                    cooldown=max(0,cooldown-1)
                    if legal and resource and not cooldown:
                        resource-=1;cooldown=10;normal_draws+=1
                        g=engagement_geometry(a,b)
                        if e.weapon.attempt_hit(rng,g.distance,g.boresight_azimuth_error,g.boresight_elevation_error):first=step
                elif s.update([[legal]],[[1500]],step):first=step
            result.append(dict(pattern=name,version='3.'+version[1:],samples=len(angles),
                geometric_exposure_samples=exposure,first_event_step=first,hit_normal_draws=normal_draws,
                semantics='same exogenous pair states; latch first event; no reward/death propagation'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=20);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if min(args.seeds,args.workers)<1:p.error('positive seeds and workers required')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('fresh output required')
    args.output.mkdir(parents=True,exist_ok=True);rows=[];events=[]
    specs=[(v,p,s) for v in ('343','344','345') for p in POLICIES for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(episode,*x) for x in specs]
        for i,f in enumerate(as_completed(futures),1):
            r,ev=f.result();rows.append(r);events.extend(ev)
            if i%10==0:print(f'{i}/{len(specs)} completed',flush=True)
    rows.sort(key=lambda r:(r['version'],r['policy'],r['seed']))
    events.sort(key=lambda r:(r['version'],r['policy'],r['seed'],r['step'],r['side'],r['attacker']))
    paired=[]
    for policy in POLICIES:
        for seed in range(args.seeds):
            group=[r for r in rows if r['policy']==policy and r['seed']==seed]
            assert len({r['initial_sha256'] for r in group})==1
            paired.append(dict(policy=policy,seed=seed,initial_states_equal=True,
                all_trajectories_equal=len({r['trajectory_sha256'] for r in group})==1))
    summaries=[];risks=[]
    for version in ('3.43','3.44','3.45'):
        for policy in POLICIES:
            group=[r for r in rows if r['version']==version and r['policy']==policy];n=len(group)
            s=dict(version=version,policy=policy,episodes=n)
            for key in ('red_win','blue_win','draw','timeout','duration_s','red_losses','blue_losses',
                        'r1_sum','r2_sum','r3_sum','r4_sum',*[side+'_'+x for side in ('red','blue') for x in ('attack_kills',*DWELL_METRICS)]):
                values=[r[key] for r in group if r[key] is not None]
                s[key+'_mean']=float(np.mean(values)) if values else None
            for side in ('red','blue'):
                s[side+'_exposure_fraction']=sum(r[side+'_eligible_pairs'] for r in group)/max(1,sum(r[side+'_live_pairs'] for r in group))
                for field in ('first_geometry_s','first_dwell_complete_s'):
                    values=[r[side+'_'+field] for r in group if r[side+'_'+field] is not None]
                    s[side+'_'+field+'_mean']=float(np.mean(values)) if values else None
                    s[side+'_'+field+'_episodes']=len(values)
            for k in range(1,5):s[f'r{k}_nonzero_fraction']=sum(r[f'r{k}_nonzero_samples'] for r in group)/max(1,sum(r['live_red_agent_steps'] for r in group))
            if version!='3.43':
                for condition,label in ((policy=='zero' and s['red_win_mean']>=.8,'zero_win_rate_at_least_80pct'),
                    (s['blue_dwell_completed_mean']==0,'blue_no_completions'),
                    (s['red_dwell_completed_mean']==0,'red_no_completions'),
                    (s['timeout_mean']>=.5,'timeouts_at_least_50pct'),(s['draw_mean']>=.8,'draws_at_least_80pct')):
                    if condition:risks.append(dict(version=version,policy=policy,risk=label))
            summaries.append(s)
    for name,data in (('episodes',rows),('events',events),('comparison',summaries),('paired_initialization',paired),('fixed_trajectory',fixed_trajectory_probe())):
        write_csv(args.output/f'{name}.csv',data)
    payload=dict(summary=summaries,risks=risks,episodes=len(rows),initial_states_equal=True,
        trajectory_pairing='only initial states paired; event outcomes may branch trajectories',
        event_counters='v3.43 dwell fields N/A; v3.44/v3.45 legacy fire/hit fields zero compatibility placeholders',
        formal_training_started=False,ledger_max_abs_error=max(r['ledger_max_abs_error'] for r in rows))
    (args.output/'summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
    print('version policy    W/L/D          seconds losses R/B dwell complete R/B',flush=True)
    for s in summaries:
        print(f"{s['version']} {s['policy']:8} {s['red_win_mean']:.0%}/{s['blue_win_mean']:.0%}/{s['draw_mean']:.0%} "
              f"{s['duration_s_mean']:.2f} {s['red_losses_mean']:.2f}/{s['blue_losses_mean']:.2f} "
              f"{s['red_dwell_completed_mean']}/{s['blue_dwell_completed_mean']}",flush=True)
    print('RISKS '+json.dumps(risks),flush=True)


if __name__=='__main__':main()
