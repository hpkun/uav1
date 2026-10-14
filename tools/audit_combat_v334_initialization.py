"""CUDA-gated geometry, straight-Red persistence and 200-episode fixed-policy diagnostics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.fixed_policy import NearestTargetPursuitPolicy
from env.v334_config import FORMATION_LOCAL_OFFSETS
from env.v334_scenario import random_combat_states_v334
from env.v323_reward import body_pair_geometry,body_levels
from env.v322_reward import distance_weights
from env.v36_geometry import angle_within
def distribution(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not a.size:
        return dict(count=0)
    return dict(count=len(a), mean=float(a.mean()), std=float(a.std()),
                **{k:float(v) for k,v in zip(('min','p05','median','p95','max'), np.quantile(a,[0,.05,.5,.95,1]))})


def target_vector(e):
    return [e.fixed_policy.nearest_target_index(b,e.red) if b.alive else None for b in e.blue]


def distinct(vector):
    return len({i for i in vector if i is not None})


def vector_key(v):
    return json.dumps(v,separators=(',',':'))


def funnel(e):
    # Read-only pre-fire live-pair counts. These never enter environment reward/info,
    # observations, policy, ranking or checkpoints.
    g=body_pair_geometry(e.red,e.blue)
    live=e.red_alive_mask.astype(bool)[:,None]&e.blue_alive_mask.astype(bool)[None,:]
    near=live&(g['distance']<=e.config['reward']['tactical_distance_max'])
    _,w4=distance_weights(g['distance'],e.config['reward'],e.config['weapon'])
    aa30=angle_within(np.abs(g['aa']),np.pi/6)
    actual=near&aa30&(body_levels(g,e.config['reward'],'offense_reward')*w4>0)
    masks=dict(live_pairs=live,distance_le_tactical=near,actual_r4_offense_positive=actual)
    for degree in (90,60,30):masks[f'distance_and_aa_le_{degree}']=near&angle_within(np.abs(g['aa']),np.deg2rad(degree))
    for degree in (30,15,5):
        align=angle_within(np.abs(g['body_azimuth']),np.deg2rad(degree))&angle_within(np.abs(g['body_elevation']),np.deg2rad(degree))
        masks[f'body_alignment_le_{degree}']=live&align
        masks[f'distance_and_body_alignment_le_{degree}']=near&align
        masks[f'distance_aa30_and_body_alignment_le_{degree}']=near&aa30&align
    return {k:int(v.sum()) for k,v in masks.items()}


def geometry_audit(e,seeds):
    baseline=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v332.yaml')
    rows=[];reference=None
    for seed in seeds:
        _,info=e.reset(seed);baseline.reset(seed)
        r=np.array([s.as_array() for s in e.red]);b=np.array([s.as_array() for s in e.blue])
        oldr=np.array([s.as_array() for s in baseline.red]);oldb=np.array([s.as_array() for s in baseline.blue])
        alpha=info['radial_angle'];u=np.array([np.cos(alpha),np.sin(alpha)]);t=np.array([-u[1],u[0]])
        cr,cb=r[:,:2].mean(0),b[:,:2].mean(0)
        np.testing.assert_allclose(cr,-4000*u,atol=1e-9);np.testing.assert_allclose(cb,4000*u,atol=1e-9)
        np.testing.assert_array_equal(b[:,:2],-r[:,:2])
        for states,center,forward,lateral,old in ((r,cr,u,t,oldr),(b,cb,-u,-t,oldb)):
            projected=np.column_stack(((states[:,:2]-center)@forward,(states[:,:2]-center)@lateral))
            np.testing.assert_allclose(projected,FORMATION_LOCAL_OFFSETS,atol=1e-9)
            np.testing.assert_array_equal(states[:,2:],old[:,2:])
        radius=np.linalg.norm(np.vstack((r[:,:2],b[:,:2])),axis=1)
        assert radius.max()<5000
        d=np.linalg.norm(b[:,None,:3]-r[None,:,:3],axis=-1)
        v=target_vector(e);rv=[e.fixed_policy.nearest_target_index(x,e.blue) for x in e.red]
        assert v==rv==[0,0,0,0]
        margin=np.sort(d,axis=1)[:,1]-d[:,0];assert margin.min()>300
        if reference is None:reference=d
        else:np.testing.assert_allclose(d,reference,atol=1e-9)
        rows.append(dict(seed=seed,alpha=alpha,red_center=cr.tolist(),blue_center=cb.tolist(),
            center_separation=float(np.linalg.norm(cr-cb)),maximum_radius=float(radius.max()),
            blue_targets=v,red_nearest_indices=rv,distinct_pursued_red=distinct(v),
            nearest_distances=d[:,0].tolist(),second_nearest_distances=np.sort(d,axis=1)[:,1].tolist(),
            nearest_margins=margin.tolist(),red_state=r.tolist(),blue_state=b.tolist()))
    class Axis:
        def uniform(self,low,high):return 0.
    red,blue,_=random_combat_states_v334(Axis(),**e.config['scenario'])
    standard=np.array([[np.linalg.norm(b.as_array()[:3]-r.as_array()[:3]) for r in red] for b in blue])
    return dict(seeds=len(rows),all_checks_passed=True,standard_alpha_0_blue_to_red_distance_matrix=standard.tolist(),
        maximum_initial_horizontal_radius=max(r['maximum_radius'] for r in rows),
        center_separation=distribution([r['center_separation'] for r in rows]),
        nearest_margin=distribution([m for r in rows for m in r['nearest_margins']]),
        initial_blue_target_frequency=dict(Counter(vector_key(r['blue_targets']) for r in rows)),rows=rows)


def simulate(e,seeds,zero_red,limit=None):
    rows=[];policy=NearestTargetPursuitPolicy(e.config['blue_policy'],e.config['action'])
    original=e.dense_combat_reward;counts={};samples=[]
    def dense():
        diagnostic=funnel(e)
        for k,v in diagnostic.items():counts[k]=counts.get(k,0)+v
        samples.append(dict(step=e.steps,**diagnostic))
        return original()
    e.dense_combat_reward=dense
    try:
        for seed in seeds:
            e.reset(seed);counts.clear();samples.clear()
            initial=target_vector(e);sequence=[dict(state_step=0,time_s=0.,targets=initial,distinct=distinct(initial))]
            previous=initial;first_switch=None;switches=0;persistence=0;lost_pattern=False
            first_pursued={str(i):None for i in (1,2,3)};snapshots={};first_fire=None
            original_entry=e._entry_attempts
            def entry(attackers,targets,states,side):
                nonlocal first_fire
                attempts=original_entry(attackers,targets,states,side)
                if attempts and first_fire is None:
                    vector=[None if x<0 else int(x) for x in e.blue_step_target_indices]
                    first_fire=dict(step=e.steps,time_s=e.steps*e.dt,side=side,
                        blue_decision_targets=vector,distinct=distinct(vector))
                return attempts
            e._entry_attempts=entry
            terminated=truncated=False
            try:
                while limit is None or e.steps<limit:
                    state_step=e.steps
                    current=target_vector(e)
                    if state_step in (0,10,20,50):snapshots[str(state_step*e.dt)]=dict(targets=current,distinct=distinct(current))
                    actions=np.zeros((4,3)) if zero_red else policy.team_actions(e.red,e.blue)
                    _,_,terminated,truncated,info=e.step(actions)
                    chosen=[None if x<0 else int(x) for x in info['blue_pursuit_target_indices']]
                    changed=sum(p is not None and c is not None and p!=c for p,c in zip(previous,chosen))
                    switches+=changed
                    if changed and first_switch is None:first_switch=dict(step=e.steps,decision_state_step=state_step,time_s=state_step*e.dt)
                    if chosen==[0,0,0,0] and not lost_pattern:persistence+=1
                    else:lost_pattern=True
                    for i in (1,2,3):
                        if i in chosen and first_pursued[str(i)] is None:
                            first_pursued[str(i)]=dict(step=e.steps,decision_state_step=state_step,time_s=state_step*e.dt)
                    previous=chosen
                    # Selection used for action step k uses state k-1; this timestamp is explicit.
                    sequence.append(dict(action_step=e.steps,state_step=state_step,time_s=state_step*e.dt,
                        targets=chosen,distinct=distinct(chosen)))
                    if terminated or truncated:break
            finally:e._entry_attempts=original_entry
            assert switches==info['blue_target_switches']
            row=dict(seed=seed,initial_blue_targets=initial,episode_length=e.steps,
                red_win=info['red_win'],blue_win=info['blue_win'],draw=info['draw'],timeout=truncated,
                mutual_destruction=not e.red_alive_mask.any() and not e.blue_alive_mask.any(),
                **{f'{side}_{k}':info[f'{side}_{k}'] for side in ('red','blue') for k in ('attack_kills','fire_attempts','weapon_hits')},
                first_target_switch=first_switch,target_switches=switches,
                all_four_target_r0_persistence_steps=persistence,
                persistence_right_censored=not lost_pattern,
                first_pursued_red=first_pursued,snapshots=snapshots,first_fire=first_fire,
                target_sequence=sequence,r4_opportunity_funnel=dict(counts),funnel_by_step=list(samples))
            rows.append(row)
            if (seed+1)%10==0:print(f"{'zero-Red' if zero_red else 'fixed-vs-fixed'}: {seed+1}/{len(seeds)}",flush=True)
    finally:e.dense_combat_reward=original
    totals={k:sum(r[k] for r in rows) for k in ('red_win','blue_win','draw','timeout','mutual_destruction',
        'red_attack_kills','blue_attack_kills','red_fire_attempts','blue_fire_attempts','red_weapon_hits','blue_weapon_hits')}
    totals.update(mean_episode_length=float(np.mean([r['episode_length'] for r in rows])),
        mean_target_switches=float(np.mean([r['target_switches'] for r in rows])),
        first_target_switch_step=distribution([None if r['first_target_switch'] is None else r['first_target_switch']['step'] for r in rows]),
        first_target_switch_time_s=distribution([None if r['first_target_switch'] is None else r['first_target_switch']['time_s'] for r in rows]),
        no_target_switch_episodes=sum(r['first_target_switch'] is None for r in rows),
        persistence_steps=distribution([r['all_four_target_r0_persistence_steps'] for r in rows]),
        persistence_right_censored_episodes=sum(r['persistence_right_censored'] for r in rows),
        initial_blue_target_frequency=dict(Counter(vector_key(r['initial_blue_targets']) for r in rows)),
        first_fire_target_frequency=dict(Counter(vector_key(r['first_fire']['blue_decision_targets']) for r in rows if r['first_fire'])),
        first_fire_time_s=distribution([r['first_fire']['time_s'] for r in rows if r['first_fire']]),
        distinct_pursued_red_over_time={k:dict(Counter(str(r['snapshots'][k]['distinct']) for r in rows if k in r['snapshots'])) for k in ('0.0','1.0','2.0','5.0')},
        target_vectors_over_time={k:dict(Counter(vector_key(r['snapshots'][k]['targets']) for r in rows if k in r['snapshots'])) for k in ('0.0','1.0','2.0','5.0')},
        first_pursued_red={str(i):dict(step=distribution([None if r['first_pursued_red'][str(i)] is None else r['first_pursued_red'][str(i)]['step'] for r in rows]),
            time_s=distribution([None if r['first_pursued_red'][str(i)] is None else r['first_pursued_red'][str(i)]['time_s'] for r in rows]),
            not_pursued_in_observed_horizon=sum(r['first_pursued_red'][str(i)] is None for r in rows)) for i in (1,2,3)})
    totals['r4_opportunity_funnel']={k:sum(r['r4_opportunity_funnel'][k] for r in rows) for k in counts}
    return dict(episodes=len(rows),red_policy='zero actions' if zero_red else 'nearest-target pure pursuit',
        horizon_steps=limit,totals=totals,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v334_validation/initialization_audit.json')
    args=p.parse_args()
    if args.episodes<200:raise ValueError('at least seeds 0-199 required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    e=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v334.yaml');seeds=range(args.episodes)
    result=dict(device=torch.cuda.get_device_name(0),environment_version='3.34',baseline='3.32',
        timing='Action step k selects at pre-advance state k-1, time=(k-1)*dt; fire time uses post-advance k*dt.',
        funnel_note='Counts of live Red-Blue pair-step opportunities sampled before firing; body-only counts are not distance/aspect gated. Conditional counts also reported. Diagnostic only.',
        geometry=geometry_audit(e,seeds))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    result['straight_red_100_steps']=simulate(e,seeds,zero_red=True,limit=100)
    args.output.write_text(json.dumps(result,indent=2))
    result['fixed_policy']=simulate(e,seeds,zero_red=False)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k]['totals'] for k in ('straight_red_100_steps','fixed_policy')},indent=2),flush=True)


if __name__=='__main__':main()
