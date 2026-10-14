"""Same-state reward A/B and bit-identical paired weapon/trajectory audit."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import math
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.geometry import engagement_geometry
from env.v323_reward import body_pair_geometry,body_levels,tactical_rewards_v323
from env.v322_reward import distance_weights
from env.v36_geometry import angle_within
from env.v337_reward import tactical_pair_rewards_v337,tactical_rewards_v337
from env.fixed_policy import NearestTargetPursuitPolicy
from env.models import AircraftState
from tools.audit_combat_v336_weapon_geometry import policy_actions


COMPONENTS=('guide','tactical_offense','tactical_defense')


def geometry_pair(d=2000,ata=0,ha=0,theta=0,aa=0):
    bearing,elevation,pitch=np.deg2rad([ata,ha,theta])
    a=AircraftState(0,0,-3000,225,pitch,0)
    b=AircraftState(d*np.cos(elevation)*np.cos(bearing),d*np.cos(elevation)*np.sin(bearing),
                    -3000-d*np.sin(elevation),225,0,math.radians(ata+aa))
    return a,b


def body_pairs(red,blue,reward,weapon):
    g=body_pair_geometry(red,blue);r=body_pair_geometry(blue,red)
    live=np.asarray([a.alive for a in red])[:,None]&np.asarray([b.alive for b in blue])[None,:]
    w3,w4=distance_weights(g['distance'],reward,weapon);_,rw4=distance_weights(r['distance'],reward,weapon)
    guide=np.where(live&angle_within(abs(g['body_azimuth']),np.pi/6)&angle_within(abs(g['body_elevation']),np.pi/6),reward['guide_reward']*w3,0.)
    offense=np.where(live&(g['distance']<=reward['tactical_distance_max'])&angle_within(abs(g['aa']),reward['tactical_aspect_max']),body_levels(g,reward,'offense_reward')*w4,0.)
    defense=np.where(live.T&(r['distance']<=reward['tactical_distance_max'])&angle_within(abs(r['aa']),reward['tactical_aspect_max']),body_levels(r,reward,'threat_penalty')*rw4,0.)
    return dict(guide=guide,tactical_offense=offense,tactical_defense=defense)


def compare_rewards(red,blue,reward,weapon,emit=None,**context):
    old=body_pairs(red,blue,reward,weapon);new=tactical_pair_rewards_v337(red,blue,reward,weapon)
    old_agent=tactical_rewards_v323(red,blue,reward,weapon);new_agent=tactical_rewards_v337(red,blue,reward,weapon)
    live=np.asarray([a.alive for a in red])[:,None]&np.asarray([b.alive for b in blue])[None,:]
    counts=Counter()
    for k in COMPONENTS:
        mask=live.T if k=='tactical_defense' else live
        delta=new[k]-old[k]
        counts[k+'_pair_agreement']=int(((delta==0)&mask).sum())
        counts[k+'_pair_changed']=int(((delta!=0)&mask).sum())
        counts[k+'_pair_delta_sum']=float(delta.sum())
        counts[k+'_pair_abs_delta_sum']=float(abs(delta).sum())
        ad=new_agent[k]-old_agent[k];alive=np.asarray([a.alive for a in red])
        counts[k+'_agent_agreement']=int(((ad==0)&alive).sum())
        counts[k+'_agent_changed']=int(((ad!=0)&alive).sum())
        counts[k+'_agent_delta_sum']=float(ad.sum())
        # Independently validate reconstruction against the actual frozen function.
        aggregate=old[k].min(axis=0) if k=='tactical_defense' else old[k].max(axis=1)
        np.testing.assert_array_equal(aggregate,old_agent[k])
        if emit is not None:
            for i,j in np.argwhere((delta!=0)&mask):
                a,b=(blue[i],red[j]) if k=='tactical_defense' else (red[i],blue[j])
                g=engagement_geometry(a,b)
                emit(dict(**context,component=k,attacker_side='blue' if k=='tactical_defense' else 'red',
                    attacker=int(i),target=int(j),distance_m=g.distance,ata_deg=math.degrees(g.ata),
                    ha_deg=math.degrees(g.ha),aa_deg=math.degrees(g.aa),
                    body_azimuth_deg=math.degrees(g.boresight_azimuth_error),
                    body_elevation_deg=math.degrees(g.boresight_elevation_error),attacker_theta_deg=math.degrees(a.theta),
                    v336_pair_reward=float(old[k][i,j]),v337_pair_reward=float(new[k][i,j]),difference=float(delta[i,j])))
    d=new_agent['tactical']-old_agent['tactical']
    counts['total_tactical_delta_sum']=float(d.sum())
    counts['total_tactical_abs_delta_sum']=float(abs(d).sum())
    counts['total_tactical_agent_changed']=int((d!=0).sum())
    return counts,old_agent,new_agent


def assert_frozen_transition(a,b,x,y):
    assert a.red==b.red and a.blue==b.blue and x[2:4]==y[2:4]
    assert a.rng.bit_generator.state==b.rng.bit_generator.state
    assert a.combat_counts==b.combat_counts
    np.testing.assert_array_equal(x[0],y[0])
    for side in ('red','blue'):
        for suffix in ('_ammo','_fire_cooldown_steps'):
            np.testing.assert_array_equal(getattr(a,side+suffix),getattr(b,side+suffix))
    for field in ('r1_rewards','r2_rewards','outcome_rewards','safe_rewards','adv_rewards'):
        np.testing.assert_array_equal(x[4][field],y[4][field])


def rollout(mode,seeds,emit):
    a=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v336.yaml')
    b=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v337.yaml')
    policy=NearestTargetPursuitPolicy(a.config['blue_policy'],a.config['action'])
    original=a.dense_combat_reward;total=Counter();rows=[];snapshot={}
    def dense():
        c,old,new=compare_rewards(a.red,a.blue,a.config['reward'],a.config['weapon'],emit,
            source=mode,seed=snapshot['seed'],step=a.steps)
        total.update(c);snapshot.update(old=old,new=new)
        return original()
    a.dense_combat_reward=dense
    try:
        for seed in seeds:
            x,_=a.reset(seed);y,_=b.reset(seed);np.testing.assert_array_equal(x,y);snapshot['seed']=seed
            while True:
                actions=policy_actions(a,policy,mode)
                x=a.step(actions);y=b.step(actions.copy())
                assert_frozen_transition(a,b,x,y)
                for k in COMPONENTS:
                    np.testing.assert_array_equal(x[4][k+'_rewards'],snapshot['old'][k])
                    np.testing.assert_array_equal(y[4][k+'_rewards'],snapshot['new'][k])
                if x[2] or x[3]:break
            rows.append(dict(seed=seed,steps=a.steps,zero_transition_weapon_rng_difference=True,
                red_win=bool(x[4]['red_win']),blue_win=bool(x[4]['blue_win']),draw=bool(x[4]['draw']),
                combat_counts=a.combat_counts,red_ammo=a.red_ammo.tolist(),blue_ammo=a.blue_ammo.tolist()))
            print(f'{mode}: seed {seed} ({a.steps} steps) zero-difference passed',flush=True)
    finally:del a.dense_combat_reward
    return dict(episodes=len(rows),same_state_reward_comparison=dict(total),
        state_observation_weapon_rng_ammo_cooldown_event_outcome_zero_difference=True,rows=rows)


def controlled(reward,weapon,emit):
    totals=Counter()
    for d in (500,1500,2500,3500):
        for ata in (-35,-30,-15,-5,0,5,15,30,35):
            for ha in (-35,-30,-15,0,15,30,35):
                for theta in (-20,0,20):
                    for reverse in (False,True):
                        a,b=geometry_pair(d,ata,ha,theta)
                        red,blue=([b],[a]) if reverse else ([a],[b])
                        c,_,_=compare_rewards(red,blue,reward,weapon,emit,source='controlled_grid',seed=-1,step=0)
                        totals.update(c)
    return dict(cases=1512,statistics=dict(totals))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/v337_validation/reward_audit');args=p.parse_args()
    if args.episodes<=0:raise ValueError('episodes must be positive')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory for repository audits; no CPU fallback')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise FileExistsError('Use a fresh output directory')
    e=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v337.yaml')
    fields=['source','seed','step','component','attacker_side','attacker','target','distance_m','ata_deg','ha_deg','aa_deg',
        'body_azimuth_deg','body_elevation_deg','attacker_theta_deg','v336_pair_reward','v337_pair_reward','difference']
    with (out/'changed_pairs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        result=dict(device=torch.cuda.get_device_name(0),seeds=list(range(args.episodes)),
            scope='Both rewards evaluated at identical authoritative pre-fire states. Paired rollout assertions forbid any transition/weapon/RNG difference. Blue policy unchanged; pitch probe changes only diagnostic Red actions.',
            grid=controlled(e.config['reward'],e.config['weapon'],writer.writerow),rollouts={})
        for mode in ('fixed','pitch_probe'):
            result['rollouts'][mode]=rollout(mode,range(args.episodes),writer.writerow)
            (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
