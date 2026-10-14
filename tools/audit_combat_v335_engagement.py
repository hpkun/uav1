"""Same-seed v3.34/v3.35 fixed-pursuit comparison; read-only firing and R4 diagnostics."""
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
from env.geometry import engagement_geometry
from tools.audit_combat_v334_initialization import funnel,distribution,target_vector


def fixed_episodes(e,seeds):
    red_policy=NearestTargetPursuitPolicy(e.config['blue_policy'],e.config['action'])
    original_dense=e.dense_combat_reward;original_entry=e._entry_attempts
    counts={};by_step=[];first_fire={};first_miss={};first_volley={}
    def dense():
        diagnostic=funnel(e)
        for k,v in diagnostic.items():counts[k]=counts.get(k,0)+v
        by_step.append(dict(step=e.steps,**diagnostic))
        return original_dense()
    def entry(attackers,targets,states,side):
        attempts=original_entry(attackers,targets,states,side)
        if attempts and first_fire[side] is None:
            first_fire[side]=dict(step=e.steps,time_s=e.steps*e.dt,
                distances_m=[engagement_geometry(attackers[i],targets[j]).distance for i,j,_ in attempts])
            first_volley[side]=dict(attempts=len(attempts),hits=sum(hit for _,_,hit in attempts))
        if first_miss[side] is None and any(not hit for _,_,hit in attempts):
            first_miss[side]=dict(step=e.steps,time_s=e.steps*e.dt)
        return attempts
    e.dense_combat_reward=dense;e._entry_attempts=entry
    rows=[]
    try:
        for seed in seeds:
            obs,_=e.reset(seed);initial=target_vector(e)
            assert initial==[0,0,0,0]
            counts.clear();by_step.clear()
            for side in ('red','blue'):
                first_fire[side]=first_miss[side]=first_volley[side]=None
            while True:
                _,_,terminated,truncated,info=e.step(red_policy.team_actions(e.red,e.blue))
                if terminated or truncated:break
            misses=[m for m in first_miss.values() if m is not None]
            first_any_miss=min(misses,key=lambda m:m['step']) if misses else None
            row=dict(seed=seed,initial_blue_targets=initial,episode_length=e.steps,duration_s=e.steps*e.dt,
                timeout=truncated,mutual_destruction=not e.red_alive_mask.any() and not e.blue_alive_mask.any(),
                **{k:info[k] for k in ('red_win','blue_win','draw','termination_reason')},
                first_miss=first_any_miss,
                continued_after_first_miss=first_any_miss is not None and e.steps>first_any_miss['step'],
                steps_after_first_miss=None if first_any_miss is None else e.steps-first_any_miss['step'],
                r4_opportunity_funnel=dict(counts),funnel_by_step=list(by_step),
                red_reward_components={k:float(e.episode_reward_components[k].sum()) for k in ('guide','tactical_offense','tactical_defense')})
            for side in ('red','blue'):
                for field in ('fire_attempts','weapon_hits','attack_kills','ammo_used'):
                    row[f'{side}_{field}']=int(e.combat_counts[side][field])
                row[f'{side}_first_fire']=first_fire[side]
                row[f'{side}_first_volley']=first_volley[side]
                row[f'{side}_first_miss']=first_miss[side]
                step=info[f'{side}_first_fire_window_step']
                row[f'{side}_first_fire_window_step']=step
                row[f'{side}_remaining_ammo']=getattr(e,side+'_ammo').tolist()
                assert row[f'{side}_ammo_used']==24-sum(row[f'{side}_remaining_ammo'])==row[f'{side}_fire_attempts']
            rows.append(row)
            if (seed+1)%10==0:print(f'v{e.environment_version}: {seed+1}/{len(seeds)}',flush=True)
    finally:e.dense_combat_reward=original_dense;e._entry_attempts=original_entry
    totals={k:sum(r[k] for r in rows) for k in ('red_win','blue_win','draw','timeout','mutual_destruction',
        'red_attack_kills','blue_attack_kills','red_fire_attempts','blue_fire_attempts','red_weapon_hits','blue_weapon_hits','red_ammo_used','blue_ammo_used')}
    totals.update(mean_episode_length=float(np.mean([r['episode_length'] for r in rows])),
        mean_episode_duration_s=float(np.mean([r['duration_s'] for r in rows])),
        mutual_destruction_rate=totals['mutual_destruction']/len(rows),
        initial_blue_target_frequency=dict(Counter(json.dumps(r['initial_blue_targets']) for r in rows)),
        episodes_with_miss=sum(r['first_miss'] is not None for r in rows),
        continued_after_first_miss_episodes=sum(r['continued_after_first_miss'] for r in rows),
        steps_after_first_miss=distribution([r['steps_after_first_miss'] for r in rows]),
        first_miss_time_s=distribution([None if r['first_miss'] is None else r['first_miss']['time_s'] for r in rows]),
        episodes_continued_after_initial_volley=sum(r['episode_length']>min(r[f'{side}_first_fire']['step'] for side in ('red','blue') if r[f'{side}_first_fire']) for r in rows))
    for side in ('red','blue'):
        totals[f'{side}_hit_rate']=totals[f'{side}_weapon_hits']/max(1,totals[f'{side}_fire_attempts'])
        fires=[r[f'{side}_first_fire'] for r in rows if r[f'{side}_first_fire']]
        totals[f'{side}_first_fire_time_s']=distribution([r['time_s'] for r in fires])
        totals[f'{side}_first_fire_distance_m']=distribution([d for r in fires for d in r['distances_m']])
        totals[f'{side}_first_fire_window_step']=distribution([r[f'{side}_first_fire_window_step'] for r in rows])
        totals[f'{side}_first_volley_attempts']=sum(r[f'{side}_first_volley']['attempts'] for r in rows if r[f'{side}_first_volley'])
        totals[f'{side}_first_volley_hits']=sum(r[f'{side}_first_volley']['hits'] for r in rows if r[f'{side}_first_volley'])
    totals['combined_hit_rate']=(totals['red_weapon_hits']+totals['blue_weapon_hits'])/max(1,totals['red_fire_attempts']+totals['blue_fire_attempts'])
    totals['r4_opportunity_funnel']={k:sum(r['r4_opportunity_funnel'][k] for r in rows) for k in counts}
    totals['red_reward_components']={k:sum(r['red_reward_components'][k] for r in rows) for k in ('guide','tactical_offense','tactical_defense')}
    return dict(episodes=len(rows),totals=totals,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v335_validation/engagement_audit.json')
    args=p.parse_args()
    if args.episodes<200:raise ValueError('at least seeds 0-199 required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory for repository audits; no CPU fallback')
    envs={v:MultiUAVCombatEnv(ROOT/f'configs/combat_environment_v{v}.yaml') for v in ('334','335')}
    reset_rows=[]
    for seed in range(args.episodes):
        a,_=envs['334'].reset(seed);b,_=envs['335'].reset(seed)
        np.testing.assert_array_equal(a,b)
        assert envs['334'].red==envs['335'].red and envs['334'].blue==envs['335'].blue
        assert target_vector(envs['334'])==target_vector(envs['335'])==[0,0,0,0]
        reset_rows.append(dict(seed=seed,identical_state=True,blue_nearest_indices=[0,0,0,0]))
    result=dict(device=torch.cuda.get_device_name(0),same_initial_seeds=list(range(args.episodes)),
        identical_initial_states=True,reset_checks=reset_rows,
        policy='Both teams nearest-target pure pursuit; inherited Red/Blue firing target coupling unchanged.',
        funnel_note='Same unchanged v3.34 funnel at pre-fire stage; diagnostic only. Pair-step totals also include live-pair denominators.',
        miss_note='A miss is an existing attempt_hit=False. Post-miss durations are conditional on an observed miss; no-miss episodes are excluded.',versions={})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for v,e in envs.items():
        result['versions'][v]=fixed_episodes(e,range(args.episodes))
        args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({v:r['totals'] for v,r in result['versions'].items()},indent=2),flush=True)


if __name__=='__main__':main()
