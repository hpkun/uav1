"""CUDA-gated 200-seed initialization and scenario-level fixed-pursuit comparison."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.fixed_policy import NearestTargetPursuitPolicy
from env.geometry import engagement_geometry
from env.v333_reward import tactical_rewards_v333
from env.v323_reward import tactical_rewards_v323


def distribution(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not a.size:
        return dict(count=0)
    return dict(count=len(a), mean=float(a.mean()), std=float(a.std()),
                **{k:float(v) for k,v in zip(('min','p05','median','p95','max'), np.quantile(a,[0,.05,.5,.95,1]))})


def initialization(e, seeds):
    rows=[]
    for seed in seeds:
        _,info=e.reset(seed)
        red=np.array([s.as_array() for s in e.red]);blue=np.array([s.as_array() for s in e.blue])
        rc,bc=red[:,:2].mean(0),blue[:,:2].mean(0)
        alpha=info['radial_angle'];u=np.array([np.cos(alpha),np.sin(alpha)]);v=np.array([-u[1],u[0]])
        radius=e.config['scenario']['center_radius']
        np.testing.assert_allclose(rc,-radius*u,atol=1e-9)
        np.testing.assert_allclose(bc,radius*u,atol=1e-9)
        assert np.all(np.linalg.norm(np.vstack((red[:,:2],blue[:,:2])),axis=1)<5000)
        for team,center in ((red,rc),(blue,bc)):
            np.testing.assert_allclose((team[:,:2]-center)@v,[-300,-100,100,300],atol=1e-9)
            np.testing.assert_allclose((team[:,:2]-center)@u,0,atol=1e-9)
            np.testing.assert_array_equal(team[:,2],np.full(4,-3000.))
            np.testing.assert_array_equal(team[:,3],np.full(4,225.))
            np.testing.assert_array_equal(team[:,4],np.zeros(4))
        np.testing.assert_allclose(red[:,:2]+blue[::-1,:2],0,atol=1e-9)
        d=np.linalg.norm(red[:,None,:3]-blue[None,:,:3],axis=-1)
        rows.append(dict(seed=seed,diameter_angle=alpha,red_center_norm=float(np.linalg.norm(rc)),
            blue_center_norm=float(np.linalg.norm(bc)),center_separation=float(np.linalg.norm(rc-bc)),
            max_aircraft_radius=float(np.linalg.norm(np.vstack((red[:,:2],blue[:,:2])),axis=1).max()),
            nearest_opposing_distances=d.min(axis=1).tolist(),red_state=red.tolist(),blue_state=blue.tolist()))
    assert len(set(r['diameter_angle'] for r in rows))==len(rows)
    return dict(seeds=len(rows),all_checks_passed=True,random_orientation=distribution([r['diameter_angle'] for r in rows]),
                center_separation=distribution([r['center_separation'] for r in rows]),
                nearest_opposing_distance=distribution([d for r in rows for d in r['nearest_opposing_distances']]),rows=rows)


def fixed_episodes(e, seeds):
    policy=NearestTargetPursuitPolicy(e.config['blue_policy'],e.config['action'])
    reward_fn=tactical_rewards_v333 if e.environment_version=='3.33' else tactical_rewards_v323
    original_entry=e._entry_attempts;original_dense=e.dense_combat_reward
    first_fire={};blue_components={}
    def entry(attackers,targets,states,side):
        attempts=original_entry(attackers,targets,states,side)
        if attempts and first_fire[side] is None:
            first_fire[side]=dict(step=e.steps,time_s=e.steps*e.dt,
                distances_m=[engagement_geometry(attackers[i],targets[j]).distance for i,j,_ in attempts])
        return attempts
    def dense():
        d=reward_fn(e.blue,e.red,e.config['reward'],e.config['weapon'])
        for key in blue_components:blue_components[key]+=float(d[key].sum())
        return original_dense()
    e._entry_attempts=entry;e.dense_combat_reward=dense
    rows=[]
    try:
        for seed in seeds:
            first_fire.update(red=None,blue=None)
            blue_components.update(guide=0.,tactical_offense=0.,tactical_defense=0.)
            e.reset(seed)
            while True:
                _,_,terminated,truncated,info=e.step(policy.team_actions(e.red,e.blue))
                if terminated or truncated:break
            row=dict(seed=seed,episode_length=e.steps,episode_duration_s=e.steps*e.dt,
                timeout=truncated,mutual_destruction=not e.red_alive_mask.any() and not e.blue_alive_mask.any(),
                **{k:info[k] for k in ('red_win','blue_win','draw','red_attack_kills','blue_attack_kills','termination_reason')})
            for side in ('red','blue'):
                for field in ('fire_attempts','weapon_hits'):row[f'{side}_{field}']=info[f'{side}_{field}']
                step=info[f'{side}_first_fire_window_step']
                row[f'{side}_first_fire_window_step']=step
                row[f'{side}_first_fire_window_time_s']=None if step is None else step*e.dt
                row[f'{side}_first_fire']=first_fire[side]
                row[f'{side}_fire_window_episode']=step is not None
            row['red_components']={k:float(e.episode_reward_components[k].sum()) for k in blue_components}
            row['blue_equivalent_components']=dict(blue_components)
            rows.append(row)
            if (seed+1)%10==0:print(f'v{e.environment_version}: {seed+1}/{len(seeds)}',flush=True)
    finally:
        e._entry_attempts=original_entry;e.dense_combat_reward=original_dense
    keys=('red_win','blue_win','draw','timeout','mutual_destruction','red_attack_kills','blue_attack_kills',
          'red_fire_attempts','blue_fire_attempts','red_weapon_hits','blue_weapon_hits')
    totals={k:sum(r[k] for r in rows) for k in keys}
    totals.update(mean_episode_length=float(np.mean([r['episode_length'] for r in rows])),
                  mean_episode_duration_s=float(np.mean([r['episode_duration_s'] for r in rows])),
                  mutual_destruction_rate=totals['mutual_destruction']/len(rows))
    for side in ('red','blue'):
        totals[f'{side}_hit_rate']=totals[f'{side}_weapon_hits']/max(1,totals[f'{side}_fire_attempts'])
        totals[f'{side}_fire_window_episode_rate']=float(np.mean([r[f'{side}_fire_window_episode'] for r in rows]))
        for field in ('first_fire_window_step','first_fire_window_time_s'):
            totals[f'{side}_{field}']=distribution([r[f'{side}_{field}'] for r in rows])
        fires=[r[f'{side}_first_fire'] for r in rows if r[f'{side}_first_fire'] is not None]
        totals[f'{side}_first_fire_time_s']=distribution([r['time_s'] for r in fires])
        totals[f'{side}_first_fire_distance_m']=distribution([d for r in fires for d in r['distances_m']])
        components='red_components' if side=='red' else 'blue_equivalent_components'
        totals[components]={k:sum(r[components][k] for r in rows) for k in blue_components}
    return dict(episodes=len(rows),totals=totals,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v333_validation/engagement_audit.json')
    args=p.parse_args()
    if args.episodes<200:raise ValueError('at least seeds 0-199 required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    seeds=range(args.episodes)
    envs={v:MultiUAVCombatEnv(ROOT/f'configs/combat_environment_v{v}.yaml') for v in ('332','333')}
    result=dict(device=torch.cuda.get_device_name(0),comparison_kind='scenario-level; center radius differs, not same trajectory',
        blue_geometry_note='Blue-equivalent reward is a diagnostic sampled at the same pre-fire stage; Blue receives no training reward.',
        hit_probability={str(d):envs['333'].weapon.expected_hit_probability(d,0,0) for d in (1000,2000,3000,4000)},
        initialization={v:initialization(e,seeds) for v,e in envs.items()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for v,e in envs.items():
        result[v]=fixed_episodes(e,seeds)
        args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({v:result[v]['totals'] for v in envs},indent=2),flush=True)


if __name__=='__main__':main()
