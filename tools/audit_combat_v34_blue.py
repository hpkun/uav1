"""Fixed-seed Blue-only behavioral audit, no learned policies or tuning."""
import argparse, json, multiprocessing as mp
from pathlib import Path
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from env.v30_policy import SensorLimitedPursuitPolicy
from algorithm.common.evaluator import aggregate_combat_records
from tools.audit_combat_v30 import verify_transition
from tools._capture_v33_reference import canonical, states

def paired(seed, mode):
    a=MultiUAVCombatEnv('configs/combat_environment_v33.yaml')
    b=MultiUAVCombatEnv('configs/combat_environment_v34.yaml')
    oa,ia=a.reset(seed);ob,ib=b.reset(seed)
    assert canonical((states(a),oa))==canonical((states(b),ob))
    rng=np.random.default_rng(seed+100000000)
    while True:
        red=rng.uniform(-1,1,(5,3)) if mode=='RANDOM' else np.zeros((5,3))
        blue=a.fixed_policy.team_actions(a.blue,a.red)
        oa,ra,ta,tra,ia=a.step(red,blue)
        ob,rb,tb,trb,ib=b.step(red,blue)
        assert canonical((states(a),oa,ra,ta,tra,a.rng.bit_generator.state))==canonical((states(b),ob,rb,tb,trb,b.rng.bit_generator.state))
        assert canonical({k:v for k,v in ia.items() if k!='environment_version'})==canonical({k:v for k,v in ib.items() if k!='environment_version'})
        np.testing.assert_array_equal(a.weapon.eligibility(a.red,a.blue,a.red_ammo)[0],b.weapon.eligibility(b.red,b.blue,b.red_ammo)[0])
        if ta or tra:return a.steps

def episode(task):
    version,mode,seed=task
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml');e.reset(seed)
    # Always the same memoryless v3.3 Red policy, never v3.4's Blue policy.
    red=SensorLimitedPursuitPolicy(e.config['blue_policy'],e.config['action'],e.config['sensor'],e.config['scenario'])
    returns=np.zeros(5);bilateral=False
    counts=dict.fromkeys(('search','pursuit','evade','guard'),0)
    while True:
        action=np.zeros((5,3)) if mode=='ZERO' else red.team_actions(e.red,e.blue)
        if version=='33':
            from env.sensor import sensor_geometry
            visible=sensor_geometry(e.blue,e.red,e.config['sensor'])[0]
            for i,s in enumerate(e.blue):
                if not s.alive:continue
                guard=np.hypot(s.x,s.y)>=4500 or s.altitude<=500 or s.altitude>=5500
                counts['guard' if guard else 'pursuit' if visible[i].any() else 'search']+=1
        obs,r,t,tr,info=e.step(action);returns+=r
        verify_transition(e,obs,r,info)
        bilateral|=info['red_fire_window_pairs']>0 and info['blue_fire_window_pairs']>0
        if t or tr:
            row={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in info.items()}
            if version=='33':
                row.update({f'blue_{k}_agent_steps':v for k,v in counts.items()},blue_evade_episode=False)
            row.update(seed=seed,mode=mode,episode_return=float(returns.sum()),mean_agent_episode_return=float(returns.mean()),bilateral_fire_window=bilateral)
            return row

def summary(rows):
    s=aggregate_combat_records(rows)
    s['median_episode_length']=float(np.median([r['episode_length'] for r in rows]))
    s['bilateral_fire_window_episode_rate']=float(np.mean([r['bilateral_fire_window'] for r in rows]))
    s['same_step_first_fire_window_rate']=float(np.mean([r['red_first_fire_window_step'] is not None and r['red_first_fire_window_step']==r['blue_first_fire_window_step'] for r in rows]))
    for side in ('red','blue'):
        for event in ('fire_window','kill'):
            vals=[r[f'{side}_first_{event}_step'] for r in rows if r[f'{side}_first_{event}_step'] is not None]
            s[f'{side}_first_{event}_step_stats']=dict(count=len(vals),mean=float(np.mean(vals)) if vals else None,median=float(np.median(vals)) if vals else None)
    counts={k:sum(r[f'blue_{k}_agent_steps'] for r in rows) for k in ('search','pursuit','evade','guard')}
    total=sum(counts.values())
    s['blue_mode_agent_step_fractions']={k:v/total if total else 0 for k,v in counts.items()}
    s['blue_evade_episode_rate']=float(np.mean([r['blue_evade_episode'] for r in rows]))
    s['blue_noncombat_loss_episode_rate']=float(np.mean([r['blue_boundary_exits']+r['blue_ground_losses']>0 for r in rows]))
    s['blue_noncombat_loss_aircraft_fraction']=sum(r['blue_boundary_exits']+r['blue_ground_losses'] for r in rows)/(5*len(rows))
    return s

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=16);p.add_argument('--episodes',type=int,default=1000);p.add_argument('--seed-base',type=int,default=39000000)
    args=p.parse_args()
    from algorithm.rmappo.protocol import require_cuda
    require_cuda('cuda')
    if args.episodes<1000:raise ValueError('at least 1000 common seeds required')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise RuntimeError('fresh output required')
    results={}
    with mp.get_context('spawn').Pool(args.workers) as pool:
        for version in ('33','34'):
            for mode in ('ZERO','PURSUIT'):
                rows=[]
                for i,row in enumerate(pool.imap_unordered(episode,[(version,mode,args.seed_base+i) for i in range(args.episodes)],chunksize=4)):
                    rows.append(row)
                    if (i+1)%100==0:print(f'[BLUE AUDIT] v{version} {mode}: {i+1}/{args.episodes}',flush=True)
                rows.sort(key=lambda r:r['seed']);results[f'v{version}_{mode}']=summary(rows)
                (out/f'v{version}_{mode}_episodes.json').write_text(json.dumps(rows))
                (out/'progress.json').write_text(json.dumps(results,indent=2))
    (out/'summary.json').write_text(json.dumps(dict(common_seeds=args.episodes,seed_base=args.seed_base,groups=results,parameters_tuned=False,formal_training_started=False),indent=2))

if __name__=='__main__':main()
