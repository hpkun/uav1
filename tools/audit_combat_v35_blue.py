"""1000 common-seed paper Blue audit; no learned policy or parameter sweep."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from env.v30_policy import SensorLimitedPursuitPolicy
from tools.audit_combat_v30 import verify_transition
from tools.audit_combat_v34_blue import episode as episode33, summary as legacy_summary

def episode(task):
    version,mode,seed=task
    if version=='33':return episode33(task)
    e=MultiUAVCombatEnv('configs/combat_environment_v35.yaml');e.reset(seed)
    old=load_config('configs/combat_environment_v33.yaml')
    red=SensorLimitedPursuitPolicy(old['blue_policy'],old['action'],old['sensor'],old['scenario'])
    returns=np.zeros(5);bilateral=False
    while True:
        action=np.zeros((5,3)) if mode=='ZERO' else red.team_actions(e.red,e.blue)
        obs,r,t,tr,info=e.step(action);returns+=r;verify_transition(e,obs,r,info)
        assert info['blue_boundary_exits']==info['blue_ground_losses']==info['blue_ceiling_losses']==0
        assert all(not s.alive or (np.hypot(s.x,s.y)<=5000 and 0<s.altitude<=6000) for s in e.blue)
        bilateral|=info['red_fire_window_pairs']>0 and info['blue_fire_window_pairs']>0
        if t or tr:
            row={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in info.items()}
            row.update(seed=seed,mode=mode,episode_return=float(returns.sum()),mean_agent_episode_return=float(returns.mean()),bilateral_fire_window=bilateral)
            return row

def summary(rows):
    if rows[0]['environment_version']=='3.3':return legacy_summary(rows)
    # Reuse existing combat and conditional event-time statistics; adapt only
    # the old mode-specific adapter and replace it with v3.5's real diagnostics.
    compatible=[dict(r,blue_evade_episode=False,**{f'blue_{k}_agent_steps':0 for k in ('search','pursuit','evade','guard')}) for r in rows]
    s=legacy_summary(compatible);s.pop('blue_evade_episode_rate');s.pop('blue_mode_agent_step_fractions')
    names=('detected_pursuit','detected_escape','guerrilla','centripetal')
    counts={k:sum(r[f'blue_{k}_agent_steps'] for r in rows) for k in names};total=sum(counts.values())
    s['blue_mode_agent_step_fractions']={k:v/total if total else 0. for k,v in counts.items()}
    s['blue_noncombat_deaths']=sum(r['blue_boundary_exits']+r['blue_ground_losses'] for r in rows)
    for k in ('boundary','horizontal','ground','ceiling'):
        s[f'blue_{k}_returns_total']=sum(r[f'blue_{k}_returns'] for r in rows)
    return s

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=16);p.add_argument('--episodes',type=int,default=1000);p.add_argument('--seed-base',type=int,default=40000000)
    a=p.parse_args()
    from algorithm.rmappo.protocol import require_cuda
    require_cuda('cuda')
    if a.episodes<1000:raise ValueError('at least 1000 common seeds required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise RuntimeError('fresh output required')
    results={}
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for version in ('33','35'):
            for mode in ('ZERO','PURSUIT'):
                rows=[]
                for i,row in enumerate(pool.imap_unordered(episode,[(version,mode,a.seed_base+j) for j in range(a.episodes)],chunksize=4)):
                    rows.append(row)
                    if (i+1)%100==0:print(f'[BLUE AUDIT] v{version} {mode}: {i+1}/{a.episodes}',flush=True)
                rows.sort(key=lambda r:r['seed']);results[f'v{version}_{mode}']=summary(rows)
                (out/f'v{version}_{mode}_episodes.json').write_text(json.dumps(rows));(out/'progress.json').write_text(json.dumps(results,indent=2))
    (out/'summary.json').write_text(json.dumps(dict(common_seeds=a.episodes,seed_base=a.seed_base,groups=results,parameters_tuned=False,formal_training_started=False),indent=2))

if __name__=='__main__':main()
