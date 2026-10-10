"""Fixed-seed v3.5/v3.6 hardware/reward audit with nonmutating geometry observers."""
import argparse,json,multiprocessing as mp
from pathlib import Path
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from env.sensor import sensor_geometry
from env.v36_geometry import combat_geometry,angle_within
from env.v30_policy import SensorLimitedPursuitPolicy
from tools.audit_combat_v30 import verify_transition
from tools.audit_combat_v35_blue import summary as shared_summary

class AuditMixin:
    def reset(self,seed=None):
        result=super().reset(seed)
        self.radar_first=dict(red=None,blue=None)
        self.geometry_entries={s:dict(total=0,ata30=0,ha30=0,both30=0) for s in ('red','blue')}
        self.first_fire_geometry=dict(red=None,blue=None)
        return result

    def dense_combat_reward(self):
        result=super().dense_combat_reward()
        for side,team,targets in (('red',self.red,self.blue),('blue',self.blue,self.red)):
            if self.radar_first[side] is None and sensor_geometry(team,targets,self.config['sensor'])[0].any():self.radar_first[side]=self.steps
        return result

    def _entry_attempts(self,attackers,targets,fire_states,side):
        eligible,_=self.weapon.eligibility(attackers,targets,getattr(self,f'{side}_ammo'))
        entries=eligible & ~fire_states.previous_eligible
        g=combat_geometry(attackers,targets);stats=self.geometry_entries[side]
        ata=angle_within(g['ata'],np.pi/6);ha=angle_within(g['ha'],np.pi/6)
        stats['total']+=int(entries.sum());stats['ata30']+=int((entries&ata).sum());stats['ha30']+=int((entries&ha).sum());stats['both30']+=int((entries&ata&ha).sum())
        result=super()._entry_attempts(attackers,targets,fire_states,side)
        if result and self.first_fire_geometry[side] is None:
            self.first_fire_geometry[side]={k:[float(g[k][i,j]) for i,j,_ in result] for k in ('ata','ha','aa')}
        return result

def episode(task):
    version,mode,seed=task;config=load_config(f'configs/combat_environment_v{version}.yaml')
    base=type(MultiUAVCombatEnv.__new__(MultiUAVCombatEnv,config))
    e=type('AuditedCombat',(AuditMixin,base),{})(config);e.reset(seed)
    # Same Red controller and thresholds in both versions; only its shared
    # radar input naturally follows each version's symmetric hardware.
    old=load_config('configs/combat_environment_v33.yaml')
    red=SensorLimitedPursuitPolicy(old['blue_policy'],config['action'],config['sensor'],config['scenario'])
    returns=np.zeros(5);bilateral=False
    while True:
        actions=np.zeros((5,3)) if mode=='ZERO' else red.team_actions(e.red,e.blue)
        obs,r,t,tr,info=e.step(actions);verify_transition(e,obs,r,info);returns+=r
        bilateral|=info['red_fire_window_pairs']>0 and info['blue_fire_window_pairs']>0
        assert info['blue_boundary_exits']==info['blue_ground_losses']==0
        if t or tr:
            row={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in info.items()}
            row.update(seed=seed,mode=mode,episode_return=float(returns.sum()),mean_agent_episode_return=float(returns.mean()),bilateral_fire_window=bilateral,
                first_radar_detection=e.radar_first,geometry_entries=e.geometry_entries,first_fire_geometry=e.first_fire_geometry)
            return row

def summary(rows):
    s=shared_summary(rows)
    for side in ('red','blue'):
        vals=[r['first_radar_detection'][side] for r in rows if r['first_radar_detection'][side] is not None]
        s[f'{side}_first_radar_detection_stats']=dict(count=len(vals),mean=float(np.mean(vals)) if vals else None,median=float(np.median(vals)) if vals else None)
        for event in ('attempt',):
            vals=[r[f'{side}_first_{event}_step'] for r in rows if r[f'{side}_first_{event}_step'] is not None]
            s[f'{side}_first_{event}_stats']=dict(count=len(vals),mean=float(np.mean(vals)) if vals else None,median=float(np.median(vals)) if vals else None)
        sums={k:sum(r['geometry_entries'][side][k] for r in rows) for k in ('total','ata30','ha30','both30')}
        s[f'{side}_entry_geometry']=dict(**sums,**{f'{k}_fraction':sums[k]/sums['total'] if sums['total'] else None for k in ('ata30','ha30','both30')})
        s[f'{side}_first_fire_mean_degrees']={k:float(np.rad2deg(np.mean([v for r in rows if r['first_fire_geometry'][side] is not None for v in r['first_fire_geometry'][side][k]]))) for k in ('ata','ha','aa')}
    return s

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--episodes',type=int,default=1000);p.add_argument('--workers',type=int,default=16);p.add_argument('--seed-base',type=int,default=41000000)
    a=p.parse_args()
    from algorithm.rmappo.protocol import require_cuda
    require_cuda('cuda')
    if a.episodes<1000:raise ValueError('at least 1000 common seeds required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise RuntimeError('fresh output required')
    groups={}
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for version in ('35','36'):
            for mode in ('ZERO','PURSUIT'):
                rows=[]
                for i,row in enumerate(pool.imap_unordered(episode,[(version,mode,a.seed_base+j) for j in range(a.episodes)],chunksize=4)):
                    rows.append(row)
                    if (i+1)%100==0:print(f'[V36 AUDIT] v{version} {mode}: {i+1}/{a.episodes}',flush=True)
                rows.sort(key=lambda r:r['seed']);groups[f'v{version}_{mode}']=summary(rows)
                (out/f'v{version}_{mode}_episodes.json').write_text(json.dumps(rows));(out/'progress.json').write_text(json.dumps(groups,indent=2))
    (out/'summary.json').write_text(json.dumps(dict(common_seeds=a.episodes,seed_base=a.seed_base,groups=groups,parameters_tuned=False,formal_training_started=False),indent=2))

if __name__=='__main__':main()
