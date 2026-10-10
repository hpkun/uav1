"""Descriptive paired initialization audit; no training or parameter selection."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse,gzip,json,multiprocessing as mp,time
from collections import Counter
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from env.math_utils import wrap_angle
from env.sensor import sensor_geometry
from tools.audit_combat_v30 import summarize,verify_transition
from tools.audit_v30_initial_policy_baselines import jsonable,wilson

CONFIGS=None
def initialize(configs):
    global CONFIGS
    CONFIGS=configs

def moments(values):
    a=np.asarray(values,dtype=float)
    return dict(count=len(a),mean=float(a.mean()) if len(a) else None,
        std=float(a.std()) if len(a) else None,median=float(np.median(a)) if len(a) else None,
        p10=float(np.percentile(a,10)) if len(a) else None,p90=float(np.percentile(a,90)) if len(a) else None)

def team_geometry(e):
    result={}
    for side in ('red','blue'):
        p=np.asarray([[s.x,s.y] for s in getattr(e,side) if s.alive],dtype=float)
        result[side]=dict(centroid_radius=float(np.linalg.norm(p.mean(0))) if len(p) else None,
            spread=float(np.linalg.norm(p-p.mean(0),axis=1).mean()) if len(p) else None)
    return result

class Probe:
    def __init__(self,e):
        self.e=e;self.original=e._entry_attempts;e._entry_attempts=self.entry
        self.first_detection=dict(red=None,blue=None)
        self.funnel=dict(reset=team_geometry(e),first_detection=None,first_fire_window=None)
        self.first_pair_radius=None
        self.observe()
    def observe(self):
        for side,other in (('red','blue'),('blue','red')):
            if self.first_detection[side] is None and sensor_geometry(getattr(self.e,side),getattr(self.e,other),self.e.config['sensor'])[0].any():
                self.first_detection[side]=self.e.steps
                if self.funnel['first_detection'] is None:self.funnel['first_detection']=team_geometry(self.e)
    def entry(self,attackers,targets,states,side):
        if side=='red':self.observe()
        if self.funnel['first_fire_window'] is None:
            eligible,dist=self.e.weapon.eligibility(attackers,targets,getattr(self.e,f'{side}_ammo'))
            if eligible.any():
                i,j=min(np.argwhere(eligible),key=lambda p:(dist[tuple(p)],int(p[0]),int(p[1])))
                self.funnel['first_fire_window']=team_geometry(self.e)
                self.first_pair_radius=float(np.hypot((attackers[i].x+targets[j].x)/2,(attackers[i].y+targets[j].y)/2))
        return self.original(attackers,targets,states,side)

def episode(task):
    version,mode,seed=task
    e=MultiUAVCombatEnv(CONFIGS[version]);obs,_=e.reset(seed);probe=Probe(e)
    rng=np.random.default_rng(seed+100_000_000);returns=np.zeros(5)
    while True:
        actions=(np.zeros((5,3)) if mode=='ZERO' else rng.uniform(-1,1,(5,3)) if mode=='RANDOM'
                 else e.fixed_policy.team_actions(e.red,e.blue))
        obs,reward,terminated,truncated,info=e.step(actions)
        verify_transition(e,obs,reward,info);returns+=reward
        if terminated or truncated:
            assert truncated==info['timeout']
            assert np.isclose(returns.sum(),sum(info[f'episode_{n}_total'] for n in ('event','outcome','adv','safe')),atol=2e-4)
            return jsonable(dict(info,version=version,mode=mode,seed=seed,episode_return=returns.sum(),
                mean_agent_episode_return=returns.mean(),funnel=probe.funnel,first_pair_radius=probe.first_pair_radius,
                **{f'{s}_first_detection_step':probe.first_detection[s] for s in ('red','blue')}))

def initialization_stats(config,count,seed_base):
    e=MultiUAVCombatEnv(config);data={s:{k:[] for k in ('speed','altitude','heading_error','lateral','minimum_spacing')} for s in ('red','blue')}
    ranks={s:np.zeros((5,5),dtype=int) for s in data}
    for seed in range(seed_base,seed_base+count):
        _,info=e.reset(seed);alpha=info['radial_angle'];axis=np.array([-np.sin(alpha),np.cos(alpha)])
        for side in data:
            states=getattr(e,side);lateral=np.array([[s.x,s.y] for s in states])@axis
            values=dict(speed=[s.v for s in states],altitude=[s.altitude for s in states],
                heading_error=[wrap_angle(s.psi-alpha-(np.pi if side=='blue' else 0)) for s in states],lateral=lateral,
                minimum_spacing=[np.diff(np.sort(lateral)).min()])
            for k,v in values.items():data[side][k].extend(v)
            rank=np.argsort(np.argsort(lateral));ranks[side][np.arange(5),rank]+=1
    result={s:{k:moments(v) for k,v in data[s].items()} for s in data}
    for side in data:
        result[side].update(rank_frequency_by_index=(ranks[side]/count).tolist(),
            leftmost_frequency=(ranks[side][:,0]/count).tolist(),rightmost_frequency=(ranks[side][:,4]/count).tolist())
    result['blue_minus_red']={k:{m:result['blue'][k][m]-result['red'][k][m] for m in ('mean','std','median')} for k in data['red']}
    return result

def group_summary(rows):
    result=summarize(rows);result['red_win_95_ci']=wilson(sum(r['red_win'] for r in rows),len(rows))
    result['median_episode_length']=float(np.median([r['episode_length'] for r in rows]))
    reasons=('red_win_elimination','blue_win_elimination','draw_mutual_destruction','red_win_timeout_survivors','blue_win_timeout_survivors','draw_timeout_equal_survivors')
    counts=Counter(r['termination_reason'] for r in rows);result['termination_reasons']={r:counts[r] for r in reasons}
    for side in ('red','blue'):
        result[f'{side}_kill_count_distribution']={k:sum(r[f'{side}_attack_kills']==k for r in rows) for k in range(6)}
        for event in ('detection','fire_window','attempt','hit','kill'):
            result[f'{side}_first_{event}']=moments([r[f'{side}_first_{event}_step'] for r in rows if r[f'{side}_first_{event}_step'] is not None])
        result[f'{side}_first_fire_to_end']=moments([r['episode_length']-r[f'{side}_first_fire_window_step'] for r in rows if r[f'{side}_first_fire_window_step'] is not None])
    result['first_fire_to_end']=moments([r['episode_length']-min(t for t in (r['red_first_fire_window_step'],r['blue_first_fire_window_step']) if t is not None) for r in rows if r['red_first_fire_window_step'] is not None or r['blue_first_fire_window_step'] is not None])
    result['center_funnel']={moment:{side:{key:moments([r['funnel'][moment][side][key] for r in rows if r['funnel'][moment] and r['funnel'][moment][side][key] is not None]) for key in ('centroid_radius','spread')} for side in ('red','blue')} for moment in ('reset','first_detection','first_fire_window')}
    radii=[r['first_pair_radius'] for r in rows if r['first_pair_radius'] is not None]
    result['first_pair_midpoint_radius']=moments(radii)
    # Descriptive spatial bands only, not acceptance criteria.
    result['first_pair_radius_fractions']={str(radius):float(np.mean(np.array(radii)<=radius)) if radii else None for radius in (500,1000,2000)}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=1000);p.add_argument('--resets',type=int,default=10000)
    p.add_argument('--seed-base',type=int,default=32000000);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--output',required=True);a=p.parse_args()
    if min(a.episodes,a.resets,a.workers)<=0:raise ValueError('counts must be positive')
    from algorithm.rmappo.protocol import require_cuda
    require_cuda('cuda')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    configs={v:load_config(ROOT/f'configs/combat_environment_v{v.replace(".","")}.yaml') for v in ('3.0','3.1')}
    init=initialization_stats(configs['3.1'],a.resets,a.seed_base)
    (output/'initialization.json').write_text(json.dumps(init,indent=2))
    tasks=[(v,m,a.seed_base+i) for i in range(a.episodes) for v in configs for m in ('ZERO','RANDOM','MIRROR')]
    groups={f'{v}_{m}':[] for v in configs for m in ('ZERO','RANDOM','MIRROR')};start=time.monotonic()
    with gzip.open(output/'episodes.jsonl.gz','wt') as raw,mp.get_context('spawn').Pool(a.workers,initializer=initialize,initargs=(configs,)) as pool:
        for n,row in enumerate(pool.imap_unordered(episode,tasks,chunksize=1),1):
            raw.write(json.dumps(row)+'\n');groups[f'{row["version"]}_{row["mode"]}'].append(row)
            if n%50==0:
                print(f'[AUDIT] {n}/{len(tasks)} elapsed={time.monotonic()-start:.1f}s',flush=True)
                (output/'progress.json').write_text(json.dumps(dict(completed=n,total=len(tasks),elapsed=time.monotonic()-start)))
    summaries={k:group_summary(sorted(v,key=lambda r:r['seed'])) for k,v in groups.items()}
    delta={}
    for mode in ('ZERO','RANDOM','MIRROR'):
        old,new=summaries[f'3.0_{mode}'],summaries[f'3.1_{mode}']
        d={k:new[k]-old[k] for k in ('red_win_rate','blue_win_rate','timeout_rate','average_episode_length','median_episode_length','average_red_loss','average_blue_loss')}
        d['signed_red_blue_win_gap']=(new['red_win_rate']-new['blue_win_rate'])-(old['red_win_rate']-old['blue_win_rate'])
        for side in ('red','blue'):
            for event in ('detection','fire_window','attempt','hit','kill'):
                key=f'{side}_first_{event}';d[key]=new[key]['mean']-old[key]['mean'] if new[key]['mean'] is not None and old[key]['mean'] is not None else None
        d['first_fire_to_end']=new['first_fire_to_end']['mean']-old['first_fire_to_end']['mean'];delta[mode]=d
    result=dict(initialization=init,groups=summaries,delta_v31_minus_v30=delta,seed_base=a.seed_base,
        episodes_per_group=a.episodes,reset_count=a.resets,spread_definition='mean horizontal distance of living aircraft from own team centroid',
        first_pair_definition='nearest eligible pair midpoint, Red tie priority; observed before combat resolution',
        first_fire_definition='first legal weapon window; missing events excluded from timing means',
        center_moments_definition='both teams at first detection/window by either side',
        formal_training_started=False,parameter_sweep=False,elapsed_seconds=time.monotonic()-start)
    (output/'summary.json').write_text(json.dumps(result,indent=2));print('[COMPLETE]',flush=True)

if __name__=='__main__':main()
