"""v3.40/v3.41 lethal comparison reusing the verified v3.40 observer/protocol."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from tools.preflight_v340 import episode, first
from tools.audit_v339_attack_geometry import POLICIES, vector_geometry, write_csv

DEFAULT=ROOT/'outputs/v341_hit_calibration_preflight'
SHARED={'env/config.py','env/combat_env.py','algorithm/mappo/runner.py',
        'algorithm/common/evaluator.py','algorithm/common/checkpoint.py'}


def read_csv(path):
    text_fields={'episode_id','version','policy','side','termination_reason','trace','trace_sha256'}
    rows=[]
    with path.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            for key,value in row.items():
                if key in text_fields:continue
                if value=='':row[key]=None
                elif value in ('True','False'):row[key]=value=='True'
                else:
                    try:row[key]=int(value)
                    except ValueError:
                        try:row[key]=float(value)
                        except ValueError:pass
            rows.append(row)
    return rows


def reuse_v340(out):
    old=ROOT/'outputs/v340_preflight';summary=json.loads((old/'summary.json').read_text())
    assert summary['seeds']==list(range(20)) and summary['lethal'] and summary['device']=='cpu'
    checked=0
    for name,expected in summary['source_sha256'].items():
        name=name.replace('\\','/')
        source=subprocess.check_output(['git','show','HEAD:'+name]) if name in SHARED else (ROOT/name).read_bytes()
        variants=(source,source.replace(b'\r\n',b'\n'),source.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
        assert any(hashlib.sha256(s).hexdigest()==expected for s in variants),('old source fingerprint',name)
        checked+=1
    rows=[r for r in read_csv(old/'episodes.csv') if r['version']=='3.40']
    assert {(r['policy'],r['seed']) for r in rows}=={(p,s) for p in POLICIES for s in range(20)}
    ids={r['episode_id'] for r in rows}
    for row in rows:
        path=ROOT/row['trace'].replace('\\','/')
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['trace_sha256']
        with np.load(path) as data:
            e=MultiUAVCombatEnv(load_config(ROOT/'configs/combat_environment_v340.yaml'));e.reset(row['seed'])
            from tools.audit_v339_attack_geometry import states
            np.testing.assert_array_equal(data['start'][0],states(e))
            assert len(data['post'])==round(row['duration_s']/.1) and data['done'][-1].any()
        target=out/'trajectory_data'/path.name;shutil.copyfile(path,target)
        row.update(trace=target.relative_to(ROOT).as_posix(),source='verified_v340_reuse')
    planes=[r for r in read_csv(old/'aircraft.csv') if r['episode_id'] in ids]
    shots=[r for r in read_csv(old/'shots.csv') if r['episode_id'] in ids]
    proof=dict(source_summary_sha256=hashlib.sha256((old/'summary.json').read_bytes()).hexdigest(),
               source_files_verified=checked,traces_verified=len(rows),reset_states_verified=len(rows),
               shared_version_registration_sources='original HEAD matches saved fingerprints allowing LF/CRLF checkout conversion only',
               protocol='same red_actions, PreFireObserver, 20 reset seeds, formal lethal mechanics')
    (out/'reuse_validation.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
    return rows,planes,shots


def enrich(row,planes,shots,out):
    with np.load(ROOT/row['trace'].replace('\\','/')) as data:
        gs=[vector_geometry(data['pre'],s) for s in (0,1)]
        row['first_2km_entry_s']=first(gs[0]['live']&(gs[0]['distance']<=2000))
        row['first_1km_entry_s']=first(gs[0]['live']&(gs[0]['distance']<=1000))
        fire=data['attempt_targets']>=0
        initial=int(np.flatnonzero(fire.any(axis=(1,2)))[0])
        volley=slice(initial,min(initial+10,len(fire)))
        losses=data['death_cause']==4
        for side,label in enumerate(('red','blue')):
            row[label+'_first_tick_shots']=int(fire[initial,side].sum())
            row[label+'_first_tick_hits']=int(data['hit'][initial,side].sum())
            row[label+'_first_tick_kills']=int(losses[initial,1-side].sum())
            row[label+'_initial_volley_shots']=int(fire[volley,side].sum())
            row[label+'_initial_volley_hits']=int(data['hit'][volley,side].sum())
            row[label+'_initial_volley_kills']=int(losses[volley,1-side].sum())
            row[label+'_exhausted_aircraft']=int((data['ammo_post'][:,side]==0).any(0).sum())
            row[label+'_alive_armed_final']=int(((data['post'][-1,side,:,6]>0)&(data['ammo_post'][-1,side]>0)).sum())
            row[label+'_ammo_on_survivors']=int((data['ammo_post'][-1,side]*(data['post'][-1,side,:,6]>0)).sum())
            entered=np.maximum.accumulate((gs[side]['live']&(gs[side]['distance']<=1000)).any(2),axis=0)
            # Each victim counted once even if several credited shooters hit it.
            pre_entry_credited=data['credited_kills'][:,side]&~entered[:,:,None]
            row[label+'_kills_by_shooters_before_own_first_1km_entry']=int(pre_entry_credited.any(1).sum())
            for plane in planes:
                if plane['episode_id']==row['episode_id'] and plane['side']==label:
                    i=plane['aircraft']
                    plane['exhausted_anytime']=bool((data['ammo_post'][:,side,i]==0).any())
                    plane['alive_and_armed_final']=bool(data['post'][-1,side,i,6] and data['ammo_post'][-1,side,i]>0)
            for band in ('near','far'):
                subset=[s for s in shots if s['episode_id']==row['episode_id'] and s['side']==label and
                        ((s['distance_m']<=1000) if band=='near' else (1000<s['distance_m']<=2000))]
                row[label+'_'+band+'_shots']=len(subset)
                row[label+'_'+band+'_hits']=sum(s['hit'] for s in subset)
            live=gs[side]['live'];tail=live&(abs(gs[side]['aa'])<=np.pi/6+1e-12)
            full=tail&(gs[side]['distance']<4000)&(abs(gs[side]['ata'])<=np.pi/6+1e-12)&(abs(gs[side]['ha'])<=np.pi/6+1e-12)
            row[label+'_tail_aa_agent_steps_after_first_kill']=int(tail[round(row['first_kill_s']/.1):].any(2).sum()) if row['first_kill_s'] is not None else 0
            row[label+'_full_r4_geometry_agent_steps_after_first_kill']=int(full[round(row['first_kill_s']/.1):].any(2).sum()) if row['first_kill_s'] is not None else 0
        # Episode-global pre-entry casualties use the same <=1km criterion for all teams.
        before_global=np.arange(len(fire)) < (round(row['first_1km_entry_s']/.1)-1 if row['first_1km_entry_s'] is not None else len(fire))
        row['weapon_deaths_before_any_1km_entry']=int(losses[before_global].sum())
        row['shots_before_any_1km_entry']=int(fire[before_global].sum())
        row['seconds_after_first_kill']=row['duration_s']-row['first_kill_s'] if row['first_kill_s'] is not None else None
        # Validate all saved real mechanisms without changing the observer or transition.
        assert hashlib.sha256((ROOT/row['trace']).read_bytes()).hexdigest()==row['trace_sha256']
        np.testing.assert_array_equal(data['start'][1:],data['post'][:-1])
        np.testing.assert_array_equal(data['ammo_pre']-data['ammo_post'],fire.astype(int))
        assert np.isfinite(data['post']).all() and np.isfinite(data['dense']).all()
        for side,g in enumerate(gs):
            for t,i in zip(*np.where(fire[:,side])):
                j=data['attempt_targets'][t,side,i]
                assert g['live'][t,i,j] and g['distance'][t,i,j]<=2000
                assert abs(g['ata'][t,i,j])<=np.pi/6+1e-12 and abs(g['ha'][t,i,j])<=np.pi/6+1e-12
                assert data['ammo_pre'][t,side,i]>0 and data['cooldown_pre'][t,side,i]<=1


def summarize(rows):
    summaries=[]
    for version in ('3.40','3.41'):
        for policy in POLICIES:
            group=[r for r in rows if r['version']==version and r['policy']==policy]
            s=dict(version=version,policy=policy,episodes=len(group))
            for key in ('duration_s','first_2km_entry_s','first_fire_s','first_hit_s','first_kill_s',
                        'first_positive_r4_s','first_armed_positive_r4_s','seconds_after_first_kill'):
                values=[r[key] for r in group if r[key] is not None]
                s[key+'_mean']=float(np.mean(values)) if values else None
                s[key+'_median']=float(np.median(values)) if values else None
                s[key+'_observed_n']=len(values)
            for key in ('red_win','blue_win','draw','timeout','r4_offense_episode','r4_defense_episode','armed_positive_r4_episode'):
                s[key+'_count']=sum(r[key] for r in group)
            numeric=[k for k,v in group[0].items() if isinstance(v,(int,float)) and not isinstance(v,bool) and k not in ('seed',)]
            for key in numeric:
                if all(r[key] is not None for r in group):
                    s.setdefault(key+'_mean',float(np.mean([r[key] for r in group])))
                    if 'first_' not in key or 'shots_before' in key:s[key+'_total']=sum(r[key] for r in group)
            for side in ('red','blue'):
                for band in ('','near_','far_'):
                    shots=sum(r[side+'_'+band+'shots'] for r in group)
                    hits=sum(r[side+'_'+band+'hits'] for r in group)
                    s[side+'_'+band+'hit_rate']=hits/shots if shots else None
            shots=sum(r['red_shots']+r['blue_shots'] for r in group)
            hits=sum(r['red_hits']+r['blue_hits'] for r in group)
            s['overall_hit_rate']=hits/shots if shots else None
            summaries.append(s)
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--rerun-v340',action='store_true',help='use if verified old data is unavailable')
    args=p.parse_args();out=args.output.resolve();(out/'trajectory_data').mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    if args.rerun_v340:rows,planes,shots=[],[],[]
    else:rows,planes,shots=reuse_v340(out)
    versions=('340','341') if args.rerun_v340 else ('341',)
    specs=[(v,p,s,str(out)) for v in versions for p in POLICIES for s in range(20)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(episode,*spec) for spec in specs]
        for i,f in enumerate(as_completed(jobs),1):
            r,a,b=f.result();r['trace']=r['trace'].replace('\\','/');r['source']='new_formal_lethal'
            rows.append(r);planes.extend(a);shots.extend(b)
            print(f'{i}/{len(specs)} {r["episode_id"]} {r["duration_s"]:.1f}s',flush=True)
    rows.sort(key=lambda r:r['episode_id'])
    lookup={r['episode_id']:r for r in rows}
    for s in shots:s.update(version=lookup[s['episode_id']]['version'],policy=lookup[s['episode_id']]['policy'],seed=lookup[s['episode_id']]['seed'])
    for r in rows:enrich(r,planes,shots,out)
    summaries=summarize(rows)
    write_csv(out/'episodes.csv',rows);write_csv(out/'aircraft.csv',planes);write_csv(out/'shots.csv',shots)
    write_csv(out/'comparison.csv',summaries)
    paired=[]
    for policy in POLICIES:
        for seed in range(20):
            a,b=[next(r for r in rows if r['version']==v and r['policy']==policy and r['seed']==seed) for v in ('3.40','3.41')]
            with np.load(ROOT/a['trace']) as x,np.load(ROOT/b['trace']) as y:np.testing.assert_array_equal(x['start'][0],y['start'][0])
            paired.append(dict(policy=policy,seed=seed,reset_states_exact=True,**{
                k+'_delta':b[k]-a[k] if a[k] is not None and b[k] is not None else None
                for k in ('duration_s','first_fire_s','first_kill_s','red_shots','blue_shots','red_kills','blue_kills','r4_offense_sum')}))
    write_csv(out/'paired_deltas.csv',paired)
    sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for folder in ('env','algorithm','configs','tools') for p in (ROOT/folder).rglob('*') if p.suffix in ('.py','.yaml')}
    result=dict(episodes=len(rows),new_episodes=len(specs),reused_episodes=len(rows)-len(specs),
        seeds=list(range(20)),policies=POLICIES,device='cpu',lethal=True,formal_3m_training_started=False,
        elapsed_s=time.perf_counter()-start,summaries=summaries,source_sha256=sources,
        definitions=dict(initial_volley='global first-fire sample through next 0.9s, 10 samples, excludes second shot at 1s cooldown',
            first_tick='only global first-fire sample, may include just one team',kills='unique victims, not summed multi-shooter credits',
            pre_own_1km='per shooter, never previously within <=1km of any living target; current sample included',
            pre_any_1km='before first pre-fire sample with any living Red/Blue pair <=1km; never-entry episodes remain censored',
            near='distance<=1000m',far='1000<distance<=2000m',r4='Red actual positive offense / negative defense separated',
            armed_r4='positive Red recipient has ammo_pre>0; no cooldown or weapon-geometry requirement',
            tail_after_kill='strictly later samples; AA-only separated from full nonzero-distance R4 geometry',
            trace='unchanged v340 NPZ: start/pre/post [step,side,aircraft,x/y/z/v/theta/psi/alive]; dense [step,R3/offense/defense,Red]; death 0none 1ground 2horizontal 3ceiling 4weapon'))
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('Finished',len(rows),'episodes',flush=True)


if __name__=='__main__':main()
