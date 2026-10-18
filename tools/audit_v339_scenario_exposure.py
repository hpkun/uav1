"""CPU scenario/R4 experiment. Formal files and old traces are immutable.

Only instance-level nonlethal resolution and reset-time Red heading are intervened.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import AbstractContextManager
from copy import deepcopy
import argparse,hashlib,json,math,os,subprocess,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.audit_v339_attack_geometry import (config,environment,states,red_actions,PreFireObserver,vector_geometry,runs,write_csv,POLICIES)
from env.math_utils import wrap_angle
from env.models import AircraftState
from env.v337_reward import tactical_pair_rewards_v337

OLD=ROOT/'outputs/v339_attack_geometry_audit'
LETHAL='formal_lethal';NONLETHAL='nonlethal_counterfactual'

class NonlethalSettlement(AbstractContextManager):
    """Do not run the formal kill resolver; leave hits/ammo/cooldown/RNG untouched.

    Returning empty kill dictionaries also correctly suppresses kill/loss rewards.
    Noncombat resolution is never wrapped or disabled. No aircraft is resurrected.
    """
    def __init__(self,e,enabled=True):self.e=e;self.enabled=enabled;self.suppressed=[]
    def __enter__(self):
        e=self.e;self.present='_resolve_combat' in e.__dict__;self.previous=e.__dict__.get('_resolve_combat')
        if self.enabled:
            def resolve(red,blue):
                mask=np.zeros((2,4),bool)
                for side,attempts in enumerate((red,blue)):
                    targets=e.blue if side==0 else e.red
                    for _,j,hit in attempts:
                        if hit and targets[j].alive:mask[side,j]=True
                self.suppressed.append(mask)
                return {},{}
            e._resolve_combat=resolve
        return self
    def __exit__(self,*args):
        if self.enabled:
            if self.present:self.e._resolve_combat=self.previous
            else:del self.e._resolve_combat

def setup(seed=0,offset=0.,scenario='default'):
    e=environment(seed);before=states(e)
    if scenario!='default':
        count=1 if '1v1' in scenario else 4;distance=float(scenario.rsplit('_',1)[1])
        for side,label in enumerate(('red','blue')):
            team=[]
            for i in range(4):
                y=0. if count==1 else (-450.,-150.,150.,450.)[i]
                team.append(AircraftState((-1 if side==0 else 1)*distance/2,y,-3000.,225.,0.,0.,i<count))
            setattr(e,label,team)
    for own in e.red:own.psi=float(wrap_angle(own.psi+math.radians(offset)))
    obs=e._observations() # rebuild after authorized initial-state intervention
    assert obs.shape==(4,45) and np.isfinite(obs).all()
    return e,obs,before

def ident(spec):return f'{spec["scenario_type"]}_{spec["red_policy"]}_{spec["lethality_mode"]}_h{spec["initial_heading_offset_deg"]:+g}_s{spec["seed"]}'

def make_specs(smoke=False):
    specs=[]
    def add(seed,policy,mode,offset=0.,scenario='default'):
        specs.append(dict(seed=seed,scenario_type=scenario,red_policy=policy,lethality_mode=mode,initial_heading_offset_deg=offset))
    for seed in range(3 if smoke else 50):
        for policy in POLICIES:
            for mode in (LETHAL,NONLETHAL):add(seed,policy,mode)
    for offset in (45.,-45.,90.,-90.):
        for seed in range(3 if smoke else 20):
            for policy in ('zero','pursuit'):
                for mode in (LETHAL,NONLETHAL):add(seed,policy,mode,offset)
    for scenario in ('tail_1v1_1500','tail_1v1_2500','tail_4v4_1500','tail_4v4_2500'):
        for seed in range(1 if smoke else 3):
            for mode in (LETHAL,NONLETHAL):add(seed,'pursuit',mode,scenario=scenario)
    return specs

def sample_first(mask,dt=.1):
    where=np.flatnonzero(np.asarray(mask).reshape(len(mask),-1).any(axis=1))
    return float((where[0]+1)*dt) if len(where) else None

def pair_conditions(g):
    live=g['live'];tail=live&(abs(g['aa'])<=math.pi/6+1e-12)
    near=tail&(g['distance']<=3000);ata=near&(abs(g['ata'])<=math.pi/6+1e-12)
    full=ata&(abs(g['ha'])<=math.pi/6+1e-12)
    return dict(live=live,tail_aspect=tail,tail_distance=near,tail_distance_ata=ata,
        r4_geometry=full,r4_nonzero_pair=full&(g['distance']<3000),weapon_geometry=live&(g['distance']<=1000)&(abs(g['ata'])<=math.pi/6+1e-12)&(abs(g['ha'])<=math.pi/6+1e-12))

def analyze(data,spec,source,terminal_reason=None):
    dt=.1;n=len(data['pre']);context=dict(episode_id=ident(spec),**spec)
    pre,start,post=data['pre'],data['start'],data['post'];initial=start[0]
    gs=[vector_geometry(pre,k) for k in (0,1)];conditions=[pair_conditions(g) for g in gs]
    row=dict(context,source=source,episode_duration_s=n*dt,steps=n,terminated=bool(data['done'][-1,0]),truncated=bool(data['done'][-1,1]),termination_reason=terminal_reason,
        first_fire_time_s=sample_first(data['attempt_targets']>=0),first_hit_time_s=sample_first(data['hit']),
        first_kill_time_s=sample_first(data['credited_kills']),first_any_death_time_s=sample_first(data['death_cause']>0),
        first_suppressed_kill_time_s=sample_first(data['suppressed_kills']) if 'suppressed_kills' in data else None)
    assert n<=1000 and np.isfinite(pre).all() and np.isfinite(post).all()
    assert data['done'][-1].any() and not data['done'][:-1].any()
    np.testing.assert_array_equal(pre[...,:6],post[...,:6]);np.testing.assert_array_equal(start[1:],post[:-1])
    np.testing.assert_array_equal(data['ammo_pre']-data['ammo_post'],(data['attempt_targets']>=0).astype(int))
    np.testing.assert_array_equal(data['cooldown_post'],np.where(data['attempt_targets']>=0,10,np.maximum(data['cooldown_pre']-1,0)))
    geom_rows=[];events=[];windows=[];serial={}
    # Reconstruct actual Red offense/defense from stored authoritative pre-fire geometry.
    pair_values=[]
    for side,g in enumerate(gs):
        c=conditions[side];off=np.zeros_like(g['distance']);threat=off.copy()
        for deg,positive,negative in ((30,.01,-.015),(15,.02,-.025),(5,.1,-.15)):
            angle=c['live']&(abs(g['ata'])<=math.radians(deg)+1e-12)&(abs(g['ha'])<=math.radians(deg)+1e-12)
            off=np.where(angle,positive,off);threat=np.where(angle,negative,threat)
        weight=np.clip((3000-g['distance'])/2000,0,1)
        pair_values.append((np.where(c['tail_distance'],off*weight,0),np.where(c['tail_distance'],threat*weight,0)))
    expected_off=pair_values[0][0].max(axis=2);expected_def=pair_values[1][1].min(axis=1)
    err=max(float(abs(expected_off-data['dense'][:,1]).max()),float(abs(expected_def-data['dense'][:,2]).max()))
    assert err<1e-12;row['r4_reconstruction_max_error']=err
    positive=data['dense'][:,1]>0;negative=data['dense'][:,2]<0
    red_den=(conditions[0]['live'].any(axis=2)).sum()
    row.update(first_positive_r4_time_s=sample_first(positive),first_negative_r4_time_s=sample_first(negative),
        r4_positive_agent_steps=int(positive.sum()),r4_negative_agent_steps=int(negative.sum()),
        r4_positive_agent_fraction=float(positive.sum()/red_den) if red_den else None,
        r4_negative_agent_fraction=float(negative.sum()/red_den) if red_den else None,
        r4_positive_total=float(data['dense'][:,1].sum()),r4_negative_total=float(data['dense'][:,2].sum()),
        r4_total=float(data['dense'][:,1:].sum()),r4_positive_episode_time_s=float(positive.any(axis=1).sum()*dt),
        r4_negative_episode_time_s=float(negative.any(axis=1).sum()*dt))
    for side,label in enumerate(('red','blue')):
        g=gs[side];c=conditions[side];live=c['live'];den=int(live.sum());agent_den=int(live.any(axis=2).sum())
        initial_c=pair_conditions(vector_geometry(initial[None],side));counts=dict(context,side=label,live_pair_steps=den,live_agent_steps=agent_den)
        for name,mask in c.items():
            counts[name+'_pair_steps']=int(mask.sum());counts[name+'_pair_fraction']=float(mask.sum()/den) if den else None
            counts[name+'_agent_steps']=int(mask.any(axis=2).sum());counts[name+'_agent_fraction']=float(mask.any(axis=2).sum()/agent_den) if agent_den else None
            first=0. if initial_c[name].any() else sample_first(mask)
            row[f'first_{name}_{label}_time_s']=first;row[f'{label}_{name}_pair_steps']=int(mask.sum())
            counts['initial_'+name+'_pairs']=int(initial_c[name].sum())
            if name in ('tail_aspect','r4_nonzero_pair'):
                for i in range(4):
                    for j in range(4):
                        for a,b in runs(mask[:,i,j]):
                            windows.append(dict(context,side=label,condition=name,attacker=i,target=j,
                                first_sample_time_s=(a+1)*dt,last_sample_time_s=b*dt,duration_s=(b-a)*dt,
                                distance_mean_m=float(g['distance'][a:b,i,j].mean()),aa_abs_max_deg=float(np.degrees(abs(g['aa'][a:b,i,j])).max())))
        geom_rows.append(counts)
        cause=data['death_cause'][:,side];alive=start[:,side,:,6]>0;alive_n=int(alive.sum())
        heading=np.arctan2(np.sin(post[:,side,:,5]-start[:,side,:,5]),np.cos(post[:,side,:,5]-start[:,side,:,5]))
        row.update({label+'_initial_alive':int(initial[side,:,6].sum()),label+'_survivors':int(post[-1,side,:,6].sum()),
            label+'_shots':int((data['attempt_targets'][:,side]>=0).sum()),label+'_hits':int(data['hit'][:,side].sum()),
            label+'_kills':int(data['credited_kills'][:,side].any(axis=1).sum()),
            label+'_boundary_losses':int(((cause==2)|(cause==3)).sum()),label+'_ground_losses':int((cause==1).sum()),label+'_ceiling_losses':int((cause==3).sum()),
            label+'_low_speed_fraction':float(((start[:,side,:,3]<=160)&alive).sum()/alive_n),
            label+'_absolute_turns_sum':float((abs(heading)*alive).sum()/(2*math.pi)),
            label+'_max_agent_absolute_turns':float((abs(heading)*alive).sum(axis=0).max()/(2*math.pi)),
            label+'_heading_saturation_fraction':float(((abs(data['actions'][:,side,:,0])>=1-1e-7)&alive).sum()/alive_n),
            label+'_pitch_saturation_fraction':float(((abs(data['actions'][:,side,:,1])>=1-1e-7)&alive).sum()/alive_n),
            label+'_first_death_time_s':sample_first(cause>0),label+'_all_dead_time_s':sample_first(post[:,side,:,6].sum(axis=1)==0)})
        targets=data['red_targets'] if side==0 else data['blue_targets']
        valid=(targets[1:]>=0)&(targets[:-1]>=0)&alive[1:]&alive[:-1]
        row[label+'_target_switches']=int(((targets[1:]!=targets[:-1])&valid).sum())
        assert int(initial[side,:,6].sum())-int(post[-1,side,:,6].sum())==int((cause>0).sum())
        row[f'first_range_3km_{label}_time_s']=0. if (initial_c['live']&(vector_geometry(initial[None],side)['distance']<=3000)).any() else sample_first(live&(g['distance']<=3000))
        row[f'first_range_1km_{label}_time_s']=0. if (initial_c['live']&(vector_geometry(initial[None],side)['distance']<=1000)).any() else sample_first(live&(g['distance']<=1000))
        if live.any():
            flat=np.where(live,abs(g['aa']),np.inf);t,i,j=np.unravel_index(np.argmin(flat),flat.shape)
            row.update({label+'_closest_tail_abs_aa_deg':float(np.degrees(abs(g['aa'][t,i,j]))),label+'_closest_tail_time_s':float((t+1)*dt),
                label+'_closest_tail_attacker':int(i),label+'_closest_tail_target':int(j),label+'_closest_tail_distance_m':float(g['distance'][t,i,j]),
                label+'_closest_tail_heading_difference_deg':float(np.degrees(wrap_angle(pre[t,1-side,j,5]-pre[t,side,i,5]))),
                label+'_closest_tail_own_speed_m_s':float(pre[t,side,i,3]),label+'_closest_tail_target_speed_m_s':float(pre[t,1-side,j,3])})
        for name in ('distance','ata','ha','aa','live'):
            serial[label+'_'+name]=g[name]
        serial[label+'_tail_aspect']=c['tail_aspect'];serial[label+'_r4_nonzero_pair']=c['r4_nonzero_pair']
    row['first_tail_aspect_time_s']=min((row[f'first_tail_aspect_{s}_time_s'] for s in ('red','blue') if row[f'first_tail_aspect_{s}_time_s'] is not None),default=None)
    row['tail_pair_steps']=row['red_tail_aspect_pair_steps']+row['blue_tail_aspect_pair_steps']
    row['boundary_losses']=row['red_boundary_losses']+row['blue_boundary_losses'];row['ground_losses']=row['red_ground_losses']+row['blue_ground_losses'];row['ceiling_losses']=row['red_ceiling_losses']+row['blue_ceiling_losses']
    row['target_switches']=row['red_target_switches']+row['blue_target_switches']
    for name,mask in [('positive_r4',positive),('negative_r4',negative)]:
        for i in range(4):
            for a,b in runs(mask[:,i]):windows.append(dict(context,side='red_reward_recipient',condition=name,attacker=i,target=None,first_sample_time_s=(a+1)*dt,last_sample_time_s=b*dt,duration_s=(b-a)*dt))
    # First events retain causal phase and pair identity for geometry/reward triggers.
    for name,value in row.items():
        if name.startswith('first_') and name.endswith('_time_s') and value is not None:
            event=dict(context,event=name,time_s=value,phase='initial_geometry' if value==0 else 'post_move_pre_fire' if any(x in name for x in ('tail','r4','range','geometry')) else 'weapon_or_post_step')
            if name in ('first_positive_r4_time_s','first_negative_r4_time_s'):
                k=round(value/dt)-1;side=0 if 'positive' in name else 1;matches=np.argwhere(conditions[side]['r4_nonzero_pair'][k]);i,j=matches[0]
                event.update(side=('red','blue')[side],attacker=int(i),target=int(j),distance_m=float(gs[side]['distance'][k,i,j]),
                    ata_deg=float(np.degrees(gs[side]['ata'][k,i,j])),ha_deg=float(np.degrees(gs[side]['ha'][k,i,j])),aa_deg=float(np.degrees(gs[side]['aa'][k,i,j])))
            events.append(event)
    if spec['lethality_mode']==NONLETHAL:
        assert not data['credited_kills'].any() and not (data['death_cause']==4).any()
    return row,events,geom_rows,windows,serial

def run_job(payload):
    spec,out,old=payload;out=Path(out);old=Path(old);key=ident(spec)
    reuse=spec['scenario_type']=='default' and spec['initial_heading_offset_deg']==0 and spec['lethality_mode']==LETHAL
    e,obs,original=setup(spec['seed'],spec['initial_heading_offset_deg'],spec['scenario_type'])
    if reuse:
        path=old/'traces'/f'{spec["red_policy"]}_seed{spec["seed"]}.npz'
        with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        np.testing.assert_array_equal(data['start'][0],states(e));source=str(path)
        red_alive,blue_alive=data['post'][-1,:, :,6].sum(axis=1)
        reason='draw_mutual_destruction' if red_alive==blue_alive==0 else 'red_win_elimination' if blue_alive==0 else 'blue_win_elimination' if red_alive==0 else 'timeout'
    else:
        with NonlethalSettlement(e,spec['lethality_mode']==NONLETHAL) as intervention:
            with PreFireObserver(e) as observer:
                while True:
                    a,t=red_actions(e,spec['red_policy']);before=states(e);result=e.step(a);observer.finish(before,a,t,result)
                    observer.current['rng_state_sha256']=hashlib.sha256(json.dumps(e.rng.bit_generator.state,sort_keys=True).encode()).hexdigest()
                    if result[2] or result[3]:break
                assert observer.calls==dict(dense=e.steps,resolve=e.steps);data=observer.arrays()
            data['suppressed_kills']=np.asarray(intervention.suppressed) if intervention.enabled else np.zeros((e.steps,2,4),bool)
        data['initial_observation']=obs;source='new_actual_step_rollout';reason=result[4]['termination_reason']
    data['initial_observation']=obs
    row,events,geoms,windows,serial=analyze(data,spec,source,reason)
    data.update(serial);data['original_reset_states']=original
    path=out/'trajectory_data'/f'{key}.npz';np.savez_compressed(path,**data)
    row['trajectory_file']=str(path);row['trajectory_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return row,events,geoms,windows

def validate_reuse(old=OLD):
    s=json.loads((old/'summary.json').read_text(encoding='utf-8'));assert s['completed_episodes']==200 and s['config']==config()
    for f,h in s['audit_sources_sha256'].items():assert hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h,f
    return dict(previous_config_verified=True,previous_episodes=200,source_hashes_verified=True,old_summary_sha256=hashlib.sha256((old/'summary.json').read_bytes()).hexdigest())

def paired_until_fork(seed,policy,offset=0.):
    a,_,_=setup(seed,offset);b,_,_=setup(seed,offset);steps=0
    with NonlethalSettlement(b):
        with PreFireObserver(a) as oa,PreFireObserver(b) as ob:
            while True:
                aa,targets=red_actions(a,policy);bb,tb=red_actions(b,policy);np.testing.assert_array_equal(aa,bb);np.testing.assert_array_equal(targets,tb)
                before_a,before_b=states(a),states(b);np.testing.assert_array_equal(before_a,before_b)
                x,y=a.step(aa),b.step(bb);oa.finish(before_a,aa,targets,x);ob.finish(before_b,bb,tb,y);steps+=1
                for field in ('pre','actions','ammo_pre','ammo_post','cooldown_pre','cooldown_post','blue_targets','attempt_targets','hit','dense'):
                    np.testing.assert_array_equal(oa.current[field],ob.current[field])
                assert a.rng.bit_generator.state==b.rng.bit_generator.state
                if oa.current['credited_kills'].any():
                    np.testing.assert_array_equal(states(a)[...,:6],states(b)[...,:6])
                    assert not np.array_equal(states(a)[...,6],states(b)[...,6])
                    return dict(seed=seed,red_policy=policy,initial_heading_offset_deg=offset,matched_steps_including_prefire_fork=steps,
                        fork_time_s=steps*.1,fork_reason='first_successful_weapon_kill_changes_alive_and_event_reward_only_at_this_step',rng_equal_at_fork=True)
                np.testing.assert_array_equal(states(a),states(b));np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1]);assert x[2:4]==y[2:4]
                assert a.combat_counts==b.combat_counts
                if x[2] or x[3]:return dict(seed=seed,red_policy=policy,initial_heading_offset_deg=offset,matched_steps_including_prefire_fork=steps,fork_time_s=None,fork_reason='no_weapon_kill_before_episode_end',rng_equal_at_fork=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/v339_scenario_exposure_audit');p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke',action='store_true');p.add_argument('--paired-validation',action='store_true')
    args=p.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError('Use a fresh output directory')
    os.environ['CUDA_VISIBLE_DEVICES']='-1';out.mkdir(parents=True);(out/'trajectory_data').mkdir()
    summary=dict(environment_version='3.39',device='CPU',config=config(),smoke=args.smoke,baseline_reuse=validate_reuse(),started=time.time(),completed=False)
    specs=make_specs(args.smoke);summary['planned_unique_episodes']=len(specs);rows=[];events=[];geoms=[];windows=[]
    if args.paired_validation:
        validations=[]
        cases=[(seed,policy,0.) for seed in range(3) for policy in POLICIES]+[(0,policy,h) for h in (45.,-45.,90.,-90.) for policy in ('zero','pursuit')]
        for seed,policy,h in cases:
            validations.append(paired_until_fork(seed,policy,h));print(f'paired validation {len(validations)}/{len(cases)}',flush=True)
        (out/'paired_validation.json').write_text(json.dumps(validations,indent=2),encoding='utf-8');summary['paired_validation_cases']=len(validations)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(run_job,(spec,str(out),str(OLD))):spec for spec in specs}
        for future in as_completed(pending):
            row,ev,g,w=future.result();rows.append(row);events.extend(ev);geoms.extend(g);windows.extend(w)
            if len(rows)%20==0 or len(rows)==1:
                print(f'{len(rows)}/{len(specs)} episodes; {time.time()-summary["started"]:.1f}s; {row["episode_id"]}',flush=True)
                summary['completed_unique_episodes']=len(rows);(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    rows.sort(key=lambda r:r['episode_id']);write_csv(out/'episode_summary.csv',rows);write_csv(out/'exposure_events.csv',events);write_csv(out/'geometry_exposure.csv',geoms);write_csv(out/'exposure_windows.csv',windows)
    summary.update(completed=True,completed_unique_episodes=len(rows),reused_formal_episodes=sum(r['source']!='new_actual_step_rollout' for r in rows),
        new_rollout_episodes=sum(r['source']=='new_actual_step_rollout' for r in rows),steps=sum(r['steps'] for r in rows),elapsed_s=time.time()-summary['started'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k not in ('config','baseline_reuse')},indent=2),flush=True)

if __name__=='__main__':main()
