"""Offline speed diagnostics from canonical traces; no rollout or policy changes."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from env.geometry import engagement_geometry
from env.math_utils import angle_difference
from env.models import AircraftState
from env.v336_weapon import PaperATAHAWeapon
from tools.combat_trace import read_trace

ANALYSIS_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

SPEED_EDGES = [150,175,200,215,235,250,275,300.000001]
SPEED_LABELS = ['150-175','175-200','200-215','215-235','235-250','250-275','275-300']
DISTANCE_LABELS = ['<=500','500-1000','1000-2000','2000-3000','3000-4000','>4000']
HEADING_LABELS = ['<15','15-30','30-60','60-90','90-135','>135']
EVENTS = ['first_3km','first_1km','first_fire_window','fire','hit','kill',
          'threat_entry','pursued_heading_error_gt60','pursued_turn_2s_gt60']


def episode_samples(trace, metadata):
    """Pre-transition alive rows; effects/events refer to the ensuing transition.

    Nearest Blue is only an offline diagnostic, never a reconstructed policy target.
    Blue pursuit identities come exclusively from canonical info-derived indices.
    """
    red, blue = trace['red_kinematics'],trace['blue_kinematics']
    ra,ba = trace['red_alive'],trace['blue_alive']
    T,N = red.shape[0]-1,red.shape[1]
    weapon = PaperATAHAWeapon(**metadata['weapon_config'])
    labels = ['v','v_next','dv','a_v','a_yaw','a_pitch','x','y','z','theta','psi',
              'distance','ata','ha','aa','elevation_error','body_az','body_el','closing_speed',
              'next_distance_same_target','blue_speed','forward_projection_blue']
    values={name:np.full((T,N),np.nan) for name in labels}
    flags={name:np.zeros((T,N),dtype=bool) for name in ['fire_window','eligible_ammo',
        'threat_window','armed_threat','pursued_by_blue','fire','hit','kill','alive_after']}
    nearest=np.full((T,N),-1,dtype=int)
    pursued=trace['blue_pursuit_target_indices']
    assert metadata['blue_target_available'], 'authoritative Blue pursuit is required'
    for name,index in [('x',0),('y',1),('z',2),('v',3),('theta',4),('psi',5)]:values[name][:]=red[:-1,:,index]
    values['v_next'][:]=red[1:,:,3]; values['dv'][:]=np.diff(red[:,:,3],axis=0)
    for name,index in [('a_yaw',0),('a_pitch',1),('a_v',2)]:values[name][:]=trace['red_actions'][:,:,index]
    flags['alive_after'][:]=ra[1:]
    initial_ammo=metadata['initial_ammo']
    for t in range(T):
        rs=[AircraftState(*s,alive=bool(a)) for s,a in zip(red[t],ra[t])]
        bs=[AircraftState(*s,alive=bool(a)) for s,a in zip(blue[t],ba[t])]
        blue_indices=np.flatnonzero(ba[t])
        if not len(blue_indices):continue
        for i in np.flatnonzero(ra[t]):
            flags['pursued_by_blue'][t,i]=bool(np.any(pursued[t]==i))
            distances=np.linalg.norm(blue[t,blue_indices,:3]-red[t,i,:3],axis=1)
            j=int(blue_indices[np.argmin(distances)]); nearest[t,i]=j
            g=engagement_geometry(rs[i],bs[j])
            for name, val in [('distance',g.distance),('ata',g.ata),('ha',g.ha),('aa',g.aa),
                ('body_az',g.boresight_azimuth_error),('body_el',g.boresight_elevation_error),
                ('elevation_error',angle_difference(g.ha,rs[i].theta)),('blue_speed',bs[j].v)]:values[name][t,i]=val
            los=(blue[t,j,:3]-red[t,i,:3])/max(g.distance,1e-12)
            values['closing_speed'][t,i]=-float(np.dot(bs[j].velocity_vector()-rs[i].velocity_vector(),los))
            values['forward_projection_blue'][t,i]=float(np.dot(blue[t,j,:3]-red[t,i,:3],
                rs[i].velocity_vector()/max(rs[i].v,1e-12)))
            if ba[t+1,j]:values['next_distance_same_target'][t,i]=np.linalg.norm(blue[t+1,j,:3]-red[t+1,i,:3])
            red_ammo=initial_ammo['red'][i] if t==0 else trace['red_ammo'][t-1,i]
            for k,dist in zip(blue_indices,distances):
                if dist>weapon.range_max:continue
                # Delegate predicates to the existing authoritative weapon, not a second geometry implementation.
                red_window=weapon.qualifies(rs[i],bs[k],1)
                blue_window=weapon.qualifies(bs[k],rs[i],1)
                flags['fire_window'][t,i] |= red_window
                flags['eligible_ammo'][t,i] |= red_window and red_ammo>0
                flags['threat_window'][t,i] |= blue_window
                blue_ammo=initial_ammo['blue'][k] if t==0 else trace['blue_ammo'][t-1,k]
                flags['armed_threat'][t,i] |= blue_window and blue_ammo>0
    for event in metadata['events']:
        k=int(event['step'])-1
        if not 0<=k<T:continue
        if event['type']=='fire_attempt' and event['side']=='red':
            i=event['attacker']; flags['fire'][k,i]=True
            flags['hit'][k,i] |= bool(event['hit'])
        elif event['type']=='attack_kill' and event['side']=='red':
            for i in event['attackers']:flags['kill'][k,i]=True
    t_idx,agents=np.where(ra[:-1])
    frame=pd.DataFrame({'episode_seed':metadata['episode_seed'],'step':t_idx,
        'time_s':t_idx*metadata['dt'],'agent':agents,'nearest_blue_diagnostic':nearest[t_idx,agents],
        'alive':True,'outcome':'win' if metadata['red_win'] else 'loss' if metadata['blue_win'] else 'timeout' if 'timeout' in metadata['termination_reason'] else 'draw',
        'episode_return':metadata['episode_return']})
    for name,array in {**values,**flags}.items():frame[name]=array[t_idx,agents]
    frame['heading_deg']=np.rad2deg(np.abs(frame.ata))
    frame['pitch_deg']=np.rad2deg(frame.theta)
    # A closure/escape proxy, not proof of intent or counterfactual exploitation.
    frame['faster_escape_proxy']=(frame.v>240)&frame.pursued_by_blue&(frame.closing_speed<0)
    frame['tail_geometry']=(frame.distance<=3000)&(np.abs(frame.ata)<=np.pi/6+1e-12)&(np.abs(frame.ha)<=np.pi/6+1e-12)&(np.abs(frame.aa)<=np.pi/6+1e-12)
    return frame,values,flags


def event_indices(trace,metadata,values,flags):
    """Actual fire time is frame k; geometry entry events use observed frame time."""
    events=[]; alive=trace['red_alive']; T=len(trace['red_actions']);dt=float(trace['dt'])
    def add(name,i,k):events.append(dict(event=name,agent=int(i),frame=int(k),time_s=k*dt))
    for i in range(alive.shape[1]):
        for name,mask in [('first_3km',values['distance'][:,i]<=3000),('first_1km',values['distance'][:,i]<=1000),('first_fire_window',flags['fire_window'][:,i])]:
            hits=np.flatnonzero(mask & alive[:-1,i])
            if len(hits):add(name,i,hits[0])
        for name,mask in [('threat_entry',flags['threat_window'][:,i]),
             ('pursued_heading_error_gt60',flags['pursued_by_blue'][:,i]&(np.abs(values['ata'][:,i])>np.pi/3))]:
            mask=mask & alive[:-1,i]
            for k in np.flatnonzero(mask & ~np.r_[False,mask[:-1]]):add(name,i,k)
        width=int(round(2/dt));turn=np.zeros(T,dtype=bool)
        for k in range(width,T):
            turn[k]=(alive[k,i] and alive[k-width,i] and flags['pursued_by_blue'][k,i]
                     and abs(angle_difference(trace['red_kinematics'][k,i,5],trace['red_kinematics'][k-width,i,5]))>np.pi/3)
        for k in np.flatnonzero(turn & ~np.r_[False,turn[:-1]]):add('pursued_turn_2s_gt60',i,k)
    for e in metadata['events']:
        if e['side']!='red':continue
        if e['type']=='fire_attempt':
            add('fire',e['attacker'],e['step'])
            if e['hit']:add('hit',e['attacker'],e['step'])
        elif e['type']=='attack_kill':
            for i in e['attackers']:add('kill',i,e['step'])
    # A simultaneous kill can remove a target before the next stored alive frame.
    # Actual shot states still establish entry into <=1km / firing geometry.
    for i in range(alive.shape[1]):
        shots=[e for e in metadata['events'] if e['type']=='fire_attempt' and e['side']=='red' and e['attacker']==i]
        if not shots:continue
        k=min(int(e['step']) for e in shots)
        for name in ['first_3km','first_1km','first_fire_window']:
            previous=[e['frame'] for e in events if e['agent']==i and e['event']==name]
            if not previous or k<min(previous):
                events=[e for e in events if not (e['agent']==i and e['event']==name)]
                add(name,i,k)
    return events


def overshoot_proxies(trace):
    """Conservative longitudinal plane crossings; neither intent nor causality.

    Blue must pursue this same Red for the preceding 1s; both survive, Red heading
    changes <=15deg, range stays below 1km at crossing, Red slower than Blue.
    """
    r,b=trace['red_kinematics'],trace['blue_kinematics'];ra,ba=trace['red_alive'],trace['blue_alive']
    pursued=trace['blue_pursuit_target_indices'];dt=float(trace['dt']);width=int(round(1/dt))
    rows=[]
    for k in range(max(1,width),len(pursued)):
        for j,i in enumerate(pursued[k-1]):
            if i<0 or not (ra[k-width:k+1,i].all() and ba[k-width:k+1,j].all()):continue
            if not (pursued[k-width:k,j]==i).all() or r[k-1,i,3]>=b[k-1,j,3]:continue
            if abs(angle_difference(r[k,i,5],r[k-width,i,5]))>np.deg2rad(15):continue
            delta0=b[k-1,j,:3]-r[k-1,i,:3];delta1=b[k,j,:3]-r[k,i,:3]
            if max(np.linalg.norm(delta0),np.linalg.norm(delta1))>1000:continue
            f0=AircraftState(*r[k-1,i]).velocity_vector()/r[k-1,i,3]
            f1=AircraftState(*r[k,i]).velocity_vector()/r[k,i,3]
            if np.dot(delta0,f0)<0<=np.dot(delta1,f1):
                rows.append(dict(frame=k,time_s=k*dt,agent=int(i),blue_agent=j,
                    red_speed=float(r[k-1,i,3]),blue_speed=float(b[k-1,j,3]),distance=float(np.linalg.norm(delta1))))
    return rows


def describe(frame):
    if not len(frame):return {'count':0}
    v=frame.v.to_numpy();a=frame.a_v.to_numpy();dv=frame.dv.to_numpy()
    result=dict(count=len(frame),mean=float(v.mean()),median=float(np.median(v)),std=float(v.std()),
        min=float(v.min()),max=float(v.max()),action_mean=float(a.mean()),action_std=float(a.std()),
        dv_mean=float(dv.mean()),accelerate=float(np.mean(dv>1e-8)),decelerate=float(np.mean(dv< -1e-8)),hold=float(np.mean(np.abs(dv)<=1e-8)))
    result.update({f'p{q}':float(np.percentile(v,q)) for q in [5,10,25,50,75,90,95]})
    for key,mask in {'v_lt200':v<200,'v_lt210':v<210,'v_210_240':(v>=210)&(v<=240),
        'v_gt240':v>240,'v_gt250':v>250,'v_gt275':v>275,'v_near_min':v<=151,'v_near_max':v>=299,
        'av_abs_lt01':np.abs(a)<.1,'av_gt05':a>.5,'av_lt_neg05':a<-.5,'av_gt09':a>.9,'av_lt_neg09':a<-.9,
        'pitch_and_speed_negative_saturation':(frame.a_pitch.to_numpy()<-.9)&(a<-.9)}.items():result[key]=float(np.mean(mask))
    for lo,hi,label in zip(SPEED_EDGES[:-1],SPEED_EDGES[1:],SPEED_LABELS):result['bin_'+label]=float(np.mean((v>=lo)&(v<hi)))
    for flag in ['fire_window','threat_window','pursued_by_blue','faster_escape_proxy','tail_geometry']:
        result[flag+'_rate']=float(frame[flag].mean())
    result['cap_blocked_action_rate']=float(np.mean((np.abs(a)>.1)&(np.abs(dv)<1e-8)&((v<=150+1e-8)|(v>=300-1e-8))))
    result['closing_speed_mean']=float(frame.closing_speed.mean()) if frame.closing_speed.notna().any() else None
    return result


def grouped_statistics(frame):
    result=[]
    def add(group,condition,subset):result.append(dict(group=group,condition=str(condition),**describe(subset)))
    add('overall','all',frame)
    for i,f in frame.groupby('agent'):add('agent',f'R{i+1}',f)
    distance=pd.cut(frame.distance,[-np.inf,500,1000,2000,3000,4000,np.inf],labels=DISTANCE_LABELS)
    heading=pd.cut(frame.heading_deg,[-np.inf,15,30,60,90,135,np.inf],labels=HEADING_LABELS,right=False)
    for label in DISTANCE_LABELS:add('distance',label,frame[distance==label])
    for label in HEADING_LABELS:add('heading',label,frame[heading==label])
    # 3x3: near <1km; medium 1..3km; far >3km. Angular medium includes 60deg.
    dc=np.where(frame.distance<1000,'near',np.where(frame.distance<=3000,'medium','far'))
    hc=np.where(frame.heading_deg<30,'small',np.where(frame.heading_deg<=60,'medium','large'))
    for d in ['near','medium','far']:
        for h in ['small','medium','large']:
            mask=(dc==d)&(hc==h)
            add('distance_heading',d+'_'+h,frame[mask])
            add('distance_heading_win',d+'_'+h,frame[mask&(frame.outcome=='win')])
            add('distance_heading_loss',d+'_'+h,frame[mask&(frame.outcome=='loss')])
    for name,mask in [('climb_gt10',frame.pitch_deg>10),('level_pm10',np.abs(frame.pitch_deg)<=10),('dive_lt_neg10',frame.pitch_deg< -10)]:add('pitch',name,frame[mask])
    for name in ['win','loss','draw','timeout']:add('outcome',name,frame[frame.outcome==name])
    episode_keys=['policy_run','episode_seed'] if 'policy_run' in frame else ['episode_seed']
    episode_returns=frame[episode_keys+['episode_return']].drop_duplicates()
    low,high=episode_returns.episode_return.quantile([.25,.75])
    key=pd.MultiIndex.from_frame(frame[episode_keys])
    low_keys=pd.MultiIndex.from_frame(episode_returns.loc[episode_returns.episode_return<=low,episode_keys])
    high_keys=pd.MultiIndex.from_frame(episode_returns.loc[episode_returns.episode_return>=high,episode_keys])
    add('return_quartile','bottom25',frame[key.isin(low_keys)])
    add('return_quartile','top25',frame[key.isin(high_keys)])
    for name,mask in [('pursued',frame.pursued_by_blue),('threatened',frame.threat_window),
        ('large_heading',frame.heading_deg>60),('fire_transition',frame.fire),('kill_transition',frame.kill)]:add('situation',name,frame[mask])
    return result


def analyze_run(job,root):
    root=Path(root)
    directory=root/'traces'/job['run_id']; out=root/'analysis'/job['run_id'];out.mkdir(parents=True,exist_ok=True)
    summary_path=out/'statistics.csv'
    sources=sorted(directory.glob('episode_*/metadata.json'))
    if not sources:raise RuntimeError(f'no traces: {directory}')
    protocol=json.loads((root/'protocol.json').read_text())
    expected_seeds=set(protocol['evaluation_seeds'])
    if len(sources)!=len(expected_seeds):raise RuntimeError(f'incomplete checkpoint audit: {job["run_id"]}')
    actual_seeds={json.loads(p.read_text())['episode_seed'] for p in sources}
    if actual_seeds!=expected_seeds:raise RuntimeError('evaluation seed mismatch')
    frames=[];event_rows=[];aligned=defaultdict(list);episodes=[];overshoots=[]
    started=time.monotonic()
    for number,meta in enumerate(sources):
        m=json.loads(meta.read_text());trace=read_trace(meta.with_name('episode_trace.npz'),m)
        assert m['checkpoint_sha256']==job['checkpoint_sha256'] and m['environment_config_sha256']==job['environment_config_sha256']
        frame,values,flags=episode_samples(trace,m)
        events=event_indices(trace,m,values,flags)
        for crossing in overshoot_proxies(trace):overshoots.append(dict(episode_seed=m['episode_seed'],**crossing))
        for event in events:
            i,k=event['agent'],event['frame'];row=dict(episode_seed=m['episode_seed'],**event)
            for lag in [-20,0,20]:
                q=k+lag;valid=0<=q<len(trace['red_alive']) and (trace['red_alive'][q,i] or
                    (lag==0 and event['event'] in ['fire','hit','kill','first_3km','first_1km','first_fire_window'] and k>0 and trace['red_alive'][k-1,i]))
                row[f'v_{lag}']=float(trace['red_kinematics'][q,i,3]) if valid else None
                row[f'av_{lag}']=float(trace['red_actions'][q-1,i,2]) if valid and q>0 else None
            row['survives_2s']=row['v_20'] is not None
            event_rows.append(row)
            for lag in range(-20,21):
                q=k+lag
                if not (0<=q<len(trace['red_alive']) and (trace['red_alive'][q,i] or
                    (lag==0 and event['event'] in ['fire','hit','kill','first_3km','first_1km','first_fire_window'] and k>0 and trace['red_alive'][k-1,i]))):continue
                av=float(trace['red_actions'][q-1,i,2]) if q>0 else np.nan
                aligned[event['event'],lag].append((float(trace['red_kinematics'][q,i,3]),av))
        episodes.append(dict(episode_seed=m['episode_seed'],return_=m['episode_return'],red_win=m['red_win'],
            blue_win=m['blue_win'],draw=m['draw'],steps=m['episode_steps'],
            red_fire_attempts=m['red_fire_attempts'],red_hits=m['red_weapon_hits'],
            red_kills=m['red_attack_kills'],blue_kills=m['blue_attack_kills'],
            mean_speed=float(frame.v.mean()),low_speed_fraction=float((frame.v<200).mean()),
            large_heading_low_speed=float((frame.loc[frame.heading_deg>60,'v']<200).mean()) if (frame.heading_deg>60).any() else None,
            high_speed_escape_fraction=float(frame.faster_escape_proxy.mean())))
        er=episodes[-1]
        for label,mask in [('near',frame.distance<1000),('far',frame.distance>3000),
                           ('turn',frame.heading_deg>60),('aligned',frame.heading_deg<30)]:
            er[label+'_speed_median']=float(frame.loc[mask,'v'].median()) if mask.any() else None
            er[label+'_action_mean']=float(frame.loc[mask,'a_v'].mean()) if mask.any() else None
        for name,a,b in [('near_minus_far','near','far'),('turn_minus_aligned','turn','aligned')]:
            er[name]=er[a+'_speed_median']-er[b+'_speed_median'] if er[a+'_speed_median'] is not None and er[b+'_speed_median'] is not None else None
        frames.append(frame)
        if (number+1)%25==0:print(f"analyze {job['run_id']}: {number+1}/{len(sources)}, {time.monotonic()-started:.1f}s",flush=True)
    frame=pd.concat(frames,ignore_index=True)
    stats=pd.DataFrame(grouped_statistics(frame))
    for key in ['run_id','algorithm','training_seed','kind']:stats[key]=job[key]
    stats.to_csv(summary_path,index=False)
    frame.to_csv(out/'alive_red_samples.csv.gz',index=False,compression='gzip')
    pd.DataFrame(episodes).to_csv(out/'episodes.csv',index=False)
    event_frame=pd.DataFrame(event_rows,columns=['episode_seed','event','agent','frame','time_s',
        'v_-20','av_-20','v_0','av_0','v_20','av_20','survives_2s'])
    event_frame.to_csv(out/'event_samples.csv',index=False)
    pd.DataFrame(overshoots,columns=['episode_seed','frame','time_s','agent','blue_agent','red_speed','blue_speed','distance']).to_csv(out/'overshoot_proxies.csv',index=False)
    comparisons=[]
    for event,subset in event_frame.groupby('event'):
        complete=subset.dropna(subset=['v_-20','v_0','v_20'])
        action_complete=subset.dropna(subset=['av_-20','av_0','av_20'])
        comparisons.append(dict(event=event,total_events=len(subset),complete_speed_triplets=len(complete),
            v_before=float(complete['v_-20'].mean()),v_at=float(complete.v_0.mean()),v_after=float(complete.v_20.mean()),
            v_after_minus_before=float((complete.v_20-complete['v_-20']).mean()),
            complete_action_triplets=len(action_complete),av_before=float(action_complete['av_-20'].mean()),
            av_at=float(action_complete.av_0.mean()),av_after=float(action_complete.av_20.mean()),
            two_second_survival_fraction=float(subset.survives_2s.mean())))
    pd.DataFrame(comparisons).to_csv(out/'event_comparisons.csv',index=False)
    erows=[]
    for (event,lag),pairs in aligned.items():
        array=np.asarray(pairs)
        erows.append(dict(event=event,lag_s=lag*.1,count=len(array),speed_mean=float(array[:,0].mean()),
            speed_median=float(np.median(array[:,0])),speed_p25=float(np.percentile(array[:,0],25)),
            speed_p75=float(np.percentile(array[:,0],75)),action_count=int(np.isfinite(array[:,1]).sum()),
            action_mean=float(np.nanmean(array[:,1])) if np.isfinite(array[:,1]).any() else None))
    pd.DataFrame(erows).to_csv(out/'event_curves.csv',index=False)
    # Equal episode weighting complements pooled aircraft-step statistics.
    e=pd.DataFrame(episodes)
    corr=e[['return_','mean_speed','low_speed_fraction','large_heading_low_speed','high_speed_escape_fraction']].corr(method='spearman')
    corr.to_csv(out/'episode_spearman.csv')
    contrasts=[];rng=np.random.default_rng(1027)
    for condition in ['near_minus_far','turn_minus_aligned']:
        for outcome,mask in [('all',np.ones(len(e),bool)),('win',e.red_win),('loss',e.blue_win)]:
            x=e.loc[mask,condition].dropna().to_numpy()
            if not len(x):continue
            means=np.mean(rng.choice(x,size=(2000,len(x)),replace=True),axis=1)
            contrasts.append(dict(condition=condition,outcome=outcome,episode_count=len(x),
                mean=float(x.mean()),median=float(np.median(x)),fraction_negative=float(np.mean(x<0)),
                bootstrap_mean_ci_low=float(np.percentile(means,2.5)),bootstrap_mean_ci_high=float(np.percentile(means,97.5))))
    pd.DataFrame(contrasts).to_csv(out/'episode_condition_contrasts.csv',index=False)
    checks=dict(run_id=job['run_id'],episodes=len(e),alive_agent_steps=len(frame),
        analysis_source_sha256=ANALYSIS_SOURCE_SHA256,
        expected_alive_agent_steps=sum(int(np.load(x.with_name('episode_trace.npz'),allow_pickle=False)['red_alive'][:-1].sum()) for x in sources),
        actual_fires=int(frame.fire.sum()),metadata_fires=int(e.red_fire_attempts.sum()),
        actual_hits=int(frame.hit.sum()),metadata_hits=int(e.red_hits.sum()),
        win_rate=float(e.red_win.mean()),mean_return=float(e.return_.mean()),
        mean_episode_speed_equal_episode_weight=float(e.mean_speed.mean()),
        blue_speed_225_verified=True)
    checks['overshoot_proxy_events']=len(overshoots)
    checks['fire_event_anchor_count']=next((r['count'] for r in erows if r['event']=='fire' and r['lag_s']==0),0)
    for meta in sources:
        with np.load(meta.with_name('episode_trace.npz'),allow_pickle=False) as t:
            assert np.all(t['blue_kinematics'][:,:,3][t['blue_alive']]==225.)
    assert checks['alive_agent_steps']==checks['expected_alive_agent_steps']
    assert checks['actual_fires']==checks['metadata_fires'] and checks['actual_hits']==checks['metadata_hits']
    assert checks['fire_event_anchor_count']==checks['metadata_fires']
    residual=np.abs(frame.v_next-np.clip(frame.v+2.5*frame.a_v,150,300))
    checks['speed_mapping_max_abs_error']=float(residual.max())
    assert residual.max()<1e-6
    (out/'checks.json').write_text(json.dumps(checks,indent=2))
    return checks


def plot(root,jobs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tools.audit_red_speed import ALGORITHMS
    figdir=root/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    all_stats=[];all_episodes=[];all_events=[]
    for job in jobs:
        out=root/'analysis'/job['run_id']
        if not (out/'statistics.csv').exists():continue
        all_stats.append(pd.read_csv(out/'statistics.csv'))
        e=pd.read_csv(out/'episodes.csv');e['run_id']=job['run_id'];e['algorithm']=job['algorithm'];e['kind']=job['kind'];e['training_seed']=job['training_seed'];all_episodes.append(e)
        e=pd.read_csv(out/'event_curves.csv');e['run_id']=job['run_id'];e['algorithm']=job['algorithm'];e['kind']=job['kind'];e['training_seed']=job['training_seed'];all_events.append(e)
    stats=pd.concat(all_stats,ignore_index=True);stats.to_csv(root/'all_statistics.csv',index=False)
    pd.concat(all_episodes,ignore_index=True).to_csv(root/'all_episodes.csv',index=False)
    events=pd.concat(all_events,ignore_index=True);events.to_csv(root/'all_event_curves.csv',index=False)
    primary=stats[stats.kind=='best']
    def layout(title):
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        fig.suptitle(title+' | best checkpoints, 100 episodes/seed',fontweight='bold')
        for ax,label in zip(axes.flat,ALGORITHMS):ax.set_title(label);ax.grid(alpha=.18)
        return fig,list(axes.flat)
    def save(fig,name):
        for ext in ['png','svg']:fig.savefig(figdir/f'{name}.{ext}',dpi=170)
        plt.close(fig)
    for name,group,labels in [('speed_vs_distance','distance',DISTANCE_LABELS),('speed_vs_heading','heading',HEADING_LABELS)]:
        fig,axes=layout(name.replace('_',' '))
        for ax,label in zip(axes,ALGORITHMS):
            for seed in (1,2,3):
                s=primary[(primary.algorithm==label)&(primary.training_seed==seed)&(primary.group==group)].set_index('condition').reindex(labels)
                ax.plot(range(len(labels)),s['mean'],marker='o',label=f'seed {seed}')
            ax.set_xticks(range(len(labels)),labels,rotation=20);ax.set_ylabel('Mean speed (m/s)');ax.axhline(225,color='gray',ls='--',lw=.7);ax.legend()
        save(fig,name)
    fig,axes=layout('Speed distribution (alive Red aircraft-steps)')
    for ax,label in zip(axes,ALGORITHMS):
        s=primary[(primary.algorithm==label)&(primary.group=='overall')]
        for _,r in s.iterrows():
            density=np.array([r['bin_'+x] for x in SPEED_LABELS])/np.diff(SPEED_EDGES)
            ax.stairs(density,SPEED_EDGES,label=f"seed {r.training_seed}")
        ax.set(xlabel='Speed (m/s)',ylabel='Probability density');ax.axvline(225,color='gray',ls='--',lw=.7);ax.legend()
    save(fig,'speed_histogram')
    fig,axes=layout('Distance x heading speed (equal mean of seed medians)')
    for ax,label in zip(axes,ALGORITHMS):
        s=primary[(primary.algorithm==label)&(primary.group=='distance_heading')].groupby('condition')['median'].mean()
        matrix=np.array([[s.get(d+'_'+h,np.nan) for h in ['small','medium','large']] for d in ['near','medium','far']])
        im=ax.imshow(matrix,vmin=150,vmax=300,cmap='viridis');fig.colorbar(im,ax=ax,label='Median speed (m/s)')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',color='white' if matrix[i,j]<225 else 'black')
        ax.set_xticks(range(3),['<30 deg','30-60 deg','>60 deg']);ax.set_yticks(range(3),['<1 km','1-3 km','>3 km']);ax.grid(False)
    save(fig,'distance_heading_heatmap')
    fig,axes=layout('Speed action distribution')
    action_bins=np.linspace(-1,1,21)
    for ax,label in zip(axes,ALGORITHMS):
        for job in jobs:
            if job['kind']!='best' or job['algorithm']!=label:continue
            f=pd.read_csv(root/'analysis'/job['run_id']/'alive_red_samples.csv.gz',usecols=['a_v'])
            hist,_=np.histogram(f.a_v,bins=action_bins)
            ax.stairs(hist/hist.sum(),action_bins,label=f"seed {job['training_seed']}")
        ax.set(xlabel='a_v',ylabel='Fraction');ax.legend()
    save(fig,'speed_action_distribution')
    fig,axes=layout('Actual fire aligned speed (surviving samples only)')
    for ax,label in zip(axes,ALGORITHMS):
        for seed in (1,2,3):
            s=events[(events.kind=='best')&(events.algorithm==label)&(events.training_seed==seed)&(events.event=='fire')].sort_values('lag_s')
            ax.plot(s.lag_s,s.speed_mean,label=f'seed {seed}')
        ax.axvline(0,color='gray',ls='--');ax.set(xlabel='Seconds relative to actual fire',ylabel='Mean speed (m/s)');ax.legend()
    save(fig,'fire_event_aligned_speed')
    fig,axes=layout('Win vs loss speed distribution (pooled per outcome)')
    for ax,label in zip(axes,ALGORITHMS):
        s=primary[(primary.algorithm==label)&(primary.group=='outcome')]
        for outcome in ['win','loss']:
            q=s[(s.condition==outcome)&(s['count']>0)]
            if len(q):
                y=[np.average(q['bin_'+x],weights=q['count']) for x in SPEED_LABELS]
                ax.plot(range(7),y,marker='o',label=outcome)
        ax.set_xticks(range(7),SPEED_LABELS,rotation=25);ax.set_ylabel('Fraction');ax.legend()
    save(fig,'win_loss_speed_distribution')
    return stats


def pooled_tables(root,jobs):
    """Descriptive aircraft-step pooling; retain per-seed tables for stability checks."""
    from tools.audit_red_speed import ALGORITHMS
    algorithm_rows=[];small_frames=[];action_hist=[]
    for algorithm in ALGORITHMS:
        frames=[]
        for job in jobs:
            if job['algorithm']!=algorithm or job['kind']!='best':continue
            frame=pd.read_csv(root/'analysis'/job['run_id']/'alive_red_samples.csv.gz')
            frame['policy_run']=job['run_id']
            hist,edges=np.histogram(frame.a_v,bins=np.linspace(-1,1,21))
            action_hist.extend(dict(run_id=job['run_id'],algorithm=algorithm,training_seed=job['training_seed'],
                lower=float(lo),upper=float(hi),count=int(n)) for lo,hi,n in zip(edges[:-1],edges[1:],hist))
            frames.append(frame)
        frame=pd.concat(frames,ignore_index=True)
        for row in grouped_statistics(frame):algorithm_rows.append(dict(algorithm=algorithm,**row))
        small_frames.append(frame[['v','a_v','distance','heading_deg','outcome']])
    pd.DataFrame(algorithm_rows).to_csv(root/'algorithm_statistics.csv',index=False)
    pd.DataFrame(action_hist).to_csv(root/'action_histogram.csv',index=False)
    pooled=pd.concat(small_frames,ignore_index=True);rows=[]
    dc=np.where(pooled.distance<1000,'near',np.where(pooled.distance<=3000,'medium','far'))
    hc=np.where(pooled.heading_deg<30,'small',np.where(pooled.heading_deg<=60,'medium','large'))
    for outcome in ['all','win','loss']:
        for d in ['near','medium','far']:
            for h in ['small','medium','large']:
                mask=(dc==d)&(hc==h)
                if outcome!='all':mask &= pooled.outcome==outcome
                f=pooled[mask];n=len(f)
                rows.append(dict(outcome=outcome,distance=d,heading=h,count=n,
                    mean=float(f.v.mean()) if n else None,median=float(f.v.median()) if n else None,
                    p25=float(f.v.quantile(.25)) if n else None,p75=float(f.v.quantile(.75)) if n else None,
                    action_mean=float(f.a_v.mean()) if n else None))
    pd.DataFrame(rows).to_csv(root/'global_distance_heading.csv',index=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',default='outputs/red_speed_audit_v338_20261017')
    p.add_argument('--run-id');p.add_argument('--plot-only',action='store_true')
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--analyze-only',action='store_true')
    p.add_argument('--skip-existing',action='store_true')
    a=p.parse_args();root=ROOT/a.input_dir
    jobs=[j for j in json.loads((root/'inventory.json').read_text()) if j['status']=='available' and (not a.run_id or j['run_id']==a.run_id)]
    if not a.plot_only:
        selected=[]
        for job in jobs:
            directory=root/'traces'/job['run_id']
            if not (directory/'episode_summary.json').exists():
                print('not complete yet:',job['run_id'],flush=True);continue
            if len(json.loads((directory/'episode_summary.json').read_text()))!=len(json.loads((root/'protocol.json').read_text())['evaluation_seeds']):
                print('partial summary, not complete yet:',job['run_id'],flush=True);continue
            check=root/'analysis'/job['run_id']/'checks.json'
            if a.skip_existing and check.exists() and json.loads(check.read_text()).get('analysis_source_sha256')==ANALYSIS_SOURCE_SHA256:
                print('existing verified analysis:',job['run_id'],flush=True);continue
            selected.append(job)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(analyze_run,j,str(root)) for j in selected]
            for future in as_completed(futures):print(json.dumps(future.result()),flush=True)
    if not a.run_id and not a.analyze_only:
        pooled_tables(root,jobs)
        plot(root,jobs)


if __name__=='__main__':main()
