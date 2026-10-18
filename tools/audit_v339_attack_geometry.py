"""CPU audit of immutable v3.39: real rollouts, pre-fire snapshots, offline ranges.

The observer forwards each wrapped method once and never calls a policy or RNG.
"""
from pathlib import Path
from copy import deepcopy
import argparse, csv, hashlib, json, math, os, platform, subprocess, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from env.models import AircraftState
from env.geometry import engagement_geometry
from env.v337_reward import tactical_pair_rewards_v337

POLICIES = ('zero', 'pursuit', 'lateral', 'vertical')
RANGES = (1000, 2000, 4000)
LIMIT = math.pi / 6

def config(): return load_config(ROOT / 'configs/combat_environment_v339.yaml')
def environment(seed=0):
    e = MultiUAVCombatEnv(config()); e.reset(seed); return e
def wrap(x): return np.arctan2(np.sin(x), np.cos(x))
def states(e): return np.asarray([[list(s.as_array()) + [float(s.alive)] for s in team] for team in (e.red, e.blue)])
def pair(distance=500., ata=0., ha=0., pitch=0., aa=0.):
    az, el = np.deg2rad([ata, ha])
    a = AircraftState(0., 0., -3000., 225., math.radians(pitch), 0.)
    b = AircraftState(distance*math.cos(el)*math.cos(az), distance*math.cos(el)*math.sin(az),
                      -3000.-distance*math.sin(el), 225., 0., math.radians(ata+aa))
    return a, b

def independent_angles(a, b):
    r = b.as_array()[:3] - a.as_array()[:3]
    d = float(np.linalg.norm(r)); horizontal = math.hypot(r[0], r[1])
    bearing = math.atan2(r[1], r[0]) if horizontal else a.psi
    # Independent velocity direction; exact zero LOS uses formal aligned convention.
    alpha = math.acos(float(np.clip(a.velocity_vector() @ r / (a.v*d), -1, 1))) if d else 0.
    return d, float(wrap(bearing-a.psi)), math.atan2(-r[2], horizontal), float(wrap(b.psi-bearing)), alpha

def independent_probability(distance, ata, ha, weapon):
    h = math.pi * math.exp(-distance / weapon.effective_hit_distance)
    intervals = [((-h-angle)/scale, (h-angle)/scale) for angle, scale in
                 ((ata, weapon.azimuth_noise_scale), (ha, weapon.elevation_noise_scale))]
    lo, hi = max(x[0] for x in intervals), min(x[1] for x in intervals)
    if lo > hi: return 0.
    # Independent deterministic Gaussian quadrature, no SciPy or random samples.
    nodes, weights = np.polynomial.legendre.leggauss(64)
    z = (hi-lo)*nodes/2 + (hi+lo)/2
    return float((hi-lo)/2 * (weights @ np.exp(-z*z/2)) / math.sqrt(2*math.pi))

def red_actions(e, mode):
    """Nearest 3D LOS, constant 225 m/s; two explicit simple angle probes."""
    out = np.zeros((4, 3), dtype=np.float32); selected = np.full(4, -1, dtype=int)
    command = e.config['action']['command']; t = e.steps*e.dt
    for i, a in enumerate(e.red):
        if not a.alive: continue
        candidates = [(sum((b.as_array()[:3]-a.as_array()[:3])**2), j) for j,b in enumerate(e.blue) if b.alive]
        if not candidates: continue
        j = min(candidates)[1]; selected[i] = j
        if mode == 'zero': continue
        b = e.blue[j]; dx,dy,dz = b.x-a.x,b.y-a.y,b.z-a.z
        heading = math.atan2(dy,dx); pitch = math.atan2(-dz,math.hypot(dx,dy))
        if mode == 'lateral' and 12. <= t < 20.: heading += math.pi/6
        if mode == 'vertical':
            pitch = math.pi/6 if 8. <= t < 12. else -math.pi/6 if 16. <= t < 20. else 0.
        out[i] = np.clip([wrap(heading-a.psi)/command['heading_delta_max'],
                          (pitch-a.theta)/command['pitch_delta_max'],
                          (225.-a.v)/command['speed_delta_max']], -1, 1)
    return out, selected

class PreFireObserver:
    """Read at dense reward (after boundary, before both shots); resolve once."""
    def __init__(self, e): self.e=e; self.records=[]; self.originals={}; self.calls={'dense':0,'resolve':0}
    def __enter__(self):
        e=self.e
        for name in ('dense_combat_reward','_resolve_combat'):
            self.originals[name]=(name in e.__dict__,e.__dict__.get(name),getattr(e,name))
        def dense():
            result=self.originals['dense_combat_reward'][2](); self.calls['dense']+=1
            self.current=dict(pre=states(e), ammo_pre=np.stack((e.red_ammo,e.blue_ammo)).copy(),
                cooldown_pre=np.stack((e.red_fire_cooldown_steps,e.blue_fire_cooldown_steps)).copy(),
                blue_targets=e.blue_step_target_indices.copy(),
                dense=np.stack([result[1][k] for k in ('guide','tactical_offense','tactical_defense')]))
            return result
        def resolve(red_attempts,blue_attempts):
            attempts=np.full((2,4),-1,dtype=np.int16); hits=np.zeros((2,4),dtype=bool)
            for side,items in enumerate((red_attempts,blue_attempts)):
                for i,j,hit in items: attempts[side,i]=j; hits[side,i]=hit
            result=self.originals['_resolve_combat'][2](red_attempts,blue_attempts); self.calls['resolve']+=1
            credited=np.zeros((2,4,4),dtype=bool)
            for side,items in enumerate(result):
                for target,shooters in items.items(): credited[side,shooters,target]=True
            self.current.update(attempt_targets=attempts,hit=hits,credited_kills=credited)
            return result
        e.dense_combat_reward=dense; e._resolve_combat=resolve
        return self
    def finish(self, start, actions, red_targets, result):
        e=self.e; row=self.current
        row.update(start=start, post=states(e), actions=np.stack((result[4]['executed_red_actions'],result[4]['executed_blue_actions'])),
            red_targets=red_targets.copy(),ammo_post=np.stack((e.red_ammo,e.blue_ammo)).copy(),
            cooldown_post=np.stack((e.red_fire_cooldown_steps,e.blue_fire_cooldown_steps)).copy(),
            rewards=result[1].copy(),done=np.asarray(result[2:4],dtype=bool))
        # 0=none,1=ground,2=horizontal,3=ceiling,4=weapon; ground priority matches resolver.
        cause=np.zeros((2,4),dtype=np.uint8)
        for side in range(2):
            for i in range(4):
                if start[side,i,6] and not row['pre'][side,i,6]:
                    s=row['pre'][side,i]; cause[side,i]=1 if -s[2]<=0 else 3 if -s[2]>6000 else 2
                elif row['pre'][side,i,6] and not row['post'][side,i,6]: cause[side,i]=4
        row['death_cause']=cause; self.records.append(row)
    def arrays(self): return {k:np.asarray([r[k] for r in self.records]) for k in self.records[0]}
    def __exit__(self,*args):
        for name,(present,previous,_) in self.originals.items():
            if present: setattr(self.e,name,previous)
            else: delattr(self.e,name)

def vector_geometry(pre,side):
    """Independent array LOS/velocity math on real pre-fire states only."""
    a=pre[:,side]; b=pre[:,1-side]; delta=b[:,None,:,:3]-a[:,:,None,:3]
    horizontal=np.hypot(delta[...,0],delta[...,1]); distance=np.linalg.norm(delta,axis=-1)
    bearing=np.where(horizontal>0,np.arctan2(delta[...,1],delta[...,0]),a[:,:,None,5])
    ata=wrap(bearing-a[:,:,None,5]); ha=np.arctan2(-delta[...,2],horizontal); aa=wrap(b[:,None,:,5]-bearing)
    def velocity(s):
        return np.stack((s[...,3]*np.cos(s[...,4])*np.cos(s[...,5]),s[...,3]*np.cos(s[...,4])*np.sin(s[...,5]),-s[...,3]*np.sin(s[...,4])),axis=-1)
    va,vb=velocity(a),velocity(b); relative=vb[:,None]-va[:,:,None]
    cosine=np.divide((va[:,:,None]*delta).sum(-1),a[:,:,None,3]*distance,out=np.ones_like(distance),where=distance>0)
    alpha=np.arccos(np.clip(cosine,-1,1)); live=(a[:,:,None,6]>0)&(b[:,None,:,6]>0)
    return dict(distance=distance,ata=ata,ha=ha,aa=aa,alpha=alpha,live=live,
        pitch=np.broadcast_to(a[:,:,None,4],distance.shape),relative_speed=np.linalg.norm(relative,axis=-1),
        relative_velocity=relative,
        closing_speed=np.divide(-(relative*delta).sum(-1),distance,out=np.zeros_like(distance),where=distance>0))

def masks(g,range_max):
    in_range=g['live']&(g['distance']>=0)&(g['distance']<=range_max)
    ata=in_range&(np.abs(g['ata'])<=LIMIT+1e-12)
    return dict(live=g['live'],distance=in_range,distance_ata=ata,
                full=ata&(np.abs(g['ha'])<=LIMIT+1e-12),cone=in_range&(g['alpha']<=LIMIT+1e-12))

def runs(mask):
    """Half-open integer runs; one sample contributes one dt, including singleton."""
    padded=np.r_[False,np.asarray(mask,dtype=bool),False].astype(int)
    return list(zip(np.flatnonzero(np.diff(padded)==1),np.flatnonzero(np.diff(padded)==-1)))

def windows(mask,g,context,dt,level):
    rows=[]
    for i in range(mask.shape[1]):
        targets=range(mask.shape[2]) if level=='pair' else [None]
        for j in targets:
            active=mask[:,i].any(axis=1) if j is None else mask[:,i,j]
            for start,end in runs(active):
                subset=mask[start:end,i].copy()
                if j is not None: subset[:]=False;subset[:,j]=mask[start:end,i,j]
                row=dict(**context,level=level,attacker=i,target=j,first_sample_time_s=(start+1)*dt,
                    last_sample_time_s=end*dt,start_time_s=(start+1)*dt,end_time_s=end*dt,
                    duration_s=(end-start)*dt,samples=end-start,targets_seen=','.join(map(str,np.flatnonzero(subset.any(axis=0)))))
                for field in ('distance','ata','ha','alpha','pitch','relative_speed','closing_speed'):
                    values=g[field][start:end,i][subset]; factor=180/math.pi if field in ('ata','ha','alpha','pitch') else 1.
                    row[field+'_mean']=float(values.mean()*factor);row[field+'_min']=float(values.min()*factor);row[field+'_max']=float(values.max()*factor)
                rows.append(row)
    return rows

def write_csv(path,rows):
    rows=list(rows)
    if not rows: path.write_text('',encoding='utf-8');return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def analyze_episode(data,seed,policy,info,counts,dt=.1):
    rows=[]; window_rows=[]; increments=[]; pre=data['pre']; start=data['start']; n=len(pre)
    for side,label in enumerate(('red','blue')):
        g=vector_geometry(pre,side); live=g['live']; pair_den=int(live.sum());agent_den=int(live.any(axis=2).sum())
        alive_start=start[:,side,:,6]>0; alive_den=int(alive_start.sum())
        base=dict(seed=seed,policy=policy,side=label,steps=n,seconds=n*dt,red_win=bool(info['red_win']),blue_win=bool(info['blue_win']),
            draw=bool(info['draw']),termination_reason=info['termination_reason'],timeout=bool(data['done'][-1,1]),
            initial_alive=int(start[0,side,:,6].sum()),survivors=int(data['post'][-1,side,:,6].sum()),live_pair_steps=pair_den,live_agent_steps=agent_den,
            low_speed_fraction=float(((start[:,side,:,3]<=160)&alive_start).sum()/alive_den),
            heading_saturation_fraction=float(((np.abs(data['actions'][:,side,:,0])>=1-1e-7)&alive_start).sum()/alive_den),
            pitch_saturation_fraction=float(((np.abs(data['actions'][:,side,:,1])>=1-1e-7)&alive_start).sum()/alive_den))
        for key in ('fire_attempts','weapon_hits','attack_kills','boundary_exits','ground_losses','ceiling_losses','ammo_used'):base['actual_'+key]=int(counts[label][key])
        base['actual_horizontal_losses']=int((data['death_cause'][:,side]==2).sum())
        assert base['actual_boundary_exits']==base['actual_horizontal_losses']+base['actual_ceiling_losses']
        assert base['actual_fire_attempts']==int((data['attempt_targets'][:,side]>=0).sum())
        assert base['actual_weapon_hits']==int(data['hit'][:,side].sum())
        assert base['initial_alive']-base['survivors']==base['actual_boundary_exits']+base['actual_ground_losses']+int((data['death_cause'][:,side]==4).sum())
        baseline=masks(g,1000)
        for radius in RANGES:
            m=masks(g,radius); row=dict(base,range_m=radius)
            for key,mask in m.items():
                row['pair_'+key]=int(mask.sum()); row['pair_'+key+'_fraction']=float(mask.sum()/pair_den) if pair_den else None
                agent=mask.any(axis=2);row['agent_'+key]=int(agent.sum());row['agent_'+key+'_fraction']=float(agent.sum()/agent_den) if agent_den else None
                hits=np.flatnonzero(agent.any(axis=1));row['first_'+key+'_time_s']=float((hits[0]+1)*dt) if len(hits) else None
                row['episode_has_'+key]=bool(len(hits))
            for model in ('full','cone'):
                wr=windows(m[model],g,dict(seed=seed,policy=policy,side=label,range_m=radius,model=model),dt,'agent')
                row[model+'_agent_window_count']=len(wr);row[model+'_agent_window_total_s']=sum(r['duration_s'] for r in wr)
                window_rows.extend(wr);window_rows.extend(windows(m[model],g,dict(seed=seed,policy=policy,side=label,range_m=radius,model=model),dt,'pair'))
            # Actual 1-km ammo/cooldown and Blue target binding funnel, NOT range counterfactual shots.
            if radius==1000:
                ammo=data['ammo_pre'][:,side,:,None]>0;ready=data['cooldown_pre'][:,side,:,None]<=1
                eligible=m['full']&ammo
                row['actual_eligible_agent_steps']=int(eligible.any(axis=2).sum())
                row['actual_ready_agent_steps']=int((eligible&ready).any(axis=2).sum())
                if side==1:
                    selected=data['blue_targets'];binding=np.arange(4)[None,None,:]==selected[:,:,None]
                    row['actual_bound_ready_agent_steps']=int((eligible&ready&binding).any(axis=2).sum())
                    row['blue_other_target_only_agent_steps']=int((eligible.any(axis=2)&~(eligible&binding).any(axis=2)).sum())
                else:row['actual_bound_ready_agent_steps']=row['actual_ready_agent_steps']
                assert row['actual_bound_ready_agent_steps']==row['actual_fire_attempts']
            rows.append(row)
            if radius>1000:
                for lower in ([1000] if radius==2000 else [1000,2000]):
                    previous=baseline if lower==1000 else masks(g,lower)
                    extra_distance=m['distance']&~previous['distance'];extra_full=m['full']&~previous['full']
                    inc=dict(seed=seed,policy=policy,side=label,baseline_range_m=lower,range_m=radius,added_distance_pairs=int(extra_distance.sum()),
                        added_full_pairs=int(extra_full.sum()),added_rejected_by_angles=int((extra_distance&~m['full']).sum()))
                    assert inc['added_distance_pairs']==inc['added_full_pairs']+inc['added_rejected_by_angles']
                    for name in ('distance','ata','ha','alpha','pitch','relative_speed','closing_speed'):
                        values=g[name][extra_full];factor=180/math.pi if name in ('ata','ha','alpha','pitch') else 1.
                        for q,labelq in ((.1,'p10'),(.5,'median'),(.9,'p90')):inc[name+'_'+labelq]=float(np.quantile(values,q)*factor) if len(values) else None
                    increments.append(inc)
    return rows,window_rows,increments

def static_audit(output):
    e=environment();rows=[];error=0.
    for pitch in (-60,-45,-30,0,30,45,60):
        for ha in (-45,-30,0,30,45):
            for ata in (-45,-30,-15,0,15,30,45):
                for distance in (500,1000,1500,2000,4000):
                    a,b=pair(distance,ata,ha,pitch);g=engagement_geometry(a,b);ind=independent_angles(a,b)
                    err=max(abs(g.distance-ind[0]),abs(wrap(g.ata-ind[1])),abs(g.ha-ind[2]),abs(wrap(g.aa-ind[3])),abs(g.off_boresight-ind[4]));error=max(error,err)
                    current=abs(g.ata)<=LIMIT+1e-12 and abs(g.ha)<=LIMIT+1e-12; cone=ind[4]<=LIMIT+1e-12
                    row=dict(requested_distance_m=distance,distance_m=g.distance,pitch_deg=pitch,requested_ata_deg=ata,requested_ha_deg=ha,
                        ata_deg=math.degrees(g.ata),ha_deg=math.degrees(g.ha),aa_deg=math.degrees(g.aa),alpha3d_deg=math.degrees(ind[4]),
                        current_angles=current,cone_angles=cone,formal_1km_eligible=e.weapon.qualifies(a,b,6),formula_error=err)
                    for radius in RANGES:row[f'current_{radius}']=current and g.distance<=radius;row[f'cone_{radius}']=cone and g.distance<=radius
                    rows.append(row)
    write_csv(output/'geometry_grid.csv',rows)
    probabilities=[]
    for d in (500,1000,1500,2000,4000):
        for ata,ha in ((0,0),(15,0),(30,0),(30,30),(30,-30),(0,30)):
            x,y=np.deg2rad([ata,ha]);actual=e.weapon.expected_hit_probability(d,x,y);expected=independent_probability(d,x,y,e.weapon)
            assert abs(actual-expected)<1e-14
            probabilities.append(dict(distance_m=d,ata_deg=ata,ha_deg=ha,probability=actual,independent_probability=expected,formal_range=d<=1000,
                                      interpretation='current unchanged function; offline extrapolation beyond 1000m'))
    write_csv(output/'hit_probability.csv',probabilities)
    cases=[]
    specs=[('level_same_heading',500,0,0,0),('equal_height_pitch_plus45',500,0,0,45),('equal_height_pitch_minus45',500,0,0,-45),
        ('higher_target',500,0,30,0),('lower_target',500,0,-30,0),('pitch_matches_los',500,0,45,45),
        ('rectangular_corner',500,30,30,0),('ata_below',500,30-1e-8,0,0),('ata_equal',500,30,0,0),('ata_above',500,30+1e-8,0,0)]
    for radius in RANGES:
        for delta in (-1e-7,0,1e-7):specs.append((f'range_{radius}_{delta:+g}',radius+delta,0,0,0))
    for name,d,ata,ha,pitch in specs:
        a,b=pair(d,ata,ha,pitch);g=engagement_geometry(a,b)
        cases.append(dict(case=name,attacker=list(a.as_array()),target=list(b.as_array()),distance_m=g.distance,ata_deg=math.degrees(g.ata),
            ha_deg=math.degrees(g.ha),aa_deg=math.degrees(g.aa),alpha3d_deg=math.degrees(g.off_boresight),formal_1km_eligible=e.weapon.qualifies(a,b,6),
            **{f'geometry_{r}':bool(g.distance<=r and abs(g.ata)<=LIMIT+1e-12 and abs(g.ha)<=LIMIT+1e-12) for r in RANGES}))
    # Degenerate LOS conventions are formal implementation facts, not physical alignment.
    for name,delta in [('coincident',(0,0,0)),('near_coincident',(1e-12,0,0)),('above',(0,0,-500)),('below',(0,0,500)),('near_vertical',(1e-12,0,-500))]:
        a,b=pair(pitch=60);b.x,b.y,b.z=a.x+delta[0],a.y+delta[1],a.z+delta[2];g=engagement_geometry(a,b)
        assert np.isfinite([g.distance,g.ata,g.ha,g.aa,g.off_boresight]).all()
        cases.append(dict(case=name,attacker=list(a.as_array()),target=list(b.as_array()),distance_m=g.distance,ata_deg=math.degrees(g.ata),ha_deg=math.degrees(g.ha),
            aa_deg=math.degrees(g.aa),alpha3d_deg=math.degrees(g.off_boresight),formal_1km_eligible=e.weapon.qualifies(a,b,6)))
    reward=[]
    for distance in (500,1000,1500,2000,3000,4000):
        for ata,ha,aa,pitch in ((0,0,0,0),(0,0,0,60),(30,30,0,0),(0,45,0,45),(0,0,180,0)):
            a,b=pair(distance,ata,ha,pitch,aa);p=tactical_pair_rewards_v337([a],[b],e.config['reward'],e.config['weapon']);g=engagement_geometry(a,b)
            w3=np.clip((g.distance-1000)/2000,0,1);w4=np.clip((3000-g.distance)/2000,0,1)
            guide=.001*w3 if abs(g.ata)<=LIMIT+1e-12 and abs(g.ha)<=LIMIT+1e-12 else 0.
            offense=0.
            for deg,value in ((30,.01),(15,.02),(5,.1)):
                if abs(g.ata)<=math.radians(deg)+1e-12 and abs(g.ha)<=math.radians(deg)+1e-12:offense=value
            offense=offense*w4 if abs(g.aa)<=LIMIT+1e-12 and g.distance<=3000 else 0.
            assert abs(guide-p['guide'][0,0])<1e-14 and abs(offense-p['tactical_offense'][0,0])<1e-14
            reward.append(dict(distance_m=distance,ata_deg=ata,ha_deg=ha,aa_deg=aa,pitch_deg=pitch,guide=guide,offense=offense,
                defense=float(p['tactical_defense'][0,0]),formal_eligible=e.weapon.qualifies(a,b,6)))
    write_csv(output/'reward_geometry.csv',reward)
    (output/'representative_geometry_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    assert error<1e-10
    return dict(grid_cases=len(rows),max_formula_error=error,probability_cases=len(probabilities),reward_cases=len(reward),representative_cases=len(cases))

def controlled_cases(output):
    """Two explanatory actual-step fixtures; excluded from default reset batches."""
    rows=[]
    for kind in ('multiple_eligible_red_targets','blue_binding_other_target'):
        e=environment(0);dead=lambda:AircraftState(0,0,-3000,225,0,0,False)
        e.red=[dead() for _ in range(4)];e.blue=[dead() for _ in range(4)]
        if kind=='multiple_eligible_red_targets':
            e.red[0]=AircraftState(0,0,-3000,225,0,0)
            e.blue[0]=AircraftState(500,0,-3000,225,0,0);e.blue[1]=AircraftState(700,0,-3000,225,0,0)
        else:
            e.blue[0]=AircraftState(0,0,-3000,225,0,0)
            e.red[0]=AircraftState(-300,0,-3000,225,0,math.pi);e.red[1]=AircraftState(500,0,-3000,225,0,0)
        with PreFireObserver(e) as obs:
            before=states(e);actions=np.zeros((4,3),np.float32);result=e.step(actions);obs.finish(before,actions,np.full(4,-1),result);data=obs.arrays()
        np.savez_compressed(output/f'{kind}.npz',**data)
        side=0 if kind=='multiple_eligible_red_targets' else 1;g=vector_geometry(data['pre'],side);m=masks(g,1000)
        for j in np.flatnonzero(data['pre'][0,1-side,:,6]):
            rows.append(dict(case=kind,side=('red','blue')[side],attacker=0,target=int(j),
                attacker_state=data['pre'][0,side,0].tolist(),target_state=data['pre'][0,1-side,j].tolist(),
                actual_action=data['actions'][0,side,0].tolist(),distance_m=float(g['distance'][0,0,j]),
                ata_deg=float(np.degrees(g['ata'][0,0,j])),ha_deg=float(np.degrees(g['ha'][0,0,j])),
                aa_deg=float(np.degrees(g['aa'][0,0,j])),alpha3d_deg=float(np.degrees(g['alpha'][0,0,j])),
                geometric_1km=bool(m['full'][0,0,j]),blue_step_target=int(data['blue_targets'][0,0]) if side else None,
                actual_fire_target=int(data['attempt_targets'][0,side,0]),actual_hit=bool(data['hit'][0,side,0])))
        if side==0:assert m['full'][0,0].sum()==2 and data['attempt_targets'][0,0,0]==0
        else:assert not m['full'][0,0,0] and m['full'][0,0,1] and data['attempt_targets'][0,1,0]==-1
    (output/'controlled_target_cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')

def run_episode(seed,policy,trace_path=None):
    e=environment(seed)
    with PreFireObserver(e) as observer:
        while True:
            actions,targets=red_actions(e,policy);before=states(e); result=e.step(actions)
            observer.finish(before,actions,targets,result)
            if result[2] or result[3]:break
        assert observer.calls==dict(dense=e.steps,resolve=e.steps)
        data=observer.arrays()
    if trace_path:np.savez_compressed(trace_path,**data)
    rows,w,inc=analyze_episode(data,seed,policy,result[4],e.combat_counts,e.dt)
    return rows,w,inc

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/v339_attack_geometry_audit')
    p.add_argument('--seeds',type=int,default=50);p.add_argument('--policies',nargs='+',choices=POLICIES,default=list(POLICIES));p.add_argument('--smoke',action='store_true')
    args=p.parse_args();out=args.output_dir.resolve()
    if out.exists():raise FileExistsError('Fresh output directory required; historical output is protected')
    os.environ['CUDA_VISIBLE_DEVICES']='-1';out.mkdir(parents=True);(out/'traces').mkdir()
    summary=dict(environment_version='3.39',device='CPU',python=platform.python_version(),numpy=np.__version__,config=config(),
        formal_weapon_range_m=1000,offline_ranges_m=list(RANGES),seeds=list(range(args.seeds)),policies=args.policies,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),started=time.time(),completed_episodes=0,smoke=args.smoke)
    summary['static']=static_audit(out);controlled_cases(out);episodes=[];all_windows=[];increments=[]
    for policy in args.policies:
        for seed in range(args.seeds):
            rows,w,inc=run_episode(seed,policy,out/'traces'/f'{policy}_seed{seed}.npz');episodes.extend(rows);all_windows.extend(w);increments.extend(inc)
            summary['completed_episodes']+=1
            if (seed+1)%5==0 or seed==0:print(f'{policy}: {seed+1}/{args.seeds}, completed {summary["completed_episodes"]}, elapsed {time.time()-summary["started"]:.1f}s',flush=True)
        write_csv(out/'episode_summary.csv',episodes);write_csv(out/'engagement_windows.csv',all_windows);write_csv(out/'incremental_opportunities.csv',increments)
        (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    summary['elapsed_s']=time.time()-summary['started'];summary['completed']=True
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    print(f'Completed {summary["completed_episodes"]} episodes in {summary["elapsed_s"]:.1f}s',flush=True)

if __name__=='__main__':main()
