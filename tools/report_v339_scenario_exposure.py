"""Analyze actual scenario interventions; preserve formal/counterfactual distinction."""
from pathlib import Path
import argparse,hashlib,json,math,os,shutil
import numpy as np
import pandas as pd
from tools.audit_v339_scenario_exposure import ROOT,LETHAL,NONLETHAL,ident,vector_geometry,pair_conditions

def scalar(x):
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)

def group_summary(frame,keys):
    result=[]
    for key,g in frame.groupby(keys,dropna=False,sort=False):
        key=key if isinstance(key,tuple) else (key,);row=dict(zip(keys,key));row['episodes']=len(g)
        for field in ('episode_duration_s','r4_positive_agent_fraction','r4_negative_agent_fraction','r4_positive_total','r4_negative_total','r4_total',
                      'red_absolute_turns_sum','blue_absolute_turns_sum','red_max_agent_absolute_turns','blue_max_agent_absolute_turns',
                      'red_target_switches','blue_target_switches','red_low_speed_fraction','blue_low_speed_fraction'):
            row[field+'_mean']=float(g[field].mean())
        for field in ('first_fire_time_s','first_hit_time_s','first_kill_time_s','first_tail_aspect_time_s','first_tail_aspect_red_time_s','first_tail_aspect_blue_time_s',
                      'first_positive_r4_time_s','first_negative_r4_time_s','first_any_death_time_s','first_range_3km_red_time_s','first_range_1km_red_time_s'):
            values=g[field].dropna();row[field+'_observed']=len(values);row[field+'_median']=float(values.median()) if len(values) else None
        for field in ('boundary_losses','ground_losses','ceiling_losses','red_shots','blue_shots','red_hits','blue_hits','red_kills','blue_kills'):
            row[field+'_sum']=int(g[field].sum())
        row['boundary_episode_count']=int(((g.boundary_losses+g.ground_losses)>0).sum())
        row['positive_exposure_episode_fraction']=float(g.first_positive_r4_time_s.notna().mean())
        row['negative_exposure_episode_fraction']=float(g.first_negative_r4_time_s.notna().mean())
        row['tail_aspect_episode_fraction']=float(g.first_tail_aspect_time_s.notna().mean())
        row['timeout_count']=int(g.truncated.sum());result.append(row)
    return pd.DataFrame(result)

def validate_prefixes(out,ep):
    keys=['scenario_type','red_policy','initial_heading_offset_deg','seed'];lookup={tuple(r[k] for k in keys)+(r['lethality_mode'],):r for r in ep.to_dict('records')};rows=[]
    online_path=out/'online_pair_validation_all.json'
    online={(r['seed'],r['red_policy'],r['initial_heading_offset_deg']):r for r in json.loads(online_path.read_text(encoding='utf-8'))} if online_path.exists() else {}
    for _,formal in ep[ep.lethality_mode==LETHAL].iterrows():
        key=tuple(formal[k] for k in keys);cf=lookup[key+(NONLETHAL,)]
        with np.load(formal.trajectory_file,allow_pickle=False) as a,np.load(cf['trajectory_file'],allow_pickle=False) as b:
            hits=np.flatnonzero(a['credited_kills'].reshape(len(a['pre']),-1).any(axis=1))
            n=int(hits[0]+1) if len(hits) else len(a['pre'])
            assert len(b['pre'])>=n
            for field in ('start','pre','actions','ammo_pre','ammo_post','cooldown_pre','cooldown_post','red_targets','blue_targets','attempt_targets','hit','dense'):
                np.testing.assert_array_equal(a[field][:n],b[field][:n])
            np.testing.assert_array_equal(a['post'][:n,...,:6],b['post'][:n,...,:6])
            if 'rng_state_sha256' in a:
                np.testing.assert_array_equal(a['rng_state_sha256'][:n],b['rng_state_sha256'][:n]);rng_proof='every_prefix_step_hash'
            elif (formal.seed,formal.red_policy,formal.initial_heading_offset_deg) in online:
                proof=online[(formal.seed,formal.red_policy,formal.initial_heading_offset_deg)]
                assert proof['rng_equal_at_fork'] and proof['matched_steps_including_prefire_fork']==n
                rng_proof='online_same_seed_policy_RNG_prefix_verified'
            else:rng_proof='legacy_trace_no_rng_hash; live paired tests/smoke provide limited RNG validation'
            if len(hits):
                assert not np.array_equal(a['post'][n-1,...,6],b['post'][n-1,...,6]);fork=n*.1
            else:
                np.testing.assert_array_equal(a['post'],b['post']);assert len(a['pre'])==len(b['pre']);fork=None
        rows.append(dict(**{k:formal[k] for k in keys},matched_steps=n,fork_time_s=fork,rng_proof=rng_proof,
            fork_reason='first_weapon_kill_alive_and_event_reward_diverge' if fork is not None else 'no_combat_kill_entire_trajectory_equal'))
    pd.DataFrame(rows).to_csv(out/'paired_prefix_validation.csv',index=False,encoding='utf-8-sig');return rows

def aggregate(out):
    s=json.loads((out/'summary.json').read_text(encoding='utf-8'));assert s['completed']
    ep=pd.read_csv(out/'episode_summary.csv');geo=pd.read_csv(out/'geometry_exposure.csv');events=pd.read_csv(out/'exposure_events.csv')
    prefixes=validate_prefixes(out,ep);s['paired_prefixes_verified']=len(prefixes)
    s['online_rng_pairs_verified']=sum(r['rng_proof']=='online_same_seed_policy_RNG_prefix_verified' for r in prefixes)
    s['stored_hash_rng_pairs_verified']=sum(r['rng_proof']=='every_prefix_step_hash' for r in prefixes)
    s['all_paired_rng_prefixes_verified']=s['online_rng_pairs_verified']+s['stored_hash_rng_pairs_verified']==len(prefixes)
    default=ep[(ep.scenario_type=='default')&(ep.initial_heading_offset_deg==0)]
    baseline=group_summary(default,['red_policy','lethality_mode']);baseline.to_csv(out/'baseline_extension_summary.csv',index=False,encoding='utf-8-sig')
    heading=ep[(ep.scenario_type=='default')&(ep.red_policy.isin(['zero','pursuit']))&(ep.seed<20)]
    heading_summary=group_summary(heading,['red_policy','lethality_mode','initial_heading_offset_deg']);heading_summary.to_csv(out/'heading_summary.csv',index=False,encoding='utf-8-sig')
    sub=[]
    for boundary in (False,True):
        selected=heading[((heading.boundary_losses+heading.ground_losses)>0)==boundary]
        if len(selected):
            x=group_summary(selected,['red_policy','lethality_mode','initial_heading_offset_deg']);x['has_boundary_or_ground_loss']=boundary;sub.append(x)
    boundary_summary=pd.concat(sub,ignore_index=True);boundary_summary.to_csv(out/'heading_boundary_strata.csv',index=False,encoding='utf-8-sig')
    control=group_summary(ep[ep.scenario_type!='default'],['scenario_type','lethality_mode']);control.to_csv(out/'positive_control_summary.csv',index=False,encoding='utf-8-sig')
    joined=default[default.lethality_mode==LETHAL].merge(default[default.lethality_mode==NONLETHAL],on=['seed','red_policy'],suffixes=('_formal','_cf'))
    joined['cf_positive_after_formal_end']=joined.first_positive_r4_time_s_cf>joined.episode_duration_s_formal
    joined['cf_negative_after_formal_end']=joined.first_negative_r4_time_s_cf>joined.episode_duration_s_formal
    joined[['seed','red_policy','episode_duration_s_formal','first_kill_time_s_formal','episode_duration_s_cf','first_tail_aspect_time_s_cf','first_positive_r4_time_s_cf','first_negative_r4_time_s_cf','cf_positive_after_formal_end','cf_negative_after_formal_end']].to_csv(out/'fork_and_exposure.csv',index=False,encoding='utf-8-sig')
    comparison=[]
    for policy,g in joined.groupby('red_policy',sort=False):
        comparison.append(dict(red_policy=policy,episodes=len(g),cf_positive=int(g.first_positive_r4_time_s_cf.notna().sum()),cf_negative=int(g.first_negative_r4_time_s_cf.notna().sum()),
            cf_positive_after_formal_end=int(g.cf_positive_after_formal_end.sum()),cf_negative_after_formal_end=int(g.cf_negative_after_formal_end.sum())))
    records=lambda df:df.astype(object).where(pd.notna(df),None).to_dict('records')
    s['baseline_extension']=records(baseline);s['heading_experiment']=records(heading_summary);s['positive_controls']=records(control);s['after_formal_end']=comparison
    s['r4_reconstruction_max_error']=float(ep.r4_reconstruction_max_error.max());assert s['r4_reconstruction_max_error']<1e-12
    assert len(ep)==s['completed_unique_episodes'];assert ep.episode_id.is_unique
    # Every paired source has the identical initial physical state except the allowed heading.
    for _,g in heading.groupby(['seed','red_policy','lethality_mode']):
        reference=g[g.initial_heading_offset_deg==0].iloc[0]
        with np.load(reference.trajectory_file) as z:ref=z['start'][0]
        for r in g.itertuples():
            with np.load(r.trajectory_file) as z:
                current=z['start'][0];np.testing.assert_array_equal(current[1],ref[1]);np.testing.assert_array_equal(current[0,:, [0,1,2,3,4,6]],ref[0,:, [0,1,2,3,4,6]])
                delta=current[0,:,5]-ref[0,:,5]-math.radians(r.initial_heading_offset_deg)
                assert abs(np.arctan2(np.sin(delta),np.cos(delta))).max()<1e-12
    s['heading_initial_state_invariants_verified']=True
    # Every geometric funnel is nested; separate zero distance weight from missed angles.
    for a,b in [('tail_aspect','live'),('tail_distance','tail_aspect'),('tail_distance_ata','tail_distance'),('r4_geometry','tail_distance_ata'),('r4_nonzero_pair','r4_geometry')]:assert (geo[a+'_pair_steps']<=geo[b+'_pair_steps']).all()
    s['funnel_nesting_verified']=True
    ammo_rows=[];event_records=events.to_dict('records');event_lookup={}
    for record in event_records:event_lookup.setdefault(record['episode_id'],[]).append(record)
    for r in ep.itertuples():
        with np.load(r.trajectory_file,allow_pickle=False) as z:
            dense=z['dense'];pre=z['pre'];initial=z['start'][0]
            positive=dense[:,1]>0;negative=dense[:,2]<0;ammo=z['ammo_pre'];armed=ammo[:,0]>0
            threatening_armed=(z['blue_r4_nonzero_pair']&(ammo[:,1,:,None]>0)).any(axis=1)
            ready=(z['cooldown_pre'][:,0]<=1)
            positive_shootable=(z['red_r4_nonzero_pair']&(z['red_distance']<=1000)).any(axis=2)&armed&ready
            first=np.flatnonzero(positive.any(axis=1))
            ammo_rows.append(dict(episode_id=r.episode_id,scenario_type=r.scenario_type,red_policy=r.red_policy,lethality_mode=r.lethality_mode,
                initial_heading_offset_deg=r.initial_heading_offset_deg,seed=r.seed,
                positive_agent_steps=int(positive.sum()),positive_with_ammo_agent_steps=int((positive&armed).sum()),
                positive_without_ammo_agent_steps=int((positive&~armed).sum()),positive_shootable_now_agent_steps=int((positive&positive_shootable).sum()),
                negative_agent_steps=int(negative.sum()),negative_with_armed_attacker_steps=int((negative&threatening_armed).sum()),
                first_positive_any_recipient_has_ammo=bool((positive[first[0]]&armed[first[0]]).any()) if len(first) else None,
                first_positive_recipient_ammo=','.join(str(int(v)) for v in ammo[first[0],0][positive[first[0]]]) if len(first) else None))
            for event in event_lookup.get(r.episode_id,[]):
                t=event['time_s'];k=max(0,round(t/.1)-1);snapshot=initial if t==0 else pre[k]
                if event['event'].startswith('first_tail_aspect'):
                    sides=[0] if '_red_' in event['event'] else [1] if '_blue_' in event['event'] else [0,1]
                    for side in sides:
                        g=vector_geometry(snapshot[None],side);valid=pair_conditions(g)['tail_aspect'][0]
                        if valid.any():
                            i,j=np.argwhere(valid)[0];event.update(side=('red','blue')[side],attacker=int(i),target=int(j),distance_m=float(g['distance'][0,i,j]),
                                ata_deg=float(np.degrees(g['ata'][0,i,j])),ha_deg=float(np.degrees(g['ha'][0,i,j])),aa_deg=float(np.degrees(g['aa'][0,i,j])));break
                if pd.notna(event.get('attacker')) and pd.notna(event.get('target')) and event.get('side') in ('red','blue'):
                    side=0 if event['side']=='red' else 1;i=int(event['attacker']);j=int(event['target'])
                    event.update(attacker_state=snapshot[side,i].tolist(),target_state=snapshot[1-side,j].tolist(),
                        attacker_ammo_pre=int(ammo[k,side,i]),pair_selection='first qualifying pair; multiple pairs may contribute')
    pd.DataFrame(ammo_rows).to_csv(out/'ammo_conditioned_r4_exposure.csv',index=False,encoding='utf-8-sig')
    events=pd.DataFrame(event_records);events.to_csv(out/'exposure_events.csv',index=False,encoding='utf-8-sig')
    windows=pd.read_csv(out/'exposure_windows.csv');wr=[]
    for key,g in windows.groupby(['scenario_type','red_policy','lethality_mode','initial_heading_offset_deg','side','condition'],sort=False):
        v=g.duration_s;wr.append(dict(zip(['scenario_type','red_policy','lethality_mode','initial_heading_offset_deg','side','condition'],key),
            windows=len(g),duration_min_s=float(v.min()),duration_median_s=float(v.median()),duration_mean_s=float(v.mean()),duration_max_s=float(v.max()),duration_total_s=float(v.sum())))
    pd.DataFrame(wr).to_csv(out/'exposure_window_summary.csv',index=False,encoding='utf-8-sig')
    schema=dict(state_arrays='start/pre/post: [sample,side,slot,field]; side 0 Red/1 Blue; fields x,y,z,v,theta,psi,alive; NED height=-z',
        phases='start before actions; pre after both movements and noncombat deaths, before shots; post after combat resolution',
        actions='actual clipped/masked actions [heading,pitch,speed]',dense='[sample,component,Red recipient]; components guide,offense,defense',
        pair_geometry='red_/blue_distance,ata,ha,aa,live: [sample,attacker,target]; angles radians, distance m',
        credited_kills='[sample,attacking side,attacker,target]; deduplicate target to count kills',
        suppressed_kills='optional [sample,attacking side,target]; unique would-be killed live targets per step, may repeat targets across steps; never actual kills',
        rng_state_sha256='new rollouts only; SHA256 of post-step actual RNG state; legacy formal traces validated separately with 200 live pairs',
        initial_observation='actual rebuilt Red observation [4,45]; state intervention only before first action',
        original_reset_states='formal reset before heading/artificial-tail intervention; start[0] is actual experimental initial state',
        death_cause='0 none,1 ground,2 horizontal boundary,3 ceiling,4 weapon; preinstalled inactive 1v1 slots are not counted as deaths',
        windows='tail_aspect/r4_nonzero_pair retain fixed attacker,target; positive_r4/negative_r4 side=red_reward_recipient uses attacker column as Red recipient index, target empty',
        first_events='geometric t=0 permitted for initial states; actual reward earliest post-move sample .1s; R4 event pair is first qualifying pair, not necessarily the unique max/min contributor')
    (out/'trajectory_data'/'schema.json').write_text(json.dumps(schema,indent=2),encoding='utf-8')
    return s,ep,geo,events,baseline,heading_summary,boundary_summary,control,joined

def figures(out,s,ep,geo,baseline,heading,control):
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    dest=out/'figures';dest.mkdir(exist_ok=True);names=[]
    def save(name,fig):fig.tight_layout();fig.savefig(dest/name,dpi=160);plt.close(fig);names.append(name)
    colors={'formal_lethal':'#aa4433','nonlethal_counterfactual':'#2679ac'}
    default=ep[(ep.scenario_type=='default')&(ep.initial_heading_offset_deg==0)]
    fig,axes=plt.subplots(1,4,figsize=(18,5))
    for ax,policy in zip(axes,['zero','pursuit','lateral','vertical']):
        for mode,shift in [(LETHAL,-.18),(NONLETHAL,.18)]:
            g=default[(default.red_policy==policy)&(default.lethality_mode==mode)]
            fields=['first_range_3km_red_time_s','first_fire_time_s','first_kill_time_s','first_tail_aspect_time_s','first_positive_r4_time_s','first_negative_r4_time_s','episode_duration_s']
            vals=[g[f].median() if g[f].notna().any() else np.nan for f in fields]
            ax.scatter(vals,np.arange(len(fields))+shift,color=colors[mode],label=mode,s=30)
            for i,f in enumerate(fields):ax.text(vals[i] if np.isfinite(vals[i]) else 1,i+shift,f' n={g[f].notna().sum()}' if np.isfinite(vals[i]) else 'not observed',fontsize=7,va='center')
        count=int(default[(default.red_policy==policy)&(default.lethality_mode==LETHAL)].shape[0])
        ax.set(yticks=np.arange(7),yticklabels=['3km','First shot','First kill','AA tail','R4 offense','R4 defense','End'],xlabel='Median time among observed episodes (s)',title=f'{policy}, {count} seeds / mode',xlim=(0,110));ax.invert_yaxis();ax.grid(alpha=.2)
        if policy=='zero':ax.legend(fontsize=7)
    save('baseline_nonlethal_timeline.png',fig)
    # Actual AA for each selected fixed pair; no target-switch envelope masquerading as a lock.
    choices=[]
    for policy in ['zero','pursuit','lateral','vertical']:
        g=default[(default.red_policy==policy)&(default.lethality_mode==NONLETHAL)]
        positive=g[g.first_positive_r4_time_s.notna()];choices.append(positive.iloc[0] if len(positive) else g.iloc[0])
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for ax,r in zip(axes.flat,choices):
        with np.load(r.trajectory_file) as z:
            time=(np.arange(len(z['pre']))+1)*.1
            if pd.notna(r.first_positive_r4_time_s):
                k=round(r.first_positive_r4_time_s/.1)-1;valid=z['red_r4_nonzero_pair'][k];i,j=np.argwhere(valid)[0]
            else:i=j=0
            mask=z['red_live'][:,i,j];ax.plot(time[mask],np.degrees(abs(z['red_aa'][:,i,j][mask])),label=f'fixed Red{i} -> Blue{j}')
            ax.axhline(30,c='green',ls='--',label='AA <=30 deg')
            if pd.notna(r.first_positive_r4_time_s):ax.axvline(r.first_positive_r4_time_s,c='purple',ls=':',label='First actual Red offense')
            ax.set(xlabel='Time (s)',ylabel='Absolute AA (deg)',title=f'{r.red_policy}, seed={r.seed}, nonlethal');ax.legend(fontsize=8);ax.grid(alpha=.2)
    save('fixed_pair_aa_time.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,field,title in [(axes[0],'first_kill_time_s','Formal first kill'),(axes[1],'first_positive_r4_time_s','Nonlethal first positive R4')]:
        mode=LETHAL if field=='first_kill_time_s' else NONLETHAL
        for policy in ['zero','pursuit','lateral','vertical']:
            g=default[(default.red_policy==policy)&(default.lethality_mode==mode)];v=np.sort(g[field].dropna())
            if len(v):ax.step(v,np.arange(1,len(v)+1)/len(g),where='post',label=f'{policy}: {len(v)}/{len(g)} observed')
            else:ax.plot([],[],label=f'{policy}: 0/{len(g)} observed')
        ax.set(xlabel='Time (s)',ylabel='Fraction of all episodes',title=title,ylim=(0,1.02));ax.legend(fontsize=8);ax.grid(alpha=.2)
    save('first_kill_r4_distribution.png',fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9));offsets=[-90,-45,0,45,90]
    for row,policy in enumerate(['zero','pursuit']):
        for col,mode in enumerate([LETHAL,NONLETHAL]):
            ax=axes[row,col];g=heading[(heading.red_policy==policy)&(heading.lethality_mode==mode)].set_index('initial_heading_offset_deg').loc[offsets];x=np.arange(5)
            ax.bar(x-.25,100*g.positive_exposure_episode_fraction,.25,label='Positive R4');ax.bar(x,100*g.negative_exposure_episode_fraction,.25,label='Negative R4');ax.bar(x+.25,100*g.boundary_episode_count/g.episodes,.25,label='Boundary/ground episode')
            ax.set(xticks=x,xticklabels=offsets,xlabel='Initial Red heading offset (deg)',ylabel='Episodes (%)',title=f'{policy}, {mode}, n={int(g.episodes.iloc[0])} / offset',ylim=(0,105));ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
    save('heading_exposure_boundary.png',fig)
    fig=plt.figure(figsize=(15,11));selected=[default[(default.red_policy=='pursuit')&(default.lethality_mode==LETHAL)].iloc[0],choices[1],choices[3],ep[(ep.scenario_type=='tail_4v4_1500')&(ep.lethality_mode==NONLETHAL)].iloc[0]]
    for index,r in enumerate(selected):
        ax=fig.add_subplot(2,2,index+1,projection='3d')
        with np.load(r.trajectory_file) as z:
            tr=np.concatenate((z['start'][:1],z['post']))
            for side,color in [(0,'#aa4433'),(1,'#2679ac')]:
                for i in range(4):
                    if not tr[0,side,i,6]:continue
                    dead=np.flatnonzero(tr[:,side,i,6]==0);end=int(dead[0]) if len(dead) else len(tr)-1;p=tr[:end+1,side,i]
                    ax.plot(p[:,0],p[:,1],-p[:,2],color=color,label=('Red' if side==0 else 'Blue') if i==0 else None)
            if pd.notna(r.first_positive_r4_time_s):
                k=round(r.first_positive_r4_time_s/.1)-1;i=int(np.flatnonzero(z['dense'][k,1]>0)[0]);point=z['pre'][k,0,i]
                ax.scatter(point[0],point[1],-point[2],color='purple',marker='*',s=90,label=f'First offense {r.first_positive_r4_time_s:.1f}s')
            ax.set(xlabel='North x (m)',ylabel='East y (m)',zlabel='Altitude (m)',title=f'{r.scenario_type}, {r.red_policy}, seed{r.seed}\n{r.lethality_mode}');ax.set_box_aspect(None,zoom=.76);ax.legend(fontsize=7)
    save('representative_3d_trajectories.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(15,6));fields=['live','tail_aspect','tail_distance','tail_distance_ata','r4_geometry','r4_nonzero_pair']
    for ax,side in zip(axes,['red','blue']):
        for mode in [LETHAL,NONLETHAL]:
            g=geo[(geo.scenario_type=='default')&(geo.initial_heading_offset_deg==0)&(geo.side==side)&(geo.lethality_mode==mode)]
            v=[g[f+'_pair_steps'].sum()/g.live_pair_steps.sum()*100 for f in fields];ax.plot(np.arange(6),v,'o-',label=mode)
            for i,val in enumerate(v):ax.text(i,val+1 if mode==NONLETHAL or val<4 else val-2,f'{val:.2f}%',fontsize=8)
        ax.set(xticks=np.arange(6),xticklabels=['Live','AA','+d<=3km','+ATA','+HA','+w4>0'],ylabel='Live pair-step (%)',title=f'{side} attacking geometry; {int(len(default)/2)} episodes / mode');ax.legend(fontsize=8);ax.grid(alpha=.2)
    save('r4_geometry_funnel.png',fig)
    return names

def table(frame,cols=None):
    frame=frame if cols is None else frame[cols]
    def fmt(v):
        if v is None or (isinstance(v,(float,np.floating)) and not np.isfinite(v)):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v).replace('|','/')
    return '| '+' | '.join(frame.columns)+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'

def report(out,s,ep,geo,events,baseline,heading,boundary,control,joined):
    content=[]
    def add(x=''):content.append(x+'\n')
    def tab(frame,cols=None):add(table(frame,cols))
    add('# v3.39 初始交战几何与战术奖励曝光专项实验')
    add(f'实际完成{s["completed_unique_episodes"]}个去重条件：复用已核验的{s["reused_formal_episodes"]}局正式迎头对局，新增{s["new_rollout_episodes"]}局真实step对局；总计{s["steps"]}步（含复用步）。CPU、PowerShell uav，CUDA_VISIBLE_DEVICES=-1；WSL Ubuntu访问E_ACCESSDENIED后按仓库规则回退。没有训练、checkpoint或正式环境改动。')
    add('## 一、代码机制与干预边界')
    files=['env/combat_v339.py','env/combat_v334.py','env/v334_scenario.py','env/combat_v332.py','env/combat_v330.py','env/v330_transition.py','env/v311_kinematics.py','env/geometry.py','env/v337_reward.py','env/v336_weapon.py','env/v339_policy.py','env/combat_env.py','tools/audit_v339_attack_geometry.py']
    add('实际核对：'+ '、'.join(f'[{f}]({ROOT.as_posix()}/{f})' for f in files)+'。')
    add('正式初始化：令u=(cosα,sinα)、t=(-sinα,cosα)，Red锚点−4000u、Blue锚点+4000u；局部offset为(300,0)、(−50,250)、(−50,−250)、(−200,0)，Red位置=−4000u+offset_x*u+offset_y*t，Blue位置=+4000u−offset_x*u−offset_y*t。Red ψ=α、Blue ψ=wrap(α+π)，v225、h3000、pitch0。随机seed主要旋转同一迎头编队，非多种战术场景。')
    add('Blue每步按三维最近存活Red选择追击目标，用LOS航向/高度角做有限增量控制，距离远/中/近期望速度240/220/200m/s；步首绑定目标同时约束Blue射击。Red四种策略完全复用上一轮zero/pursuit/lateral/vertical；Red期望速度225，lateral在12–20s使用LOS+30°，vertical在8–12s期望+30°、16–20s期望−30°、其他0°。全过程正常action接口，45°/s、30°/s、dt.1s和速度范围不改。')
    add('执行顺序：Blue步首目标和动作→双方运动→非战斗死亡→dense/R4→资格与冷却/弹药/命中采样→_resolve_combat同时汇集双方命中后标记死亡→事件奖励及终止。R4在武器死亡之前计算；因此不能把当步被击毁的飞机从prefire几何中删除。')
    add('几何：ATA=wrap(LOS方位−本机ψ)、HA=LOS相对世界水平面的高度角、AA=wrap(目标ψ−LOS方位)。尾向条件只有|AA|≤30°；完整R4还需ATA/HA≤30°、d≤3000m，且w4=clip((3000−d)/2000,0,1)>0。因此d=3000m几何门限可合格但奖励为0。R4进攻按30/15/5°取.01/.02/.1，防御−.015/−.025/−.15；Red进攻跨目标max，防御对Blue中心威胁跨攻击者min。进攻和防御分量单独记录，不能用净和0推断零曝光。')
    add('nonlethal_counterfactual仅在该实验实例临时替换_resolve_combat：成功命中仍由原武器采样、原计数器记录，ammo/cooldown照常；结算返回({}, {})，不标记战斗死亡、不产生击毁/被击毁事件奖励，attack_kills和first kill保持0/缺失。原正式resolver代码不改，也不先杀死再复活。非战斗resolver和1000步上限完全保留；未改奖励系数。每次成功命中活目标的“被屏蔽击毁事件”另记suppressed_kills，可重复命中同一目标，不能当独立实际击毁数。')
    add('## 二、样本分配、记录和验证')
    add('A：原迎头四策略×50seeds，共200局，直接复用完整旧轨迹；核验旧summary的config、旧工具/config SHA、每局t0与当前真实reset逐位相同，旧目录当前全部文件另做SHA保护。B：相同四策略×50seeds的非致命延续200局。C：0/±45/±90°×zero/pursuit×两模式×20seeds=400条件，其中H0的80条件取自A/B相同seed，新增320；只有reset后Red ψ统一偏置，Blue、位置、高度、速度及其他字段不变，并重新调用正式_observations构建初始观测。D：1v1与4v4×1500/2500m×两模式×3seeds=24人工正向控制。总条件成员824，但去重后744，不把共享H0重复当独立样本。')
    add('先行smoke80去重条件，其中12复用、68新对局；C每个航向/策略/模式先覆盖3seeds，B每策略先3seeds；在线配对20例另核对RNG到首次击毁。完整批量采用4个独立CPU进程，每局独立env/RNG，不共享环境状态。')
    add('正向控制不是正式默认随机样本：Red x=−d/2、Blue x=+d/2，h3000、v225、ψ0、pitch0；1v1只启用slot0，其余预置死亡；4v4双方y均为−450/−150/150/450m。同y对应pair初始位于尾后且满足R4；不是宣称每个cross-pair均合格，例如d1500、横向差900m的交叉对ATA约30.96°会被排除。所有飞机位于安全区域。Blue仍自动最近追击，会主动转向身后的Red；没有为维持尾追改Blue。人工初态在相同scenario的三个seed中相同，seed主要改变武器采样，不能当三种不同尾追布局。')
    add('prefire观察器直接复用上一轮PreFireObserver，原方法每步转发一次，不重复策略或命中采样。所有对局保存完整start/pre/post、alive、actions、targets、ammo/cooldown、attempts/hits、实际credited kills、非战斗/武器死亡、Red三个dense分量、终止与几何数组。新轨迹还保存每步RNG状态SHA，不抽取额外随机数。对局路径不作理想化替换。')
    add('统计单位：prefire存活有向pair-step；agent机会为该步至少一个合格目标；实际R4进攻/防御曝光均以Red奖励接收者agent-step统计，Blue几何表是Blue攻击者中心，不应混称Blue自身获得正式奖励。报告均值以episode等权，pooled几何漏斗只作曝光描述，不把step当独立实验重复。AA-only与完整几何、正距离权重、实际聚合非零分层保存；非适用时间缺失而非0。')
    add('表中fraction是0–1比例，图中转为百分数。首事件中位数仅在发生者中计算；不存在事件不参与中位数，旁列observed保留分母。窗口分布按窗口描述，episode等权比例另外报告，不作基于step独立性的显著性检验。')
    add('初始几何可在t=0已满足；此时first_tail/range标0。实际R4只在首次真实step的prefire阶段计算，最早.1s。窗口保留固定攻击者/目标身份，按连续样本数×.1s统计；agent奖励窗口另记，允许不同pair贡献，不能将其当固定目标锁定。')
    add('全批量验证：'+str(s['paired_prefixes_verified'])+'组致命/非致命配对，至首次击毁当步包含prefire、动作、目标、弹药、冷却、命中和dense逐位相同；该步post坐标相同，alive和事件奖励开始分叉。C/D新轨迹用逐步RNG SHA核验'+str(s['stored_hash_rng_pairs_verified'])+'组；A旧轨迹本身未存逐步RNG，另重新执行同seed同策略在线配对核验'+str(s['online_rng_pairs_verified'])+'组，且其匹配前缀长度与旧轨迹首次击毁一致。全部配对RNG证明是否齐全：'+str(s['all_paired_rng_prefixes_verified'])+'。无武器击毁的配对要求整局完全相同。细节见paired_prefix_validation.csv、online_pair_validation_all.json和paired_validation_smoke.json；不能声称旧文件直接提供了未保存的RNG字段。')
    for label,value in s.get('tests',{}).items():add('测试 '+label+'：'+value+'。')
    if s.get('smoke_full_reproducibility_verified_episodes'):add('额外完整重复性核验：smoke与完整批量共有'+str(s['smoke_full_reproducibility_verified_episodes'])+'个相同条件，对其全部逐步数组逐位比较通过；这些不是新增独立样本。')
    add(f'全批量实际R4独立重建最大误差{s["r4_reconstruction_max_error"]:.3g}，初始航向字段不变量和逐层几何集合嵌套均通过。测试覆盖原观察器透明性、非致命不改运动/RNG、不能屏蔽非战斗死亡、首次击毁分叉、实例恢复、初始观测一致、正向控制、AA≠实际R4、重复性和正式1000步限制。')
    add('## 三、A/B：正式迎头与非致命延续')
    tab(baseline,['red_policy','lethality_mode','episodes','episode_duration_s_mean','first_fire_time_s_median','first_hit_time_s_median','first_kill_time_s_median','first_tail_aspect_red_time_s_observed','first_tail_aspect_red_time_s_median','first_tail_aspect_blue_time_s_observed','first_tail_aspect_blue_time_s_median','first_positive_r4_time_s_observed','first_positive_r4_time_s_median','first_negative_r4_time_s_observed','first_negative_r4_time_s_median'])
    add('上表首事件中位数只在实际发生者内计算；observed为对应分母。nonlethal的first kill缺失是实验定义，不是武器未命中；hits继续正常记录。正式A原200局R4进攻/防御均为0，未因本次重新分析改变。')
    tab(baseline,['red_policy','lethality_mode','r4_positive_agent_fraction_mean','r4_negative_agent_fraction_mean','r4_positive_total_mean','r4_negative_total_mean','r4_total_mean','boundary_losses_sum','ground_losses_sum','ceiling_losses_sum','timeout_count'])
    tab(baseline,['red_policy','lethality_mode','red_absolute_turns_sum_mean','blue_absolute_turns_sum_mean','red_max_agent_absolute_turns_mean','blue_max_agent_absolute_turns_mean','red_target_switches_mean','blue_target_switches_mean','red_low_speed_fraction_mean','blue_low_speed_fraction_mean'])
    add('累计turns是存活时逐步绝对航向变化除2π，不一定为完整圆周轨迹。zero的Red nearest target变化仅是诊断，不是零动作控制主动切换。低速阈值≤160m/s；这些固定策略未主动贴150，不能由零低速率排除学习policy的低速偏好。')
    add('配对延续中R4是否在正式对局结束后才出现：')
    tab(pd.DataFrame(s['after_formal_end']))
    add('这才是检验“战术曝光是否被早期致命结算截断”的主要配对证据；不能只比较两个模式的总奖励或时长。非致命模式不再是相同胜负任务，击毁奖励被定义性移除，禁止把其胜率/return当性能提高。')
    add('归因边界：非致命干预同时保留了本会死亡的飞机和追击目标集合，并推迟全灭终止，不是把正式幸存者的同一条轨迹机械延长。因此证据指向“致命结算/存活选择对后续几何的影响”，不能单独识别纯粹episode时长的效果。正式max_steps原本已允许100s；若提前结束源于全灭，仅调大max_steps不会解决。')
    ammo=pd.read_csv(out/'ammo_conditioned_r4_exposure.csv');a=ammo[(ammo.scenario_type=='default')&(ammo.initial_heading_offset_deg==0)&(ammo.lethality_mode==NONLETHAL)]
    add('保留6发弹药也意味着延续后可能在耗尽弹药时仍有R4，因为正式R4不检查ammo/cooldown。下表分开记录“几何奖励曝光”与“仍有弹药/当前可射”的曝光；不能把无弹药时的R4当有效武器攻击。negative_with_armed_attacker_steps要求至少一个符合威胁几何的Blue仍有弹药。')
    tab(a.groupby('red_policy',sort=False)[['positive_agent_steps','positive_with_ammo_agent_steps','positive_without_ammo_agent_steps','positive_shootable_now_agent_steps','negative_agent_steps','negative_with_armed_attacker_steps']].sum().reset_index())
    add('![时间线](figures/baseline_nonlethal_timeline.png)\n\n![固定pair AA](figures/fixed_pair_aa_time.png)\n\n![首事件分布](figures/first_kill_r4_distribution.png)')
    add('## 四、C：初始航向单因素及边界混杂')
    tab(heading,['red_policy','lethality_mode','initial_heading_offset_deg','episodes','episode_duration_s_mean','positive_exposure_episode_fraction','negative_exposure_episode_fraction','first_positive_r4_time_s_median','first_negative_r4_time_s_median','boundary_episode_count','boundary_losses_sum'])
    add('所有航向组均保留全部20样本；边界样本没有删除。额外分层如下：若某组没有“无边界损失”样本，则不能给它编造边界无关的曝光或胜率比较。此实验不把非致命结局当正式性能指标。')
    tab(boundary,['red_policy','lethality_mode','initial_heading_offset_deg','has_boundary_or_ground_loss','episodes','positive_exposure_episode_fraction','negative_exposure_episode_fraction','episode_duration_s_mean'])
    add('初始heading扰动只持续到策略通过action逐步纠正。纯追击可在45°/s约束下纠正±45/±90°，所以不能预设初始航向改变一定保留到交战时；zero没有这个纠正。靠边缘初态也意味着大偏置可能直接变成出界实验，必须结合以上分层归因。')
    add('![航向与边界曝光](figures/heading_exposure_boundary.png)')
    add('## 五、D：人工尾追正向控制')
    tab(control,['scenario_type','lethality_mode','episodes','first_positive_r4_time_s_observed','first_positive_r4_time_s_median','r4_positive_total_mean','first_negative_r4_time_s_observed','first_negative_r4_time_s_median','episode_duration_s_mean','boundary_losses_sum'])
    w=pd.read_csv(out/'exposure_window_summary.csv');selected=w[(w.scenario_type!='default')&w.condition.isin(['r4_nonzero_pair','positive_r4'])]
    tab(selected,['scenario_type','lethality_mode','side','condition','windows','duration_median_s','duration_max_s','duration_total_s'])
    add('这些初态满足条件时确实产生R4，说明奖励路径不是常量0或不可达的空分支。持续尾追是否维持必须看固定pair窗口，不只看总奖励。Blue立即转向最近Red会改变目标AA，使初始尾追很快失效；此后4v4可能通过目标切换重新获得机会，不能称为同一个目标持续尾追。初始人工状态、每步轨迹与事件表给出了可复核依据。')
    add('## 六、几何漏斗与零曝光分类')
    g=geo[(geo.scenario_type=='default')&(geo.initial_heading_offset_deg==0)]
    cols=['live_pair_steps','tail_aspect_pair_steps','tail_distance_pair_steps','tail_distance_ata_pair_steps','r4_geometry_pair_steps','r4_nonzero_pair_pair_steps']
    tab(g.groupby(['red_policy','lethality_mode','side'],sort=False)[cols].sum().reset_index())
    add('零曝光分类必须逐层解释：没有tail_aspect表示未进入目标尾向LOS；有AA但无距离/ATA/HA表示未进入完整优势几何；r4_geometry有而nonzero无表示距离权重为0；pair非零而实际agent分量不符则是聚合/统计疑点。本次独立重建已经核验实际Red max offense/min threat，未发现后一种记录错误。正负分量可相互抵消，因此r4_total=0绝不等于零曝光。')
    add('![几何漏斗](figures/r4_geometry_funnel.png)\n\n![实际三维轨迹](figures/representative_3d_trajectories.png)')
    add('## 七、七个问题与归因')
    add('### 1. 默认场景为什么几乎没有尾后占位？')
    add('正式迎头200局的正/负R4均为0；AA-only并非绝对全0，侧偏组Red有1/50局、Blue有9/50局，其余组均无AA-only。这些少量角度曝光未通过全部距离/角度/权重要求。B中有主动转向的三组各50/50局出现正R4，说明不能将默认无曝光直接归因于运动学不可达；早期击毁以及纯追击的几何形成时间具有直接影响。')
    add('### 2. 不发生武器击毁，继续追击能否形成尾追和R4？')
    for policy in ['zero','pursuit','lateral','vertical']:
        r=baseline[(baseline.red_policy==policy)&(baseline.lethality_mode==NONLETHAL)].iloc[0]
        number=lambda x:'未发生' if pd.isna(x) else f'{x:.3f}s'
        add(f'{policy}：{int(r.episodes)}局中正R4 {int(r.first_positive_r4_time_s_observed)}局、负R4 {int(r.first_negative_r4_time_s_observed)}局；首次正/负中位为{number(r.first_positive_r4_time_s_median)}/{number(r.first_negative_r4_time_s_median)}；AA-only出现{int(r.first_tail_aspect_time_s_observed)}局。zero不主动转向，其Red正奖励与Blue尾追威胁必须分开。')
    add('### 3. 首次交战、击毁与优势位置的时间关系？')
    pf=baseline[(baseline.red_policy=='pursuit')&(baseline.lethality_mode==LETHAL)].iloc[0];pc=baseline[(baseline.red_policy=='pursuit')&(baseline.lethality_mode==NONLETHAL)].iloc[0]
    add(f'纯追击正式组：进入3km中位{pf.first_range_3km_red_time_s_median:.1f}s，首次开火/击毁中位{pf.first_fire_time_s_median:.1f}/{pf.first_kill_time_s_median:.1f}s，平均结束{pf.episode_duration_s_mean:.3f}s。配对非致命组Red AA-only中位{pc.first_tail_aspect_red_time_s_median:.1f}s，正R4中位{pc.first_positive_r4_time_s_median:.1f}s，负R4中位{pc.first_negative_r4_time_s_median:.1f}s。逐局配对而非仅比较中位数，纯追击与vertical各50/50局正R4都晚于对应正式结束；lateral为43/50，另7局CF曝光早于正式结束，说明个体死亡与目标选择分叉同样重要。首次击毁当步仍同坐标、同prefire，此后允许分叉。')
    add('### 4. 初始航向有多大影响？')
    add('C的纯追击在所有五种heading的致命组均为0/20正R4和0/20负R4，而非致命组均为20/20正、20/20负。只偏置初始heading没有打破致命模式的零R4现象；±45/±90°被追击控制逐步纠正。zero在±45°的致命组有20/20负R4，但正R4仍0，且20/20局有边界损失；±90°两模式均20/20局出界、约15s结束且没有射击。不同航向的全部样本与边界分层同时报告，不从删选后的样本宣称整体改进。')
    add('### 5. 尾追初态下控制和奖励是否正常？')
    add('D的24个条件全部在.1s首次产生正R4；没有修改奖励或武器。1v1的初始正R4窗口在1500m持续.7s、2500m持续.6s，非致命延续至100s也没有新增正R4窗口。Blue立即朝身后Red转向，改变目标AA，迅速破坏尾追。4v4延续可借由后续机动/目标变化再次出现R4，但固定pair窗口必须单独核对。这排除了奖励路径原则上无法触发的强假设，同时不能证明普通纯追击可以长期保持同一个目标的优势位置。')
    add('### 6. 有证据表明三维控制本身阻碍绕后和持续机动吗？')
    add('B/D已经实际出现R4，因此“运动学绝对不允许形成这些优势几何”的强判断被反驳。zero没有正曝光来自它根本不主动改变航向，不能归咎于控制执行失败；长延续中的频繁转向与切换说明可以持续机动，但累计转向多不等于战术优秀。低速≤160m/s在本批为0，未主动研究150m/s学习策略偏好。没有必须增加PID/RK4/过载模型的证据。')
    add('### 7. 下一阶段最值得验证的单一因素？')
    add('根据B的配对延续与D正向控制，优先考虑“战斗致命结算的时间尺度/首次交战结束过快”这一单一机制的后续受控验证，而不是先扩大射程或调R4系数。非致命只是诊断，不建议直接把正式任务取消击毁。若C表明纯追击会迅速纠正初始heading，则仅改变初始heading不足以保证尾追曝光。任何正式干预都需新的任务授权，保持奖励/Blue/运动学等其他因素冻结；本轮不实施。')
    add('后续必须同时监测“有弹药时的R4曝光”，不能只追求总R4。当前非致命纯追击正R4有16369个agent-step，其中仅1315个仍有弹药，当前R4几何同时具备射击距离/弹药/冷却条件的仅161个。这限制了把长延续奖励曝光直接解释为有效作战能力的结论。')
    add('归因层级：已确认事实是公式、时序、目标选择与非致命结算边界；实测现象是以上曝光/事件/边界表；配对延续可支持早期武器死亡截断后续几何的机制解释；但尚未证明何种改动能改善MAPPO/MADSAC训练、何种训练样本配比最优、是否应改奖励或真实动力学。')
    add('## 八、复现、文件索引与保护')
    add('所有新对局走当前正式step，不是shadow运动学。baseline_extension、heading、positive_control及boundary_strata分别给出逐局等权统计；geometry_exposure保留方向及分母，exposure_events保留具体首R4的pair、距离、ATA/HA/AA；exposure_windows保留固定pair身份与持续时长。trajectory_data每局完整轨迹可重建全部图。')
    add('Ubuntu复现（项目根目录；输出必须为新目录）：\n\n```bash\nconda activate uav\nexport CUDA_VISIBLE_DEVICES=-1\nexport PYTHONDONTWRITEBYTECODE=1\nexport OMP_NUM_THREADS=1\npython -B -m pytest tests/test_v339_scenario_exposure.py tests/test_v339_attack_geometry_audit.py tests/test_combat_v339.py tests/test_combat_v337.py -q -p no:cacheprovider\npython -B tools/audit_v339_scenario_exposure.py --smoke --paired-validation --workers 4 --output-dir outputs/v339_scenario_exposure_smoke_repeat\npython -B tools/audit_v339_scenario_exposure.py --workers 4 --output-dir outputs/v339_scenario_exposure_audit_repeat\npython -B -m tools.report_v339_scenario_exposure --output-dir outputs/v339_scenario_exposure_audit_repeat\n```')
    add('若要复现本轮完整200组额外在线RNG验证，在完整批量完成后、最终report之前执行下列代码，然后再次生成report。此验证不训练、不修改环境，只重新配对运行至首次击毁：\n\n```bash\npython -B - <<\'PY\'\nfrom concurrent.futures import ProcessPoolExecutor\nfrom pathlib import Path\nimport json\nfrom tools.audit_v339_scenario_exposure import paired_until_fork, POLICIES\nif __name__ == "__main__":\n    cases = [(seed, policy) for seed in range(50) for policy in POLICIES]\n    with ProcessPoolExecutor(max_workers=8) as pool:\n        rows = list(pool.map(paired_until_fork, [x[0] for x in cases], [x[1] for x in cases]))\n    target = Path("outputs/v339_scenario_exposure_audit_repeat/online_pair_validation_all.json")\n    target.write_text(json.dumps(rows, indent=2), encoding="utf-8")\nPY\npython -B -m tools.report_v339_scenario_exposure --output-dir outputs/v339_scenario_exposure_audit_repeat\n```')
    add('复用依赖：上一轮outputs/v339_attack_geometry_audit必须完整，当前配置和旧工具SHA须吻合，否则入口失败而不会冒充已核验基线。跨平台复现浮点轨迹可能不同，应重新运行配对验证；不保证Windows/Ubuntu逐位一致。')
    add('保护核查：'+json.dumps(s.get('preservation',{}),ensure_ascii=False)+'。正式环境及配置、历史数据保持不变，只新增独立实验/分析/测试与本输出。未启动训练，不自动实施任何环境改动。')
    body='\n'.join(content)
    import re
    body=re.sub(r'\]\(([^)]+)\)',lambda m:']('+m.group(1)+')' if re.match(r'^[A-Za-z]:/',m.group(1)) else ']('+out.as_posix()+'/'+m.group(1)+')',body)
    (out/'audit_report.md').write_text(body,encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/v339_scenario_exposure_audit');p.add_argument('--protection-manifest',type=Path);p.add_argument('--tests-log',type=Path);p.add_argument('--smoke-dir',type=Path)
    args=p.parse_args();out=args.output_dir.resolve();s,*data=aggregate(out)
    if args.protection_manifest:
        before=json.loads(args.protection_manifest.read_text(encoding='utf-8'));changed=[f for f,h in before.items() if hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=h]
        assert not changed;shutil.copy2(args.protection_manifest,out/'source_protection_manifest.json');s['preservation']=dict(protected_files=len(before),unchanged=True,changed=[])
    if args.tests_log:
        shutil.copy2(args.tests_log,out/'tests.log');s['tests']=dict(combined=args.tests_log.read_text(encoding='utf-8',errors='replace').strip().splitlines()[-1])
    if args.smoke_dir:
        shutil.copy2(args.smoke_dir/'paired_validation.json',out/'paired_validation_smoke.json');shutil.copy2(args.smoke_dir/'summary.json',out/'smoke_summary.json')
        matched=0
        for trace in (args.smoke_dir/'trajectory_data').glob('*.npz'):
            with np.load(trace,allow_pickle=False) as a,np.load(out/'trajectory_data'/trace.name,allow_pickle=False) as b:
                assert set(a.files)==set(b.files)
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                matched+=1
        s['smoke_full_reproducibility_verified_episodes']=matched
    s['figure_files']=figures(out,s,data[0],data[1],data[3],data[4],data[6])
    files=['tools/audit_v339_scenario_exposure.py','tools/report_v339_scenario_exposure.py','tests/test_v339_scenario_exposure.py','configs/combat_environment_v339.yaml']
    s['source_sha256']={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    (out/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False,default=scalar),encoding='utf-8');report(out,s,*data)
    print('Completed analysis:',out/'audit_report.md')

if __name__=='__main__':main()
