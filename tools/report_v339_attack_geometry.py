"""Aggregate completed real trajectories and render the v3.39 attack audit."""
from pathlib import Path
import argparse,hashlib,json,math,os,re,shutil
import numpy as np
import pandas as pd
from tools.audit_v339_attack_geometry import ROOT,POLICIES,RANGES,vector_geometry,masks,write_csv

def json_scalar(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported JSON value {type(value)}')

def aggregate(out):
    s=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    if not s.get('completed'):raise ValueError('Cannot report a partial run as complete')
    ep=pd.read_csv(out/'episode_summary.csv');win=pd.read_csv(out/'engagement_windows.csv');inc=pd.read_csv(out/'incremental_opportunities.csv')
    rate=[]
    for (policy,side,radius),g in ep.groupby(['policy','side','range_m'],sort=False):
        row=dict(policy=policy,side=side,range_m=int(radius),episodes=len(g),live_pair_steps=int(g.live_pair_steps.sum()),live_agent_steps=int(g.live_agent_steps.sum()))
        for stage in ('distance','distance_ata','full','cone'):
            row['pair_'+stage]=int(g['pair_'+stage].sum())
            row['pair_'+stage+'_pooled_fraction']=float(g['pair_'+stage].sum()/g.live_pair_steps.sum())
            row['pair_'+stage+'_episode_mean']=float(g['pair_'+stage+'_fraction'].mean())
            row['pair_'+stage+'_episode_std']=float(g['pair_'+stage+'_fraction'].std(ddof=1)) if len(g)>1 else None
            row['agent_'+stage+'_pooled_fraction']=float(g['agent_'+stage].sum()/g.live_agent_steps.sum())
            row['agent_'+stage+'_episode_mean']=float(g['agent_'+stage+'_fraction'].mean())
            row['episode_has_'+stage+'_fraction']=float(g['episode_has_'+stage].mean())
            times=g['first_'+stage+'_time_s'].dropna()
            row['first_'+stage+'_time_median_s']=float(times.median()) if len(times) else None
            row['first_'+stage+'_time_observed_episodes']=len(times)
        rate.append(row)
    rates=pd.DataFrame(rate);rates.to_csv(out/'range_summary.csv',index=False,encoding='utf-8-sig')
    wr=[]
    for key,g in win.groupby(['policy','side','range_m','model','level'],sort=False):
        values=g.duration_s;row=dict(zip(['policy','side','range_m','model','level'],key))
        row.update(windows=len(values),total_s=float(values.sum()),minimum_s=float(values.min()),median_s=float(values.median()),mean_s=float(values.mean()),p90_s=float(values.quantile(.9)),
            gt_0_5_s_fraction=float((values>.5+1e-9).mean()),gt_1_s_fraction=float((values>1+1e-9).mean()),gt_2_s_fraction=float((values>2+1e-9).mean()))
        wr.append(row)
    ws=pd.DataFrame(wr);ws.to_csv(out/'window_summary.csv',index=False,encoding='utf-8-sig')
    outcomes=[]
    for policy in s['policies']:
        red=ep[(ep.policy==policy)&(ep.side=='red')&(ep.range_m==1000)];blue=ep[(ep.policy==policy)&(ep.side=='blue')&(ep.range_m==1000)]
        row=dict(policy=policy,episodes=len(red),red_win=int(red.red_win.sum()),blue_win=int(red.blue_win.sum()),draw=int(red.draw.sum()),timeouts=int(red.timeout.sum()),mean_seconds=float(red.seconds.mean()))
        for side,g in [('red',red),('blue',blue)]:
            for field in ('fire_attempts','weapon_hits','attack_kills','boundary_exits','ground_losses','ceiling_losses'):
                row[side+'_'+field]=int(g['actual_'+field].sum())
            row[side+'_boundary_episode_fraction']=float(((g.actual_boundary_exits+g.actual_ground_losses)>0).mean())
            for field in ('low_speed_fraction','heading_saturation_fraction','pitch_saturation_fraction'):row[side+'_'+field]=float(g[field].mean())
            row[side+'_survivors_mean']=float(g.survivors.mean())
        outcomes.append(row)
    outcome=pd.DataFrame(outcomes);outcome.to_csv(out/'policy_summary.csv',index=False,encoding='utf-8-sig')
    addition=inc.groupby(['policy','side','baseline_range_m','range_m'],sort=False)[['added_distance_pairs','added_full_pairs','added_rejected_by_angles']].sum().reset_index()
    addition['added_full_fraction_of_added_distance']=addition.added_full_pairs/addition.added_distance_pairs
    addition.to_csv(out/'range_increment_summary.csv',index=False,encoding='utf-8-sig')
    grid=pd.read_csv(out/'geometry_grid.csv');confusion=pd.crosstab(grid.current_angles,grid.cone_angles).reindex(index=[True,False],columns=[True,False],fill_value=0)
    s['geometry_confusion_angle_only']={'both':int(confusion.loc[True,True]),'current_only':int(confusion.loc[True,False]),'cone_only':int(confusion.loc[False,True]),'neither':int(confusion.loc[False,False])}
    s['geometry_confusion_by_range']={}
    for radius in RANGES:
        a,b=grid[f'current_{radius}'].astype(bool),grid[f'cone_{radius}'].astype(bool)
        s['geometry_confusion_by_range'][str(radius)]=dict(both=int((a&b).sum()),current_only=int((a&~b).sum()),cone_only=int((~a&b).sum()),neither=int((~a&~b).sum()))
    s['range_summary']=rate;s['window_summary']=wr;s['policy_summary']=outcomes;s['range_increment_summary']=addition.to_dict('records')
    # Every stored snapshot contributes to actual pitch/geometry and kinematic diagnostics.
    diagnostics=[];pitch_rows=[];checks=dict(trace_files=0,recorded_steps=0,reward_reconstruction_max_error=0.,action_bound_ok=True,ammo_cooldown_match_recorded_shots=True,
        pair_masks_nested=True,source_geometry_samples=0,max_source_geometry_error=0.)
    geometry_dir=out/'pair_geometry';geometry_dir.mkdir(exist_ok=True)
    from env.geometry import engagement_geometry
    from env.models import AircraftState
    for policy in s['policies']:
        for seed in s['seeds']:
            with np.load(out/'traces'/f'{policy}_seed{seed}.npz',allow_pickle=False) as data:
                pre=data['pre'];start=data['start'];post=data['post'];actions=data['actions'];n=len(pre)
                assert np.isfinite(pre).all() and np.isfinite(start).all() and np.isfinite(post).all()
                checks['trace_files']+=1;checks['recorded_steps']+=n
                assert len(data['rewards'])==n and data['done'][-1].any() and not data['done'][:-1].any()
                checks['action_bound_ok'] &= bool(np.isfinite(actions).all() and (abs(actions)<=1).all())
                shot=data['attempt_targets']>=0
                np.testing.assert_array_equal(data['ammo_pre']-data['ammo_post'],shot.astype(int))
                np.testing.assert_array_equal(data['cooldown_post'],np.where(shot,10,np.maximum(data['cooldown_pre']-1,0)))
                np.testing.assert_array_equal(pre[...,:6],post[...,:6])
                np.testing.assert_array_equal(start[1:],post[:-1])
                previously_dead=start[...,6]==0
                np.testing.assert_array_equal(start[...,:6][previously_dead],post[...,:6][previously_dead])
                # Independent pair rewards; compares actual pre-fire dense components.
                gs=[vector_geometry(pre,side) for side in (0,1)]
                # Persist every directed pair and every sample, not only aggregate windows.
                derived={name:np.stack([g[name] for g in gs],axis=1) for name in gs[0]}
                derived['time_s']=(np.arange(n)+1)*.1;derived['range_hypotheses_m']=np.asarray(RANGES)
                for stage in ('distance','distance_ata','full','cone'):
                    derived['mask_'+stage]=np.stack([np.stack([masks(g,r)[stage] for r in RANGES],axis=1) for g in gs],axis=1)
                derived['actual_1km_eligible']=derived['mask_full'][:,:,0]&(data['ammo_pre'][...,None]>0)
                np.savez_compressed(geometry_dir/f'{policy}_seed{seed}.npz',**derived)
                for side,g in enumerate(gs):
                    mm=[masks(g,r) for r in RANGES]
                    assert np.all(~mm[0]['full']|mm[1]['full']) and np.all(~mm[1]['full']|mm[2]['full'])
                    for frame in np.unique(np.linspace(0,n-1,5,dtype=int)):
                        a=AircraftState(*pre[frame,side,0,:6],alive=bool(pre[frame,side,0,6]));b=AircraftState(*pre[frame,1-side,0,:6],alive=bool(pre[frame,1-side,0,6]));actual=engagement_geometry(a,b)
                        circular=lambda x:abs(math.atan2(math.sin(x),math.cos(x)))
                        error=max(abs(actual.distance-g['distance'][frame,0,0]),circular(actual.ata-g['ata'][frame,0,0]),abs(actual.ha-g['ha'][frame,0,0]),circular(actual.aa-g['aa'][frame,0,0]))
                        checks['max_source_geometry_error']=max(checks['max_source_geometry_error'],float(error));checks['source_geometry_samples']+=1
                    alive=start[:,side,:,6]>0;turn=np.diff(np.concatenate((start[:1,side,:,5],post[:,side,:,5]),axis=0),axis=0)
                    turn=np.arctan2(np.sin(turn),np.cos(turn))
                    flips=(actions[1:,side,:,0]*actions[:-1,side,:,0]<-.25)&alive[1:]&alive[:-1]
                    total_heading=float(np.abs(turn[alive]).sum())
                    diagnostics.append(dict(policy=policy,seed=seed,side=('red','blue')[side],
                        absolute_heading_turns_sum=total_heading/(2*math.pi),max_agent_absolute_turns=float(np.max((abs(turn)*alive).sum(axis=0))/(2*math.pi)),
                        heading_sign_reversals=int(flips.sum()),minimum_speed_m_s=float(start[:,side,:,3][alive].min()),
                        maximum_abs_pitch_deg=float(np.degrees(abs(start[:,side,:,4][alive])).max()),
                        full_1km_pair_steps=int(mm[0]['full'].sum()),
                        full_1km_tail_pair_steps=int((mm[0]['full']&(abs(g['aa'])<=math.pi/6+1e-12)).sum()),
                        full_1km_headon_pair_steps=int((mm[0]['full']&(abs(g['aa'])>=5*math.pi/6-1e-12)).sum()),
                        red_guide_total=float(data['dense'][:,0].sum()) if side==0 else None,
                        red_offense_total=float(data['dense'][:,1].sum()) if side==0 else None,
                        red_defense_total=float(data['dense'][:,2].sum()) if side==0 else None,
                        agent_steps_without_geometry=int((g['live'].any(axis=2)&~mm[0]['full'].any(axis=2)).sum())))
                    for lo,hi in [(-60,-30),(-30,-5),(-5,5),(5,30),(30,60.0001)]:
                        binned_pitch=np.round(np.degrees(g['pitch']),6)
                        selected=g['live']&(binned_pitch>=lo)&(binned_pitch<hi)
                        pitch_rows.append(dict(policy=policy,seed=seed,side=('red','blue')[side],pitch_lo_deg=lo,pitch_hi_deg=hi,
                            live_pairs=int(selected.sum()),current_1km=int((mm[0]['full']&selected).sum()),cone_1km=int((mm[0]['cone']&selected).sum()),
                            current_only=int((mm[0]['full']&~mm[0]['cone']&selected).sum()),cone_only=int((mm[0]['cone']&~mm[0]['full']&selected).sum())))
                pair_rewards=[]
                for g in gs:
                    valid=g['live']&(abs(g['ata'])<=math.pi/6+1e-12)&(abs(g['ha'])<=math.pi/6+1e-12)
                    guide=np.where(valid,.001*np.clip((g['distance']-1000)/2000,0,1),0.)
                    offense=np.zeros_like(g['distance']);defense=offense.copy()
                    for deg,off,deff in ((30,.01,-.015),(15,.02,-.025),(5,.1,-.15)):
                        tier=g['live']&(abs(g['ata'])<=math.radians(deg)+1e-12)&(abs(g['ha'])<=math.radians(deg)+1e-12)
                        offense=np.where(tier,off,offense);defense=np.where(tier,deff,defense)
                    gate=(g['distance']<=3000)&(abs(g['aa'])<=math.pi/6+1e-12)
                    weight=np.clip((3000-g['distance'])/2000,0,1)
                    pair_rewards.append((guide,np.where(gate,offense*weight,0),np.where(gate,defense*weight,0)))
                expected=np.stack((pair_rewards[0][0].max(axis=2),pair_rewards[0][1].max(axis=2),pair_rewards[1][2].min(axis=1)),axis=1)
                checks['reward_reconstruction_max_error']=max(checks['reward_reconstruction_max_error'],float(abs(expected-data['dense']).max()))
    assert checks['reward_reconstruction_max_error']<1e-12 and checks['max_source_geometry_error']<1e-9
    assert checks['trace_files']==s['completed_episodes'] and checks['action_bound_ok']
    pd.DataFrame(diagnostics).to_csv(out/'maneuver_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pitch_rows).to_csv(out/'actual_pitch_geometry.csv',index=False,encoding='utf-8-sig');s['recording_checks']=checks
    (geometry_dir/'schema.json').write_text(json.dumps(dict(axes='sample, side(red=0/blue=1), attacker, target; mask arrays insert range after side',
        angles='ata,ha,aa,alpha,pitch in radians',distance='m',relative_speed='m/s',relative_velocity='NED m/s; final axis xyz',
        closing_speed='m/s; positive means approaching',live='both alive at true post-move pre-fire stage',time_s='post-move sample time',
        eligible='formal 1km geometry and pre-fire ammo; excludes cooldown and Blue target binding'),indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False,default=json_scalar),encoding='utf-8')
    return s,ep,win,rates,ws,outcome,addition

def figures(out,s,ep,win,rates,ws,outcomes):
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[]
    def save(name,fig):fig.tight_layout();fig.savefig(out/name,dpi=160);plt.close(fig);names.append(name)
    grid=pd.read_csv(out/'geometry_grid.csv');fig,axes=plt.subplots(2,4,figsize=(15,8))
    for ax,pitch in zip(axes.flat,[-60,-45,-30,0,30,45,60]):
        g=grid[(grid.pitch_deg==pitch)&(grid.requested_distance_m==500)].copy();g['difference']=g.current_angles.astype(int)-g.cone_angles.astype(int)
        pivot=g.pivot(index='requested_ha_deg',columns='requested_ata_deg',values='difference')
        ax.imshow(pivot,origin='lower',vmin=-1,vmax=1,cmap='coolwarm',aspect='auto')
        ax.set(xticks=np.arange(7),xticklabels=[-45,-30,-15,0,15,30,45],yticks=np.arange(5),yticklabels=[-45,-30,0,30,45],xlabel='ATA grid (deg)',ylabel='LOS HA grid (deg)',title=f'Pitch={pitch:+d} deg; d=500m')
    axes.flat[-1].axis('off');axes.flat[-1].text(.05,.8,'Red: ATA/HA only\nBlue: 3D cone only\nLight grey: agreement\n\n7 pitches, 35 angular cells each\nAngles use 1e-12 rad tolerance',va='top')
    save('geometry_difference_heatmap.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5));g=grid[grid.requested_distance_m==500].groupby('pitch_deg')[['current_angles','cone_angles']].mean()
    axes[0].plot(g.index,100*g.current_angles,'o-',label='ATA/HA');axes[0].plot(g.index,100*g.cone_angles,'s-',label='3D cone');axes[0].set(xlabel='Pitch (deg)',ylabel='Angular cells accepted (%)',title='Static grid: 35 equally weighted cells / pitch')
    actual=pd.read_csv(out/'actual_pitch_geometry.csv').groupby(['pitch_lo_deg','pitch_hi_deg'])[['live_pairs','current_1km','cone_1km']].sum().reset_index();active=actual.live_pairs>0;actual=actual[active];x=np.arange(len(actual))
    axes[1].bar(x-.18,100*actual.current_1km/actual.live_pairs,.36,label='ATA/HA');axes[1].bar(x+.18,100*actual.cone_1km/actual.live_pairs,.36,label='3D cone')
    axes[1].set(xticks=x,xticklabels=[f'[{a:g},{min(b,60):g})' for a,b in zip(actual.pitch_lo_deg,actual.pitch_hi_deg)],xlabel='Attacker pitch bin (deg)',ylabel='Live pair-steps accepted (%)',title=f'Actual trajectories: {s["completed_episodes"]} episodes, 1km')
    axes[1].set_xticklabels([f'[{a:g},{min(b,60):g})\nn={n:,}' for a,b,n in zip(actual.pitch_lo_deg,actual.pitch_hi_deg,actual.live_pairs)],fontsize=9)
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    save('pitch_geometry_comparison.png',fig)
    fig,axes=plt.subplots(2,4,figsize=(19,8))
    for row,side in enumerate(('red','blue')):
        for col,policy in enumerate(s['policies']):
            ax=axes[row,col];g=rates[(rates.policy==policy)&(rates.side==side)].sort_values('range_m');x=np.arange(3)
            for k,(field,label) in enumerate([('distance','Distance'),('distance_ata','+ATA'),('full','+HA'),('cone','3D cone')]):
                ax.bar(x+(k-1.5)*.19,100*g['pair_'+field+'_pooled_fraction'],.19,label=label)
            ax.set(xticks=x,xticklabels=['1','2','4'],xlabel='Offline range (km)',ylabel='Live pair-step (%)',title=f'{policy}, {side}, n={int(g.episodes.iloc[0])} episodes');ax.grid(axis='y',alpha=.2)
            if row==0 and col==0:ax.legend(fontsize=8)
    save('range_opportunity_funnel.png',fig)
    fig,axes=plt.subplots(2,4,figsize=(19,8))
    for row,side in enumerate(('red','blue')):
        for col,policy in enumerate(s['policies']):
            ax=axes[row,col]
            for radius in RANGES:
                v=np.sort(win[(win.policy==policy)&(win.side==side)&(win.range_m==radius)&(win.model=='full')&(win.level=='agent')].duration_s.to_numpy())
                if len(v):ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=f'{radius/1000:g}km, {len(v)} windows')
            ax.set(xlabel='Agent-level window duration (s)',ylabel='Empirical CDF',title=f'{policy}, {side}; 50 episodes');ax.legend(fontsize=7);ax.grid(alpha=.2)
    save('window_duration_cdf.png',fig)
    fig=plt.figure(figsize=(15,11))
    for index,policy in enumerate(s['policies']):
        ax=fig.add_subplot(2,2,index+1,projection='3d');data=np.load(out/'traces'/f'{policy}_seed0.npz');tr=np.concatenate((data['start'][:1],data['post']),axis=0)
        for side,color in ((0,'#b43232'),(1,'#2477ac')):
            for agent in range(4):
                alive=tr[:,side,agent,6]>0;dead=np.flatnonzero(~alive);end=int(dead[0]) if len(dead) else len(tr)-1;p=tr[:end+1,side,agent]
                ax.plot(p[:,0],p[:,1],-p[:,2],color=color,label=('Red' if side==0 else 'Blue') if agent==0 else None)
                ax.scatter(p[-1,0],p[-1,1],-p[-1,2],color=color,marker='x',s=25)
        ax.set(xlabel='North x (m)',ylabel='East y (m)',zlabel='Altitude (m)',title=f'{policy}, actual seed=0, original 1km weapon');ax.set_box_aspect(None,zoom=.78);ax.legend(fontsize=8)
    save('fixed_policy_3d_trajectories.png',fig)
    fig,axes=plt.subplots(1,2,figsize=(16,6));g=ep[ep.range_m==1000]
    fields=['agent_live','agent_distance','agent_distance_ata','agent_full','actual_eligible_agent_steps','actual_ready_agent_steps','actual_bound_ready_agent_steps','actual_fire_attempts','actual_weapon_hits','actual_attack_kills']
    labels=['Live agent-step','Distance','+ATA','+HA','Ammo','Cooldown','Target binding','Shots','Hits','Kills']
    for ax,side in zip(axes,('red','blue')):
        h=g[g.side==side];v=[h[f].sum() for f in fields];ax.barh(np.arange(len(v)),v,color='#3274a1');ax.set(yticks=np.arange(len(v)),yticklabels=labels,xlabel='Counts (agent-steps, then actual events)',title=f'{side}: {s["completed_episodes"]} actual 1km episodes');ax.invert_yaxis()
        for i,x in enumerate(v):ax.text(x,i,f' {int(x)}',va='center',fontsize=8)
        ax.set_xlim(0,max(v)*1.17)
    save('actual_shooting_funnel.png',fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5));x=np.arange(len(s['policies']));order=outcomes.set_index('policy').loc[list(s['policies'])]
    for side,shift in (('red',-.18),('blue',.18)):
        axes[0].bar(x+shift,100*order[side+'_low_speed_fraction'],.36,label=side)
        axes[1].bar(x+shift,100*order[side+'_boundary_episode_fraction'],.36,label=side)
        r=rates[(rates.range_m==1000)&(rates.side==side)].set_index('policy').loc[list(s['policies'])]
        axes[2].bar(x+shift,100*r.agent_full_episode_mean,.36,label=side)
    for ax,title in zip(axes,['Speed <=160 m/s, living start-agent steps','Episodes with boundary or ground loss','1km agent opportunity, episode equal mean']):
        ax.set(xticks=x,xticklabels=s['policies'],ylabel='Percent (%)',title=title);ax.legend();ax.grid(axis='y',alpha=.2)
    for ax in axes[:2]:
        ax.set_ylim(0,1 if max(p.get_height() for p in ax.patches)==0 else None)
        if max(p.get_height() for p in ax.patches)==0:ax.text(.5,.5,'Red = Blue = 0% in all four groups\n50 episodes per group',transform=ax.transAxes,ha='center')
    save('maneuver_risk_opportunities.png',fig)
    fig,ax=plt.subplots(figsize=(9,5));p=pd.read_csv(out/'hit_probability.csv')
    for (ata,ha),g in p.groupby(['ata_deg','ha_deg']):ax.plot(g.distance_m,g.probability,'o-',label=f'ATA={ata} deg, HA={ha} deg')
    ax.axvline(1000,color='black',ls='--',label='Formal range limit');ax.set(xlabel='Distance (m)',ylabel='Conditional hit probability',title='Frozen current parameters; >1km is offline extrapolation');ax.legend(fontsize=8);ax.grid(alpha=.2)
    save('conditional_hit_probability.png',fig)
    return names

def markdown_table(frame,columns=None):
    frame=frame if columns is None else frame[columns]
    def cell(x):
        if x is None or (isinstance(x,(float,np.floating)) and not np.isfinite(x)):return '—'
        if isinstance(x,(float,np.floating)):return f'{x:.6e}' if 0<abs(x)<1e-8 else f'{x:.6f}'
        return str(x).replace('|','/')
    return '| '+' | '.join(frame.columns)+' |\n|'+'|'.join(['---']*len(frame.columns))+'|\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'

def report(out,s,ep,win,rates,ws,outcomes,addition):
    text=[]
    def add(x=''):text.append(x+'\n')
    def table(x,cols=None):add(markdown_table(x,cols))
    def link(label,path):return f'[{label}]({(out/path).as_posix()})'
    add('# v3.39 三维攻击几何、交战窗口及固定规则机动专项审计')
    add(f'本次实际完成{len(s["policies"])}组×{len(s["seeds"])} seeds={s["completed_episodes"]}局正式配置对局，另有8局smoke和2个受控目标选择案例。seeds={s["seeds"][0]}–{s["seeds"][-1]}。实际累计{int(s["recording_checks"]["recorded_steps"])}个环境step。全程CPU、Windows PowerShell uav（WSL Ubuntu访问E_ACCESSDENIED）；未使用任何checkpoint或启动RL训练。所有真实射击均来自原1km环境，2/4km只作相同状态下离线反事实计算。')
    add('## 一、源码、论文与执行时序')
    sources=['configs/combat_environment_v339.yaml','env/combat_v339.py','env/combat_v332.py','env/v311_kinematics.py','env/geometry.py','env/v336_weapon.py','env/v337_reward.py','env/v338_observation.py','env/v339_policy.py','env/v310_policy.py','env/v330_transition.py','env/combat_v326.py','env/models.py','env/combat_v330.py','env/combat_v323.py','env/v322_reward.py','env/v334_scenario.py','tools/audit_v339_single_aircraft.py','tests/test_v339_single_aircraft_audit.py']
    add('实际读取的主要实现：'+ '、'.join(f'[{p}]({ROOT.as_posix()}/{p})' for p in sources)+'。同时核对docs的v339/v336/v337说明、已有weapon/reward审计与测试。')
    add('v339→v338→v337→v336→v335→v334→v332继续使用直接三维运动学。action=[航向增量,俯仰增量,速度增量]，clip[-1,1]；dt=.1s；45°/s、30°/s、±25m/s²；v∈[150,300]、pitch∈[−60°,60°]。速度和角度先更新，位置用新状态积分；没有PID、滚转或过载动态。本轮不重做或改写上一轮运动学结论。')
    add('令Δp=(dx,dy,dz)，ρ=sqrt(dx²+dy²)，d=||Δp||，β=atan2(dy,dx)。当前精确公式：\n\n```text\nATA = wrap(β - ψ_attacker)\nHA  = atan2(-dz, ρ)\nAA  = wrap(ψ_target - β)\nα3D = acos(clip(v_hat · r_hat,-1,1))\ncos α3D = sinθ sinHA + cosθ cosHA cosATA\n```')
    add('ATA/AA是水平角，HA是LOS相对世界水平面的高度角；HA不减本机pitch。AA描述目标航向与攻击者指向目标的水平LOS关系，AA小对应本机大体朝向目标尾部的几何，并不是3D前向夹角。v338敌机观测使用同一个AA/ATA/HA，own pitch另有自身观测槽。α3D只是审计对照，不替换正式武器。ρ=0时β回退本机ψ；d=0时formal off_boresight定义0，HA=atan2(0,0)=0；这是退化点约定，并不能说真实零距离视线物理可定义。')
    paper=next(ROOT.glob('Li*2023*.pdf'))
    add(f'论文原件：[{paper.name}]({paper.as_posix()})。已用本地Poppler重新渲染并目视核对第6页Eq.(6)/Fig.2和第7页Eq.(7)/(8)/Fig.4，页面保存在'+link('page6','paper_reference-06.png')+'、'+link('page7','paper_reference-07.png')+'。论文将HA用于描述垂直LOS位置关系，并以独立ATA、HA阈值定义攻击，而不是单一3D圆锥。因此pitch不直接进入相同位置下的HA，与此建模语义相符；不能直接报成程序bug。signed wrap、1km、6发、1秒冷却、数值命中标定及项目距离奖励均须与完整论文复现区分。')
    add('真实时序：①步首Blue最近存活Red目标及自动动作（各一次team_actions）；②双方动作clip并屏蔽死亡slot；③双方移动；④非战斗损失；⑤dense reward；⑥统计武器资格；⑦Red/Blue分别扣冷却、选择目标、扣弹药、各次射击取一次共享高斯；⑧同时汇集双方attempts后结算死亡；⑨奖励/终止/info。prefire观察放在⑤原dense调用返回处，因此记录的是移动后、非战斗死亡后、双方尚未射击的同一快照。Red已扣弹后Blue才扣弹，但此间无状态死亡；双方开火机会以同一prefire alive为准。')
    add('资格：双方存活、ammo>0、0≤d≤1000m、|ATA|≤30°、|HA|≤30°。距离严格包含端点，无额外距离容差；角度继承1e−12 rad舍入容差。资格不含冷却。每步先将冷却max(c−1,0)，再判断是否为0，所以prefire记录c≤1可发射；发射后置10步。Red每机最多1发，选择最近合格目标，距离/index打破平局；Blue只准攻击步首绑定的最近追击目标，不会因它不可攻击而改射另一个可攻击目标。Blue速度远/中/近为240/220/200m/s。')
    add('## 二、实验协议、统计单位与验证')
    add('四组共用原config、初始化、Blue策略、1000步/100s上限和seeds0–49：zero为全零动作；pursuit按最近存活Blue的3D LOS逐步控制，速度目标固定225m/s；lateral与pursuit相同，但12≤t<20s时desired heading=LOS+30°；vertical同样追击航向、固定225m/s，8≤t<12s期望pitch+30°、16≤t<20s期望−30°，其余期望0°，含有限速改平。所有命令除以当前每步增量后clip，通过正常action接口执行，没有设置飞行状态。若对局提前结束，未执行的后段动作不计作完成。zero不自动改平；本协议初始pitch恰好为0。')
    add('默认reset实际只随机旋转一组点对称的同高度迎头4v4编队：锚点相距8km，初高3000m、初速225m/s、pitch0。seeds同时影响武器RNG，但不是50种独立复杂战术场景。不同Red动作会改变路径和取随机数次数，不能说不同策略的逐发命中随机数严格配对；本报告的射程比较则严格复用每局同一条已实现轨迹。')
    add('pair比例分母=双方在prefire均存活的有向敌对pair-step，含弹药耗尽/冷却中的几何机会；agent比例分母=存活攻击者且至少有一名存活对手的agent-step，分子=该步至少一个目标满足条件。episode比例以每组50局为分母。另报告每局自身比例的等权均值和标准差，避免把step当独立实验样本。pooled是曝光加权描述，不是统计显著性检验。首次时间是该局首次满足条件的postmove采样时刻；未发生为缺失，不填0，汇总中位数只在发生的局中计算，旁列局数。')
    add('连续窗口按离散采样定义：同一攻击者至少有一个满足条件的目标构成agent窗口，可在窗口内切换目标；pair窗口要求固定目标。first/end时间分别为首/末满足条件样本时刻，duration=样本数×dt，单步窗口为.1s，因此duration=end−start+dt。不能据此保证两个采样时刻之间每一瞬间都满足条件。局终止、死亡或无机会即结束，不补出终止后的轨迹，不混合不同episode。窗口分布以窗口为描述单位，不视为独立实验重复；episode等权结果单独给出。')
    add('每局traces/*.npz保存start/pre/post状态（含alive）、执行动作、引导最近目标/实际Blue绑定目标、ammo/cooldown的pre/post值、actual attempt target/hit、每个目标的credited killers、死亡原因、dense reward与终止标记。pair_geometry/*.npz由同一真实prefire状态确定性派生，完整保存每step双方16个有向pair的距离、ATA/HA/AA/α、pitch、相对速度向量/模、closing speed、alive及三种射程的各层mask，含实际1km弹药资格。schema.json给出轴和单位，角度为rad。列顺序state=[x,y,z,v,theta,psi,alive]，NED z=-height；axis side0=Red/1=Blue。死亡cause0无、1地面、2水平、3天花板、4武器；ceiling已含于boundary，不重复算损失。相对速度报告||v_target−v_own||，closing speed=−v_rel·LOS，后者正值接近。zero组red_targets只是最近目标的诊断，并不表示zero策略主动追击。')
    for name,entry in s.get('tests',{}).items():add(f'测试 {name}：{entry["summary"]}。')
    add('新增87项测试覆盖独立角度/球面恒等式、同位不同pitch、距离/角度边界、退化LOS、共享高斯积分、Blue绑定、Red多目标、同step同时击毁的prefire alive、ammo/cooldown与几何分离、分母/目标切换窗口、三类非战斗死亡，以及四策略各一整局带/不带观察器状态/奖励/action/计数/策略diagnostics/ammo/cooldown/RNG逐步完全一致。相关现有317项通过；本轮现有测试无失败，没有改历史预期。')
    add('全批量离线核对：'+json.dumps(s['recording_checks'],ensure_ascii=False)+'。新增只读观察的原方法调用次数在每局断言为每step各一次dense/resolve；不调用attempt_hit/策略以便“补记录”，不额外抽样。实际ammo减少量逐step等于shots，冷却更新逐step吻合；死亡飞机坐标冻结，非战斗/武器死亡与结局守恒；真实prefire奖励按独立pair公式重建。')
    add('## 三、实验A：静态三维几何')
    add(f'网格共{s["static"]["grid_cases"]}项：7种pitch（含补充±45°）×5种HA×7种ATA×5种距离。最大独立公式误差{s["static"]["max_formula_error"]:.3g}；这是所选离散格点的计数，不能解释为真实策略出现概率。角度混淆矩阵暂不加距离门限，每个角度格点在5种距离重复：')
    c=s['geometry_confusion_angle_only'];table(pd.DataFrame([['ATA/HA满足',c['both'],c['current_only']],['ATA/HA不满足',c['cone_only'],c['neither']]],columns=['当前模型','3D前向满足','3D前向不满足']))
    add('在相同距离门限下的完整条件混淆计数（超范围两者均不满足，不应误读成角度一致性提高）：')
    table(pd.DataFrame([dict(range_m=r,**v) for r,v in s['geometry_confusion_by_range'].items()]))
    add('平飞时cosα=cosHA cosATA，所以矩形角门限的角落会超出30°圆锥：ATA=HA=30°得到α=41.409622°。同高度且ATA0时，pitch0/±45°分别α0/45°，但ATA=HA=0、正式武器资格和条件命中率均不变。pitch=HA=45°且ATA0时α0，但HA45°违反正式门限；纯正上/正下方HA≈±90°被拒绝。俯仰既不是无用动作，也不是机头圆锥武器的直接瞄准控制：它通过速度方向改变后续位置、LOS高度角、距离和敌方几何。')
    cases=pd.DataFrame(json.loads((out/'representative_geometry_cases.json').read_text(encoding='utf-8')))
    table(cases,['case','distance_m','ata_deg','ha_deg','aa_deg','alpha3d_deg','formal_1km_eligible'])
    add('各例准确六维初态、1/2/4km判定保存在'+link('representative_geometry_cases.json','representative_geometry_cases.json')+'；d=1000的浮点球坐标构造可能产生1e−13m级偏差，CSV同时保留requested与实际d，以严格实际距离比较为准。边界测试使用明确的轴向状态或超差偏移，未人为放宽正式距离门限。实际pitch分箱仅先按1e−6°舍入避免±30°处浮点尾数制造极小伪分箱；全部正式资格和角度记录仍使用原始数值，图中n为每箱live pair-step数。')
    add('!['+'静态差异热图'+'](geometry_difference_heatmap.png)\n\n![俯仰分层](pitch_geometry_comparison.png)')
    add('## 四、实验B：同轨迹1/2/4km交战窗口')
    add('下表ratio均为0–1的小数；pair为pooled曝光比例，episode_mean为每局比例等权均值。所有range_m=2000/4000的数值均为反事实几何；没有重新开火、杀死飞机或延长实际轨迹。')
    table(rates,['policy','side','range_m','episodes','pair_distance_pooled_fraction','pair_distance_ata_pooled_fraction','pair_full_pooled_fraction','pair_full_episode_mean','agent_full_episode_mean','episode_has_full_fraction'])
    add('首次进入距离/完整条件的时间（秒；中位数只在发生者内计算，count保留其分母）：')
    table(rates,['policy','side','range_m','first_distance_time_median_s','first_distance_time_observed_episodes','first_full_time_median_s','first_full_time_observed_episodes'])
    add('agent-level正式角度模型的连续窗口分布：')
    selected=ws[(ws.model=='full')&(ws.level=='agent')]
    table(selected,['policy','side','range_m','windows','minimum_s','median_s','mean_s','p90_s','gt_0_5_s_fraction','gt_1_s_fraction','gt_2_s_fraction'])
    add('完整3D-cone对照及pair-level固定目标窗口分布见'+link('window_summary.csv','window_summary.csv')+'；逐窗口详情见'+link('engagement_windows.csv','engagement_windows.csv')+'。目标切换不会把agent窗口误称固定目标锁定；targets_seen保留切换标识。episode_summary另有每局window count/total_s，不把少数长局的大量step冒充更多实验重复。')
    add('!['+'射程分层机会'+'](range_opportunity_funnel.png)\n\n![窗口时长CDF](window_duration_cdf.png)')
    add('距离增量与角度筛选可精确拆分：新增距离pair数ΔD=新增完整机会ΔG+新增距离内被ATA/HA排除的pair数。由于角度门限在三种range中没有改变，跨range的ΔG全部来自放宽距离条件后原本就满足角度的状态；“角度贡献”应理解为新增距离段的保留率ΔG/ΔD，而不能说角度本身被改进。')
    table(addition,['policy','side','baseline_range_m','range_m','added_distance_pairs','added_full_pairs','added_rejected_by_angles','added_full_fraction_of_added_distance'])
    add('新增机会逐局的距离、ATA/HA、α、pitch、相对速度和closing speed的p10/median/p90保存在'+link('incremental_opportunities.csv','incremental_opportunities.csv')+'，同时包含1→2、1→4、2→4km；无新增者为缺失。它们描述已实现状态的机会分布，不用于预测新增命中/胜率。')
    add('## 五、实验C：命中概率及奖励一致性')
    add('每次attempt只取一个ε~N(0,1)，同时用于ATA与HA，不能把两角的单独概率相乘。h(d)=πexp(−d/558.1106265512473)，c4=c5=.9984711359155588。将两个不等式各化为ε的区间，交集为空则P=0；否则P=Φ(min upper)−Φ(max lower)。本工具另用64点Gauss-Legendre对标准正态密度做确定性积分交叉核对expected_hit_probability，30个概率场景误差<1e−14，没有随机抽样。')
    table(pd.read_csv(out/'hit_probability.csv'),['distance_m','ata_deg','ha_deg','probability','formal_range'])
    add('1500/2000/4000m只是保持当前全部函数/数值参数的离线条件计算，正式环境不在这些距离开火；不代表论文的2/4km标定。相同符号的ATA/HA可共享同一ε抵消，两角异号且接近门限则可能区间无交叠。1000m时(ATA,HA)=(30°,30°)的P≈.352865，而(30°,−30°)仅约8.87e−17；两者几何门限都满足。这种符号组合的概率差异由共享噪声数学结构造成，数值函数执行正确，不是把现有模型算错。它是否诱发上下/转向组合偏好尚需实际policy验证，不能静默改成独立噪声。')
    add('奖励使用同一ATA/HA/AA：w3=clip((d−1000)/2000,0,1)、w4=clip((3000−d)/2000,0,1)。引导=.001w3且ATA/HA≤30°，没有3km硬下限；进攻在d≤3000且|AA|≤30°时按ATA/HA的30/15/5°取.01/.02/.1再乘w4；防御以反向Blue中心重算几何，系数−.015/−.025/−.15，取最强威胁。Red跨目标聚合max引导、max进攻、min防御。1–3km连续交接、3km以上仍有引导是有意塑形，与1km发射资格不同不构成bug；AA也只在战术塑形中有额外门限，武器没有AA资格门限。')
    table(pd.read_csv(out/'reward_geometry.csv'),['distance_m','ata_deg','ha_deg','aa_deg','pitch_deg','guide','offense','defense','formal_eligible'])
    add(f'真实全部轨迹dense重建最大误差{s["recording_checks"]["reward_reconstruction_max_error"]:.3g}。奖励和武器共享“世界HA不扣pitch”的语义，未发现二者代码角度输入冲突；但两者也都不直接奖励/约束3D机头对准，若研究目标要求前向武器则是任务定义差异。')
    add('![条件命中概率](conditional_hit_probability.png)')
    add('## 六、实验D：固定策略实际对局')
    table(outcomes,['policy','episodes','red_win','blue_win','draw','timeouts','mean_seconds','red_fire_attempts','red_weapon_hits','red_attack_kills','blue_fire_attempts','blue_weapon_hits','blue_attack_kills'])
    table(outcomes,['policy','red_survivors_mean','blue_survivors_mean','red_boundary_exits','blue_boundary_exits','red_ground_losses','blue_ground_losses','red_ceiling_losses','blue_ceiling_losses'])
    table(outcomes,['policy','red_low_speed_fraction','blue_low_speed_fraction','red_heading_saturation_fraction','blue_heading_saturation_fraction','red_pitch_saturation_fraction','blue_pitch_saturation_fraction'])
    add('这些是原1km实际发射、命中和击毁，shots/hits/kills分开统计；多个命中可共同命中同一个目标，hit不保证增加一次击毁。真实射击漏斗在agent-step尺度列出距离、ATA、HA、弹药、冷却、Blue目标绑定，再接actual shots/hits/kills事件数量，避免把pair机会直接除成射击转化率。')
    funnel=ep[ep.range_m==1000].groupby(['policy','side'],sort=False)[['agent_live','agent_distance','agent_distance_ata','agent_full','actual_eligible_agent_steps','actual_ready_agent_steps','actual_bound_ready_agent_steps','actual_fire_attempts','actual_weapon_hits','actual_attack_kills']].sum().reset_index();table(funnel)
    add('![实际射击漏斗](actual_shooting_funnel.png)\n\n![真实三维轨迹](fixed_policy_3d_trajectories.png)\n\n![低速和边界](maneuver_risk_opportunities.png)')
    add('绕后不是“转弯动作已执行”就算成功：最近目标纯追击及固定30°侧偏并不具有尾追转换保证。报告不把lateral探针包装成高级战术；真正AA占位变化、累计转向和无几何机会agent-step见maneuver_diagnostics与原始trace。低速阈值定义≤160m/s，本组Red固定225、Blue最低目标200，因此没有探测主动贴150m/s的策略收益；低速占比为0不能证明不存在RL低速盘旋偏好。')
    maneuver=pd.read_csv(out/'maneuver_diagnostics.csv')
    aspect=maneuver.groupby(['policy','side'],sort=False)[['full_1km_pair_steps','full_1km_tail_pair_steps','full_1km_headon_pair_steps']].sum().reset_index()
    aspect['tail_fraction_of_full']=aspect.full_1km_tail_pair_steps/aspect.full_1km_pair_steps
    aspect['headon_fraction_of_full']=aspect.full_1km_headon_pair_steps/aspect.full_1km_pair_steps
    add('绕后/迎头的实际证据：tail定义完整1km几何且|AA|≤30°，headon定义完整1km几何且|AA|≥150°；都是有向pair-step描述，不宣称持续尾追锁定。')
    table(aspect)
    add('实际Red dense reward累计量（各50局，完整开火前状态；不含击杀/损失事件奖励）：')
    table(maneuver[maneuver.side=='red'].groupby('policy',sort=False)[['red_guide_total','red_offense_total','red_defense_total']].sum().reset_index())
    add('武器允许迎头开火，而进攻/防御战术塑形另要求AA≤30°，因此迎头有效射击不必得到尾部占位奖励。这是明确的塑形目标差异，不能直接当几何输入不一致。默认快杀迎头场景是否使绕后塑形缺少曝光，是比“不能转弯”更直接的待检机制。')
    add('受控案例：多合格目标时Red只射最近者；Blue步首绑定不可攻击目标、另一目标可攻击时仍不射。准确移动后状态与动作、角度和实际发射目标如下；这些人工初态只用于解释机制，不计入200局默认reset统计：')
    controlled=pd.DataFrame(json.loads((out/'controlled_target_cases.json').read_text(encoding='utf-8')));table(controlled,['case','side','target','distance_m','ata_deg','ha_deg','aa_deg','alpha3d_deg','geometric_1km','blue_step_target','actual_fire_target','actual_hit'])
    add('准确状态及每个case的一步trace见controlled_target_cases.json及两个具名npz。死亡原因/终止及cooldown断言详见专项测试；不能用step返回后的alive代替以上prefire alive，互相同时击毁的测试明确保留双方当步几何机会。')
    add('## 七、七个研究问题的直接回答')
    add('### 问题1：ATA/HA与三维方向是否存在容易诱发非预期策略的差异？')
    add(f'有明确的几何语义差异：静态格点current-only={c["current_only"]}、cone-only={c["cone_only"]}，同高pitch±45°仍可ATA=HA=0开火。它允许相对水平面有利而机头不直接指向目标的状态；反向也会排除机头对准的陡升目标。该差异与论文独立角定义和当前配置相符，属于建模语义，尚未证明被任何学习policy利用。真实pitch分层计数另存actual_pitch_geometry.csv，避免把静态格点频率当实战频率。')
    add('### 问题2：三种射程的机会与窗口？')
    for radius in RANGES:
        r=ep[(ep.range_m==radius)&(ep.side=='red')];v=win[(win.range_m==radius)&(win.side=='red')&(win.model=='full')&(win.level=='agent')].duration_s
        add(f'{radius/1000:g}km：混合四策略的Red有效pair={int(r.live_pair_steps.sum())}，距离合格{int(r.pair_distance.sum())}，完整几何{int(r.pair_full.sum())}（pooled {r.pair_full.sum()/r.live_pair_steps.sum():.4%}）；episode等权pair均值{r.pair_full_fraction.mean():.4%}，agent均值{r.agent_full_fraction.mean():.4%}；agent窗口{len(v)}个，中位{v.median():.3f}s、均值{v.mean():.3f}s、p90 {v.quantile(.9):.3f}s。跨策略混合仅作总览，各策略/双方细分表是主要证据。')
    add('### 问题3：4km增量多少来自距离、多少被角度筛掉？')
    for lower in (1000,2000):
        a=addition[(addition.baseline_range_m==lower)&(addition.range_m==4000)&(addition.side=='red')]
        add(f'Red {lower/1000:g}→4km：新增距离pair {int(a.added_distance_pairs.sum())}，其中新增完整几何{int(a.added_full_pairs.sum())}，角度排除{int(a.added_rejected_by_angles.sum())}，保留率{a.added_full_pairs.sum()/a.added_distance_pairs.sum():.4%}。角度门限未改变；所有新增完整机会发生在被新增距离覆盖的相同真实状态。')
    add('### 问题4：1km下纯追击能否形成真实攻击，瓶颈是什么？')
    p=ep[(ep.policy=='pursuit')&(ep.side=='red')&(ep.range_m==1000)];v=win[(win.policy=='pursuit')&(win.side=='red')&(win.range_m==1000)&(win.model=='full')&(win.level=='agent')].duration_s
    add(f'在本50局迎头编队协议中，Red纯追击{int((p.actual_fire_attempts>0).sum())}/{len(p)}局发生实际射击，共{int(p.actual_fire_attempts.sum())}发、{int(p.actual_weapon_hits.sum())}次命中、{int(p.actual_attack_kills.sum())}次击毁；有完整几何的episode比例{p.episode_has_full.mean():.2%}，agent窗口中位{v.median():.3f}s，≤.5s窗口占比{(v<=.5+1e-9).mean():.2%}。这直接反驳“本场景1km根本无法形成攻击”的强判断，但不定义训练所需的“足够”。')
    add('具体瓶颈按上文漏斗与时序区分：远段接近过程无距离资格；进入后ATA/HA、cooldown和ammo再筛选，Blue另有目标绑定；射击后的条件概率与多机同目标造成hits/kills差异。不能把所有live-step无攻击都称机动失败，因为8km迎头接近本就需要时间。也不能由增加远距离几何机会推断真正4km对局胜率。')
    add('### 问题5：高转向率、低速半径及俯仰是否表现为明显异常？')
    diag=pd.read_csv(out/'maneuver_diagnostics.csv');add(f'本200局边界损失总计Red={int(outcomes.red_boundary_exits.sum())}、Blue={int(outcomes.blue_boundary_exits.sum())}，地面Red={int(outcomes.red_ground_losses.sum())}、Blue={int(outcomes.blue_ground_losses.sum())}；实际最小存活速度{diag.minimum_speed_m_s.min():.1f}m/s、最大绝对pitch {diag.maximum_abs_pitch_deg.max():.3f}°，单机最多累计绝对转向{diag.max_agent_absolute_turns.max():.3f}圈。未发现NaN、瞬移或动作路径异常；这不等于机动能力已物理合理。')
    add('四组大多在首次接近附近就被真实武器结算，垂直动作序列的后半段可能未到执行时刻，边界/长追击样本受到终止截断；本批次不能证明更长对局或近边界初态没有问题。150m/s低速盘旋未主动覆盖，不能用本组零低速率排除该机制。高等效机动量级的数学事实沿用上一轮，但其RL因果影响未知。')
    add('### 问题6：有证据必须引入PID或复杂动力学吗？')
    add('没有。当前正常action控制可完成接近、有限速转向、爬升/改平/下降的已执行部分及真实攻击；已有运动学检查与本轮记录验证没有执行不稳定证据。控制器不能自行解决世界HA与3D瞄准语义、共享噪声概率或默认迎头场景单一的问题；引入它会改变研究任务，非本次结论。')
    add('### 问题7：优先验证哪个单一因素？')
    add('最优先验证“初始相对航向/迎头交战几何”这一单一因素：在独立后续试验中只改变初始相对航向，冻结位置、速度、高度、Blue追击、武器、reward和运动学，观测是否仍快速回到迎头、尾部占位与R4是否获得实际曝光。即使纯追击抹去了初始航向差异，这也是应量化的结果，不能提前保证会改善。依据是本200局全组进攻/防御R4均为0、合格1km几何的tail pair-step均为0，默认seed主要只是旋转同一迎头布局，而不是已证明机动控制做不到。')
    add('攻击角度语义是已证实的另一建模差异，应保留为后续独立因素，不与场景或射程同时修改；本批1km实际对照仅vertical-Red有58个current-only pair-step，其他组/方向无差异，不能用静态极端格点夸大它在当前批次的主导性。更大俯仰/高度差覆盖可在场景专项之后另行设计。当前1km已能真实交战，单凭4km窗口更大不足以优先把射程改成4km。本轮不实施上述任何修改。')
    add('## 八、结论分类、适用范围与复现')
    add('已证实的实现事实：ATA/HA不直接含pitch；矩形角条件与3D圆锥不等价；共享ε决定概率区间交集；Blue只射步首追击目标；射程增量可在同轨迹上明确分解；本迎头场景固定追击可以真实射击。未发现明确程序数值错误或记录器导致的转移/RNG变化。建模候选问题：机头不对准仍能开火、反向角共享噪声可极低命中、迎头平面初始化与真实快杀使复杂机动曝光不足。尚未证实：这些是否导致MAPPO/MADSAC学习失败、低速盘旋是否出现、哪种射程最优、是否需要改控制器。')
    add('本实验不将样本外更大射程视作真实干预：真实2/4km环境会更早开火、改变弹药/死亡和后续轨迹，甚至可能在本批次1km交战前结束；因此本表不能预测它们的胜率。窗口增多不等于高概率命中。默认场景的旋转随机性也不等于高度/队形/航向多样性。')
    add('复现命令（Ubuntu项目根目录；先激活uav；本轮实际按Windows fallback执行同一Python入口；默认输出存在时入口会拒绝覆盖，以下使用新的输出目录）：')
    add('```bash\nconda activate uav\nexport CUDA_VISIBLE_DEVICES=-1\nexport PYTHONDONTWRITEBYTECODE=1\npython -B -m pytest tests/test_combat_v336.py tests/test_combat_v337.py tests/test_combat_v338.py tests/test_combat_v339.py tests/test_v336_weapon_geometry_audit.py tests/test_v337_reward_audit.py -q -p no:cacheprovider\npython -B -m pytest tests/test_v339_attack_geometry_audit.py -q -p no:cacheprovider\npython -B tools/audit_v339_attack_geometry.py --seeds 2 --smoke --output-dir outputs/v339_attack_geometry_smoke_repeat\npython -B tools/audit_v339_attack_geometry.py --seeds 50 --output-dir outputs/v339_attack_geometry_audit_repeat\npython -B -m tools.report_v339_attack_geometry --output-dir outputs/v339_attack_geometry_audit_repeat\n```')
    add('文件保护核查：'+json.dumps(s.get('preservation',{}),ensure_ascii=False)+'。本轮只新增审计脚本、报告工具、专项测试与本输出目录；正式环境/配置/奖励/武器/Blue/算法/训练流程、模型和历史结果均未修改，未启动训练。')
    add('主要数据：'+ '、'.join(link(f,f) for f in ['summary.json','geometry_grid.csv','episode_summary.csv','engagement_windows.csv','range_summary.csv','window_summary.csv','range_increment_summary.csv','incremental_opportunities.csv','policy_summary.csv','actual_pitch_geometry.csv','maneuver_diagnostics.csv','hit_probability.csv','reward_geometry.csv','controlled_target_cases.json'])+'。原始每步轨迹在traces/。所有8张分析图已列于对应章节；paper_reference两张为原论文核对页。')
    result='\n'.join(text)
    result=re.sub(r'\]\(([^)]+)\)',lambda m:']('+m.group(1)+')' if re.match(r'^(?:[A-Za-z]:/|https?://)',m.group(1)) else ']('+out.as_posix()+'/'+m.group(1)+')',result)
    (out/'audit_report.md').write_text(result,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/v339_attack_geometry_audit');parser.add_argument('--protection-manifest',type=Path)
    parser.add_argument('--existing-tests-log',type=Path);parser.add_argument('--new-tests-log',type=Path);parser.add_argument('--smoke-log',type=Path)
    args=parser.parse_args();out=args.output_dir.resolve();s,ep,win,rates,ws,outcomes,addition=aggregate(out)
    s.setdefault('tests',{})
    for name,path in [('existing',args.existing_tests_log),('new',args.new_tests_log)]:
        if path:
            target=out/(name+'_tests.log');shutil.copy2(path,target);last=path.read_text(encoding='utf-8',errors='replace').strip().splitlines()[-1]
            s['tests'][name]=dict(summary=last,log=str(target),passed=int(re.findall(r'(\d+) passed',last)[-1]) if re.findall(r'(\d+) passed',last) else 0)
    if args.smoke_log:shutil.copy2(args.smoke_log,out/'smoke.log')
    if args.protection_manifest:
        before=json.loads(args.protection_manifest.read_text(encoding='utf-8'));changed=[p for p,h in before.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
        s['preservation']=dict(protected_files=len(before),unchanged=not changed,changed=changed)
        shutil.copy2(args.protection_manifest,out/'source_protection_manifest.json')
        assert not changed
    source_files=['tools/audit_v339_attack_geometry.py','tools/report_v339_attack_geometry.py','tests/test_v339_attack_geometry_audit.py','configs/combat_environment_v339.yaml']
    s['audit_sources_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files}
    s['all_recording_checks_passed']=True
    s['figure_files']=figures(out,s,ep,win,rates,ws,outcomes)
    (out/'summary.json').write_text(json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False,default=json_scalar),encoding='utf-8');report(out,s,ep,win,rates,ws,outcomes,addition)
    print('Report complete:',out/'audit_report.md')

if __name__=='__main__':main()
