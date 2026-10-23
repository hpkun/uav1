"""Offline CSV/report analysis. Inputs are exclusively canonical trace + metadata."""
from collections import Counter
from pathlib import Path
import csv
import json
import numpy as np
from tools.combat_trace import read_trace, validate_trace


def plane(side, index):
    return f"{'Red' if side=='red' else 'Blue'} UAV {'R' if side=='red' else 'B'}{index+1}"


def analyze_trace(trace, metadata):
    validate_trace(trace,metadata)
    fires=[e for e in metadata['events'] if e['type']=='fire_attempt']
    kills=[e for e in metadata['events'] if e['type']=='attack_kill']
    losses=[e for e in metadata['events'] if e['type']=='noncombat_loss']
    dwell=metadata.get('event_semantics')=='continuous_dwell_neutralization'
    components=trace['reward_components'].sum(axis=0,dtype=np.float64)
    rows=[];combat={}
    for side in ('red','blue'):
        own=[e for e in fires if e['side']==side];credited=[e for e in kills if e['side']==side]
        count=dict(fire_attempts=len(own),weapon_hits=sum(bool(e['hit']) for e in own),attack_kills=len(credited),
            first_fire_time_s=min((e['step']*metadata['dt'] for e in own),default=None),
            first_kill_time_s=min((e['step']*metadata['dt'] for e in credited),default=None))
        for field in ('fire_attempts','weapon_hits','attack_kills'):
            if count[field]!=metadata[side+'_'+field]:raise ValueError(f'{side} event/metadata {field} mismatch')
        count['noncombat_losses']=dict(Counter(e['cause'] for e in losses if e['side']==side))
        ammo=trace.get(side+'_ammo')
        initial=(metadata.get('initial_ammo') or {}).get(side)
        count['ammo_used']=None if initial is None or ammo is None else int(np.sum(initial)-ammo[-1].sum())
        if initial is not None and count['ammo_used']!=len(own):raise ValueError('ammo use / actual fire attempts mismatch')
        combat[side]=count
        if dwell:
            completed=[e for e in metadata['events'] if e['type']=='dwell_complete' and e['side']==side]
            if len(completed)!=metadata[f'{side}_dwell_completed']:raise ValueError('dwell completion metadata mismatch')
            if len(credited)!=metadata[f'{side}_dwell_neutralizations']:raise ValueError('dwell victim metadata mismatch')
            count.update(dwell_completed=len(completed),dwell_neutralizations=len(credited),
                         first_dwell_complete_time_s=min((e['time_s'] for e in completed),default=None))
        alive=trace[side+'_alive'];states=trace[side+'_kinematics']
        if int(alive[-1].sum())!=metadata[side+'_survivors']:raise ValueError('survivor metadata mismatch')
        for i in range(states.shape[1]):
            deaths=np.flatnonzero(~alive[:,i]);last=int(deaths[0]) if len(deaths) else len(alive)-1
            path=states[:last+1,i]
            own_shots=[e for e in own if e['attacker']==i]
            own_kills=[e for e in credited if i in e['attackers']]
            loss=next((e for e in losses if e['side']==side and e['agent']==i),None)
            row=dict(side=side,agent=i,label=plane(side,i),alive=bool(alive[-1,i]),
                final_time_s=float(trace['time_s'][last]),initial_state=path[0].tolist(),final_state=path[-1].tolist(),
                path_length_m=float(np.linalg.norm(np.diff(path[:,:3],axis=0),axis=1).sum()),
                min_altitude_m=float((-path[:,2]).min()),max_altitude_m=float((-path[:,2]).max()),
                speed_range_m_s=[float(path[:,3].min()),float(path[:,3].max())],
                fire_attempts=len(own_shots),hits=sum(bool(e['hit']) for e in own_shots),
                credited_kill_events=len(own_kills),fractional_kill_credit=sum(1/len(e['attackers']) for e in own_kills),
                loss_cause=loss['cause'] if loss else 'combat' if not alive[-1,i] else None,
                remaining_ammo=None if ammo is None or (ammo[-1]<0).any() else int(ammo[-1,i]))
            if side=='red':row['reward_breakdown']=components[:,i].tolist()
            rows.append(row)
    rewards=components.sum(axis=1)
    for i,value in enumerate(rewards,1):
        if not np.isclose(value,metadata[f'episode_r{i}_total'],rtol=1e-6,atol=1e-5):raise ValueError('reward component totals mismatch')
    if not np.isclose(trace['local_rewards'].sum(),metadata['episode_return'],rtol=1e-6,atol=1e-5):raise ValueError('episode return mismatch')
    available=bool(metadata.get('blue_target_available',False))
    targets=trace.get('blue_pursuit_target_indices') if available else None
    frequencies=None if targets is None else {plane('blue',i):dict(Counter(map(int,targets[:,i]))) for i in range(targets.shape[1])}
    return dict(combat=combat,uavs=rows,reward_totals=rewards.tolist(),
        reward_sum_matches_local=bool(np.allclose(trace['local_rewards'],trace['reward_components'][:,0]+trace['reward_components'][:,1]+trace['reward_components'][:,2]+trace['reward_components'][:,3],rtol=1e-6,atol=1e-6)),
        blue_target_source=metadata.get('blue_target_source','unavailable_in_historical_schema'),
        blue_target_frequencies=frequencies)


def export_csv(trace,metadata,output):
    output=Path(output)
    fields=['step','time_s','side','agent','uav','alive','x','y','z_ned','altitude','speed_m_s','theta_rad','psi_rad','theta_deg','psi_deg','ammo','cooldown_steps','blue_pursuit_target_index']
    with (output/'all_uav_states.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for k,step in enumerate(trace['steps']):
            for side in ('red','blue'):
                for i,state in enumerate(trace[side+'_kinematics'][k]):
                    initial=(metadata.get('initial_ammo') or {}).get(side)
                    ammo=initial[i] if k==0 and initial is not None else int(trace[side+'_ammo'][k-1,i]) if k and side+'_ammo' in trace else None
                    cooldown=int(trace[side+'_cooldown'][k-1,i]) if k and side+'_cooldown' in trace else 0 if k==0 and metadata.get('cooldown_available') else None
                    target=int(trace['blue_pursuit_target_indices'][k-1,i]) if k and side=='blue' and metadata.get('blue_target_available') else None
                    w.writerow(dict(step=int(step),time_s=float(trace['time_s'][k]),side=side,agent=i,uav=plane(side,i),alive=bool(trace[side+'_alive'][k,i]),
                        x=state[0],y=state[1],z_ned=state[2],altitude=-state[2],speed_m_s=state[3],theta_rad=state[4],psi_rad=state[5],theta_deg=np.rad2deg(state[4]),psi_deg=np.rad2deg(state[5]),ammo=ammo,cooldown_steps=cooldown,blue_pursuit_target_index=target))
    dwell=metadata.get('event_semantics')=='continuous_dwell_neutralization'
    events=[e for e in metadata['events'] if e['type']==('dwell_complete' if dwell else 'fire_attempt')]
    geometry=['distance','horizontal_distance','line_of_sight','ata','ha','aa','hca','off_boresight','boresight_azimuth_error','boresight_elevation_error']
    audit=['range_pass','ata_pass','ha_pass','body_azimuth_pass','body_elevation_pass']
    if metadata.get('environment_version') in ('3.42','3.43','3.48'):
        audit += ['cone_pass','probability_angular_frame']
    fields=['step','time_s','side','attacker','target','attacker_uav','target_uav','hit',*geometry,*audit]
    if dwell:
        fields.remove('hit');fields+=['event_semantics','duration_steps','dwell_seconds','cone_pass']
    fields += [who+'_'+name for who in ('attacker','target') for name in ('x','y','z_ned','altitude','speed_m_s','theta_rad','psi_rad')]
    with (output/('dwell_events.csv' if dwell else 'fire_events.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for event in events:
            row={key:event.get(key) for key in fields}
            row['time_s']=event['step']*metadata['dt'];row['attacker_uav']=plane(event['side'],event['attacker']);row['target_uav']=plane('blue' if event['side']=='red' else 'red',event['target'])
            for who in ('attacker','target'):
                a=event[who+'_state']
                row.update({who+'_'+key:value for key,value in zip(('x','y','z_ned','altitude','speed_m_s','theta_rad','psi_rad'),(*a[:3],-a[2],*a[3:]))})
            w.writerow(row)


def write_report(trace,metadata,output):
    output=Path(output);a=analyze_trace(trace,metadata)
    lines=['# Combat replay report','',
        'All observations below come from this one canonical trace and its metadata. No second rollout or inferred tactical intent.','',
        '## Checkpoint','', '| Field | Value |','|---|---|']
    for key in ('algorithm','checkpoint_path','checkpoint_sampled_steps','training_seed','checkpoint_sha256','environment_version','environment_config_sha256','algorithm_config_sha256'):
        lines.append(f'| {key} | {metadata.get(key,"unavailable")} |')
    lines+=['','## Episode','', '| Field | Value |','|---|---|']
    for key in ('episode_seed','episode_steps','simulation_time_s','termination_reason','red_survivors','blue_survivors','red_win','blue_win','draw','episode_return'):
        lines.append(f'| {key} | {metadata[key]} |')
    if metadata.get('event_semantics')=='continuous_dwell_neutralization':
        lines+=['','Event semantics: continuous_dwell_neutralization. Fire/weapon-hit counters are zero compatibility placeholders; ammo/cooldown are unavailable.','',
            '| Side | Dwell started | Interrupted | Target switches | Completed attackers | Unique neutralizations |','|---|---:|---:|---:|---:|---:|']
        for side in ('red','blue'):
            lines.append('| '+side+' | '+' | '.join(str(metadata[f'{side}_{k}']) for k in ('dwell_started','dwell_interrupted','dwell_target_switches','dwell_completed','dwell_neutralizations'))+' |')
        lines+=['','All completion geometry is in dwell_events.csv. Per-transition dwell target/count and pre-settlement completion sample are in episode_trace.npz.','',
                '| Time (s) | Attacker | Target | Duration steps | Distance (m) | 3D alpha (rad) |','|---|---|---|---:|---:|---:|']
        for e in metadata['events']:
            if e['type']=='dwell_complete':
                lines.append(f"| {e['time_s']:.3f} | {plane(e['side'],e['attacker'])} | {plane('blue' if e['side']=='red' else 'red',e['target'])} | {e['duration_steps']} | {e['distance']:.6f} | {e['off_boresight']:.9f} |")
    lines+=['','Position units: metres. Source coordinates are NED (x,y,z); altitude=-z exactly once. Theta/psi in radians unless a column says deg. UAV CSV/event indices are zero-based; visible UAV labels are one-based.','',
        '## Reward breakdown','',
        f"reward_components[t,component,agent]; component axis=1, agent axis=2. Local reward equals sum of these components: {a['reward_sum_matches_local']}.",'',
        '| Component | Total |','|---|---:|']
    for i,value in enumerate(a['reward_totals'],1):lines.append(f'| Reward-R{i} total | {value:.12g} |')
    lines+=['','| UAV | Reward-R1 | Reward-R2 | Reward-R3 | Reward-R4 | Local return |','|---|---:|---:|---:|---:|---:|']
    for row in a['uavs']:
        if row['side']=='red':lines.append('| '+row['label']+' | '+' | '.join(f'{v:.12g}' for v in row['reward_breakdown'])+f" | {trace['local_rewards'][:,row['agent']].sum():.12g} |")
    lines+=['','## Combat and noncombat losses','',
        '| Side | Attempts | Hits | Kills | Ammo used | First fire (s) | First kill (s) | Horizontal loss | Ceiling loss | Ground loss |','|---|---:|---:|---:|---|---|---|---:|---:|---:|']
    for side,c in a['combat'].items():
        loss=c['noncombat_losses']
        lines.append(f"| {side} | {c['fire_attempts']} | {c['weapon_hits']} | {c['attack_kills']} | {c['ammo_used']} | {c['first_fire_time_s']} | {c['first_kill_time_s']} | {loss.get('horizontal_boundary',0)} | {loss.get('ceiling',0)} | {loss.get('ground',0)} |")
    lines+=['','| Noncombat time (s) | UAV | Primary cause | Horizontal / ceiling / ground exceeded |','|---|---|---|---|']
    for e in metadata['events']:
        if e['type']=='noncombat_loss':lines.append(f"| {e['time_s']:.3f} | {plane(e['side'],e['agent'])} | {e['cause']} | {e['horizontal_exceeded']} / {e['ceiling_exceeded']} / {e['ground_exceeded']} |")
    lines+=['','Environment counters are retained verbatim (ceiling may also count as boundary):', '', '`'+json.dumps(metadata.get('combat_counts',{}),sort_keys=True)+'`','',
        '## Actual fire events','',
        'All geometry angles below are radians and are recorded from authoritative engagement_geometry at the actual shot. Audit pass flags never generate or suppress attempts.','',
        '| Time (s) | Attacker | Target | Hit | Attacker (x,y,h) | Target (x,y,h) | Distance m | ATA / HA / AA |','|---|---|---|---|---|---|---:|---|']
    for e in metadata['events']:
        if e['type']!='fire_attempt':continue
        positions=['('+', '.join(f'{v:.3f}' for v in (e[k][0],e[k][1],-e[k][2]))+')' for k in ('start','end')]
        lines.append(f"| {e['step']*metadata['dt']:.3f} | {plane(e['side'],e['attacker'])} | {plane('blue' if e['side']=='red' else 'red',e['target'])} | {e['hit']} | {positions[0]} | {positions[1]} | {e['distance']:.6f} | {e.get('ata')} / {e.get('ha')} / {e.get('aa')} |")
    lines+=['','## Authoritative Blue pursuit targets','',
        f"Source: {a['blue_target_source']}. Rows are transition-aligned. -1 means no target when available; a missing capability means unavailable, never reconstructed.",'',
        '`'+json.dumps(a['blue_target_frequencies'],sort_keys=True)+'`','',
        'Full target timeline is in episode_trace.npz and the Blue rows of all_uav_states.csv (frame k>0 carries transition k-1 target).','',
        '## UAV final states and basic statistics','',
        '| UAV | Alive | Last active time (s) | Initial (x,y,h) | Last (x,y,h) | Speed | Theta / psi (deg) | Path (m) | Hits / credited kill events | Ammo | Cause |','|---|---|---:|---|---|---:|---|---:|---|---|---|']
    for row in a['uavs']:
        state=row['final_state'];initial=row['initial_state']
        point=lambda x: '('+', '.join(f'{v:.3f}' for v in (x[0],x[1],-x[2]))+')'
        lines.append(f"| {row['label']} | {row['alive']} | {row['final_time_s']:.3f} | {point(initial)} | {point(state)} | {state[3]:.3f} | {np.rad2deg(state[4]):.3f} / {np.rad2deg(state[5]):.3f} | {row['path_length_m']:.3f} | {row['hits']} / {row['credited_kill_events']} | {row['remaining_ammo']} | {row['loss_cause']} |")
    lines+=['','Shared kill events can credit multiple attackers; this is distinct from total victim count. Destroyed UAV trajectories stop at loss; their stored later states may remain frozen.','',
        'Single-episode replay is a research audit and does not replace the official evaluator or its statistical evaluation protocol.']
    if (output/'trajectory.png').is_file():
        lines+=['','![Trajectory](trajectory.png)']
    artifacts=(('MP4','episode.mp4'),('Interactive replay','interactive.html'),
               ('All UAV states','all_uav_states.csv'),('Fire events','fire_events.csv'))
    links=[f'[{label}]({name})' for label,name in artifacts if (output/name).is_file()]
    if links:lines+=['',' 路 '.join(links)]
    (output/'detailed_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return a


def export_analysis(trace_path,metadata_path,output,formats=('csv','report')):
    metadata=json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    trace=read_trace(trace_path,metadata)
    if int(trace['trace_schema_version'])!=3 and set(formats)&{'csv','report'}:
        raise ValueError('Historical schema 2 is render-only; unavailable authoritative data cannot be reconstructed')
    if 'csv' in formats:export_csv(trace,metadata,output)
    if 'report' in formats:return write_report(trace,metadata,output)
