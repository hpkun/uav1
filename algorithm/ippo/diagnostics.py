"""Read-only control and fixed-observation diagnostics; no policy/reward changes."""
from pathlib import Path
import json
import numpy as np
import torch
from algorithm.common.protocol import config_sha256

AXES=('heading','pitch','speed')

def action_statistics(actions, alive):
    live=np.asarray(actions)[np.asarray(alive)> .5]
    out={'valid_control_samples':int(len(live))}
    for i,name in enumerate(AXES):
        a=live[:,i] if len(live) else np.zeros(1)
        out.update({f'action_mean_{name}':float(a.mean()),f'action_std_{name}':float(a.std()),
            f'action_saturation_0_99_{name}':float((np.abs(a)>.99).mean())})
    return out

def rollout_action_diagnostics(batch):
    out=action_statistics(batch.actions,batch.alive_masks)
    # Consecutive controls must belong to the same episode and the agent must
    # be alive at both control instants. Terminal reward samples stay in PPO.
    pair=(batch.alive_masks[1:]>.5)&(batch.alive_masks[:-1]>.5)&(batch.dones[:-1,:,None]<.5)
    change=np.abs(np.diff(batch.actions,axis=0))[pair]
    for i,name in enumerate(AXES):
        out[f'adjacent_action_abs_delta_{name}']=float(change[:,i].mean()) if len(change) else None
    # v3.41 explicitly encodes own psi/pi, theta/(pi/3), and normalized
    # speed in fields 5, 6, 3. Decode these only for read-only diagnostics.
    # Zero observations after death cannot describe the last physical state;
    # exclude those pairs and report their coverage. Canonical traces retain
    # the exact physical states through death.
    physical=(batch.alive_masks>.5)&(batch.next_alive_masks>.5)
    heading=(batch.next_observations[...,5]-batch.observations[...,5])*np.pi
    heading=(heading+np.pi)%(2*np.pi)-np.pi
    pitch=(batch.next_observations[...,6]-batch.observations[...,6])*(np.pi/3)
    speed=batch.observations[...,3]*75+225
    for name,value,mask in [('actual_heading_delta_rad',heading,physical),
        ('actual_pitch_delta_rad',pitch,physical),('actual_speed_m_s',speed,batch.alive_masks>.5)]:
        selected=value[mask]
        out[name+'_mean']=float(selected.mean()) if len(selected) else None
        out[name+'_std']=float(selected.std()) if len(selected) else None
    out['flight_state_source']='decoded v3.41 own observation (float32); death transition angles excluded'
    out['flight_angle_samples']=int(physical.sum())
    out.update(action_source='verified_executed_red_actions',
        consecutive_pair_samples=int(pair.sum()),units='alive agent control steps',
        temporal_scope='within_rollout_same_episode')
    return out

def trace_diagnostics(trace, metadata):
    """Full single-episode trajectory; angles radians, speed m/s, time seconds.

    A sustained turn is a descriptive sign-run >=10 executed steps. A 360
    event is each full 2*pi of signed accumulation in that run, not a failure.
    """
    red=trace['red_kinematics']; alive=trace['red_alive']; actions=trace['red_actions']
    dt=float(metadata['dt']); valid=alive[:-1]
    out=action_statistics(actions,valid)
    heading=(np.diff(red[:,:,5],axis=0)+np.pi)%(2*np.pi)-np.pi
    pitch=np.diff(red[:,:,4],axis=0)
    for name,value in [('actual_heading_delta_rad',heading),('actual_pitch_delta_rad',pitch),
                       ('actual_speed_m_s',red[:-1,:,3])]:
        out[name+'_mean']=float(value[valid].mean()) if valid.any() else None
        out[name+'_std']=float(value[valid].std()) if valid.any() else None
    pair=valid[1:]&valid[:-1]
    delta=np.abs(np.diff(actions,axis=0))[pair]
    for i,name in enumerate(AXES):
        out['adjacent_action_abs_delta_'+name]=float(delta[:,i].mean()) if len(delta) else None
    sustained=0; rotations=0
    for agent in range(4):
        sign=0; count=0; total=0.
        def flush(count,total):
            return (count if count>=10 else 0),int(abs(total)//(2*np.pi))
        for step,angle in enumerate(heading[:,agent]):
            direction=int(np.sign(angle)) if abs(angle)>1e-7 and valid[step,agent] else 0
            if direction==0 or direction!=sign:
                a,b=flush(count,total); sustained+=a; rotations+=b
                count=0; total=0.
            sign=direction
            if direction:
                count+=1; total+=angle
        a,b=flush(count,total); sustained+=a; rotations+=b
    out['sustained_same_direction_turn_fraction']=sustained/max(1,int(valid.sum()))
    out['cumulative_360_turn_events']=rotations
    blue=trace['blue_kinematics']; blue_alive=trace['blue_alive']
    distance=np.linalg.norm(red[:,:,None,:3]-blue[:,None,:,:3],axis=-1)
    pairs=alive[:,:,None]&blue_alive[:,None,:]
    weapon=metadata['weapon_config']
    in_range=((distance>=weapon['range_min'])&(distance<=weapon['range_max'])&pairs).any(axis=(1,2))
    hits=np.flatnonzero(in_range)
    out['red_first_weapon_range_time_s']=float(hits[0]*dt) if len(hits) else None
    events=metadata.get('events',[])
    for event,label in [('fire_attempt','fire'),('attack_kill','kill')]:
        rows=[e for e in events if e['type']==event and e['side']=='red']
        out['red_first_'+label+'_time_s']=min((e['time_s'] for e in rows),default=None)
        if event=='fire_attempt':
            first=min(rows,key=lambda e:e['step']) if rows else {}
            out['red_first_fire_ata_rad']=first.get('ata')
            out['red_first_fire_ha_rad']=first.get('ha')
    for side in ('red','blue'):
        for field in ('attack_kills','boundary_exits','ground_losses','ceiling_losses'):
            out[side+'_'+field]=metadata.get(side+'_'+field)
    losses=[]
    for agent in range(4):
        death=np.flatnonzero(alive[:-1,agent]&~alive[1:,agent])
        step=int(death[0]+1) if len(death) else None
        cause=None
        if step is not None:
            matching=[e for e in events if e['step']==step and
                ((e['type']=='noncombat_loss' and e['side']=='red' and e['agent']==agent) or
                 (e['type']=='attack_kill' and e['side']=='blue' and e['target']==agent))]
            cause=matching[0].get('cause','weapon') if matching else 'unclassified'
        losses.append(dict(agent=agent,loss_time_s=step*dt if step is not None else None,cause=cause))
    out.update(red_losses_by_agent=losses,turn_run_min_steps=10,
        turn_classification='descriptive_only',action_source='canonical executed actions',
        temporal_scope='one complete episode',angle_units='radians',speed_units='m/s')
    return out

def save_observation_bank(path, observations, source):
    if source.get('split') not in ('training','development') or not source.get('seed_provenance'):
        raise ValueError('fixed observations need explicit non-holdout seed provenance')
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    observations=np.asarray(observations,dtype=np.float32)
    if observations.shape[-2:]!=(4,45) or not np.isfinite(observations).all():
        raise ValueError('diagnostic bank must contain finite [...,4,45] observations')
    np.savez_compressed(path,observations=observations,source_json=json.dumps(source))

def compare_checkpoints(checkpoints, bank_path, environment_config, algorithm_config, device='cuda'):
    from .evaluation import load_checkpoint
    with np.load(bank_path,allow_pickle=False) as bank:
        observations=bank['observations']; source=json.loads(str(bank['source_json']))
    if source.get('split') not in ('training','development') or not source.get('seed_provenance'):
        raise ValueError('holdout/unknown observation bank is forbidden')
    if source.get('environment_config_sha256', config_sha256(environment_config)) != config_sha256(environment_config):
        raise ValueError('diagnostic observation bank environment fingerprint mismatch')
    previous=None; rows=[]
    for checkpoint in checkpoints:
        trainer=load_checkpoint(checkpoint,algorithm_config,environment_config,device)
        with torch.no_grad():
            action=trainer.actor.deterministic(torch.as_tensor(observations,device=trainer.device)).cpu().numpy()
        row=dict(checkpoint=str(checkpoint),deterministic_actions=action.tolist(),
            action_mean=action.reshape(-1,3).mean(0).tolist(),
            action_variance=action.reshape(-1,3).var(0).tolist(),
            saturation_fraction=(np.abs(action)>.99).reshape(-1,3).mean(0).tolist(),
            previous_checkpoint_mean_abs_delta=None if previous is None else np.abs(action-previous).reshape(-1,3).mean(0).tolist(),
            persistent_saturation_fraction=None if previous is None else
                ((np.abs(action)>.99)&(np.abs(previous)>.99)&(np.sign(action)==np.sign(previous))).reshape(-1,3).mean(0).tolist())
        rows.append(row); previous=action
    return dict(observation_bank=str(Path(bank_path).resolve()),source=source,checkpoints=rows,
        convergence_claim=False,trajectory_comparison='use canonical same-seed recordings; fixed actions alone do not establish convergence')
