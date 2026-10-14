"""Read-only current flight-control audit; no training, policy or environment edits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.v329_control import action_to_control_v329,target_state_v329
from env.v311_kinematics import advance_aircraft
from env.math_utils import wrap_angle

SPEEDS=(150.,225.,300.)
HEADING=(.05,.1,.2,.3,.5,.75,1.)
PITCH=(-1.,-.75,-.5,-.25,-.1,-.05,0.,.05,.1,.25,.5,.75,1.)
COMBINED=([1,0,0],[.5,0,0],[.3,0,0],[.2,0,0],[0,1,0],[0,-1,0],
          [1,1,0],[.5,.5,0],[.3,.3,0],[1,0,1],[1,0,-1],[.5,.5,1])


def response(e,state,action):
    psi,theta,speed=target_state_v329(state,action,e.config['action'],e.spec)
    cmd=e.config['action']['controller'];g=e.dynamics.gravity
    delta=wrap_angle(psi-state.psi)
    if abs(abs(delta)-np.pi)<=1e-12:delta=np.copysign(np.pi,action[0])
    yaw=delta/cmd['heading_time_constant'];pitch=(theta-state.theta)/cmd['pitch_time_constant']
    accel=(speed-state.v)/cmd['speed_time_constant']
    raw_a=np.cos(state.theta)+state.v*pitch/g
    raw_b=state.v*np.cos(state.theta)*yaw/g
    control=action_to_control_v329(state,action,e.config['action'],e.spec,g)
    rates=e.dynamics.derivatives(state,control)
    return dict(desired_heading=psi,desired_pitch=theta,desired_speed=speed,
                commanded_yaw_deg_s=float(np.rad2deg(yaw)),commanded_pitch_deg_s=float(np.rad2deg(pitch)),
                commanded_acceleration=float(accel),raw_vertical=float(raw_a),raw_lateral=float(raw_b),
                raw_load=float(np.hypot(max(raw_a,0),raw_b)),
                vertical_floor=bool(raw_a<0),load_projection=bool(np.hypot(max(raw_a,0),raw_b)>10),
                pitch_target_clipped=bool(abs(theta-(state.theta+np.pi/3*action[1]))>1e-10),
                speed_target_clipped=bool(abs(speed-(state.v+50*action[2]))>1e-10),
                actual_yaw_deg_s=float(np.rad2deg(rates[5])),actual_pitch_deg_s=float(np.rad2deg(rates[4])),
                actual_acceleration=float(rates[3]),nz=control.nz,phi_deg=float(np.rad2deg(control.phi)))


def curve_fit(states):
    xy=states[:,:2];x,y=xy.T
    if len(x)<5 or np.linalg.matrix_rank(np.column_stack((2*x,2*y,np.ones_like(x))))<3:return None,None
    cx,cy,c=np.linalg.lstsq(np.column_stack((2*x,2*y,np.ones_like(x))),x*x+y*y,rcond=None)[0]
    radius=float(np.sqrt(max(0,cx*cx+cy*cy+c)))
    residual=float(np.sqrt(np.mean((np.hypot(x-cx,y-cy)-radius)**2)))
    return radius,residual


def crossing(time,values,target):
    reached=np.flatnonzero(values>=target-1e-10)
    if not len(reached):return None
    i=int(reached[0])
    if i==0:return 0.
    return float(np.interp(target,values[i-1:i+1],time[i-1:i+1]))


def simulate(e,speed,action,duration,stop_ground=True):
    e.reset(31);e.red=[AircraftState(0,0,-3000,speed,0,0)]+[AircraftState(0,0,-3000,speed,0,0,False) for _ in range(3)]
    states=[e.red[0].as_array()];responses=[]
    actions=np.zeros((4,3),dtype=np.float32);actions[0]=action
    first_boundary=None;pitch_state_clip=0;speed_state_clip=0
    for index in range(int(np.ceil(duration/e.dt))):
        s=e.red[0];responses.append(response(e,s,actions[0]))
        # The actual step calls this exact inherited method with four real slots.
        e._advance(e.red,actions);next_state=e.red[0]
        states.append(next_state.as_array())
        pitch_state_clip+=abs(abs(next_state.theta)-np.pi/3)<1e-9
        speed_state_clip+=abs(next_state.v-e.spec.v_min)<1e-9 or abs(next_state.v-e.spec.v_max)<1e-9
        if first_boundary is None and (np.hypot(next_state.x,next_state.y)>e.arena_radius or next_state.altitude>6000):first_boundary=(index+1)*e.dt
        if stop_ground and next_state.altitude<=0:break
    states=np.asarray(states);time=np.arange(len(states))*e.dt
    heading=np.cumsum([0]+[wrap_angle(y-x) for x,y in zip(states[:-1,5],states[1:,5])])
    rates=np.diff(heading)/e.dt
    yaw=np.array([r['actual_yaw_deg_s'] for r in responses]);pitch=np.array([r['actual_pitch_deg_s'] for r in responses])
    nz=np.array([r['nz'] for r in responses]);phi=np.array([r['phi_deg'] for r in responses])
    radius,rms=curve_fit(states);tail=states[max(0,len(states)-max(5,int(2/e.dt))):]
    tail_radius,tail_rms=curve_fit(tail)
    horizontal_velocity=states[:-1,3]*np.cos(states[:-1,4]);nonzero=np.abs(rates)>1e-12
    result=dict(speed=speed,action=list(action),duration_s=float(time[-1]),initial=responses[0],
                requested_heading_delta_deg=float(180*action[0]),requested_pitch_delta_deg=float(60*action[1]),
                actual_yaw_mean_deg_s=float(np.rad2deg(rates.mean())),actual_yaw_max_deg_s=float(np.rad2deg(np.max(np.abs(rates)))),
                actual_pitch_mean_deg_s=float(pitch.mean()),actual_pitch_max_abs_deg_s=float(np.abs(pitch).max()),
                nz_mean=float(nz.mean()),nz_max=float(nz.max()),phi_mean_deg=float(phi.mean()),phi_max_abs_deg=float(np.abs(phi).max()),
                exact_cap_fraction=float(np.mean(nz>=10-1e-10)),diagnostic_nz_ge_9_9_fraction=float(np.mean(nz>=9.9)),
                vertical_floor_fraction=float(np.mean([r['vertical_floor'] for r in responses])),
                pitch_target_clip_fraction=float(np.mean([r['pitch_target_clipped'] for r in responses])),
                pitch_state_clip_fraction=float(pitch_state_clip/len(responses)),speed_state_clip_fraction=float(speed_state_clip/len(responses)),
                altitude_change_m=float(-(states[-1,2]-states[0,2])),final_pitch_deg=float(np.rad2deg(states[-1,4])),
                final_speed_m_s=float(states[-1,3]),tail_pitch_mean_deg_s=float(pitch[-max(1,int(2/e.dt)):].mean()),
                hit_ground=bool(-states[-1,2]<=0),first_other_boundary_time_s=first_boundary,
                turn_90_s=crossing(time,heading,np.pi/2),turn_180_s=crossing(time,heading,np.pi),
                radius_fit_xy_m=radius,radius_fit_rms_m=rms,tail_radius_fit_xy_m=tail_radius,tail_radius_fit_rms_m=tail_rms,
                radius_v_over_yaw_m=float(np.median(states[:-1,3][nonzero]/np.abs(rates[nonzero]))) if nonzero.any() else None,
                radius_horizontal_v_over_yaw_m=float(np.median(horizontal_velocity[nonzero]/np.abs(rates[nonzero]))) if nonzero.any() else None,
                radius_tail_horizontal_v_over_yaw_m=float(np.median((horizontal_velocity[nonzero]/np.abs(rates[nonzero]))[-max(1,int(2/e.dt)):])) if nonzero.any() else None)
    return result,states,time


def log_snapshot(path,out):
    if not path.exists():return dict(available=False,path=str(path))
    raw=path.read_bytes();(out/'training_metrics_snapshot.jsonl').write_bytes(raw)
    rows=[];incomplete=0
    for line in raw.decode('utf-8').splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:incomplete+=1;continue
        if row.get('record_type')=='rollout_summary':rows.append(row)
    targets=[500000,1000000,1500000,2000000,2500000]
    selected=[min(rows,key=lambda r:abs(r['sampled_steps']-n)) for n in targets]+[rows[-1]] if rows else []
    keys=['sampled_steps','rollout_sampled_steps','completed_episodes_in_rollout','red_mean_nz','red_nz_saturation_fraction','red_mean_abs_phi',
          *[f'red_mean_abs_high_level_action_{k}' for k in ('psi','theta','v')],
          *[f'red_action_saturation_fraction_{k}' for k in ('psi','theta','v')],
          'mean_r3_reward','mean_r4_reward','episode_r3_total','episode_r4_total','timeout_rate',
          'average_episode_tactical_offense_total','average_episode_tactical_defense_total',
          *[f'red_funnel_{k}_agent_steps_per_env_step' for k in ('live','range','range_az','geometry','eligible')],
          'red_fire_window_episode_rate','red_attempt_episode_rate','red_kill_episode_rate','blue_kill_episode_rate',
          'win_rate','red_survivors','red_boundary_exits','red_ground_losses','red_ceiling_losses']
    return dict(available=True,path=str(path),sha256=hashlib.sha256(raw).hexdigest(),rows=len(rows),incomplete_lines=incomplete,
                selected=[{k:r.get(k) for k in keys} for r in selected],
                last10_unweighted_means={k:float(np.mean([r[k] for r in rows[-10:] if isinstance(r.get(k),(int,float))]))
                                          for k in keys if any(isinstance(r.get(k),(int,float)) for r in rows[-10:])})


def plot_audit(e,result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    grid=np.linspace(-1,1,201)
    for speed in SPEEDS:
        s=AircraftState(0,0,-3000,speed,0,0)
        axes[0,0].plot(grid,[response(e,s,[a,0,0])['actual_yaw_deg_s'] for a in grid],label=f'{speed:.0f} m/s')
        axes[0,1].plot(grid,[response(e,s,[0,a,0])['actual_pitch_deg_s'] for a in grid],label=f'{speed:.0f} m/s')
    axes[0,0].plot(grid,90*grid,'k--',alpha=.5,label='requested')
    axes[0,1].plot(grid,30*grid,'k--',alpha=.5,label='requested')
    axes[0,0].set(xlabel='Heading action',ylabel='Initial yaw rate (deg/s)',title='Level initial state: heading compression')
    axes[0,1].set(xlabel='Pitch action',ylabel='Initial pitch rate (deg/s)',title='Pitch asymmetry / floor')
    for ax in axes[0]:ax.grid(alpha=.3);ax.legend(fontsize=8)
    g=np.linspace(-1,1,81);s=AircraftState(0,0,-3000,225,0,0)
    pitch=np.array([[response(e,s,[psi,theta,0])['actual_pitch_deg_s'] for psi in g] for theta in g])
    im=axes[1,0].imshow(pitch,origin='lower',extent=(-1,1,-1,1),aspect='auto',cmap='coolwarm',vmin=-23,vmax=23)
    axes[1,0].axhline(0,color='k',lw=.7)
    axes[1,0].set(xlabel='Heading action',ylabel='Pitch action',title='Actual initial pitch rate at 225 m/s')
    fig.colorbar(im,ax=axes[1,0],label='deg/s')
    logs=result['training'].get('selected',[])
    if logs:
        x=[r['sampled_steps']/1e6 for r in logs]
        for key,label in [('red_nz_saturation_fraction','nz >= 9.9 fraction'),('red_mean_abs_high_level_action_psi','mean |a heading|'),('red_mean_abs_high_level_action_theta','mean |a pitch|'),('red_mean_abs_high_level_action_v','mean |a speed|')]:
            axes[1,1].plot(x,[r[key] for r in logs],marker='o',label=label)
        axes[1,1].set(xlabel='Sampled steps (M)',ylabel='Fraction / normalized magnitude',ylim=(0,1),title='Read-only v3.30 training snapshot')
        axes[1,1].legend(fontsize=8);axes[1,1].grid(alpha=.3)
    fig.savefig(out/'controller_mapping.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for speed in SPEEDS:
        data=np.load(out/f'heading_{int(speed)}_1.00.npz');states,time=data['states'],data['time_s']
        axes[0].plot(states[:,0],states[:,1],label=f'{speed:.0f} m/s')
        axes[1].plot(time,-states[:,2],label=f'{speed:.0f} m/s')
    axes[0].set(xlabel='X (m)',ylabel='Y (m)',title='Pure heading action [1,0,0]: XY projection',aspect='equal')
    axes[1].set(xlabel='Time (s)',ylabel='Altitude (m)',title='Same trajectories: unintended descent')
    for ax in axes:ax.grid(alpha=.3);ax.legend()
    fig.savefig(out/'pure_heading_descent.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v330_controller_audit')
    p.add_argument('--training-log',type=Path,default=ROOT/'outputs/mappo_v330_seed1_3m/training_metrics.jsonl')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory for repository audits')
    protected=[f for folder in ('env','algorithm','configs') for f in (ROOT/folder).rglob('*') if f.is_file() and f.suffix in ('.py','.yaml')]
    before={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected}
    a=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v330.yaml');b=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v331.yaml')
    legacy=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v311.yaml')
    assert legacy.spec==a.spec and legacy.config['action']['command']==a.config['action']['command']
    assert type(a)._advance is type(b)._advance
    heading=[];pitch=[];combined=[];minimum=[];threshold=[];direct=[]
    for speed in SPEEDS:
        for turn in HEADING:
            row,states,time=simulate(a,speed,[turn,0,0],max(160.,8/turn))
            heading.append(row)
            np.savez_compressed(args.output/f'heading_{int(speed)}_{turn:.2f}.npz',states=states,time_s=time)
        for turn in PITCH:
            row,_,_=simulate(a,speed,[0,turn,0],8.);pitch.append(row)
        for action in COMBINED:
            row,_,_=simulate(a,speed,action,5.);combined.append(row)
        g=a.dynamics.gravity;cap=a.config['action']['controller']['normal_load_max']
        cutoff=g*np.sqrt(cap*cap-1)/(speed*np.pi/2)
        deadzone=6*g/(np.pi*speed)
        s=AircraftState(0,0,-3000,speed,0,0)
        grid=np.linspace(-1,1,2001);responses=[response(a,s,[u,0,0]) for u in grid]
        threshold.append(dict(speed=speed,theoretical_heading_cap_action=cutoff,
                              simulated_first_cap_action=next(u for u,r in zip(grid,responses) if u>=0 and r['nz']>=10-1e-10),
                              exact_cap_action_fraction=float(np.mean([r['nz']>=10-1e-10 for r in responses])),
                              unsaturated_action_fraction=float(np.mean([r['nz']<10-1e-10 for r in responses])),
                              negative_pitch_floor_threshold=-deadzone,negative_half_compressed_fraction=1-deadzone))
        raw_b=speed*np.pi/2/g;raw_a=raw_b/np.sqrt(cap*cap-1)
        compensate=(raw_a-1)*g/speed*2/(np.pi/3)
        row,states,time=simulate(a,speed,[1,compensate,0],40.)
        radius=speed*speed/(g*np.sqrt(cap*cap-1))
        assert np.max(np.abs(states[:,2]+3000))<.01
        assert abs(row['radius_fit_xy_m']-radius)<.01
        minimum.append(dict(speed=speed,action=[1,compensate,0],radius_theory_m=radius,
                            radius_fit_m=row['radius_fit_xy_m'],yaw_deg_s=row['actual_yaw_mean_deg_s'],
                            turn_90_s=row['turn_90_s'],turn_180_s=row['turn_180_s'],nz=row['nz_mean']))
        for turn in (.1,.3,.5,1.):
            state=s.copy();psi=0.;trace=[state.as_array()];history=[0.]
            for i in range(100):
                previous=state.psi;state=advance_aircraft(state,[turn,0,0],.1,a.config['action']['command'],a.spec)
                psi+=np.pi*turn # +pi is direction ambiguous after wrapping; commanded winding is known.
                history.append(psi);trace.append(state.as_array())
            radius_fit,rms=curve_fit(np.array(trace))
            current=next(r for r in heading if r['speed']==speed and r['action'][0]==turn) if turn in HEADING else None
            current_1s,states,_=simulate(a,speed,[turn,0,0],1.)
            direct.append(dict(speed=speed,a_psi=turn,direct_one_step_heading_deg=180*turn,
                               current_one_step_heading_deg=float(np.rad2deg(wrap_angle(states[1,5]-states[0,5]))),
                               direct_1s_commanded_winding_deg=1800*turn,
                               current_1s_winding_deg=current_1s['actual_yaw_mean_deg_s'],
                               direct_90_s=float(np.ceil(.5/turn-1e-12)*.1),direct_180_s=float(np.ceil(1/turn-1e-12)*.1),
                               current_90_s=current['turn_90_s'],current_180_s=current['turn_180_s'],
                               direct_polygon_radius_m=radius_fit,direct_polygon_radius_theory_m=speed*.1/(2*np.sin(np.pi*turn/2)),
                               direct_radius_fit_rms_m=rms,current_horizontal_radius_fit_m=current['radius_fit_xy_m'],
                               direct_two_position_oscillation=turn==1.))
    speeds=[]
    for av in (-1.,-.5,-.25,0.,.25,.5,1.):
        row,states,time=simulate(a,225.,[0,0,av],40.)
        speeds.append(dict(action=av,initial=row['initial'],final_speed=float(states[-1,3]),
                           time_within_1m_s_of_150=crossing(time,225-states[:,3],74.),
                           time_within_1m_s_of_300=crossing(time,states[:,3]-225,74.),
                           exact_150_s=crossing(time,225-states[:,3],75.),exact_300_s=crossing(time,states[:,3]-225,75.),
                           initial_225_s=0.,speed_target_clip_fraction=float(np.mean([abs(states[i,3]+50*av-np.clip(states[i,3]+50*av,150,300))>1e-10 for i in range(len(states)-1)]))))
    # One genuine step verifies the reviewed _advance -> controller -> RK4 path.
    step_checks=[]
    for version in ('330','331'):
        e=MultiUAVCombatEnv(ROOT/f'configs/combat_environment_v{version}.yaml');e.reset(31)
        e.red=[AircraftState(0,0,-3000,225,0,0)]+[AircraftState(0,0,-3000,225,0,0,False) for _ in range(3)]
        expected=e.integrator.step(e.red[0],action_to_control_v329(e.red[0],np.array([.3,.3,0],dtype=np.float32),e.config['action'],e.spec),e.dynamics,e.spec)
        actions=np.zeros((4,3));actions[0]=[.3,.3,0];e.step(actions)
        assert np.array_equal(e.red[0].as_array(),expected.as_array())
        step_checks.append(dict(version=version,exact_step_state_match=True,advance_owner=type(e)._advance.__qualname__))
    assert before=={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in protected}
    result=dict(heading=heading,pitch=pitch,speed=speeds,combined=combined,minimum_level_turn=minimum,
                thresholds=threshold,direct_comparison=direct,actual_step_checks=step_checks,
                training=log_snapshot(args.training_log,args.output),protected_source_config_hashes_unchanged=True,
                convention='Long sweeps isolate inherited _advance; no horizontal arena termination. Ground crossing ends heading runs. No artificial altitude stabilization. Minimum LEVEL turns separately use feasible compensating pitch actions. Pitch/heading rates are finite differences or instantaneous derivatives as labelled. Signed NED theta: positive climbs.')
    (args.output/'controller_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for name in ('heading','pitch','speed','combined','minimum_level_turn','thresholds','direct_comparison'):
        rows=result[name]
        with (args.output/f'{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            keys=sorted(set().union(*(r.keys() for r in rows)));writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
            writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    plot_audit(a,result,args.output)
    print(json.dumps(dict(minimum=minimum,thresholds=threshold,log_available=result['training']['available'],protected_hashes_unchanged=True),indent=2))


if __name__=='__main__':main()
