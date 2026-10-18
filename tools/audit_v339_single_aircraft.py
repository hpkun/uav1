"""CPU-only, fresh-output validation of the actual v3.39 motion path.

No training, policy checkpoint, physical configuration edits, or controller changes.
"""
from pathlib import Path
import argparse,csv,hashlib,inspect,json,math,os,platform,subprocess,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.v311_kinematics import advance_aircraft

def config():return load_config(ROOT/'configs/combat_environment_v339.yaml')
def state(speed=225.,height=3000.,heading=0.,pitch=0.,x=0.,y=0.,alive=True):
    return AircraftState(x,y,-height,speed,pitch,heading,alive)
def angle_error(a,b):return np.arctan2(np.sin(np.asarray(a)-b),np.cos(np.asarray(a)-b))
def isolated_env(cfg,s):
    e=MultiUAVCombatEnv(cfg)
    e.red=[s.copy()]+[state(alive=False) for _ in range(3)]
    e.blue=[state(alive=False) for _ in range(4)]
    return e
def trajectory(cfg,s,action,seconds):
    """Actual v339._advance dispatch only; no boundary/weapon/outcome stage."""
    e=isolated_env(cfg,s);a=np.zeros((4,3));a[0]=action
    rows=[s.as_array()]
    for _ in range(round(seconds/e.dt)):
        e._advance(e.red,a);rows.append(e.red[0].as_array())
    return np.asarray(rows)
def independent_series(cfg,s,action,steps):
    """Nonrecursive closed-form channel sequences plus an independent position sum."""
    c=cfg['action']['command'];ac=cfg['aircraft'];dt=cfg['simulation']['dt']
    a=np.minimum(1.,np.maximum(-1.,np.asarray(action,dtype=float)));n=np.arange(steps+1)
    psi=s.psi+n*c['heading_delta_max']*a[0]
    theta=np.minimum(ac['theta_max'],np.maximum(ac['theta_min'],s.theta+n*c['pitch_delta_max']*a[1]))
    speed=np.minimum(ac['v_max'],np.maximum(ac['v_min'],s.v+n*c['speed_delta_max']*a[2]))
    # atan2 convention may choose +pi; compare circular errors, never a false 2pi jump.
    heading=np.arctan2(np.sin(psi),np.cos(psi))
    xyz=np.column_stack((speed*np.cos(theta)*np.cos(psi),speed*np.cos(theta)*np.sin(psi),-speed*np.sin(theta)))
    position=np.vstack((s.as_array()[:3],s.as_array()[:3]+np.cumsum(xyz[1:]*dt,axis=0)))
    return np.column_stack((position,speed,theta,heading))
def fit_circle(states):
    xy=states[:,:2];origin=xy.mean(axis=0);q=xy-origin
    cx,cy,k=np.linalg.lstsq(np.column_stack((2*q,np.ones(len(q)))),np.sum(q*q,axis=1),rcond=None)[0]
    radius=float(np.sqrt(cx*cx+cy*cy+k));center=origin+np.array([cx,cy])
    return radius,float(np.max(np.abs(np.linalg.norm(xy-center,axis=1)-radius)))
def first_time(values,threshold,dt,mode='ge'):
    mask=values>=threshold-1e-10 if mode=='ge' else values<=threshold+1e-10
    where=np.flatnonzero(mask)
    return float(where[0]*dt) if len(where) else None
def boundary_flags(cfg,s):
    arena=cfg['arena'];h=s.altitude;r=math.hypot(s.x,s.y)
    ground=h<=arena['altitude_min'];boundary=not ground and (r>arena['radius'] or h>arena['altitude_max'])
    return dict(ground=bool(ground),boundary=bool(boundary),ceiling=bool(boundary and h>arena['altitude_max']),
                horizontal=bool(boundary and r>arena['radius']),alive=not (ground or boundary))
def quiet_full_environment(cfg,s):
    """One test Red, other Red dead; Blue safely loiters far away with explicit actions.

    Unchanged transition/weapon logic runs. Geometry keeps every shot impossible.
    """
    e=MultiUAVCombatEnv(cfg);e.reset(37)
    e.red=[s.copy()]+[state(alive=False) for _ in range(3)]
    e.blue=[state(speed=150,x=-4000)]+[state(alive=False) for _ in range(3)]
    return e
def full_boundary_case(cfg,s,action,seconds=60.):
    e=quiet_full_environment(cfg,s);a=np.zeros((4,3),dtype=np.float32);a[0]=action
    blue=np.zeros_like(a);blue[0]=[1,0,0] # explicit loiter, no automatic Blue policy
    predicted=s.copy();max_error=0.;frames=[s.as_array()];info=None
    for _ in range(round(seconds/e.dt)):
        # Analytic channel expectation is independent of the environment transition.
        expected=independent_series(cfg,s,action,e.steps+1)[-1]
        predicted=AircraftState(*expected)
        _,_,term,trunc,info=e.step(a,blue)
        actual=e.red[0];frames.append(actual.as_array())
        error=np.r_[actual.as_array()[:5]-expected[:5],angle_error(actual.psi,expected[5])]
        max_error=max(max_error,float(abs(error).max()))
        if not actual.alive or term or trunc:break
    flags=boundary_flags(cfg,e.red[0])
    counts=e.combat_counts['red'];no_fire=e.combat_counts['red']['fire_attempts']==e.combat_counts['blue']['fire_attempts']==0
    counts_match=(counts['boundary_exits']==int(flags['boundary']) and counts['ground_losses']==int(flags['ground']) and counts['ceiling_losses']==int(flags['ceiling']))
    return dict(time_s=e.steps*e.dt,state=e.red[0].as_array().tolist(),flags=flags,
        red_boundary_exits=int(counts['boundary_exits']),red_ground_losses=int(counts['ground_losses']),red_ceiling_losses=int(counts['ceiling_losses']),
        no_fire=no_fire,prediction_error=max_error,counts_match=counts_match,terminated=bool(term),truncated=bool(trunc),
        reason=info['termination_reason'],frames=np.asarray(frames))

class Audit:
    def __init__(self,cfg):self.cfg=cfg;self.rows=[];self.checks=[];self.traces={};self.boundaries=[]
    def check(self,case,name,passed,observed=None,expected=None):
        self.checks.append(dict(case=case,check=name,passed=bool(passed),observed=observed,expected=expected))
    def run(self,case,group,s,action,seconds):
        cfg=self.cfg;dt=cfg['simulation']['dt'];c=cfg['action']['command']
        out=trajectory(cfg,s,action,seconds);theory=independent_series(cfg,s,action,len(out)-1)
        pos_error=float(np.max(abs(out[:,:3]-theory[:,:3])));channel_error=np.c_[out[:,3:5]-theory[:,3:5],angle_error(out[:,5],theory[:,5])]
        err=float(np.max(abs(channel_error)));finite=bool(np.isfinite(out).all());vel=np.diff(out[:,:3],axis=0)/dt
        delta_heading=np.diff(np.unwrap(out[:,5]));pitch_rate=np.diff(out[:,4])/dt;acceleration=np.diff(out[:,3])/dt
        step_length=np.linalg.norm(np.diff(out[:,:3],axis=0),axis=1)
        a=np.clip(action,-1,1);end=AircraftState(*out[-1]);bound=boundary_flags(cfg,end)
        passed=finite and pos_error<1e-7 and err<1e-10 and float(step_length.max())<=cfg['aircraft']['v_max']*dt+1e-8
        row=dict(case_id=case,group=group,scope='isolated_kinematics_no_death_resolver',
            initial_x=s.x,initial_y=s.y,initial_z=s.z,initial_speed_m_s=s.v,initial_pitch_deg=math.degrees(s.theta),initial_heading_deg=math.degrees(s.psi),initial_alive=s.alive,
            input_heading=action[0],input_pitch=action[1],input_speed=action[2],clipped_heading=float(a[0]),clipped_pitch=float(a[1]),clipped_speed=float(a[2]),
            simulated_seconds=seconds,steps=len(out)-1,heading_rate_deg_s=float(np.degrees(delta_heading.mean()/dt)),
            pitch_rate_initial_deg_s=float(np.degrees(pitch_rate[0])),pitch_rate_max_deg_s=float(np.degrees(abs(pitch_rate).max())),
            acceleration_initial_m_s2=float(acceleration[0]),acceleration_max_abs_m_s2=float(abs(acceleration).max()),
            effective_heading_first=float(delta_heading[0]/c['heading_delta_max']),effective_pitch_first=float((out[1,4]-out[0,4])/c['pitch_delta_max']),effective_speed_first=float((out[1,3]-out[0,3])/c['speed_delta_max']),
            final_speed_m_s=end.v,final_height_m=end.altitude,final_pitch_deg=math.degrees(end.theta),final_heading_deg=math.degrees(end.psi),
            dx_first_m=float(out[1,0]-s.x),dy_first_m=float(out[1,1]-s.y),dz_first_m=float(out[1,2]-s.z),max_step_length_m=float(step_length.max()),
            position_error_max_m=pos_error,channel_error_max=err,finite=finite,isolated_end_outside_arena=not bound['alive'],
            heading_error_max_rad=float(abs(channel_error[:,2]).max()),pitch_error_max_rad=float(abs(channel_error[:,1]).max()),speed_error_max_m_s=float(abs(channel_error[:,0]).max()),
            fitted_radius_m=None,continuous_radius_m=None,discrete_radius_m=None,lateral_accel_m_s2=None,equivalent_g=None,
            boundary_status=json.dumps(bound),checks_passed=passed)
        self.check(case,'finite/formula/position_continuity',passed,dict(position_error_m=pos_error,channel_error=err),dict(position_tol_m=1e-7,channel_tol=1e-10))
        self.rows.append(row);self.traces[case]=out
        return row,out
    def boundary(self,case,s,action):
        result=full_boundary_case(self.cfg,s,action)
        self.traces[case]=result.pop('frames');result.update(case_id=case,initial_speed=s.v,initial_height=s.altitude,action=list(action))
        self.boundaries.append(result)
        ok=result['counts_match'] and result['no_fire'] and result['prediction_error']<1e-7 and not result['flags']['alive'] and result['terminated'] and not result['truncated']
        self.check(case,'full_transition_boundary_no_weapons',ok,{k:v for k,v in result.items() if k not in ['state']},None)
        final=AircraftState(*result['state'])
        self.rows.append(dict(case_id=case,group='boundary',scope='full_environment_explicit_blue_loiter',initial_x=s.x,initial_y=s.y,initial_z=s.z,initial_speed_m_s=s.v,
            initial_pitch_deg=math.degrees(s.theta),initial_heading_deg=math.degrees(s.psi),initial_alive=True,
            input_heading=action[0],input_pitch=action[1],input_speed=action[2],clipped_heading=float(np.clip(action[0],-1,1)),clipped_pitch=float(np.clip(action[1],-1,1)),clipped_speed=float(np.clip(action[2],-1,1)),
            simulated_seconds=result['time_s'],steps=round(result['time_s']/self.cfg['simulation']['dt']),final_speed_m_s=final.v,final_height_m=final.altitude,
            boundary_status=json.dumps(result['flags']),checks_passed=ok,position_error_max_m=result['prediction_error']))
        return result

def run_experiments(cfg):
    audit=Audit(cfg);dt=cfg['simulation']['dt'];speeds=[150.,225.,300.]
    for speed in speeds:
        for channel in range(3):
            for value in [-1.,-.5,0.,.5,1.,-2.,2.]:
                action=np.zeros(3);action[channel]=value
                audit.run(f'A_v{int(speed)}_c{channel}_a{value:g}','A',state(speed),action,dt)
    for speed in speeds:
        for action_value in [1.,-1.,.5,-.5]:
            name=f'B_v{int(speed)}_a{action_value:g}';row,tr=audit.run(name,'B',state(speed),[action_value,0,0],32.)
            radius,residual=fit_circle(tr);omega=float(np.diff(np.unwrap(tr[:,5])).mean()/dt)
            measured=np.degrees(np.unwrap(tr[:,5])-tr[0,5]);disc=speed*dt/(2*math.sin(abs(omega)*dt/2));cont=speed/abs(omega)
            row.update(fitted_radius_m=radius,circle_fit_max_residual_m=residual,continuous_radius_m=cont,discrete_radius_m=disc,
                radius_error_discrete_m=radius-disc,radius_error_continuous_fraction=(radius-cont)/cont,
                time90_s=first_time(abs(measured),90,dt),time180_s=first_time(abs(measured),180,dt),
                lateral_accel_m_s2=speed*abs(omega),equivalent_g=speed*abs(omega)/9.81)
            audit.check(name,'circle_fit_and_times',abs(radius-disc)<1e-7 and residual<1e-7 and abs(row['time90_s']-math.pi/2/abs(omega))<1e-9 and abs(row['time180_s']-math.pi/abs(omega))<1e-9)
            audit.check(name,'constant_speed_height_pitch',np.all(tr[:,3]==speed) and np.all(tr[:,2]==-3000) and np.all(tr[:,4]==0))
        for magnitude in [1.,.5]:
            left=audit.traces[f'B_v{int(speed)}_a{-magnitude:g}'];right=audit.traces[f'B_v{int(speed)}_a{magnitude:g}']
            audit.check('B',f'left_right_symmetry_v{speed}_a{magnitude}',np.max(abs(left[:,0]-right[:,0]))<1e-9 and np.max(abs(left[:,1]+right[:,1]))<1e-9)
    for speed in speeds:
        for pitch_action in [1.,-1.,.5,-.5]:
            for height in [3000.,100.,5900.]:
                name=f'C_v{int(speed)}_p{pitch_action:g}_h{int(height)}'
                row,tr=audit.run(name,'C',state(speed,height),[0,pitch_action,0],60.)
                degrees=abs(np.degrees(tr[:,4]));row.update(time_pitch30_s=first_time(degrees,30,dt),time_pitch60_s=first_time(degrees,60,dt),
                    vertical_speed_terminal_m_s=float(-np.diff(tr[:,2])[-1]/dt),pitch_bound_hold_fraction=float(np.mean(np.isclose(abs(tr[1:,4]),cfg['aircraft']['theta_max'],atol=1e-12,rtol=0))))
                audit.check(name,'pitch_limits_and_other_channels',np.all(tr[:,3]==speed) and np.all(tr[:,5]==0) and abs(abs(tr[-1,4])-cfg['aircraft']['theta_max'])<1e-12)
                full=audit.boundary('full_'+name,state(speed,height),[0,pitch_action,0]);row['first_full_height_boundary_s']=full['time_s']
    for speed in [150.,200.,225.,250.,300.]:
        for av in [1.,-1.,.5,-.5,0.]:
            name=f'D_v{int(speed)}_a{av:g}';row,tr=audit.run(name,'D',state(speed),[0,0,av],20.)
            row.update(time_vmin_s=first_time(tr[:,3],cfg['aircraft']['v_min'],dt,'le'),time_vmax_s=first_time(tr[:,3],cfg['aircraft']['v_max'],dt))
            monotone=np.all(np.diff(tr[:,3])>=0) if av>=0 else np.all(np.diff(tr[:,3])<=0)
            audit.check(name,'bounded_monotone_no_cross_channel',monotone and np.all(tr[:,4:]==0) and np.all(tr[:,2]==-3000))
    actions=[[1,1,0],[1,-1,0],[-1,1,0],[1,1,1],[1,1,-1],[.5,.5,.5],[0,0,0]]
    for speed in speeds:
        for index,action in enumerate(actions):
            name=f'E_v{int(speed)}_a{index}';row,tr=audit.run(name,'E',state(speed),action,20.)
            velocities=np.vstack((state(speed).velocity_vector(),np.diff(tr[:,:3],axis=0)/dt));accels=np.diff(velocities,axis=0)/dt
            mid=(velocities[1:]+velocities[:-1])/2;mag=np.linalg.norm(accels,axis=1)
            parallel=np.sum(mid*accels,axis=1)/np.linalg.norm(mid,axis=1)
            normal=np.sqrt(np.maximum(0,mag*mag-parallel*parallel))
            row.update(fd_accel_max_m_s2=float(mag.max()),fd_accel_mean_m_s2=float(mag.mean()),fd_equivalent_g_max=float(mag.max()/9.81),
                curvature_max_per_m=float((normal/np.sum(mid*mid,axis=1)).max()),
                speed_bound_fraction=float(np.mean((tr[1:,3]==150)|(tr[1:,3]==300))),
                pitch_bound_fraction=float(np.mean(np.isclose(abs(tr[1:,4]),cfg['aircraft']['theta_max'],atol=1e-12,rtol=0))))
            row['pitch_zero_increment_with_nonzero_command_fraction']=float(np.mean(abs(np.diff(tr[:,4]))<1e-12)) if action[1] else None
            row['speed_zero_increment_with_nonzero_command_fraction']=float(np.mean(abs(np.diff(tr[:,3]))<1e-12)) if action[2] else None
            for channel,field in enumerate([5,4,3]):
                single=np.zeros(3);single[channel]=action[channel];alone=trajectory(cfg,state(speed),single,20.)
                error=angle_error(tr[:,field],alone[:,field]) if channel==0 else tr[:,field]-alone[:,field]
                audit.check(name,f'channel_{channel}_independent',float(abs(error).max())<1e-12)
    for speed in speeds:audit.boundary(f'full_horizontal_v{int(speed)}',state(speed),[0,0,0])
    # Deterministic nonconstant actions, signed wrap, dead freeze, invalid commands.
    rng=np.random.default_rng(339);sequence=rng.uniform(-2,2,(1000,3))
    def variable_run():
        e=isolated_env(cfg,state());rows=[e.red[0].as_array()]
        for action in sequence:
            commands=np.zeros((4,3));commands[0]=action;e._advance(e.red,commands);rows.append(e.red[0].as_array())
        return np.asarray(rows)
    first,second=variable_run(),variable_run();audit.traces['F_repeat_random']=first
    audit.check('F','1000_step_repeat_bitwise',np.array_equal(first,second) and np.isfinite(first).all())
    e=MultiUAVCombatEnv(cfg);e.red=[state(speed=speeds[i%3],pitch=.1*i,heading=-.2*i) for i in range(4)];e.blue=[s.copy() for s in e.red]
    for action in sequence[:100]:
        commands=np.tile(action,(4,1));e._advance(e.red,commands);e._advance(e.blue,commands)
    audit.check('F','red_blue_identical_actions_states',np.array_equal([s.as_array() for s in e.red],[s.as_array() for s in e.blue]))
    for psi in [math.pi-.001,-math.pi+.001,math.pi,-math.pi,0.]:
        for av in [-1.,1.]:audit.run(f'F_wrap_{psi:g}_{av:g}','F_wrap',state(heading=psi),[av,0,0],dt)
    for pitch in [-math.pi/9,math.pi/9]:
        row,tr=audit.run(f'F_zero_action_pitch_{pitch:g}','F_coast',state(pitch=pitch,heading=math.pi/2),[0,0,0],10.)
        audit.check('F','zero_action_keeps_existing_pitch',np.all(tr[:,4]==pitch) and abs(tr[-1,2]+3000)>100)
    dead=state(alive=False);copy_dead=advance_aircraft(dead,[1,1,1],dt,cfg['action']['command'],e.spec)
    audit.check('F','dead_frozen_copy',np.array_equal(dead.as_array(),copy_dead.as_array()) and copy_dead is not dead and not copy_dead.alive)
    for index,bad in enumerate([[1,2],[1,2,3,4],[[0,0,0]],[np.nan,0,0],[0,np.inf,0],[0,0,-np.inf]]):
        try:advance_aircraft(state(),bad,dt,cfg['action']['command'],e.spec)
        except ValueError:ok=True
        else:ok=False
        audit.check('F',f'invalid_isolated_{index}',ok)
    for index,bad in enumerate([np.zeros((3,3)),np.full((4,3),np.nan),np.full((4,3),np.inf)]):
        e=quiet_full_environment(cfg,state())
        try:e.step(bad,np.zeros((4,3)))
        except ValueError:ok=True
        else:ok=False
        audit.check('F',f'invalid_full_{index}',ok)
    for height in [0.,-1e-7,1e-7,6000.,6000.+1e-7,6000.-1e-7]:
        e=MultiUAVCombatEnv(cfg);s=state(height=height);e.red=[s.copy()]+[state(alive=False) for _ in range(3)];e.blue=[s.copy()]+[state(alive=False) for _ in range(3)]
        e._resolve_noncombat_losses();expected=boundary_flags(cfg,s)
        audit.check('F',f'exact_altitude_boundary_{height!r}',all(getattr(e,side)[0].alive==expected['alive'] and e.combat_counts[side]['ceiling_losses']==int(expected['ceiling']) for side in ['red','blue']))
    for radius in [5000.,5000.+1e-7,5000.-1e-7]:
        e=MultiUAVCombatEnv(cfg);s=state(x=radius);e.red=[s.copy()]+[state(alive=False) for _ in range(3)];e.blue=[state(alive=False) for _ in range(4)];e._resolve_noncombat_losses()
        audit.check('F',f'exact_radius_{radius!r}',e.red[0].alive==boundary_flags(cfg,s)['alive'])
    # Existing function, re-executed on actual v339 instances; no historical result files.
    from tools.audit_combat_v332_kinematics import circle_audit
    for speed in speeds:
        previous=circle_audit(MultiUAVCombatEnv(cfg),speed,16.)
        audit.check('existing_circle_v339',f'v{speed:g}',abs(previous['radius_fitted_m']-previous['radius_discrete_m'])<1e-7)
    return audit

def write_results(output,audit):
    output.mkdir(parents=True,exist_ok=False)
    fields=list(dict.fromkeys(k for row in audit.rows for k in row))
    with (output/'motion_cases.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(audit.rows)
    np.savez_compressed(output/'trajectories.npz',**audit.traces)
    cfg=audit.cfg;e=MultiUAVCombatEnv(cfg)
    protected={str(ROOT/f):hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0') if f and (ROOT/f).is_file()}
    summary=dict(environment_version='3.39',execution_device='CPU',CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
        python=platform.python_version(),numpy=np.__version__,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        config=cfg,config_sha256=hashlib.sha256((ROOT/'configs/combat_environment_v339.yaml').read_bytes()).hexdigest(),
        dispatch=dict(environment=type(e).__name__,advance_module=e._advance.__func__.__module__,advance_line=inspect.getsourcelines(e._advance.__func__)[1],kinematics_module=advance_aircraft.__module__,
            has_legacy_dynamics=hasattr(e,'dynamics'),has_legacy_integrator=hasattr(e,'integrator')),
        case_count=len(audit.rows),trace_count=len(audit.traces),check_count=len(audit.checks),
        passed_checks=sum(c['passed'] for c in audit.checks),all_passed=all(c['passed'] for c in audit.checks),
        failures=[c for c in audit.checks if not c['passed']],checks=audit.checks,
        max_position_formula_error_m=max(row.get('position_error_max_m',0) for row in audit.rows),
        max_channel_formula_error=max(row.get('channel_error_max',0) for row in audit.rows),
        max_heading_error_rad=max(row.get('heading_error_max_rad',0) for row in audit.rows),
        max_pitch_error_rad=max(row.get('pitch_error_max_rad',0) for row in audit.rows),
        max_speed_error_m_s=max(row.get('speed_error_max_m_s',0) for row in audit.rows),
        boundary_cases=audit.boundaries,source_sha256=protected)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/v339_single_aircraft_audit')
    for flag in ['existing-tests-log','focused-tests-log','supplement-tests-log','protection-manifest']:parser.add_argument('--'+flag,type=Path)
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError('Use a fresh --output-dir; existing audit output is protected')
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    cfg=config();audit=run_experiments(cfg);summary=write_results(args.output_dir,audit)
    import shutil
    for name in ['existing_tests_log','focused_tests_log','supplement_tests_log']:
        original=getattr(args,name)
        if original:shutil.copy2(original,args.output_dir/(name+'.txt'))
    if args.protection_manifest:
        before=json.loads(args.protection_manifest.read_text(encoding='utf-8'))
        changed=[p for p,h in before.items() if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
        summary['preservation']=dict(protected_files=len(before),unchanged=not changed,changed=changed)
        (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    from tools.report_v339_single_aircraft import render_report
    render_report(args.output_dir)
    print(json.dumps({k:summary[k] for k in ['execution_device','case_count','check_count','passed_checks','all_passed','max_position_formula_error_m','max_channel_formula_error','failures']},indent=2))
    if not summary['all_passed']:raise SystemExit(1)

if __name__=='__main__':main()
