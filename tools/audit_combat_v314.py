"""Bounded synthetic environment control audit; no learned policy or training."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import numpy as np
import torch
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
from env.control import action_to_control, action_to_target
from env.math_utils import wrap_angle
from env.models import AircraftState
from env.v313_observation import build_team_observations_v313


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_audit(output=ROOT/'outputs/v314_validation'):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is mandatory for the control audit')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT/'configs/combat_environment_v314.yaml')
    e = MultiUAVCombatEnv(config)
    e.reset(31)
    control_rows = []
    commands = [(f'heading_{a:+g}', [a,0,0]) for a in [.1,.25,.5,1.,-.1,-.25,-.5,-1.]]
    commands += [(f'pitch_{a:+g}', [0,a,0]) for a in [.5,-.5]]
    commands += [(f'speed_{a:+g}', [0,0,a]) for a in [.5,-.5]]
    commands += [(f'replay_{a:g}', [a,.60,-.31]) for a in [.426,.442,.454]]
    for case, action in commands:
        aircraft = [AircraftState(0,0,-3000,225,0,0) for _ in range(5)]
        unwrapped = 0.
        for step in range(1,21):
            before = aircraft[0].copy()
            target = action_to_target(before, action, config['action']['command'])
            control = action_to_control(before, action, config['action'])
            phis = e._advance(aircraft, np.tile(action,(5,1)))
            after = aircraft[0]
            yaw = wrap_angle(after.psi-before.psi)
            previous_unwrapped = unwrapped
            unwrapped += yaw
            assert np.isfinite(after.as_array()).all()
            assert abs(yaw)<np.radians(10)
            assert 150 <= after.v <= 300 and abs(after.theta)<=np.pi/3
            assert control.nz<=8.+1e-12 and abs(control.phi)<=np.pi/2
            assert abs(phis[0]-control.phi)<1e-14
            control_rows.append(dict(case=case,step=step,time_s=step*e.dt,
                input_action_heading=action[0],input_action_pitch=action[1],input_action_speed=action[2],
                desired_heading_delta=np.pi*action[0],actual_heading_delta=yaw,
                desired_pitch_delta=(np.pi/3)*action[1],actual_pitch_delta=after.theta-before.theta,
                desired_speed_delta=50*action[2],actual_speed_delta=after.v-before.v,
                nx=control.nx,nz=control.nz,phi=control.phi,
                x=after.x,y=after.y,altitude=after.altitude,speed=after.v,psi=after.psi,theta=after.theta,
                unwrapped_psi=unwrapped,
                desired_unwrapped_psi=previous_unwrapped+wrap_angle(target.heading-before.psi),
                desired_pitch=target.pitch,desired_speed=target.speed))
    write_csv(output/'control_response.csv',control_rows)
    symmetry={}
    for a in [.1,.25,.5]:
        pos=[row for row in control_rows if row['case']==f'heading_{a:+g}']
        neg=[row for row in control_rows if row['case']==f'heading_{-a:+g}']
        error=max(abs(p['actual_heading_delta']+n['actual_heading_delta']) for p,n in zip(pos,neg))
        assert error<1e-12
        symmetry[str(a)]=error
    # +/-1 share the -pi tie direction by the existing wrap convention.
    pos=[row for row in control_rows if row['case']=='heading_+1']
    neg=[row for row in control_rows if row['case']=='heading_-1']
    assert all(p['actual_heading_delta']==n['actual_heading_delta'] for p,n in zip(pos,neg))

    zero_rows=[]
    e.reset(31)
    for step in range(21):
        centers=[np.mean([s.as_array()[:2] for s in side],axis=0) for side in (e.red,e.blue)]
        distance=float(np.linalg.norm(centers[1]-centers[0]))
        zero_rows.append(dict(step=step,time_s=step*e.dt,
            red_center_x=centers[0][0],red_center_y=centers[0][1],
            blue_center_x=centers[1][0],blue_center_y=centers[1][1],center_horizontal_distance=distance,
            red_phi_max=float(np.max(np.abs(e.red_last_executed_phi))),
            blue_phi_max=float(np.max(np.abs(e.blue_last_executed_phi)))))
        if step<20:
            obs,reward,term,trunc,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
            assert np.isfinite(obs).all() and np.isfinite(reward).all()
            assert not term and not trunc
    assert np.all(np.diff([r['center_horizontal_distance'] for r in zero_rows])<0)
    assert zero_rows[-1]['center_horizontal_distance']>4000
    write_csv(output/'zero_action_sanity.csv',zero_rows)

    difference=[]
    for case,action in [('heading_043',[.43,0,0]),('half_heading',[.5,0,0])]+[(f'replay_{a:g}',[a,.60,-.31]) for a in [.426,.442,.454]]:
        for version in ['313','314']:
            instance=MultiUAVCombatEnv(ROOT/f'configs/combat_environment_v{version}.yaml')
            aircraft=[AircraftState(0,0,-3000,225,0,0) for _ in range(5)]
            phi=instance._advance(aircraft,np.tile(action,(5,1)))
            after=aircraft[0]
            c=action_to_control(AircraftState(0,0,-3000,225,0,0),action,config['action']) if version=='314' else None
            difference.append(dict(case=case,version='3.'+version[1:],
                input_action_heading=action[0],input_action_pitch=action[1],input_action_speed=action[2],
                desired_heading_delta=np.pi*action[0],actual_heading_delta=wrap_angle(after.psi),
                desired_heading_delta_deg=180*action[0],actual_heading_delta_deg=float(np.degrees(wrap_angle(after.psi))),
                desired_pitch_delta=(np.pi/3)*action[1],actual_pitch_delta=after.theta,
                desired_speed_delta=50*action[2],actual_speed_delta=after.v-225,
                phi=float(phi[0]),nz=c.nz if c else '',nx=c.nx if c else ''))
    write_csv(output/'v313_v314_action_difference.csv',difference)

    e.reset(31)
    reset_phi=not e._observations()[:,4].any()
    obs,*_=e.step(np.tile([.5,0,0],(5,1)),np.zeros((5,3)))
    np.testing.assert_array_equal(obs[:,4],e.red_last_executed_phi/(np.pi/2))
    original=build_team_observations_v313(e.red,e.blue,e.config['observation'])
    masked=obs.copy();masked[:,4]=0
    np.testing.assert_array_equal(masked,original)
    checks=dict(cuda_device=torch.cuda.get_device_name(0),python=sys.executable,
        control_response_rows=len(control_rows),continuous_steps_per_command=20,
        max_single_step_yaw_deg=float(np.degrees(max(abs(r['actual_heading_delta']) for r in control_rows))),
        max_nz=max(r['nz'] for r in control_rows),max_abs_phi=max(abs(r['phi']) for r in control_rows),
        heading_symmetry_residual=symmetry,full_pi_tie_both_signs_use_minus_pi=True,
        reset_phi_zero=reset_phi,self_phi_matches_executed=True,other_55_features_exact=True,
        zero_center_distance_start=zero_rows[0]['center_horizontal_distance'],
        zero_center_distance_end=zero_rows[-1]['center_horizontal_distance'],zero_distance_strictly_decreases=True,
        first_step_difference=difference,finite_and_bounded=True,policy_executed=False)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name,case,actual,desired,factor,unit,initial in [
        ('heading','heading_+0.5','unwrapped_psi','desired_unwrapped_psi',180/np.pi,'deg',0),
        ('pitch','pitch_+0.5','theta','desired_pitch',180/np.pi,'deg',0),
        ('speed','speed_+0.5','speed','desired_speed',1,'m/s',225)]:
        rows=[r for r in control_rows if r['case']==case]
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot([0]+[r['time_s'] for r in rows],[initial]+[r[actual]*factor for r in rows],label='Actual state after RK4',marker='.',markersize=4)
        ax.plot([r['time_s']-e.dt for r in rows],[r[desired]*factor for r in rows],label='Desired target recomputed each step',linestyle='--')
        ax.set(xlabel='Time (s)',ylabel=f'{name.capitalize()} ({unit})',title=f'v3.14 {name}: relative command +0.5 every 0.1 s')
        ax.grid(alpha=.25);ax.legend();fig.tight_layout()
        fig.savefig(output/f'{name}_response.png',dpi=150);plt.close(fig)
    checks['response_plots_generated']=True
    (output/'checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    return checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/v314_validation')
    result=run_audit(parser.parse_args().output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
