"""Network-free v3.30/v3.32 motion comparison and fixed-pursuit fairness audit."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.fixed_policy import NearestTargetPursuitPolicy
from env.math_utils import wrap_angle

ACTIONS=([.1,0,0],[.2,0,0],[.3,0,0],[.5,0,0],[1,0,0],
         [0,.1,0],[0,.5,0],[0,1,0],[0,-1,0],[.5,.5,0],[1,1,0],
         [1,.05,0],[0,0,.2],[0,0,.5],[0,0,1],[0,0,-1])


def circle_audit(e,speed,seconds=16.):
    e.reset(1)
    e.red=[AircraftState(0,0,-3000,speed,0,0)]+[AircraftState(0,0,-3000,speed,0,0,False) for _ in range(3)]
    states=[e.red[0].as_array()]
    for _ in range(round(seconds/e.dt)):
        e._advance(e.red,np.tile([1.,0.,0.],(4,1)));states.append(e.red[0].as_array())
    states=np.asarray(states);x,y=states[:,:2].T
    cx,cy,c=np.linalg.lstsq(np.column_stack((2*x,2*y,np.ones_like(x))),x*x+y*y,rcond=None)[0]
    radius=float(np.sqrt(cx*cx+cy*cy+c))
    theory=speed/(np.pi/4);discrete=speed*e.dt/(2*np.sin(np.pi/80))
    assert abs(radius-discrete)<1e-7
    np.testing.assert_array_equal(states[:,2],np.full(len(states),-3000.))
    np.testing.assert_array_equal(states[:,3],np.full(len(states),speed))
    np.testing.assert_array_equal(states[:,4],np.zeros(len(states)))
    return dict(speed=speed,duration_s=seconds,action=[1,0,0],radius_continuous_m=theory,
                radius_discrete_m=discrete,radius_fitted_m=radius,altitude_constant=True,
                speed_constant=True,pitch_constant=True,no_ground_loss=True,no_vertical_compensation=True)


def direct_comparison(old,new):
    rows=[]
    for speed in (150.,225.,300.):
        for action in ACTIONS:
            row=dict(initial_speed=speed,action=list(action))
            for version,e in (('v330',old),('v332',new)):
                e.reset(1);e.red=[AircraftState(0,0,-3000,speed,0,0)]+[AircraftState(0,0,-3000,speed,0,0,False) for _ in range(3)]
                before=e.red[0].copy();a=np.zeros((4,3),dtype=np.float32);a[0]=action
                e._advance(e.red,a);after=e.red[0]
                row[version]=dict(yaw_deg_s=float(np.rad2deg(wrap_angle(after.psi-before.psi))/e.dt),
                                  pitch_deg_s=float(np.rad2deg(after.theta-before.theta)/e.dt),
                                  acceleration_m_s2=float((after.v-before.v)/e.dt),
                                  altitude_change_m=after.altitude-before.altitude,speed_change_m_s=after.v-before.v)
            expected=[45*float(a[0,0]),30*float(a[0,1]),
                      (np.clip(speed+2.5*float(a[0,2]),150,300)-speed)/new.dt]
            np.testing.assert_allclose([row['v332']['yaw_deg_s'],row['v332']['pitch_deg_s'],row['v332']['acceleration_m_s2']],expected,rtol=0,atol=1e-10)
            rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,default=200)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/v332_validation/kinematics_audit.json')
    args=parser.parse_args()
    if args.episodes<200:raise ValueError('at least 200 seeds required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory for repository audits')
    old=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v330.yaml')
    e=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v332.yaml')
    comparison=direct_comparison(old,e)
    circles=[circle_audit(e,v) for v in (150.,225.,300.)]
    red_policy=NearestTargetPursuitPolicy(e.config['blue_policy'],e.config['action'])
    rows=[]
    for seed in range(args.episodes):
        e.reset(seed)
        while True:
            _,_,terminated,truncated,info=e.step(red_policy.team_actions(e.red,e.blue))
            assert not info['controller_diagnostics'] and 'executed_red_phi' not in info
            if terminated or truncated:break
        rows.append(dict(seed=seed,episode_length=e.steps,
            **{k:info[k] for k in ('red_win','blue_win','draw','red_attack_kills','blue_attack_kills','termination_reason')},
            timeout=truncated,mutual_destruction=not e.red_alive_mask.any() and not e.blue_alive_mask.any(),
            red_noncombat_losses=info['red_boundary_exits']+info['red_ground_losses'],
            blue_noncombat_losses=info['blue_boundary_exits']+info['blue_ground_losses'],
            red_fire_window_episode=info['red_first_fire_window_step'] is not None,
            blue_fire_window_episode=info['blue_first_fire_window_step'] is not None))
        if (seed+1)%10==0:print(f'[FAIRNESS] {seed+1}/{args.episodes}',flush=True)
    totals={k:sum(r[k] for r in rows) for k in ('red_win','blue_win','draw','timeout','mutual_destruction',
            'red_attack_kills','blue_attack_kills','red_noncombat_losses','blue_noncombat_losses')}
    totals.update(mean_episode_length=float(np.mean([r['episode_length'] for r in rows])),
                  red_fire_window_episode_rate=float(np.mean([r['red_fire_window_episode'] for r in rows])),
                  blue_fire_window_episode_rate=float(np.mean([r['blue_fire_window_episode'] for r in rows])))
    result=dict(environment_version='3.32',episodes=args.episodes,comparison=comparison,circles=circles,totals=totals,rows=rows,
                note='Same nearest-target pursuit mathematics and identical motion capability. Red eligible-target selection and Blue pursuit/fire coupling remain different inherited mechanisms; no physics/controller/G saturation.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(circles=circles,totals=totals),indent=2),flush=True)


if __name__=='__main__':main()
