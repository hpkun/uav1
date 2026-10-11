"""Small fixed-seed timing comparison, no learning or parameter tuning."""
import json
from pathlib import Path
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from env.sensor import sensor_geometry
from env.v36_geometry import combat_geometry
from env.fixed_policy import NearestTargetPursuitPolicy

SEEDS=(1,31,10000000)
def trial(version,mode,seed):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml');_,info=e.reset(seed)
    d=combat_geometry(e.red,e.blue)['distance']
    output=dict(version=e.environment_version,policy=mode,seed=seed,
        initial_minimum_inter_team_distance=float(d.min()),initial_mean_inter_team_distance=float(d.mean()),
        initial_maximum_inter_team_distance=float(d.max()),first_detection_step=None)
    pursuit=NearestTargetPursuitPolicy(e.config['blue_policy'],e.config['action'])
    def detect():
        return sensor_geometry(e.red,e.blue,e.config['sensor'])[0].any() or sensor_geometry(e.blue,e.red,e.config['sensor'])[0].any()
    if detect():output['first_detection_step']=0
    while True:
        action=np.zeros((5,3)) if mode=='ZERO' else pursuit.team_actions(e.red,e.blue)
        _,_,t,tr,info=e.step(action)
        if output['first_detection_step'] is None and detect():output['first_detection_step']=e.steps
        if t or tr:break
    for name in ('fire_window','attempt','kill'):
        values=[info[f'{s}_first_{name}_step'] for s in ('red','blue') if info[f'{s}_first_{name}_step'] is not None]
        output[f'first_{name}_step']=min(values) if values else None
        for side in ('red','blue'):output[f'{side}_first_{name}_step']=info[f'{side}_first_{name}_step']
    output.update(episode_length=e.steps,termination_reason=info['termination_reason'],
        red_boundary_exits=info['red_boundary_exits'],blue_boundary_returns=info['blue_boundary_returns'])
    return output

def main():
    rows=[trial(v,m,s) for m in ('ZERO','PURSUIT') for s in SEEDS for v in ('37','38')]
    for i in range(0,len(rows),2):
        before,after=rows[i:i+2]
        assert after['first_detection_step'] is not None and after['first_detection_step']>before['first_detection_step']
        assert after['first_fire_window_step'] is not None and after['first_fire_window_step']>before['first_fire_window_step']
    path=Path('outputs/v38_validation/fixed_policy_comparison.json')
    path.write_text(json.dumps(dict(seeds=SEEDS,rows=rows,training_started=False),indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
