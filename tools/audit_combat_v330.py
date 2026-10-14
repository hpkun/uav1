"""Read-only 4v4 geometry/control and 200-seed fixed-pursuit fairness audit."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.v329_control import action_to_control_v329
from env.models import AircraftState
from env.fixed_policy import NearestTargetPursuitPolicy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/v330_validation/fairness.json')
    args=p.parse_args()
    if args.episodes<200:raise ValueError('at least 200 seeds required')
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory for repository audits')
    e=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v330.yaml');obs,info=e.reset(0)
    reset=dict(observation_shape=list(obs.shape),red=[s.as_array().tolist() for s in e.red],blue=[s.as_array().tolist() for s in e.blue],
        anchor_separation=info['nominal_anchor_separation'],adjacent_spacing=np.linalg.norm(np.diff(np.array([s.as_array()[:3] for s in e.red]),axis=0),axis=1).tolist(),
        nearest_targets=[e.fixed_policy.nearest_target_index(s,e.red) for s in e.blue],
        eligibility_shape=list(e.weapon.eligibility(e.red,e.blue,e.red_ammo)[0].shape))
    load=[]
    for a in np.linspace(-1,1,9):
        for b in np.linspace(-1,1,9):load.append(action_to_control_v329(AircraftState(0,0,-3000,225,0,0),[a,b,0],e.config['action'],e.spec).nz)
    rows=[]
    red_policy=NearestTargetPursuitPolicy(e.config['blue_policy'],e.config['action'])
    for seed in range(args.episodes):
        e.reset(seed)
        # Only the diagnostic Red action source changes: identical nearest pursuit.
        while True:
            # Same mathematical policy, separate stateless instance: do not
            # contaminate Blue's diagnostic target-switch counters with Red calls.
            actions=red_policy.team_actions(e.red,e.blue)
            _,reward,terminated,truncated,info=e.step(actions)
            if terminated or truncated:break
        rows.append(dict(seed=seed,red_win=info['red_win'],blue_win=info['blue_win'],draw=info['draw'],timeout=truncated,
            episode_length=e.steps,mutual_destruction=not e.red_alive_mask.any() and not e.blue_alive_mask.any(),
            red_attack_kills=info['red_attack_kills'],blue_attack_kills=info['blue_attack_kills'],
            red_noncombat_losses=info['red_boundary_exits']+info['red_ground_losses'],
            blue_noncombat_losses=info['blue_boundary_exits']+info['blue_ground_losses'],
            blue_target_switches=info['blue_target_switches'],termination_reason=info['termination_reason'],
            reward_components={f'r{i}':info[f'episode_r{i}_total'] for i in range(1,5)}))
        if (seed+1)%10==0:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.with_name('fairness_progress.json').write_text(json.dumps(rows))
            print(f'[FAIRNESS] {seed+1}/{args.episodes}',flush=True)
    totals={k:sum(r[k] for r in rows) for k in ('red_win','blue_win','draw','timeout','mutual_destruction',
        'red_attack_kills','blue_attack_kills','red_noncombat_losses','blue_noncombat_losses')}
    result=dict(reset=reset,controller_grid=dict(max_nz=max(load),saturation_fraction=float(np.mean(np.array(load)>=9.9))),
        totals=totals,episodes=args.episodes,rows=rows,
        note='Mutual destruction is classified as draw in v3.30; other elimination, timeout and ongoing outcome semantics remain inherited. This audit uses identical nearest-target pursuit mathematics for both teams and is not a training policy.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(totals),flush=True)


if __name__=='__main__':main()
