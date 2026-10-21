"""Compare complete historical CPU trajectories with an isolated HEAD archive.

No weights/evaluation checkpoints; only fixed-control environment transitions.
The baseline directory must be a separately extracted source tree.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

PROBE = r'''
import hashlib,json
import numpy as np
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config
result={}
for version in ('31','32','37','39','330','332','336','341'):
    for seed in (0,31):
        for mode in ('zero','oscillating'):
            e=MultiUAVCombatEnv(load_config(f'configs/combat_environment_v{version}.yaml'))
            obs,_=e.reset(seed);digest=hashlib.sha256(obs.tobytes())
            while True:
                action=np.zeros((e.team_size,3),np.float32)
                if mode=='oscillating':
                    action[:,0]=.3*np.sin(e.steps*.05)
                    action[:,1]=.3*np.sin(e.steps*.1)
                obs,reward,term,trunc,info=e.step(action)
                for array in (obs,reward,np.array([s.as_array() for s in e.red+e.blue]),
                              np.array([s.alive for s in e.red+e.blue]),e.red_ammo,e.blue_ammo):
                    digest.update(array.tobytes())
                digest.update(json.dumps(e.rng.bit_generator.state,sort_keys=True).encode())
                digest.update(bytes((term,trunc)))
                if term or trunc:break
            result[f'{version}_{seed}_{mode}']=dict(sha256=digest.hexdigest(),steps=e.steps)
print(json.dumps(result,sort_keys=True))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    root=Path(__file__).resolve().parents[1]
    a=json.loads(subprocess.check_output([sys.executable,'-c',PROBE],cwd=root,text=True))
    b=json.loads(subprocess.check_output([sys.executable,'-c',PROBE],cwd=args.baseline_root.resolve(),text=True))
    assert a==b,'historical trajectories/RNG changed relative to HEAD'
    payload=dict(episodes=len(a),all_complete_trajectory_rewards_observations_ammo_rng_exact=True,
                 compared_versions=['3.1','3.2','3.7','3.9','3.30','3.32','3.36','3.41'],results=a)
    args.output.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(f'{len(a)} historical episodes: complete state/observation/reward/ammo/RNG hashes match HEAD exactly')


if __name__=='__main__':main()
