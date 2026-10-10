"""Capture pre-hook historical trajectories, including terminal rewards."""
import hashlib,json
from pathlib import Path
import numpy as np
from env.combat_env import MultiUAVCombatEnv

def trajectory_hash(e,seed,mode):
    obs,_=e.reset(seed);h=hashlib.sha256()
    while True:
        for side in (e.red,e.blue):
            h.update(np.asarray([[s.x,s.y,s.z,s.v,s.theta,s.psi,float(s.alive)] for s in side],dtype=np.float64).tobytes())
        h.update(obs.tobytes())
        actions=np.zeros((5,3)) if mode=='ZERO' else e.fixed_policy.team_actions(e.red,e.blue)
        obs,reward,terminated,truncated,info=e.step(actions)
        h.update(reward.tobytes())
        h.update(info['termination_reason'].encode())
        h.update(np.asarray([terminated,truncated],dtype=bool).tobytes())
        if terminated or truncated:
            h.update(obs.tobytes())
            return h.hexdigest()

def capture():
    return {f'{version}_{mode}_{seed}':trajectory_hash(MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml'),seed,mode)
        for version in ('30','31') for mode in ('ZERO','MIRROR') for seed in (1,31,32000000,33000000)}

if __name__=='__main__':
    out=Path('outputs/v32_validation');out.mkdir(parents=True,exist_ok=True)
    Path('tests/v32_legacy_reference.json').write_text(json.dumps(capture(),indent=2)+'\n')
    files=[p for folder in ('env','algorithm','configs') for p in Path(folder).rglob('*') if p.suffix in ('.py','.yaml')]
    (out/'before_hashes.json').write_text(json.dumps({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
