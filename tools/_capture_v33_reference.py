"""Pre-extension state, observation, reward decomposition and full-info reference."""
import hashlib,json
from pathlib import Path
import numpy as np
from env.combat_env import MultiUAVCombatEnv

def canonical(value):
    if isinstance(value,np.ndarray):return dict(dtype=str(value.dtype),shape=value.shape,bytes=value.tobytes().hex())
    if isinstance(value,np.generic):return canonical(value.item())
    if isinstance(value,float):return {'float64_hex':value.hex()}
    if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [canonical(v) for v in value]
    return value

def states(e):
    return np.asarray([[s.x,s.y,s.z,s.v,s.theta,s.psi,float(s.alive)] for s in e.red+e.blue],dtype=np.float64)

def capture():
    result={}
    for version in ('30','31','32'):
        for mode in ('ZERO','MIRROR'):
            for seed in (1,31,32000000,33000000):
                e=MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml');obs,info=e.reset(seed);h=hashlib.sha256()
                h.update(json.dumps(canonical((states(e),obs,info)),sort_keys=True).encode())
                while True:
                    action=np.zeros((5,3)) if mode=='ZERO' else e.fixed_policy.team_actions(e.red,e.blue)
                    obs,reward,terminated,truncated,info=e.step(action)
                    h.update(json.dumps(canonical((states(e),obs,reward,terminated,truncated,info,e.rng.bit_generator.state)),sort_keys=True).encode())
                    if terminated or truncated:break
                result[f'{version}_{mode}_{seed}']=h.hexdigest()
    return result

if __name__=='__main__':
    out=Path('outputs/v33_validation');out.mkdir(parents=True,exist_ok=True)
    Path('tests/v33_legacy_reference.json').write_text(json.dumps(capture(),indent=2)+'\n')
    files=[p for folder in ('env','algorithm','configs') for p in Path(folder).rglob('*') if p.suffix in ('.py','.yaml')]
    (out/'before_hashes.json').write_text(json.dumps({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
