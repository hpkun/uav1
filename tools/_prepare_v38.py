"""Capture v3.7 trajectories before adding the initialization hook."""
import hashlib,json
from pathlib import Path
import numpy as np
import yaml
from env.combat_env import MultiUAVCombatEnv
from tools._capture_v33_reference import canonical,states

def capture37():
    result={}
    for mode in ('ZERO','MIRROR'):
        for seed in (1,31,10000000,40000000):
            e=MultiUAVCombatEnv('configs/combat_environment_v37.yaml');obs,info=e.reset(seed);h=hashlib.sha256()
            h.update(json.dumps(canonical((states(e),obs,info)),sort_keys=True).encode())
            while True:
                a=np.zeros((5,3)) if mode=='ZERO' else e.fixed_policy.team_actions(e.red,e.blue)
                obs,r,t,tr,info=e.step(a)
                h.update(json.dumps(canonical((states(e),obs,r,t,tr,info,e.rng.bit_generator.state,e.fixed_policy.rng.bit_generator.state)),sort_keys=True).encode())
                if t or tr:break
            result[f'{mode}_{seed}']=h.hexdigest()
    return result

def main():
    out=Path('outputs/v38_validation');out.mkdir(parents=True,exist_ok=True)
    (out/'before_hashes.json').write_text(json.dumps({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('env','algorithm','configs') for p in Path(folder).rglob('*') if p.suffix in ('.py','.yaml')},indent=2))
    Path('tests/v38_v37_reference.json').write_text(json.dumps(capture37(),indent=2))
    c=yaml.safe_load(Path('configs/combat_environment_v37.yaml').read_text());c['environment_version']='3.8'
    c['scenario'].pop('center_radius')
    Path('configs/combat_environment_v38.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        Path(f'configs/{name}_5v5_v38.yaml').write_bytes(Path(f'configs/{name}_5v5_v37.yaml').read_bytes())

if __name__=='__main__':main()
