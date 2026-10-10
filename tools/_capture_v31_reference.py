import hashlib,json
from pathlib import Path
import numpy as np
from env.combat_env import MultiUAVCombatEnv

def capture():
    result={}
    for version in ('23','24','25','26','27','28','29','30'):
        path='configs/combat_environment.yaml' if version=='23' else f'configs/combat_environment_v{version}.yaml'
        rows={}
        for seed in (1,31,32000000):
            e=MultiUAVCombatEnv(path);obs,info=e.reset(seed)
            h=hashlib.sha256()
            for step in range(21):
                for side in (e.red,e.blue):
                    h.update(np.asarray([[s.x,s.y,s.z,s.v,s.theta,s.psi,float(s.alive)] for s in side],dtype=np.float64).tobytes())
                h.update(obs.tobytes())
                if step<20:
                    obs,reward,_,_,_=e.step(np.zeros((e.team_size,3)))
                    h.update(reward.tobytes())
            rows[str(seed)]=h.hexdigest()
        result[version]=rows
    return result

if __name__=='__main__':
    Path('tests/v31_legacy_reference.json').write_text(json.dumps(capture(),indent=2)+'\n')
    Path('outputs/v31_validation').mkdir(parents=True,exist_ok=True)
    files=[p for folder in ('env','algorithm','configs') for p in Path(folder).rglob('*') if p.suffix in ('.py','.yaml')]
    Path('outputs/v31_validation/before_hashes.json').write_text(json.dumps({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
