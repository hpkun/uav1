"""Capture pre-hook v3.5 regression and prepare frozen v3.6 configuration."""
import json,hashlib,copy
from pathlib import Path
import numpy as np
import yaml
from tools._capture_v33_reference import canonical,states
from env.combat_env import MultiUAVCombatEnv

def capture35():
    result={}
    for mode in ('ZERO','MIRROR'):
        for seed in (1,31,10000000,40000000):
            e=MultiUAVCombatEnv('configs/combat_environment_v35.yaml');obs,info=e.reset(seed);h=hashlib.sha256()
            h.update(json.dumps(canonical((states(e),obs,info)),sort_keys=True).encode())
            while True:
                a=np.zeros((5,3)) if mode=='ZERO' else e.fixed_policy.team_actions(e.red,e.blue)
                obs,r,t,tr,info=e.step(a)
                h.update(json.dumps(canonical((states(e),obs,r,t,tr,info,e.rng.bit_generator.state,e.fixed_policy.rng.bit_generator.state)),sort_keys=True).encode())
                if t or tr:break
            result[f'{mode}_{seed}']=h.hexdigest()
    return result

def main():
    out=Path('outputs/v36_validation');out.mkdir(parents=True,exist_ok=True)
    (out/'before_hashes.json').write_text(json.dumps({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('env','algorithm','configs') for p in Path(folder).rglob('*') if p.suffix in ('.py','.yaml')},indent=2))
    Path('tests/v36_v35_reference.json').write_text(json.dumps(capture35(),indent=2))
    c=yaml.safe_load(Path('configs/combat_environment_v35.yaml').read_text());c['environment_version']='3.6'
    c['sensor']['range_max']=4000.
    c['weapon']=dict(range_min=0.,range_max=2000.,ata_max=np.pi/6,ha_max=np.pi/6,hit_probability=.7,ammo_per_aircraft=6)
    for k in ('potential_gamma','potential_scale','distance_weight','angle_weight'):c['reward'].pop(k)
    c['reward'].update(guide_reward=.001,offense_reward_30deg=.01,offense_reward_15deg=.02,offense_reward_5deg=.10,
        threat_penalty_30deg=-.015,threat_penalty_15deg=-.025,threat_penalty_5deg=-.15,tactical_aspect_max=np.pi/6)
    Path('configs/combat_environment_v36.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        Path(f'configs/{name}_5v5_v36.yaml').write_bytes(Path(f'configs/{name}_5v5_v35.yaml').read_bytes())

if __name__=='__main__':main()
