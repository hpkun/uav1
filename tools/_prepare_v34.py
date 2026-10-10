"""One-time v3.4 compatibility preparation and pre-change v3.3 reference."""
from pathlib import Path
import hashlib, json
from tools._capture_v33_reference import canonical, states
from env.combat_env import MultiUAVCombatEnv
import numpy as np

def capture33():
    result={}
    for mode in ('ZERO','MIRROR'):
        for seed in (1,31,32000000,33000000):
            e=MultiUAVCombatEnv('configs/combat_environment_v33.yaml');obs,info=e.reset(seed);h=hashlib.sha256()
            h.update(json.dumps(canonical((states(e),obs,info)),sort_keys=True).encode())
            while True:
                a=np.zeros((5,3)) if mode=='ZERO' else e.fixed_policy.team_actions(e.red,e.blue)
                obs,r,t,tr,info=e.step(a)
                h.update(json.dumps(canonical((states(e),obs,r,t,tr,info,e.rng.bit_generator.state)),sort_keys=True).encode())
                if t or tr:break
            result[f'{mode}_{seed}']=h.hexdigest()
    return result

def main():
    out=Path('outputs/v34_validation');out.mkdir(parents=True,exist_ok=True)
    (out/'before_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('env','algorithm','configs') for p in Path(folder).rglob('*') if p.suffix in ('.py','.yaml')},indent=2))
    Path('tests/v34_v33_reference.json').write_text(json.dumps(capture33(),indent=2))
    p=Path('configs/combat_environment_v33.yaml');s=p.read_text().replace("'3.3'","'3.4'")
    import yaml
    if yaml.safe_load(s)['environment_version']!='3.4':s=s.replace('"3.3"','"3.4"')
    s=s.replace('  guard_altitude_max: 5500.0','  guard_altitude_max: 5500.0\n  evade_enter_distance: 1200.0\n  evade_exit_distance: 1600.0\n  threat_radius: 1500.0\n  support_radius: 1500.0')
    Path('configs/combat_environment_v34.yaml').write_text(s)
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        Path(f'configs/{name}_5v5_v34.yaml').write_bytes(Path(f'configs/{name}_5v5_v33.yaml').read_bytes())
    for f in ('env/config.py','algorithm/common/evaluator.py'):
        p=Path(f);s=p.read_text().replace("'3.0', '3.1', '3.2', '3.3'","'3.0', '3.1', '3.2', '3.3', '3.4'")
        s=s.replace("row.get('environment_version') == '3.3'","row.get('environment_version') in ('3.3','3.4')")
        p.write_text(s)
    for name in ('rmappo','ea_mappo'):
        for f in ('protocol','runner'):
            p=Path(f'algorithm/{name}/{f}.py');s=p.read_text().replace('"3.2","3.3"','"3.2","3.3","3.4"').replace('3.2 or 3.3','3.2, 3.3 or 3.4');p.write_text(s)
    p=Path('algorithm/mappo/runner.py');p.write_text(p.read_text().replace("== '3.3' else {}","in ('3.3','3.4') else {}"))
    p=Path('algorithm/madsac/protocol.py');p.write_text(p.read_text().replace("(ENVIRONMENT_VERSION,'3.3')","(ENVIRONMENT_VERSION,'3.3','3.4')").replace('or v3.3 5v5/66D','or v3.3/v3.4 5v5/66D'))
    p=Path('algorithm/maddpg/protocol.py');p.write_text(p.read_text().replace("('3.3', (66, 3, 5))","('3.3', (66, 3, 5)), ('3.4', (66, 3, 5))").replace('or v3.3 66/3/5','or v3.3/v3.4 66/3/5'))
    p=Path('tools/smoke_formal_8v8.py');p.write_text(p.read_text().replace("if env['environment_version']=='3.3': stem += '_v33'","if env['environment_version']=='3.3': stem += '_v33'\n            if env['environment_version']=='3.4': stem += '_v34'"))
    p=Path('tests/test_combat_protocol.py');s=p.read_text()
    for stem in ('combat_environment','mappo_5v5','rmappo_5v5','ea_mappo_5v5','stea_mappo_5v5'):
        s=s.replace(f"'{stem}_v33.yaml',",f"'{stem}_v33.yaml', '{stem}_v34.yaml',")
    p.write_text(s)
    p=Path('tests/test_formal_8v8.py');p.write_text(p.read_text().replace('3.2 or 3.3','3.2, 3.3 or 3.4'))

if __name__=='__main__':main()
