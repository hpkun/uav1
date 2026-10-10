"""Capture old v3.4 reference and prepare isolated v3.5 protocol files."""
import json,hashlib
from pathlib import Path
import numpy as np
from tools._capture_v33_reference import canonical,states
from env.combat_env import MultiUAVCombatEnv

def capture34():
    result={}
    for mode in ('ZERO','MIRROR'):
        for seed in (1,31,33000000,39000000):
            e=MultiUAVCombatEnv('configs/combat_environment_v34.yaml');obs,info=e.reset(seed);h=hashlib.sha256()
            h.update(json.dumps(canonical((states(e),obs,info)),sort_keys=True).encode())
            while True:
                a=np.zeros((5,3)) if mode=='ZERO' else e.fixed_policy.team_actions(e.red,e.blue)
                obs,r,t,tr,info=e.step(a)
                h.update(json.dumps(canonical((states(e),obs,r,t,tr,info,e.rng.bit_generator.state)),sort_keys=True).encode())
                if t or tr:break
            result[f'{mode}_{seed}']=h.hexdigest()
    return result

def main():
    out=Path('outputs/v35_validation');out.mkdir(parents=True,exist_ok=True)
    (out/'before_hashes.json').write_text(json.dumps({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ('env','algorithm','configs') for p in Path(folder).rglob('*') if p.suffix in ('.py','.yaml')},indent=2))
    Path('tests/v35_v34_reference.json').write_text(json.dumps(capture34(),indent=2))
    s=Path('configs/combat_environment_v33.yaml').read_text().replace("'3.3'","'3.5'")
    s=s.replace('  guard_radius: 4500.0\n  guard_altitude_min: 500.0\n  guard_altitude_max: 5500.0','  close_distance: 1800.0\n  dispersion_threshold: 1500.0')
    Path('configs/combat_environment_v35.yaml').write_text(s)
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        Path(f'configs/{name}_5v5_v35.yaml').write_bytes(Path(f'configs/{name}_5v5_v33.yaml').read_bytes())
    p=Path('env/config.py');s=p.read_text().replace('"3.3", "3.4"','"3.3", "3.4", "3.5"').replace("'3.3', '3.4')","'3.3', '3.4', '3.5')")
    s=s.replace('    if isinstance(config, dict) and str(config.get(\'environment_version\')) == \'3.4\':',"    if isinstance(config, dict) and str(config.get('environment_version')) == '3.5':\n        from .v35_config import validate_v35\n        return validate_v35(config)\n    if isinstance(config, dict) and str(config.get('environment_version')) == '3.4':")
    p.write_text(s)
    p=Path('env/combat_env.py');s=p.read_text().replace("            if str(candidate.get('environment_version')) == '3.4':","            if str(candidate.get('environment_version')) == '3.5':\n                from .combat_v35 import CombatEnvironmentV35\n                return object.__new__(CombatEnvironmentV35)\n            if str(candidate.get('environment_version')) == '3.4':");p.write_text(s)
    for name in ('rmappo','ea_mappo'):
        for f in ('protocol','runner'):
            p=Path(f'algorithm/{name}/{f}.py');s=p.read_text().replace('"3.3","3.4"','"3.3","3.4","3.5"').replace('3.3 or 3.4','3.3, 3.4 or 3.5');p.write_text(s)
    p=Path('algorithm/mappo/runner.py');p.write_text(p.read_text().replace("in ('3.3','3.4') else {}","in ('3.3','3.4','3.5') else {}"))
    p=Path('algorithm/madsac/protocol.py');p.write_text(p.read_text().replace("(ENVIRONMENT_VERSION,'3.3','3.4')","(ENVIRONMENT_VERSION,'3.3','3.4','3.5')").replace('v3.3/v3.4 5v5/66D','v3.3/v3.4/v3.5 5v5/66D'))
    p=Path('algorithm/maddpg/protocol.py');p.write_text(p.read_text().replace("('3.4', (66, 3, 5))","('3.4', (66, 3, 5)), ('3.5', (66, 3, 5))").replace('v3.3/v3.4 66/3/5','v3.3/v3.4/v3.5 66/3/5'))
    p=Path('algorithm/common/evaluator.py');s=p.read_text().replace("'3.3', '3.4')","'3.3', '3.4', '3.5')").replace("in ('3.3','3.4')","in ('3.3','3.4','3.5')");p.write_text(s)
    p=Path('tools/smoke_formal_8v8.py');p.write_text(p.read_text().replace("if env['environment_version']=='3.4': stem += '_v34'","if env['environment_version']=='3.4': stem += '_v34'\n            if env['environment_version']=='3.5': stem += '_v35'"))
    p=Path('tests/test_combat_protocol.py');s=p.read_text()
    for stem in ('combat_environment','mappo_5v5','rmappo_5v5','ea_mappo_5v5','stea_mappo_5v5'):
        s=s.replace(f"'{stem}_v34.yaml',",f"'{stem}_v34.yaml', '{stem}_v35.yaml',")
    p.write_text(s)
    p=Path('tests/test_formal_8v8.py');p.write_text(p.read_text().replace('3.3 or 3.4','3.3, 3.4 or 3.5'))

if __name__=='__main__':main()
