"""One-time version compatibility plumbing; no algorithm parameter changes."""
from pathlib import Path

def main():
    p=Path('env/config.py');s=p.read_text().replace('"3.4", "3.5"','"3.4", "3.5", "3.6"').replace("'3.4', '3.5')","'3.4', '3.5', '3.6')")
    s=s.replace("    if isinstance(config, dict) and str(config.get('environment_version')) == '3.5':","    if isinstance(config, dict) and str(config.get('environment_version')) == '3.6':\n        from .v36_config import validate_v36\n        return validate_v36(config)\n    if isinstance(config, dict) and str(config.get('environment_version')) == '3.5':")
    p.write_text(s)
    p=Path('env/combat_env.py');s=p.read_text().replace("            if str(candidate.get('environment_version')) == '3.5':","            if str(candidate.get('environment_version')) == '3.6':\n                from .combat_v36 import CombatEnvironmentV36\n                return object.__new__(CombatEnvironmentV36)\n            if str(candidate.get('environment_version')) == '3.5':");p.write_text(s)
    for name in ('rmappo','ea_mappo'):
        for f in ('protocol','runner'):
            p=Path(f'algorithm/{name}/{f}.py');s=p.read_text().replace('"3.4","3.5"','"3.4","3.5","3.6"').replace('3.4 or 3.5','3.4, 3.5 or 3.6');p.write_text(s)
    p=Path('algorithm/mappo/runner.py');p.write_text(p.read_text().replace("in ('3.3','3.4','3.5') else {}","in ('3.3','3.4','3.5','3.6') else {}"))
    p=Path('algorithm/madsac/protocol.py');p.write_text(p.read_text().replace("(ENVIRONMENT_VERSION,'3.3','3.4','3.5')","(ENVIRONMENT_VERSION,'3.3','3.4','3.5','3.6')").replace('v3.3/v3.4/v3.5 5v5/66D','v3.3/v3.4/v3.5/v3.6 5v5/66D'))
    p=Path('algorithm/maddpg/protocol.py');p.write_text(p.read_text().replace("('3.5', (66, 3, 5))","('3.5', (66, 3, 5)), ('3.6', (66, 3, 5))").replace('v3.3/v3.4/v3.5 66/3/5','v3.3/v3.4/v3.5/v3.6 66/3/5'))
    p=Path('algorithm/common/evaluator.py');s=p.read_text().replace("'3.4', '3.5')","'3.4', '3.5', '3.6')").replace("in ('3.3','3.4','3.5')","in ('3.3','3.4','3.5','3.6')").replace("== '3.5' for row in records","in ('3.5','3.6') for row in records");p.write_text(s)
    p=Path('tools/smoke_formal_8v8.py');p.write_text(p.read_text().replace("if env['environment_version']=='3.5': stem += '_v35'","if env['environment_version']=='3.5': stem += '_v35'\n            if env['environment_version']=='3.6': stem += '_v36'"))
    p=Path('tests/test_combat_protocol.py');s=p.read_text()
    for stem in ('combat_environment','mappo_5v5','rmappo_5v5','ea_mappo_5v5','stea_mappo_5v5'):
        s=s.replace(f"'{stem}_v35.yaml',",f"'{stem}_v35.yaml', '{stem}_v36.yaml',")
    p.write_text(s)
    p=Path('tests/test_formal_8v8.py');p.write_text(p.read_text().replace('3.4 or 3.5','3.4, 3.5 or 3.6'))

if __name__=='__main__':main()
