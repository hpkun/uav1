"""One-time v3.9 config and minimal version plumbing extension."""
from pathlib import Path
import yaml


def main():
    c = yaml.safe_load(Path('configs/combat_environment_v38.yaml').read_text())
    c['environment_version'] = '3.9'
    c['sensor']['range_max'] = 10000.
    c['reward'].update(guide_distance_min=4000., tactical_distance_max=4000.)
    Path('configs/combat_environment_v39.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    for name in ('mappo', 'rmappo', 'ea_mappo', 'stea_mappo'):
        Path(f'configs/{name}_5v5_v39.yaml').write_bytes(Path(f'configs/{name}_5v5_v38.yaml').read_bytes())
    files = ['env/config.py', 'algorithm/common/evaluator.py', 'algorithm/mappo/runner.py',
        'algorithm/ea_mappo/protocol.py', 'algorithm/ea_mappo/runner.py',
        'algorithm/rmappo/protocol.py', 'algorithm/rmappo/runner.py',
        'algorithm/maddpg/protocol.py', 'algorithm/madsac/protocol.py', 'tests/test_formal_8v8.py']
    for file in files:
        p = Path(file)
        s = p.read_text()
        s = s.replace('"3.8"}', '"3.8", "3.9"}')
        s = s.replace("'3.8')", "'3.8','3.9')")
        s = s.replace("'3.8', (66, 3, 5))", "'3.8', (66, 3, 5)), ('3.9', (66, 3, 5))")
        s = s.replace('3.7 or 3.8', '3.7, 3.8 or 3.9').replace('v3.7/v3.8 ', 'v3.7/v3.8/v3.9 ')
        if file == 'env/config.py':
            marker = "    if isinstance(config, dict) and str(config.get('environment_version')) == '3.8':"
            s = s.replace(marker, "    if isinstance(config, dict) and str(config.get('environment_version')) == '3.9':\n        from .v39_config import validate_v39\n        return validate_v39(config)\n" + marker)
        p.write_text(s)
    p = Path('env/combat_env.py')
    s = p.read_text()
    marker = "            if str(candidate.get('environment_version')) == '3.8':"
    p.write_text(s.replace(marker, "            if str(candidate.get('environment_version')) == '3.9':\n                from .combat_v39 import CombatEnvironmentV39\n                return object.__new__(CombatEnvironmentV39)\n" + marker))
    p = Path('tools/smoke_formal_8v8.py')
    s = p.read_text()
    s = s.replace("            if env['environment_version']=='3.8': stem += '_v38'", "            if env['environment_version']=='3.8': stem += '_v38'\n            if env['environment_version']=='3.9': stem += '_v39'")
    p.write_text(s)


if __name__ == '__main__':
    main()
