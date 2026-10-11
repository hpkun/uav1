"""One-time creation of independent configs and version dispatch extensions."""
from pathlib import Path
import yaml
from env.v311_config import EXPECTED
from copy import deepcopy


def main():
    config = dict(environment_version='3.11', **deepcopy(EXPECTED))
    Path('configs/combat_environment_v311.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    for name in ('mappo', 'rmappo', 'ea_mappo', 'stea_mappo'):
        Path(f'configs/{name}_5v5_v311.yaml').write_bytes(Path(f'configs/{name}_5v5_v310.yaml').read_bytes())
    paths = ['env/config.py', 'algorithm/common/evaluator.py', 'algorithm/mappo/runner.py',
        'algorithm/ea_mappo/protocol.py', 'algorithm/ea_mappo/runner.py',
        'algorithm/rmappo/protocol.py', 'algorithm/rmappo/runner.py',
        'algorithm/maddpg/protocol.py', 'algorithm/madsac/protocol.py', 'tests/test_formal_8v8.py']
    for file in paths:
        p = Path(file)
        s = p.read_text()
        s = s.replace('"3.10"}', '"3.10", "3.11"}')
        s = s.replace("'3.10')", "'3.10','3.11')")
        s = s.replace("'3.10', (66, 3, 5))", "'3.10', (66, 3, 5)), ('3.11', (66, 3, 5))")
        s = s.replace('3.9 or 3.10', '3.9, 3.10 or 3.11').replace('v3.9/v3.10 ', 'v3.9/v3.10/v3.11 ')
        if file in ('algorithm/common/evaluator.py', 'algorithm/mappo/runner.py'):
            s = s.replace("== '3.10'", "in ('3.10', '3.11')")
        if file == 'env/config.py':
            marker = "    if isinstance(config, dict) and str(config.get('environment_version')) == '3.10':"
            s = s.replace(marker, "    if isinstance(config, dict) and str(config.get('environment_version')) == '3.11':\n        from .v311_config import validate_v311\n        return validate_v311(config)\n" + marker)
        p.write_text(s)
    p = Path('env/combat_env.py')
    s = p.read_text()
    marker = "            if str(candidate.get('environment_version')) == '3.10':"
    p.write_text(s.replace(marker, "            if str(candidate.get('environment_version')) == '3.11':\n                from .combat_v311 import CombatEnvironmentV311\n                return object.__new__(CombatEnvironmentV311)\n" + marker))
    p = Path('tools/smoke_formal_8v8.py')
    s = p.read_text().replace("            if env['environment_version']=='3.10': stem += '_v310'",
        "            if env['environment_version']=='3.10': stem += '_v310'\n            if env['environment_version']=='3.11': stem += '_v311'")
    p.write_text(s)
    p = Path('tests/test_combat_protocol.py')
    s = p.read_text()
    for name in ('combat_environment', 'mappo_5v5', 'rmappo_5v5', 'ea_mappo_5v5', 'stea_mappo_5v5'):
        marker = f"'{name}_v310.yaml',"
        s = s.replace(marker, marker + f" '{name}_v311.yaml',")
    p.write_text(s)


if __name__ == '__main__':
    main()
