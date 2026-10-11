"""One-time independent config creation and minimal version plumbing."""
from pathlib import Path
import yaml


def main():
    c = yaml.safe_load(Path('configs/combat_environment_v39.yaml').read_text())
    c['environment_version'] = '3.10'
    c['blue_policy'] = dict(desired_speed=250.)
    Path('configs/combat_environment_v310.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    for name in ('mappo', 'rmappo', 'ea_mappo', 'stea_mappo'):
        Path(f'configs/{name}_5v5_v310.yaml').write_bytes(Path(f'configs/{name}_5v5_v39.yaml').read_bytes())
    paths = ['env/config.py', 'algorithm/common/evaluator.py', 'algorithm/mappo/runner.py',
        'algorithm/ea_mappo/protocol.py', 'algorithm/ea_mappo/runner.py',
        'algorithm/rmappo/protocol.py', 'algorithm/rmappo/runner.py',
        'algorithm/maddpg/protocol.py', 'algorithm/madsac/protocol.py', 'tests/test_formal_8v8.py']
    for file in paths:
        p = Path(file)
        s = p.read_text()
        s = s.replace('"3.9"}', '"3.9", "3.10"}')
        s = s.replace("'3.9')", "'3.9','3.10')")
        s = s.replace("'3.9', (66, 3, 5))", "'3.9', (66, 3, 5)), ('3.10', (66, 3, 5))")
        s = s.replace('3.8 or 3.9', '3.8, 3.9 or 3.10').replace('v3.8/v3.9 ', 'v3.8/v3.9/v3.10 ')
        if file in ('algorithm/common/evaluator.py', 'algorithm/mappo/runner.py'):
            s = s.replace("== '3.9'", "in ('3.9', '3.10')")
        if file == 'env/config.py':
            marker = "    if isinstance(config, dict) and str(config.get('environment_version')) == '3.9':"
            s = s.replace(marker, "    if isinstance(config, dict) and str(config.get('environment_version')) == '3.10':\n        from .v310_config import validate_v310\n        return validate_v310(config)\n" + marker)
        p.write_text(s)
    p = Path('env/combat_env.py')
    s = p.read_text()
    marker = "            if str(candidate.get('environment_version')) == '3.9':"
    p.write_text(s.replace(marker, "            if str(candidate.get('environment_version')) == '3.10':\n                from .combat_v310 import CombatEnvironmentV310\n                return object.__new__(CombatEnvironmentV310)\n" + marker))
    p = Path('tools/smoke_formal_8v8.py')
    s = p.read_text().replace("            if env['environment_version']=='3.9': stem += '_v39'",
        "            if env['environment_version']=='3.9': stem += '_v39'\n            if env['environment_version']=='3.10': stem += '_v310'")
    p.write_text(s)
    p = Path('tests/test_combat_protocol.py')
    s = p.read_text()
    for name in ('combat_environment', 'mappo_5v5', 'rmappo_5v5', 'ea_mappo_5v5', 'stea_mappo_5v5'):
        # Lexicographic inventory: v31, v310, v32.
        marker = f"'{name}_v31.yaml',"
        s = s.replace(marker, marker + f" '{name}_v310.yaml',")
    p.write_text(s)


if __name__ == '__main__':
    main()
