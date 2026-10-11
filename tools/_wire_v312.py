"""Create independent observation configs and extend only version dispatch."""
from pathlib import Path
import yaml
from env.v312_config import OBSERVATION


def main():
    c = yaml.safe_load(Path('configs/combat_environment_v311.yaml').read_text())
    c['environment_version'] = '3.12'
    c['observation'] = OBSERVATION
    Path('configs/combat_environment_v312.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        c = yaml.safe_load(Path(f'configs/{name}_5v5_v311.yaml').read_text())
        c['network']['observation_dim'] = 56
        if name in ('ea_mappo','stea_mappo'):
            c['network'].update(self_feature_dim=7, observation_layout='madsac56')
        Path(f'configs/{name}_5v5_v312.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
    paths = ['env/config.py', 'algorithm/common/evaluator.py', 'algorithm/mappo/runner.py',
        'algorithm/ea_mappo/protocol.py', 'algorithm/ea_mappo/runner.py',
        'algorithm/rmappo/protocol.py', 'algorithm/rmappo/runner.py',
        'algorithm/madsac/protocol.py', 'tests/test_formal_8v8.py']
    for file in paths:
        p = Path(file)
        s = p.read_text().replace('"3.11"}', '"3.11", "3.12"}')
        s = s.replace("'3.11')", "'3.11','3.12')")
        s = s.replace('3.10 or 3.11', '3.10, 3.11 or 3.12')
        if file == 'algorithm/madsac/protocol.py':
            s = s.replace('5v5/66D', '5v5/66D or v3.12 5v5/56D')
        if file == 'env/config.py':
            marker = "    if isinstance(config, dict) and str(config.get('environment_version')) == '3.11':"
            s = s.replace(marker, "    if isinstance(config, dict) and str(config.get('environment_version')) == '3.12':\n        from .v312_config import validate_v312\n        return validate_v312(config)\n" + marker)
            marker = "    if str(config['environment_version']) in ('3.0'"
            s = s.replace(marker, "    if str(config['environment_version']) == '3.12':\n        return 56, 3, agents\n" + marker)
        p.write_text(s)
    p = Path('env/combat_env.py')
    s = p.read_text()
    marker = "            if str(candidate.get('environment_version')) == '3.11':"
    p.write_text(s.replace(marker, "            if str(candidate.get('environment_version')) == '3.12':\n                from .combat_v312 import CombatEnvironmentV312\n                return object.__new__(CombatEnvironmentV312)\n" + marker))
    p = Path('tools/smoke_formal_8v8.py')
    s = p.read_text().replace("            if env['environment_version']=='3.11': stem += '_v311'",
        "            if env['environment_version']=='3.11': stem += '_v311'\n            if env['environment_version']=='3.12': stem += '_v312'")
    p.write_text(s)
    p = Path('tests/test_combat_protocol.py')
    s = p.read_text()
    for name in ('combat_environment','mappo_5v5','rmappo_5v5','ea_mappo_5v5','stea_mappo_5v5'):
        marker = f"'{name}_v311.yaml',"
        s = s.replace(marker, marker + f" '{name}_v312.yaml',")
    p.write_text(s)
    p = Path('algorithm/maddpg/protocol.py')
    s = p.read_text().replace("('3.11', (66, 3, 5))", "('3.11', (66, 3, 5)), ('3.12', (56, 3, 5))")
    p.write_text(s.replace('v3.11 66/3/5', 'v3.11 66/3/5 or v3.12 56/3/5'))
    p = Path('algorithm/maddpg/trainer.py')
    s = p.read_text().replace('(66, 3, 5))', '(66, 3, 5), (56, 3, 5))')
    p.write_text(s.replace('or 66/3/5', 'or 66/3/5 or 56/3/5'))
    p = Path('algorithm/rmappo/networks.py')
    s = p.read_text().replace('(65,66,104)', '(56,65,66,104)')
    p.write_text(s.replace('obs65 or obs104', 'obs56, obs65, obs66 or obs104'))


if __name__ == '__main__':
    main()
