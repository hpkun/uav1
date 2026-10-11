"""Add v3.13 version dispatch; copy frozen parameter files byte for byte."""
from pathlib import Path


def main():
    old = Path('configs/combat_environment_v312.yaml').read_text()
    assert '3.12' in old
    Path('configs/combat_environment_v313.yaml').write_text(old.replace('3.12', '3.13', 1))
    for name in ('mappo', 'rmappo', 'ea_mappo', 'stea_mappo'):
        Path(f'configs/{name}_5v5_v313.yaml').write_bytes(Path(f'configs/{name}_5v5_v312.yaml').read_bytes())
    p = Path('env/v311_reward.py')
    s = p.read_text().replace('from .v36_geometry import combat_geometry, angle_within',
        'from .v313_geometry import combat_geometry_v313 as combat_geometry\nfrom .v36_geometry import angle_within')
    Path('env/v313_reward.py').write_text(s.replace('Equation (25)', 'v3.13 Equation (25)'))
    s = Path('env/v312_observation.py').read_text().replace('v312', 'v313').replace('v3.12', 'v3.13')
    s = s.replace('from .v36_geometry import combat_geometry', 'from .v313_geometry import combat_geometry_v313 as combat_geometry')
    s = s.replace("geometry['ha'][i,j]/np.pi", "geometry['ha'][i,j]/(np.pi/2)")
    Path('env/v313_observation.py').write_text(s)
    paths = ['env/config.py', 'algorithm/common/evaluator.py', 'algorithm/mappo/runner.py',
        'algorithm/ea_mappo/protocol.py', 'algorithm/ea_mappo/runner.py',
        'algorithm/rmappo/protocol.py', 'algorithm/rmappo/runner.py',
        'algorithm/madsac/protocol.py', 'tests/test_formal_8v8.py']
    for file in paths:
        p = Path(file)
        s = p.read_text().replace('"3.12"}', '"3.12", "3.13"}')
        s = s.replace("'3.12')", "'3.12','3.13')")
        s = s.replace('3.11 or 3.12', '3.11, 3.12 or 3.13')
        if file == 'env/config.py':
            marker = "    if isinstance(config, dict) and str(config.get('environment_version')) == '3.12':"
            s = s.replace(marker, "    if isinstance(config, dict) and str(config.get('environment_version')) == '3.13':\n        from .v313_config import validate_v313\n        return validate_v313(config)\n" + marker)
            s = s.replace("if str(config['environment_version']) == '3.12':", "if str(config['environment_version']) in ('3.12', '3.13'):")
        p.write_text(s)
    p = Path('env/combat_env.py')
    s = p.read_text()
    marker = "            if str(candidate.get('environment_version')) == '3.12':"
    p.write_text(s.replace(marker, "            if str(candidate.get('environment_version')) == '3.13':\n                from .combat_v313 import CombatEnvironmentV313\n                return object.__new__(CombatEnvironmentV313)\n" + marker))
    p = Path('algorithm/maddpg/protocol.py')
    s = p.read_text().replace("('3.12', (56, 3, 5))", "('3.12', (56, 3, 5)), ('3.13', (56, 3, 5))")
    p.write_text(s.replace('v3.12 56/3/5', 'v3.12/v3.13 56/3/5'))
    p = Path('tools/smoke_formal_8v8.py')
    s = p.read_text().replace("            if env['environment_version']=='3.12': stem += '_v312'",
        "            if env['environment_version']=='3.12': stem += '_v312'\n            if env['environment_version']=='3.13': stem += '_v313'")
    p.write_text(s)
    p = Path('tests/test_combat_protocol.py')
    s = p.read_text()
    for name in ('combat_environment', 'mappo_5v5', 'rmappo_5v5', 'ea_mappo_5v5', 'stea_mappo_5v5'):
        marker = f"'{name}_v312.yaml',"
        s = s.replace(marker, marker + f" '{name}_v313.yaml',")
    p.write_text(s)
    for oldname, newname in (('_check_v312_vector_reset', '_check_v313_vector_reset'),
                             ('_audit_v312_checkpoints', '_audit_v313_checkpoints')):
        s = Path(f'tools/{oldname}.py').read_text()
        # Shift both sides of the prior contract audit, retaining same-shape 56D.
        s = s.replace('v312', 'v313').replace('v311', 'v312').replace('3.12', '3.13').replace('3.11', '3.12')
        s = s.replace('old 66D', 'old 56D')
        Path(f'tools/{newname}.py').write_text(s)


if __name__ == '__main__':
    main()
