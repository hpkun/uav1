"""Only local formation representation differs from the strict v3.32 baseline."""
from copy import deepcopy
from .v332_config import validate_v332

FORMATION_LOCAL_OFFSETS = [[300., 0.], [-50., 250.], [-50., -250.], [-200., 0.]]


def validate_v334(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.34':
        raise ValueError('v3.34 requires environment_version 3.34')
    scenario = config.get('scenario', {})
    if scenario.get('formation_local_offsets') != FORMATION_LOCAL_OFFSETS or 'formation_offsets' in scenario:
        raise ValueError('v3.34 requires exact formation_local_offsets and no scalar formation_offsets')
    if scenario.get('center_radius') != 4000. or type(scenario.get('team_size')) is not int or scenario['team_size'] != 4:
        raise ValueError('v3.34 requires center_radius=4000 and team_size=4')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.32'
    baseline['scenario'].pop('formation_local_offsets')
    baseline['scenario']['formation_offsets'] = [-300., -100., 100., 300.]
    validate_v332(baseline)
    return config
