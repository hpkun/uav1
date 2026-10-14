"""Paper engagement scale over the strictly frozen v3.32 baseline."""
from copy import deepcopy
from .v332_config import validate_v332


CHANGES = (('scenario', 'center_radius', 4800., 4000.),
           ('weapon', 'range_max', 4000., 1000.),
           ('reward', 'guide_distance_min', 4000., 3000.),
           ('reward', 'tactical_distance_max', 4000., 3000.))


def validate_v333(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.33':
        raise ValueError('v3.33 requires environment_version 3.33')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.32'
    for section, key, current, previous in CHANGES:
        if config.get(section, {}).get(key) != current:
            raise ValueError(f'v3.33 requires {section}.{key}={current}')
        baseline[section][key] = previous
    validate_v332(baseline)
    return config
