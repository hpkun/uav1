"""Only v3.19 weapon and reward distance scales differ from frozen v3.17."""
from copy import deepcopy
from .v311_config import _frozen
from .v317_config import validate_v317


def validate_v319(config):
    if not isinstance(config, dict) or str(config.get('environment_version')) != '3.19':
        raise ValueError('v3.19 requires version 3.19')
    legacy = deepcopy(config)
    for section, field, current, previous in (
        ('weapon', 'range_max', 1000., 800.),
        ('reward', 'guide_distance_min', 3000., 4000.),
        ('reward', 'tactical_distance_max', 3000., 4000.),
    ):
        values = legacy.get(section)
        if not isinstance(values, dict):
            raise ValueError(f'v3.19 requires {section}')
        _frozen(values.get(field), current, f'{section}.{field}')
        values[field] = previous
    legacy['environment_version'] = '3.17'
    validate_v317(legacy)
    return config
