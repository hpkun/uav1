"""Freeze v3.13 values and the explicitly restored historical controller."""
from copy import deepcopy
from .v313_config import validate_v313
from .v311_config import _frozen

CONTROLLER = dict(heading_time_constant=2.0, pitch_time_constant=2.0,
                  speed_time_constant=2.0, normal_load_max=8.0)


def validate_v314(config):
    if not isinstance(config, dict) or str(config.get('environment_version')) != '3.14':
        raise ValueError('v3.14 requires version 3.14')
    legacy = deepcopy(config)
    action = legacy.get('action')
    if not isinstance(action, dict) or 'controller' not in action:
        raise ValueError('v3.14 requires action.controller')
    controller = action.pop('controller')
    _frozen(controller, CONTROLLER, 'action.controller')
    legacy['environment_version'] = '3.13'
    validate_v313(legacy)
    return config
