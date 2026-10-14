"""Strict v3.30 baseline with only direct per-step motion limits changed."""
from copy import deepcopy
import math
from .v330_config import validate_v330

ACTION = dict(command=dict(heading_delta_max=math.pi/40,
                          pitch_delta_max=math.pi/60, speed_delta_max=2.5))
BASELINE_ACTION = dict(command=dict(heading_delta_max=math.pi,
                                   pitch_delta_max=math.pi/3, speed_delta_max=50.),
                       controller=dict(heading_time_constant=2., pitch_time_constant=2.,
                                       speed_time_constant=2., normal_load_max=10.))


def validate_v332(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.32':
        raise ValueError('v3.32 requires environment_version 3.32')
    if config.get('action') != ACTION:
        raise ValueError('v3.32 requires only per-step action.command limits pi/40, pi/60, 2.5; no controller')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.30'
    baseline['action'] = deepcopy(BASELINE_ACTION)
    validate_v330(baseline)
    return config
