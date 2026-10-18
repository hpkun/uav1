"""Isolated range experiment; map only a deep validation copy to v3.39."""
from copy import deepcopy
from .v339_config import validate_v339


BLUE_POLICY = dict(near_distance_max=1000., far_distance_min=3000.,
                   near_desired_speed=200., mid_desired_speed=220., far_desired_speed=240.)


def validate_v340(config):
    if not isinstance(config, dict) or config.get('environment_version') != '3.40':
        raise ValueError('v3.40 requires environment_version 3.40')
    for section, name, value in (('weapon', 'range_max', 2000.),
                                  ('reward', 'tactical_distance_max', 4000.)):
        fields = config.get(section)
        if not isinstance(fields, dict) or fields.get(name) != value:
            raise ValueError(f'v3.40 requires {section}.{name}={value}')
    # v3.39 permits policy experiments; this controlled experiment freezes its
    # canonical five values as well as retaining all inherited schema checks.
    if config.get('blue_policy') != BLUE_POLICY:
        raise ValueError('v3.40 requires the unchanged formal v3.39 blue_policy')
    baseline = deepcopy(config)
    baseline['environment_version'] = '3.39'
    baseline['weapon']['range_max'] = 1000.
    baseline['reward']['tactical_distance_max'] = 3000.
    validate_v339(baseline)
    return config
