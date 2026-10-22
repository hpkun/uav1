"""Version-isolated dwell ablation, validated against the frozen v3.43 schema."""
from copy import deepcopy
import math
from .v343_config import validate_v343
from .v341_config import EFFECTIVE_HIT_DISTANCE
from .v344_dwell import EVENT_SEMANTICS

FIRE_CONTROL=dict(mechanism=EVENT_SEMANTICS,dwell_seconds=.5,dwell_rounding='ceil',
                  cooldown_seconds=0.,legacy_ammo_cooldown='disabled')


def validate_v344(config, *, version='3.44', angle=math.pi/6):
    if not isinstance(config,dict) or config.get('environment_version')!=version:
        raise ValueError(f'expected environment_version {version}')
    if config.get('weapon')!=dict(range_min=0.,range_max=2000.,off_boresight_max=angle):
        raise ValueError(f'{version} requires only the explicit dwell geometry fields')
    if config.get('fire_control')!=FIRE_CONTROL:
        raise ValueError(f'{version} requires 0.5-second dwell, ceil sampling and disabled legacy resource/cooldown')
    baseline=deepcopy(config);baseline['environment_version']='3.43'
    baseline['weapon']=dict(range_min=0.,range_max=2000.,off_boresight_max=math.pi/6,
        ammo_per_aircraft=6,effective_hit_distance=EFFECTIVE_HIT_DISTANCE,
        azimuth_noise_scale=.9984711359155588,elevation_noise_scale=.9984711359155588)
    baseline['fire_control']=dict(cooldown_seconds=1.)
    validate_v343(baseline)
    return config
