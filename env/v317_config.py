"""Freeze v3.16 values while giving weapon-only body axes explicit names."""
from copy import deepcopy
import math
from .v316_config import validate_v316

BODY_TO_LEGACY = {
    'boresight_azimuth_max': 'ata_max',
    'boresight_elevation_max': 'ha_max',
    'azimuth_noise_scale': 'ata_noise_scale',
    'elevation_noise_scale': 'ha_noise_scale',
}
WEAPON_KEYS = {'range_min', 'range_max', 'ammo_per_aircraft', 'effective_hit_distance', *BODY_TO_LEGACY}


def validate_v317(config):
    if not isinstance(config, dict) or str(config.get('environment_version')) != '3.17':
        raise ValueError('v3.17 requires version 3.17')
    legacy = deepcopy(config)
    weapon = legacy.get('weapon')
    if not isinstance(weapon, dict) or set(weapon) != WEAPON_KEYS:
        raise ValueError('v3.17 weapon requires only the body-frame schema')
    for name in ('boresight_azimuth_max', 'boresight_elevation_max'):
        value = weapon[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= math.pi:
            raise ValueError(f'weapon.{name} must be finite and in (0, pi]')
        if value != math.pi / 6:
            raise ValueError(f'formal weapon.{name} must equal pi/6')
    for current, previous in BODY_TO_LEGACY.items():
        weapon[previous] = weapon.pop(current)
    weapon['boresight_max'] = math.pi / 6
    legacy['environment_version'] = '3.16'
    validate_v316(legacy)
    return config
