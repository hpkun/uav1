"""Only the Blue strategy schema differs from the frozen v3.9 protocol."""
from copy import deepcopy
from .v39_config import validate_v39
from .v35_config import BLUE as TIERED_BLUE
from .v36_config import frozen

BLUE = dict(desired_speed=250.)


def validate_v310(config):
    frozen(config.get('blue_policy'), BLUE, 'v3.10 blue_policy')
    legacy = deepcopy(config)
    legacy['environment_version'] = '3.9'
    # Adapter only for historical schema validation; never used in runtime pursuit.
    legacy['blue_policy'] = dict(TIERED_BLUE)
    validate_v39(legacy)
    return config
