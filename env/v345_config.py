"""Only the dwell cone half-angle differs from v3.44."""
import math
from .v344_config import validate_v344


def validate_v345(config):
    return validate_v344(config,version='3.45',angle=math.pi/4)
