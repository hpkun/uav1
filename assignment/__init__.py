"""Target assignment policies for interception."""

from .nearest import NearestVisibleAssignment
from .distance import DistanceAssignment
from .huang import HuangGlobalAssignment

__all__ = ['NearestVisibleAssignment', 'DistanceAssignment', 'HuangGlobalAssignment']
