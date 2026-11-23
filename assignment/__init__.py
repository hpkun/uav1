"""Target assignment policies for interception."""

from .nearest import NearestVisibleAssignment
from .distance import DistanceAssignment
from .huang import HuangGlobalAssignment
from .ideal import IdealGlobalAssignment

__all__ = ['NearestVisibleAssignment', 'DistanceAssignment', 'HuangGlobalAssignment',
           'IdealGlobalAssignment']
