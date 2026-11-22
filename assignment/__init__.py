"""Target assignment policies for interception."""

from .nearest import NearestVisibleAssignment
from .distance import DistanceAssignment

__all__ = ['NearestVisibleAssignment', 'DistanceAssignment']
