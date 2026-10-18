"""v3.39 mechanics with independently validated 2 km / 4 km parameters."""
from .combat_v339 import CombatEnvironmentV339


class CombatEnvironmentV340(CombatEnvironmentV339):
    environment_version = '3.40'
