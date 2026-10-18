"""v3.40 mechanisms with only the normalized hit-distance calibration changed."""
from .combat_v340 import CombatEnvironmentV340


class CombatEnvironmentV341(CombatEnvironmentV340):
    environment_version = '3.41'
