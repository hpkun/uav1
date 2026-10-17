"""v3.38 mechanics with only Blue's automatic speed command changed."""
from .combat_v338 import CombatEnvironmentV338
from .v339_policy import DistanceTierNearestTargetBluePolicy


class CombatEnvironmentV339(CombatEnvironmentV338):
    environment_version = '3.39'

    def __init__(self, config):
        super().__init__(config)
        self.fixed_policy = DistanceTierNearestTargetBluePolicy(
            self.config['blue_policy'], self.config['action'])
