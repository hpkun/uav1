"""v3.9 mechanics with global nearest-target Blue pursuit only."""
from .combat_v39 import CombatEnvironmentV39
from .v310_policy import PaperNearestTargetBluePolicy


class CombatEnvironmentV310(CombatEnvironmentV39):
    environment_version = '3.10'

    def __init__(self, config):
        super().__init__(config)
        self.fixed_policy = PaperNearestTargetBluePolicy(self.config['blue_policy'], self.config['action'])
