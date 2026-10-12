"""v3.20 mechanics with nearest-pursuit Blue action encoding corrected."""
from .combat_v320 import CombatEnvironmentV320
from .v321_policy import V321NearestTargetBluePolicy


class CombatEnvironmentV321(CombatEnvironmentV320):
    environment_version = '3.21'

    def __init__(self, config):
        super().__init__(config)
        self.fixed_policy = V321NearestTargetBluePolicy(self.config['blue_policy'], self.config['action'], self.spec)
