"""v3.21 with continuous distance-weighted R3 only."""
from .combat_v321 import CombatEnvironmentV321
from .v322_reward import tactical_rewards_v322


class CombatEnvironmentV322(CombatEnvironmentV321):
    environment_version = '3.22'

    def dense_combat_reward(self):
        diagnostics = tactical_rewards_v322(self.red, self.blue, self.config['reward'], self.config['weapon'])
        return diagnostics['tactical'], diagnostics
