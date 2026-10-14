"""v3.36 with only R3/R4 angular inputs replaced by authoritative ATA/HA/AA."""
from .combat_v336 import CombatEnvironmentV336
from .v337_reward import tactical_rewards_v337


class CombatEnvironmentV337(CombatEnvironmentV336):
    environment_version = '3.37'

    def dense_combat_reward(self):
        diagnostics = tactical_rewards_v337(self.red, self.blue, self.config['reward'], self.config['weapon'])
        return diagnostics['tactical'], diagnostics
