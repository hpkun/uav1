"""v3.32 simplified motion with paper-aligned engagement geometry and reward."""
from .combat_v332 import CombatEnvironmentV332
from .v333_reward import tactical_rewards_v333


class CombatEnvironmentV333(CombatEnvironmentV332):
    environment_version = '3.33'

    def dense_combat_reward(self):
        diagnostics = tactical_rewards_v333(self.red, self.blue, self.config['reward'], self.config['weapon'])
        return diagnostics['tactical'], diagnostics
