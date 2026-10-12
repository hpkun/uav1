"""Direct v3.17 baseline: repeat fire with isolated reward distance weighting."""
from .combat_v317 import CombatEnvironmentV317
from .v319_reward import tactical_rewards


class CombatEnvironmentV319(CombatEnvironmentV317):
    environment_version = '3.19'

    def dense_combat_reward(self):
        diagnostics = tactical_rewards(self.red, self.blue, self.config['reward'])
        return diagnostics['tactical'], diagnostics
