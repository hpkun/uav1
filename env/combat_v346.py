"""v3.44 with only its R4 computation replaced by paper Eq. (25)."""
from .combat_v344 import CombatEnvironmentV344
from .v346_reward import tactical_rewards


class CombatEnvironmentV346(CombatEnvironmentV344):
    environment_version = '3.46'

    def dense_combat_reward(self):
        diagnostic = tactical_rewards(self.red, self.blue, self.config['reward'])
        return diagnostic['tactical'], diagnostic
