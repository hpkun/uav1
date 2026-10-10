"""v3.3 mechanisms with isolated tiered automatic Blue actions."""
from .combat_v33 import CombatEnvironmentV33
from .v34_policy import SensorLimitedTieredBluePolicy


class CombatEnvironmentV34(CombatEnvironmentV33):
    environment_version = '3.4'

    def __init__(self, config):
        super().__init__(config)
        self.fixed_policy = SensorLimitedTieredBluePolicy(self.config['blue_policy'],
            self.config['action'], self.config['sensor'], self.config['scenario'])

    def reset(self, seed=None):
        self.fixed_policy.reset()
        return super().reset(seed)

    def step(self, red_actions, blue_actions=None):
        automatic = blue_actions is None
        result = super().step(red_actions, blue_actions)
        if automatic:
            result[4].update(self.fixed_policy.diagnostics())
        return result
