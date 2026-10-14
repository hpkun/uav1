"""v3.32 with only point-symmetric staggered initial positions changed."""
from .combat_v332 import CombatEnvironmentV332
from .v334_scenario import random_combat_states_v334


class CombatEnvironmentV334(CombatEnvironmentV332):
    environment_version = '3.34'

    def initial_states(self):
        return random_combat_states_v334(self.rng, **self.config['scenario'])
