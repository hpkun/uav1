"""v3.0 combat with only an independent initialization/reset path."""
import numpy as np
from .combat_v30 import CombatEnvironmentV30
from .v31_scenario import random_combat_states_v31

class CombatEnvironmentV31(CombatEnvironmentV30):
    environment_version = '3.1'

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.red, self.blue, alpha = random_combat_states_v31(self.rng, **self.config['scenario'])
        self.red_ammo.fill(self.weapon.ammo_per_aircraft)
        self.blue_ammo.fill(self.weapon.ammo_per_aircraft)
        self.red_fire_states = self._new_fire_states()
        self.blue_fire_states = self._new_fire_states()
        self.red_last_executed_phi.fill(0.)
        self.blue_last_executed_phi.fill(0.)
        self.steps = 0
        self._reset_metrics()
        return self._observations(), dict(environment_version=self.environment_version,
            radial_angle=alpha, red_alive_mask=self.red_alive_mask, blue_alive_mask=self.blue_alive_mask)
