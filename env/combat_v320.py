"""v3.19 with isolated feasible maneuver-to-control allocation."""
import numpy as np
from .combat_v319 import CombatEnvironmentV319
from .v320_control import action_to_control_v320


class CombatEnvironmentV320(CombatEnvironmentV319):
    environment_version = '3.20'

    def _advance(self, states, actions):
        phis = np.zeros(self.team_size, dtype=np.float64)
        for index, state in enumerate(states):
            if state.alive:
                control = action_to_control_v320(state, actions[index], self.config['action'], self.spec)
                phis[index] = control.phi
                states[index] = self.integrator.step(state, control, self.dynamics, self.spec)
        return phis
