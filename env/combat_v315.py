"""Minimal feasibility-corrected desired-command environment."""
import numpy as np
from .combat_v314 import CombatEnvironmentV314
from .v315_control import action_to_control_v315
from .v315_observation import build_team_observations_v315


class CombatEnvironmentV315(CombatEnvironmentV314):
    environment_version = '3.15'

    def _advance(self, states, actions):
        phis = np.zeros(self.team_size, dtype=np.float64)
        for index, state in enumerate(states):
            if state.alive:
                control = action_to_control_v315(
                    state, actions[index], self.config['action'], self.spec)
                phis[index] = control.phi
                states[index] = self.integrator.step(
                    state, control, self.dynamics, self.spec)
        return phis

    def _observations(self):
        return build_team_observations_v315(
            self.red, self.blue, self.config['observation'])
