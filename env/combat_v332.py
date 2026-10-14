"""v3.30 combat mechanics with shared, independent rate-limited kinematics."""
import numpy as np
from .combat_v330 import CombatEnvironmentV330
from .v311_kinematics import advance_aircraft


class CombatEnvironmentV332(CombatEnvironmentV330):
    environment_version = '3.32'

    def __init__(self, config):
        super().__init__(config)
        # Parent initialization is retained for identical combat state allocation.
        # Its legacy flight engines have no role in this version.
        del self.dynamics, self.integrator

    def _advance(self, states, actions):
        for index, state in enumerate(states):
            if state.alive:
                states[index] = advance_aircraft(
                    state, actions[index], self.dt, self.config['action']['command'], self.spec)
        # Compatibility return for the unchanged transition; no roll control.
        # The observation's phi placeholder remains zero, as in v3.30.
        return np.zeros(self.team_size, dtype=np.float32)

    def _info(self, *args, **kwargs):
        info = super()._info(*args, **kwargs)
        # Do not expose compatibility zeros as physical roll diagnostics.
        info.pop('executed_red_phi', None)
        return info
