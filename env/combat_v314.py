"""v3.13 combat with desired commands, existing point-mass dynamics and RK4."""
from .combat_v313 import CombatEnvironmentV313
from .combat_env import MultiUAVCombatEnv
from .dynamics import PointMassDynamics
from .integrator import RK4Integrator
from .v314_observation import build_team_observations_v314


class CombatEnvironmentV314(CombatEnvironmentV313):
    environment_version = '3.14'

    def __init__(self, config):
        super().__init__(config)
        self.dynamics = PointMassDynamics()
        self.integrator = RK4Integrator(self.dt)

    def _advance(self, states, actions):
        return MultiUAVCombatEnv._advance(self, states, actions)

    def _observations(self):
        return build_team_observations_v314(
            self.red, self.blue, self.config['observation'],
            self.red_last_executed_phi,
        )
