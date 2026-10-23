"""v3.46 mechanics and rewards with the original v3.38 45D observations."""
from .combat_v346 import CombatEnvironmentV346
from .v338_observation import build_team_observations_v338


class CombatEnvironmentV347(CombatEnvironmentV346):
    environment_version = '3.47'

    def _observations(self):
        return build_team_observations_v338(
            self.red, self.blue, self.config['observation']
        )
