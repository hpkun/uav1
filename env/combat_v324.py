"""v3.23 transitions with signed relative direction in enemy observation slots."""
from .combat_v323 import CombatEnvironmentV323
from .v324_observation import build_team_observations_v324


class CombatEnvironmentV324(CombatEnvironmentV323):
    environment_version = '3.24'

    def _observations(self):
        return build_team_observations_v324(self.red, self.blue, self.config['observation'])
