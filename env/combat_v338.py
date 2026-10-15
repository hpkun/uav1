"""v3.37 mechanics with only enemy observation angular semantics changed."""
from .combat_v337 import CombatEnvironmentV337
from .v338_observation import build_team_observations_v338


class CombatEnvironmentV338(CombatEnvironmentV337):
    environment_version = '3.38'

    def _observations(self):
        return build_team_observations_v338(self.red, self.blue, self.config['observation'])
