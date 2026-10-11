"""v3.11 combat with only an independent 56D actor observation builder."""
from .combat_v311 import CombatEnvironmentV311
from .v312_observation import build_team_observations_v312


class CombatEnvironmentV312(CombatEnvironmentV311):
    environment_version = '3.12'

    def _observations(self):
        return build_team_observations_v312(self.red, self.blue, self.config['observation'])
