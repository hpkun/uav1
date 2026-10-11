"""v3.12 mechanics with isolated, pitch-independent height LOS geometry."""
from .combat_v312 import CombatEnvironmentV312
from .v313_geometry import combat_geometry_v313
from .v313_weapon import MADSACPaperHAWeapon
from .v313_reward import tactical_rewards
from .v313_observation import build_team_observations_v313


class CombatEnvironmentV313(CombatEnvironmentV312):
    environment_version = '3.13'

    def build_weapon(self):
        return MADSACPaperHAWeapon(**self.config['weapon'])

    def attempt_hit_for_pair(self, attacker, target):
        g = combat_geometry_v313([attacker], [target])
        return self.weapon.attempt_hit(self.rng, g['distance'][0, 0], g['ata'][0, 0], g['ha'][0, 0])

    def dense_combat_reward(self):
        diagnostics = tactical_rewards(self.red, self.blue, self.config['reward'])
        return diagnostics['tactical'], diagnostics

    def _observations(self):
        return build_team_observations_v313(self.red, self.blue, self.config['observation'])
