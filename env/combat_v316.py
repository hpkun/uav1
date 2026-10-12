"""v3.15 flight, observations and rewards with isolated v3.16 weapons."""
from .combat_v315 import CombatEnvironmentV315
from .v316_weapon import MADSACBoresightWeapon


class CombatEnvironmentV316(CombatEnvironmentV315):
    environment_version = '3.16'

    def build_weapon(self):
        return MADSACBoresightWeapon(**self.config['weapon'])

    # Inherited attempt_hit_for_pair already passes paper distance/ATA/HA to
    # self.weapon. Keep that authoritative path and override only the weapon.
