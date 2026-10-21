"""v3.41 with symmetric 3-D eligibility and body-frame probability inputs only."""
from .combat_v341 import CombatEnvironmentV341
from .combat_v326 import CombatEnvironmentV326
from .geometry import engagement_geometry
from .v36_geometry import angle_within
from .v342_weapon import ConeWeapon


class CombatEnvironmentV342(CombatEnvironmentV341):
    environment_version = '3.42'

    def __init__(self, config):
        super().__init__(config)
        if self.config['environment_version'] != self.environment_version:
            raise ValueError(f'{self.environment_version} requires matching environment identity')

    def build_weapon(self):
        return ConeWeapon(**self.config['weapon'])

    def attempt_hit_for_pair(self, attacker, target):
        g = engagement_geometry(attacker, target)
        return self.weapon.attempt_hit(self.rng, g.distance,
            g.boresight_azimuth_error, g.boresight_elevation_error)

    def _entry_attempts(self, attackers, targets, fire_states, side):
        # Jump over obsolete rectangular diagnostic wrappers only. This is the
        # same v3.26 target/cooldown/resource/sampling implementation as v3.41.
        if side == 'red':
            eligible, _ = self.weapon.eligibility(attackers, targets, self.red_ammo)
            counts = dict(live=0, range=0, range_az=0, geometry=0, eligible=0)
            for i, own in enumerate(attackers):
                live = [t for t in targets if own.alive and t.alive]
                geometry = [engagement_geometry(own, t) for t in live]
                ranged = [g for g in geometry if self.weapon.range_min <= g.distance <= self.weapon.range_max]
                cone = any(angle_within(g.off_boresight, self.weapon.off_boresight_max) for g in ranged)
                counts['live'] += bool(live)
                counts['range'] += bool(ranged)
                # Compatibility key is explicitly relabelled in info below.
                counts['range_az'] += cone
                counts['geometry'] += cone
                counts['eligible'] += int(eligible[i].any())
            self.firing_geometry_diagnostics = counts
        return CombatEnvironmentV326._entry_attempts(self, attackers, targets, fire_states, side)

    def step(self, red_actions, blue_actions=None):
        result = super().step(red_actions, blue_actions)
        result[4]['firing_geometry_semantics'] = 'range_az and geometry: 3D cone alpha <= 30deg'
        return result
