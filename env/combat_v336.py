"""v3.35 with weapon eligibility and hit inputs changed to paper ATA/HA only."""
from .combat_v335 import CombatEnvironmentV335
from .geometry import engagement_geometry
from .v336_weapon import PaperATAHAWeapon
from .v36_geometry import angle_within


class CombatEnvironmentV336(CombatEnvironmentV335):
    environment_version = '3.36'

    def build_weapon(self):
        return PaperATAHAWeapon(**self.config['weapon'])

    def attempt_hit_for_pair(self, attacker, target):
        g = engagement_geometry(attacker, target)
        return self.weapon.attempt_hit(self.rng, g.distance, g.ata, g.ha)

    def _entry_attempts(self, attackers, targets, fire_states, side):
        # Snapshot diagnostic counters before ammo consumption. The inherited
        # method still exclusively controls selection, cooldown and sampling.
        ammo = getattr(self, f'{side}_ammo')
        eligible, _ = self.weapon.eligibility(attackers, targets, ammo)
        counts = dict(live=0, range=0, range_az=0, geometry=0, eligible=0)
        for i, own in enumerate(attackers):
            flags = dict(live=False, range=False, range_az=False, geometry=False)
            for target in targets:
                if not own.alive or not target.alive:
                    continue
                g = engagement_geometry(own, target)
                in_range = self.weapon.range_min <= g.distance <= self.weapon.range_max
                ata_ok = in_range and angle_within(abs(g.ata), self.weapon.ata_max)
                full = ata_ok and angle_within(abs(g.ha), self.weapon.ha_max)
                for key, value in (('live', True), ('range', in_range), ('range_az', ata_ok), ('geometry', full)):
                    flags[key] |= bool(value)
            for key, value in flags.items():
                counts[key] += int(value)
            counts['eligible'] += int(eligible[i].any())
        attempts = super()._entry_attempts(attackers, targets, fire_states, side)
        if side == 'red':
            # Historical logging keys preserved: range_az now means ATA,
            # geometry means ATA+HA. These are agent-step OR counters.
            self.firing_geometry_diagnostics = counts
        return attempts
