"""v3.13 qualification; inherit the unchanged shared-Gaussian hit sampler."""
import numpy as np
from .v311_weapon import MADSACFourKmWeapon
from .v36_geometry import angle_within
from .v313_geometry import combat_geometry_v313


class MADSACPaperHAWeapon(MADSACFourKmWeapon):
    def eligibility(self, attackers, targets, ammo):
        g = combat_geometry_v313(attackers, targets)
        alive = np.asarray([s.alive for s in attackers])[:, None] & np.asarray([s.alive for s in targets])[None, :]
        return (alive & (np.asarray(ammo)[:, None] > 0) & (g['distance'] >= self.range_min)
            & (g['distance'] <= self.range_max) & angle_within(g['ata'], self.ata_max)
            & angle_within(g['ha'], self.ha_max)), g['distance']
