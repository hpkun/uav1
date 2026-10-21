"""Research-only 3-D cone; reuse the frozen shared-epsilon probability law."""
from dataclasses import dataclass
import numpy as np
from .geometry import engagement_geometry
from .v336_weapon import PaperATAHAWeapon
from .v36_geometry import angle_within


@dataclass(frozen=True)
class ConeWeapon:
    range_min: float
    range_max: float
    off_boresight_max: float
    ammo_per_aircraft: int
    effective_hit_distance: float
    azimuth_noise_scale: float
    elevation_noise_scale: float

    # Only the eligibility and angular reference frame change. One normal draw,
    # shared by both signed errors, and the distance law remain exactly frozen.
    attempt_hit = PaperATAHAWeapon.attempt_hit
    expected_hit_probability = PaperATAHAWeapon.expected_hit_probability

    def eligibility(self, attackers, targets, ammo):
        geometry = [[engagement_geometry(a, t) for t in targets] for a in attackers]
        distance = np.asarray([[g.distance for g in row] for row in geometry])
        alpha = np.asarray([[g.off_boresight for g in row] for row in geometry])
        live = (np.asarray([a.alive for a in attackers])[:, None]
                & np.asarray([t.alive for t in targets])[None, :])
        eligible = (live & (np.asarray(ammo)[:, None] > 0)
                    & (distance >= self.range_min) & (distance <= self.range_max)
                    & angle_within(alpha, self.off_boresight_max))
        return eligible, distance

    def qualifies(self, attacker, target, ammo):
        return bool(self.eligibility([attacker], [target], [ammo])[0][0, 0])
