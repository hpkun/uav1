"""Independent ATA/HA weapon; numerical hit calibration is frozen at v3.35."""
from dataclasses import dataclass
import math
import numpy as np
from .geometry import engagement_geometry
from .v36_geometry import angle_within


@dataclass(frozen=True)
class PaperATAHAWeapon:
    range_min: float
    range_max: float
    ata_max: float
    ha_max: float
    ammo_per_aircraft: int
    effective_hit_distance: float
    azimuth_noise_scale: float
    elevation_noise_scale: float

    # Compatibility for the inherited read-only body diagnostic wrapper.
    # v3.36 replaces its reported counters; neither alias enters eligibility.
    @property
    def boresight_azimuth_max(self):
        return self.ata_max

    @property
    def boresight_elevation_max(self):
        return self.ha_max

    def eligibility(self, attackers, targets, ammo):
        geometry = [[engagement_geometry(a, t) for t in targets] for a in attackers]
        distance = np.asarray([[g.distance for g in row] for row in geometry])
        ata = np.asarray([[g.ata for g in row] for row in geometry])
        ha = np.asarray([[g.ha for g in row] for row in geometry])
        alive = np.asarray([s.alive for s in attackers])[:, None] & np.asarray([s.alive for s in targets])[None, :]
        eligible = (alive & (np.asarray(ammo)[:, None] > 0)
                    & (distance >= self.range_min) & (distance <= self.range_max)
                    & angle_within(np.abs(ata), self.ata_max)
                    & angle_within(np.abs(ha), self.ha_max))
        return eligible, distance

    def qualifies(self, attacker, target, ammo):
        return bool(self.eligibility([attacker], [target], np.asarray([ammo]))[0][0, 0])

    def attempt_hit(self, rng, distance, ata, ha):
        epsilon = rng.normal(0., 1.)
        threshold = math.pi * math.exp(-distance / self.effective_hit_distance)
        return bool(abs(ata + self.azimuth_noise_scale * epsilon) <= threshold
                    and abs(ha + self.elevation_noise_scale * epsilon) <= threshold)

    def expected_hit_probability(self, distance, ata, ha):
        """Read-only shared-epsilon interval probability; arguments are radians."""
        threshold = math.pi * math.exp(-distance / self.effective_hit_distance)
        lower = max((-threshold - ata) / self.azimuth_noise_scale,
                    (-threshold - ha) / self.elevation_noise_scale)
        upper = min((threshold - ata) / self.azimuth_noise_scale,
                    (threshold - ha) / self.elevation_noise_scale)
        if lower > upper:
            return 0.
        return .5 * (math.erf(upper / math.sqrt(2.)) - math.erf(lower / math.sqrt(2.)))
