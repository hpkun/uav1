"""Weapon-only body-frame axes, preserving one shared Gaussian per attempt."""
from dataclasses import dataclass
import math
import numpy as np
from .geometry import engagement_geometry
from .v36_geometry import angle_within


@dataclass(frozen=True)
class BodyFrameGaussianWeapon:
    range_min: float
    range_max: float
    boresight_azimuth_max: float
    boresight_elevation_max: float
    ammo_per_aircraft: int
    effective_hit_distance: float
    azimuth_noise_scale: float
    elevation_noise_scale: float

    def eligibility(self, attackers, targets, ammo):
        geometry = [[engagement_geometry(a, t) for t in targets] for a in attackers]
        distance = np.asarray([[g.distance for g in row] for row in geometry])
        azimuth = np.asarray([[g.boresight_azimuth_error for g in row] for row in geometry])
        elevation = np.asarray([[g.boresight_elevation_error for g in row] for row in geometry])
        alive = np.asarray([s.alive for s in attackers])[:, None] & np.asarray([s.alive for s in targets])[None, :]
        eligible = (alive & (np.asarray(ammo)[:, None] > 0)
                    & (distance >= self.range_min) & (distance <= self.range_max)
                    & angle_within(np.abs(azimuth), self.boresight_azimuth_max)
                    & angle_within(np.abs(elevation), self.boresight_elevation_max))
        return eligible, distance

    def qualifies(self, attacker, target, ammo):
        return bool(self.eligibility([attacker], [target], np.asarray([ammo]))[0][0, 0])

    def attempt_hit(self, rng, distance, body_azimuth, body_elevation):
        epsilon = rng.normal(0., 1.)
        threshold = math.pi * math.exp(-distance / self.effective_hit_distance)
        return bool(abs(body_azimuth + self.azimuth_noise_scale * epsilon) <= threshold
                    and abs(body_elevation + self.elevation_noise_scale * epsilon) <= threshold)

    def expected_hit_probability(self, distance, body_azimuth, body_elevation):
        """Diagnostic only: intersect the two shared-epsilon intervals."""
        threshold = math.pi * math.exp(-distance / self.effective_hit_distance)
        lower = max((-threshold - body_azimuth) / self.azimuth_noise_scale,
                    (-threshold - body_elevation) / self.elevation_noise_scale)
        upper = min((threshold - body_azimuth) / self.azimuth_noise_scale,
                    (threshold - body_elevation) / self.elevation_noise_scale)
        if lower > upper:
            return 0.
        return .5 * (math.erf(upper / math.sqrt(2.)) - math.erf(lower / math.sqrt(2.)))
