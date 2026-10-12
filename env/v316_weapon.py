"""v3.16 paper geometry plus true boresight, with corrected Equation (8)."""
from dataclasses import dataclass
import math
import numpy as np
from .geometry import engagement_geometry
from .v313_weapon import MADSACPaperHAWeapon
from .v36_geometry import angle_within


@dataclass(frozen=True)
class MADSACBoresightWeapon(MADSACPaperHAWeapon):
    boresight_max: float

    def eligibility(self, attackers, targets, ammo):
        paper_eligible, distance = super().eligibility(attackers, targets, ammo)
        boresight = np.asarray([
            [engagement_geometry(attacker, target).off_boresight for target in targets]
            for attacker in attackers
        ])
        return paper_eligible & angle_within(boresight, self.boresight_max), distance

    def attempt_hit(self, rng, distance, ata, ha):
        epsilon = rng.normal(0., 1.)
        threshold = math.pi * math.exp(-distance / self.effective_hit_distance)
        return bool(abs(ata + self.ata_noise_scale * epsilon) <= threshold
                    and abs(ha + self.ha_noise_scale * epsilon) <= threshold)

    def expected_hit_probability(self, distance, ata, ha):
        """Diagnostic only: probability of the shared epsilon interval intersection."""
        threshold = math.pi * math.exp(-distance / self.effective_hit_distance)
        lower = max((-threshold - ata) / self.ata_noise_scale,
                    (-threshold - ha) / self.ha_noise_scale)
        upper = min((threshold - ata) / self.ata_noise_scale,
                    (threshold - ha) / self.ha_noise_scale)
        if lower > upper:
            return 0.
        return .5 * (math.erf(upper / math.sqrt(2.)) - math.erf(lower / math.sqrt(2.)))
