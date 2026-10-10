"""Shared Gaussian Equation (8); calibration is project supplied, not paper values."""
from dataclasses import dataclass
import math
import numpy as np
from .v36_geometry import combat_geometry,angle_within

EFFECTIVE_HIT_DISTANCE=1116.2212531024945
NOISE_SCALE=0.9984711359155588

def expected_hit_probability(distance,ata,ha,effective_hit_distance=EFFECTIVE_HIT_DISTANCE,
                             ata_noise_scale=NOISE_SCALE,ha_noise_scale=NOISE_SCALE):
    """Analytic diagnostic only; never used to sample a combat outcome."""
    threshold=math.pi*math.exp(-distance/effective_hit_distance)
    z=min((threshold-abs(ata))/ata_noise_scale,(threshold-abs(ha))/ha_noise_scale)
    return .5*(1.+math.erf(z/math.sqrt(2.)))

@dataclass(frozen=True)
class MADSACGaussianFiniteAmmoWeapon:
    range_min:float
    range_max:float
    ata_max:float
    ha_max:float
    effective_hit_distance:float
    ata_noise_scale:float
    ha_noise_scale:float
    ammo_per_aircraft:int

    def eligibility(self,attackers,targets,ammo):
        g=combat_geometry(attackers,targets)
        alive=np.asarray([s.alive for s in attackers])[:,None]&np.asarray([s.alive for s in targets])[None,:]
        return (alive & (np.asarray(ammo)[:,None]>0) & (g['distance']>=self.range_min)
            & (g['distance']<=self.range_max) & angle_within(g['ata'],self.ata_max)
            & angle_within(g['ha'],self.ha_max)),g['distance']

    def qualifies(self,attacker,target,ammo):
        return bool(self.eligibility([attacker],[target],np.array([ammo]))[0][0,0])

    def attempt_hit(self,rng,distance,ata,ha):
        epsilon=rng.normal(0.,1.)
        threshold=math.pi*math.exp(-distance/self.effective_hit_distance)
        return bool(abs(ata)+self.ata_noise_scale*epsilon<=threshold
                    and abs(ha)+self.ha_noise_scale*epsilon<=threshold)

    def expected_hit_probability(self,distance,ata,ha):
        return expected_hit_probability(distance,ata,ha,self.effective_hit_distance,
                                        self.ata_noise_scale,self.ha_noise_scale)
