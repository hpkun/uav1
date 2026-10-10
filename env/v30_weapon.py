"""Finite-ammunition constant-probability forward-cone weapon."""
from dataclasses import dataclass
from .sensor import sensor_geometry


@dataclass(frozen=True)
class FiniteAmmoWeapon:
    range_min: float
    range_max: float
    off_boresight_angle_max: float
    hit_probability: float
    ammo_per_aircraft: int

    def eligibility(self,attackers,targets,ammo):
        visible,distance,_,_=sensor_geometry(attackers,targets,
            {'range_max':self.range_max,'off_boresight_angle_max':self.off_boresight_angle_max})
        return visible & (distance>=self.range_min) & (ammo[:,None]>0),distance

    def qualifies(self,attacker,target,ammo):
        import numpy as np
        return bool(self.eligibility([attacker],[target],np.array([ammo]))[0][0,0])

    def attempt_hit(self,rng):
        return bool(rng.random()<self.hit_probability)
