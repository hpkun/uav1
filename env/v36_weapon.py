"""Color-neutral finite-ammunition ATA/HA-qualified constant-p weapon."""
from dataclasses import dataclass
import numpy as np
from .v36_geometry import combat_geometry,angle_within

@dataclass(frozen=True)
class MADSACStyleFiniteAmmoWeapon:
    range_min:float
    range_max:float
    ata_max:float
    ha_max:float
    hit_probability:float
    ammo_per_aircraft:int

    def eligibility(self,attackers,targets,ammo):
        g=combat_geometry(attackers,targets)
        alive=np.asarray([s.alive for s in attackers])[:,None]&np.asarray([s.alive for s in targets])[None,:]
        return (alive & (np.asarray(ammo)[:,None]>0) & (g['distance']>=self.range_min)
            & (g['distance']<=self.range_max) & angle_within(g['ata'],self.ata_max)
            & angle_within(g['ha'],self.ha_max)),g['distance']

    def qualifies(self,attacker,target,ammo):
        return bool(self.eligibility([attacker],[target],np.array([ammo]))[0][0,0])

    def attempt_hit(self,rng):return bool(rng.random()<self.hit_probability)
