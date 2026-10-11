"""v3.7 unchanged combat, with independent paper-scale initialization."""
import numpy as np
from .combat_v37 import CombatEnvironmentV37
from .v38_scenario import random_combat_states_v38
from .v36_geometry import combat_geometry
from .sensor import sensor_geometry

class CombatEnvironmentV38(CombatEnvironmentV37):
    environment_version='3.8'

    def initial_states(self):
        return random_combat_states_v38(self.rng,self.arena_radius,**self.config['scenario'])

    def reset(self,seed=None):
        obs,info=super().reset(seed)
        alpha=info['radial_angle'];u=np.array([np.cos(alpha),np.sin(alpha)])
        distance=combat_geometry(self.red,self.blue)['distance']
        info.update(red_nominal_anchor=-self.arena_radius*u,blue_nominal_anchor=self.arena_radius*u,
            nominal_anchor_separation=2*self.arena_radius,
            minimum_inter_team_distance=float(distance.min()),mean_inter_team_distance=float(distance.mean()),
            maximum_inter_team_distance=float(distance.max()),
            initial_red_visible_pairs=int(sensor_geometry(self.red,self.blue,self.config['sensor'])[0].sum()),
            initial_blue_visible_pairs=int(sensor_geometry(self.blue,self.red,self.config['sensor'])[0].sum()),
            initial_red_fire_eligible_pairs=int(self.weapon.eligibility(self.red,self.blue,self.red_ammo)[0].sum()),
            initial_blue_fire_eligible_pairs=int(self.weapon.eligibility(self.blue,self.red,self.blue_ammo)[0].sum()))
        return obs,info
