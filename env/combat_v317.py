"""v3.16 combat with weapon qualification and hit inputs in body axes."""
import numpy as np
from .combat_v316 import CombatEnvironmentV316
from .geometry import engagement_geometry
from .v317_weapon import BodyFrameGaussianWeapon


class CombatEnvironmentV317(CombatEnvironmentV316):
    environment_version = '3.17'

    def build_weapon(self):
        return BodyFrameGaussianWeapon(**self.config['weapon'])

    def reset(self, seed=None):
        observation, info = super().reset(seed)
        radius = self.config['scenario']['center_radius']
        alpha = info['radial_angle']
        direction = np.array([np.cos(alpha), np.sin(alpha)])
        info.update(red_nominal_anchor=-radius * direction,
                    blue_nominal_anchor=radius * direction,
                    nominal_anchor_separation=2 * radius)
        return observation, info

    def attempt_hit_for_pair(self, attacker, target):
        geometry = engagement_geometry(attacker, target)
        return self.weapon.attempt_hit(self.rng, geometry.distance,
                                       geometry.boresight_azimuth_error,
                                       geometry.boresight_elevation_error)
