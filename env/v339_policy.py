"""Project-calibrated speed tiers; inherited nearest-target pursuit diagnostics."""
import numpy as np

from .v310_policy import PaperNearestTargetBluePolicy


class DistanceTierNearestTargetBluePolicy(PaperNearestTargetBluePolicy):
    def action(self, own, targets):
        target_index = self.nearest_target_index(own, targets) if own.alive else None
        if target_index is None:
            return np.zeros(3, dtype=np.float32)
        target = targets[target_index]
        dx, dy, dz = target.x - own.x, target.y - own.y, target.z - own.z
        horizontal = float(np.hypot(dx, dy))
        distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        cfg = self.config
        if distance > cfg['far_distance_min']:
            desired_speed = cfg['far_desired_speed']
        elif distance > cfg['near_distance_max']:
            desired_speed = cfg['mid_desired_speed']
        else:
            desired_speed = cfg['near_desired_speed']
        return self.action_toward(
            own, float(np.arctan2(dy, dx)),
            float(np.arctan2(own.z - target.z, horizontal)), float(desired_speed))
