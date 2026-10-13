"""v3.26 with one persistent lock per Blue, invalidated only by target death."""
import numpy as np
from .combat_v326 import CombatEnvironmentV326
from .combat_v325 import CombatEnvironmentV325
from .v327_policy import V327LockedTargetBluePolicy


class CombatEnvironmentV327(CombatEnvironmentV326):
    environment_version = '3.27'

    def __init__(self, config):
        super().__init__(config)
        self.locked_target_indices = np.full(self.team_size, -1, dtype=np.int64)
        self.fixed_policy = V327LockedTargetBluePolicy(self.config['blue_policy'],
            self.config['action'], self.spec, self.blue_step_target_indices)

    def reset(self, seed=None):
        result = super().reset(seed)
        self.locked_target_indices.fill(-1)
        return result

    def step(self, red_actions, blue_actions=None):
        for i, own in enumerate(self.blue):
            old = int(self.locked_target_indices[i])
            if not own.alive:
                self.locked_target_indices[i] = -1
            elif old < 0 or not self.red[old].alive:
                target = self.fixed_policy.nearest_target_index(own, self.red)
                self.locked_target_indices[i] = -1 if target is None else target
                if old >= 0 and target is not None:
                    self.fixed_policy.target_switches += 1
        self.blue_step_target_indices[:] = self.locked_target_indices
        # Skip only v3.26's per-step nearest reselection wrapper. All physical
        # transitions and v3.26's coupled firing method remain inherited.
        result = CombatEnvironmentV325.step(self, red_actions, blue_actions)
        for i, own in enumerate(self.blue):
            if not own.alive:
                self.locked_target_indices[i] = -1
        result[4].update(blue_pursuit_target_indices=self.blue_step_target_indices.copy(),
                         blue_locked_target_indices=self.locked_target_indices.copy(),
                         blue_target_switches=self.fixed_policy.target_switches)
        return result
