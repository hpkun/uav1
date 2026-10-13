"""Unchanged pursuit guidance driven by the environment's frozen step targets."""
import numpy as np
from .v321_policy import V321NearestTargetBluePolicy


class V327LockedTargetBluePolicy(V321NearestTargetBluePolicy):
    def __init__(self, config, action_config, aircraft_spec, step_targets):
        super().__init__(config, action_config, aircraft_spec)
        self.step_targets = step_targets  # reference to the environment snapshot

    def team_actions(self, team, targets):
        actions = np.zeros((len(team), 3), dtype=np.float32)
        selected = []
        for i, own in enumerate(team):
            j = int(self.step_targets[i])
            valid = own.alive and j >= 0 and targets[j].alive
            selected.append(j if valid else None)
            if valid:
                target = targets[j]
                dx, dy = target.x - own.x, target.y - own.y
                actions[i] = self.action_toward(own, float(np.arctan2(dy, dx)),
                    float(np.arctan2(own.z - target.z, np.hypot(dx, dy))),
                    float(self.config['desired_speed']))
                self.pursuit_agent_steps += 1
        self.last_targets = selected
        return actions
