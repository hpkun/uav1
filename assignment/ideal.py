"""Initial global one-to-one distance matching with event-only replanning."""

import numpy as np

from .distance import DistanceAssignment


class IdealGlobalAssignment:
    def __init__(self):
        self.targets = np.full(5, -1, dtype=np.int64)

    def update(self, defenders, attackers, defender_alive, attacker_alive, radius, step, events=()):
        replan = step == 0 or any(event['type'] in ('capture', 'penetration', 'defender_retired') for event in events)
        if replan:
            distances = np.linalg.norm(defenders[:, None, :] - attackers[None, :, :], axis=-1)
            visible = ((distances <= radius) & defender_alive[:, None] & attacker_alive[None, :])
            self.targets = DistanceAssignment._match(distances, visible)
        return self.targets.copy(), bool(replan)
