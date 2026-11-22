"""Periodic nearest-visible allocation with event-triggered replanning."""
import numpy as np


class NearestVisibleAssignment:
    def __init__(self, period=20):
        if type(period) is not int or period < 1:
            raise ValueError('Assignment period must be a positive integer')
        self.period = period
        self.targets = np.full(5, -1, dtype=np.int64)
        self.visible = np.zeros((5, 5), dtype=bool)

    def update(self, defenders, attackers, defender_alive, attacker_alive, radius, step, events=()):
        distances = np.linalg.norm(defenders[:, None, :] - attackers[None, :, :], axis=-1)
        visible = ((distances <= radius) & defender_alive[:, None] & attacker_alive[None, :])
        lost = any(target >= 0 and not visible[i, target] for i, target in enumerate(self.targets))
        discovered = np.any(visible & ~self.visible)
        replan = (step == 0 or step % self.period == 0 or discovered or lost or
                  any(event['type'] in ('capture', 'penetration') for event in events))
        if replan:
            self.targets.fill(-1)
            for i in np.flatnonzero(defender_alive):
                candidates = np.flatnonzero(visible[i])
                if len(candidates):
                    self.targets[i] = min(candidates, key=lambda j: (distances[i, j], int(j)))
        self.visible = visible
        return self.targets.copy(), bool(replan)
