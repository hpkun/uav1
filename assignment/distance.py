"""Low-frequency distance assignment for the five-on-five interception task."""
from functools import lru_cache

import numpy as np


class DistanceAssignment:
    """Replan on a period or system-level target events, then hold assignments."""

    def __init__(self, period=20, method='distance_matching'):
        if type(period) is not int or period < 1:
            raise ValueError('Assignment period must be a positive integer')
        if method not in ('distance_matching', 'independent_nearest'):
            raise ValueError('Unknown distance assignment method')
        self.period = period
        self.method = method
        self.targets = np.full(5, -1, dtype=np.int64)
        self.detected_targets = np.zeros(5, dtype=bool)

    @staticmethod
    def _match(distances, visible):
        """Exact minimum-cost maximum-cardinality matching; five agents need no solver dependency."""
        @lru_cache(None)
        def solve(i, used):
            if i == 5:
                return 0, 0.0, ()
            options = [-1] + [int(j) for j in np.flatnonzero(visible[i]) if not used & (1 << j)]
            candidates = []
            for j in options:
                count, cost, suffix = solve(i + 1, used | (1 << j) if j >= 0 else used)
                candidates.append((count + (j >= 0), cost + (distances[i, j] if j >= 0 else 0.0),
                                   (j,) + suffix))
            return min(candidates, key=lambda row: (-row[0], row[1],
                                                    tuple(5 if j < 0 else j for j in row[2])))

        return np.asarray(solve(0, 0)[2], dtype=np.int64)

    def update(self, defenders, attackers, defender_alive, attacker_alive, radius, step, events=()):
        distances = np.linalg.norm(defenders[:, None, :] - attackers[None, :, :], axis=-1)
        visible = (distances <= radius) & defender_alive[:, None] & attacker_alive[None, :]
        detected_now = visible.any(axis=0)
        new_target_detected = bool(np.any(detected_now & ~self.detected_targets))
        self.detected_targets |= detected_now
        replan = (step == 0 or step % self.period == 0 or new_target_detected or
                  any(event['type'] in ('capture', 'penetration') for event in events))
        if replan:
            if self.method == 'distance_matching':
                self.targets = self._match(distances, visible)
            else:
                self.targets.fill(-1)
                for i in np.flatnonzero(defender_alive):
                    candidates = np.flatnonzero(visible[i])
                    if len(candidates):
                        self.targets[i] = min(candidates, key=lambda j: (distances[i, j], int(j)))
        return self.targets.copy(), bool(replan)
