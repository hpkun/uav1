"""v3.17 weapons with one attempt per continuous attacker-level window."""
import numpy as np
from .combat_v317 import CombatEnvironmentV317
from .weapon import FireState


class CombatEnvironmentV318(CombatEnvironmentV317):
    environment_version = '3.18'

    def _new_fire_states(self):
        return [FireState() for _ in range(self.team_size)]

    def _entry_attempts(self, attackers, targets, fire_states, side):
        ammo = getattr(self, f'{side}_ammo')
        current, distance = self.weapon.eligibility(attackers, targets, ammo)
        attempts = []
        for i, fire_state in enumerate(fire_states):
            indices = np.flatnonzero(current[i])
            if not indices.size:
                fire_state.armed = True
            elif fire_state.armed and ammo[i] > 0:
                j = int(min(indices, key=lambda index: (distance[i, index], index)))
                ammo[i] -= 1
                attempts.append((i, j, self.attempt_hit_for_pair(attackers[i], targets[j])))
                fire_state.armed = False
        self.combat_counts[side]['ammo_used'] += len(attempts)
        self.combat_counts[side]['fire_attempts'] += len(attempts)
        hits = sum(hit for _, _, hit in attempts)
        self.combat_counts[side]['weapon_hits'] += hits
        for key, yes in (('attempt', bool(attempts)), ('hit', bool(hits))):
            if yes and self.first_steps[side][key] is None:
                self.first_steps[side][key] = self.steps
        return attempts
