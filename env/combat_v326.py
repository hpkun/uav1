"""v3.25 with Blue fire bound to the step-start nearest pursuit target."""
import numpy as np
from .combat_v325 import CombatEnvironmentV325


class CombatEnvironmentV326(CombatEnvironmentV325):
    environment_version = '3.26'

    def __init__(self, config):
        super().__init__(config)
        self.blue_step_target_indices = np.full(self.team_size, -1, dtype=np.int64)

    def reset(self, seed=None):
        result = super().reset(seed)
        self.blue_step_target_indices.fill(-1)
        return result

    def step(self, red_actions, blue_actions=None):
        # Inherited automatic policy uses this same selector on unchanged states
        # before dynamics. No target is reselected in the Blue weapon path.
        for i, own in enumerate(self.blue):
            target = self.fixed_policy.nearest_target_index(own, self.red) if own.alive else None
            self.blue_step_target_indices[i] = -1 if target is None else target
        result = super().step(red_actions, blue_actions)
        result[4]['blue_pursuit_target_indices'] = self.blue_step_target_indices.copy()
        return result

    def _entry_attempts(self, attackers, targets, fire_states, side):
        if side != 'blue':
            return super()._entry_attempts(attackers, targets, fire_states, side)
        cooldown = self.blue_fire_cooldown_steps
        np.maximum(cooldown - 1, 0, out=cooldown)
        attempts = []
        for i, attacker in enumerate(attackers):
            j = int(self.blue_step_target_indices[i])
            if j < 0 or not attacker.alive or not targets[j].alive or cooldown[i] > 0:
                continue
            if self.weapon.qualifies(attacker, targets[j], self.blue_ammo[i]):
                self.blue_ammo[i] -= 1
                cooldown[i] = self.cooldown_steps
                attempts.append((i, j, self.attempt_hit_for_pair(attacker, targets[j])))
        self.combat_counts['blue']['ammo_used'] += len(attempts)
        self.combat_counts['blue']['fire_attempts'] += len(attempts)
        hits = sum(hit for _, _, hit in attempts)
        self.combat_counts['blue']['weapon_hits'] += hits
        for key, yes in (('attempt', bool(attempts)), ('hit', bool(hits))):
            if yes and self.first_steps['blue'][key] is None:
                self.first_steps['blue'][key] = self.steps
        return attempts
