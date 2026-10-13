"""v3.22 with Blue ground death, body reward geometry and firing cooldown."""
import numpy as np
from .combat_v322 import CombatEnvironmentV322
from .v323_reward import tactical_rewards_v323


class CombatEnvironmentV323(CombatEnvironmentV322):
    environment_version = '3.23'

    def __init__(self, config):
        super().__init__(config)
        self.cooldown_steps = round(self.config['fire_control']['cooldown_seconds'] / self.dt)
        self.red_fire_cooldown_steps = np.zeros(self.team_size, dtype=np.int64)
        self.blue_fire_cooldown_steps = np.zeros(self.team_size, dtype=np.int64)

    def reset(self, seed=None):
        result = super().reset(seed)
        self.red_fire_cooldown_steps.fill(0)
        self.blue_fire_cooldown_steps.fill(0)
        return result

    def _resolve_noncombat_losses(self):
        blue_ground = []
        for index, state in enumerate(self.blue):
            if state.alive and state.altitude <= self.config['arena']['altitude_min']:
                state.alive = False
                blue_ground.append(index)
        self.combat_counts['blue']['ground_losses'] += len(blue_ground)
        # Dead Blue bypass every old return; Red and other Blue returns stay exact.
        red_exits, blue_exits, red_ground, _ = super()._resolve_noncombat_losses()
        return red_exits, blue_exits, red_ground, blue_ground

    def dense_combat_reward(self):
        diagnostics = tactical_rewards_v323(self.red, self.blue, self.config['reward'], self.config['weapon'])
        return diagnostics['tactical'], diagnostics

    def _entry_attempts(self, attackers, targets, fire_states, side):
        cooldown = getattr(self, f'{side}_fire_cooldown_steps')
        np.maximum(cooldown - 1, 0, out=cooldown)
        ammo = getattr(self, f'{side}_ammo')
        eligible, distance = self.weapon.eligibility(attackers, targets, ammo)
        attempts = []
        for index in range(self.team_size):
            if cooldown[index] > 0:
                continue
            indices = np.flatnonzero(eligible[index])
            if indices.size:
                target = int(min(indices, key=lambda j: (distance[index, j], j)))
                ammo[index] -= 1
                cooldown[index] = self.cooldown_steps
                attempts.append((index, target, self.attempt_hit_for_pair(attackers[index], targets[target])))
        self.combat_counts[side]['ammo_used'] += len(attempts)
        self.combat_counts[side]['fire_attempts'] += len(attempts)
        hits = sum(hit for _, _, hit in attempts)
        self.combat_counts[side]['weapon_hits'] += hits
        for key, yes in (('attempt', bool(attempts)), ('hit', bool(hits))):
            if yes and self.first_steps[side][key] is None:
                self.first_steps[side][key] = self.steps
        return attempts
