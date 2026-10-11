"""Isolated direct-motion, repeat-fire MADSAC-style five-versus-five version."""
import numpy as np
from .combat_v310 import CombatEnvironmentV310
from .config import load_config, validate_config, aircraft_spec, environment_dimensions
from .v310_policy import PaperNearestTargetBluePolicy
from .v311_kinematics import advance_aircraft
from .v311_scenario import random_combat_states_v311
from .v311_weapon import MADSACFourKmWeapon
from .v311_reward import tactical_rewards


class CombatEnvironmentV311(CombatEnvironmentV310):
    environment_version = '3.11'

    def __init__(self, config):
        # Deliberately do not construct a controller, dynamics or integrator.
        self.config = validate_config(config) if isinstance(config, dict) else load_config(config)
        if str(self.config['environment_version']) != self.environment_version:
            raise ValueError('v3.11 requires version 3.11')
        self.observation_dim, self.action_dim, self.team_size = environment_dimensions(self.config)
        self.spec = aircraft_spec(self.config)
        self.dt = float(self.config['simulation']['dt'])
        self.max_steps = int(self.config['simulation']['max_steps'])
        self.arena_radius = float(self.config['arena']['radius'])
        self.fixed_policy = PaperNearestTargetBluePolicy(self.config['blue_policy'], self.config['action'])
        self.weapon = self.build_weapon()
        self.red_ammo = np.full(5, self.weapon.ammo_per_aircraft, dtype=np.int64)
        self.blue_ammo = self.red_ammo.copy()
        self.red, self.blue = [], []
        self.rng = np.random.default_rng()
        self.steps = 0
        self.red_last_executed_phi = np.zeros(5, dtype=np.float32)
        self.blue_last_executed_phi = self.red_last_executed_phi.copy()
        self.red_fire_states = self.blue_fire_states = None
        self._reset_metrics()

    def build_weapon(self):
        return MADSACFourKmWeapon(**self.config['weapon'])

    def _new_fire_states(self):
        # Compatibility argument for inherited step only; no pair eligibility state.
        return None

    def initial_states(self):
        return random_combat_states_v311(self.rng, **self.config['scenario'])

    def reset(self, seed=None):
        obs, info = super().reset(seed)
        alpha = info['radial_angle']
        u = np.array([np.cos(alpha), np.sin(alpha)])
        # Replace inherited chord diagnostics with actual compact formation centres.
        info.update(red_nominal_anchor=-4800.*u, blue_nominal_anchor=4800.*u,
                    nominal_anchor_separation=9600., initialization_protocol='random_diameter_compact_5v5')
        return obs, info

    def _advance(self, states, actions):
        for i, state in enumerate(states):
            if state.alive:
                states[i] = advance_aircraft(state, actions[i], self.dt, self.config['action']['command'], self.spec)
        return np.zeros(5, dtype=np.float32)

    def dense_combat_reward(self):
        diagnostics = tactical_rewards(self.red, self.blue, self.config['reward'])
        return diagnostics['tactical'], diagnostics

    def _entry_attempts(self, attackers, targets, fire_states, side):
        # The method name is an inherited hook; fire_states is never inspected.
        ammo = getattr(self, f'{side}_ammo')
        current, distance = self.weapon.eligibility(attackers, targets, ammo)
        attempts = []
        for i in range(5):
            indices = np.flatnonzero(current[i])
            if indices.size:
                j = int(min(indices, key=lambda j: (distance[i, j], j)))
                ammo[i] -= 1
                attempts.append((i, j, self.attempt_hit_for_pair(attackers[i], targets[j])))
        self.combat_counts[side]['ammo_used'] += len(attempts)
        self.combat_counts[side]['fire_attempts'] += len(attempts)
        hits = sum(hit for _, _, hit in attempts)
        self.combat_counts[side]['weapon_hits'] += hits
        for key, yes in (('attempt', bool(attempts)), ('hit', bool(hits))):
            if yes and self.first_steps[side][key] is None:
                self.first_steps[side][key] = self.steps
        return attempts

    def _outcome(self, truncated):
        red, blue = int(self.red_alive_mask.sum()), int(self.blue_alive_mask.sum())
        if blue == 0:
            return True, False, False, 'red_win_elimination'
        if red == 0:
            return False, True, False, 'blue_win_elimination'
        if truncated:
            return False, True, False, 'blue_win_timeout'
        return False, False, False, 'ongoing'
