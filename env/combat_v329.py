"""v3.28 mechanics with shared high-level increment action control only."""
import numpy as np
from .combat_v328 import CombatEnvironmentV328
from .v310_policy import PaperNearestTargetBluePolicy
from .v329_control import action_to_control_v329
from .geometry import engagement_geometry
from .v36_geometry import angle_within


class CombatEnvironmentV329(CombatEnvironmentV328):
    environment_version = '3.29'

    def __init__(self, config):
        super().__init__(config)
        self.fixed_policy = PaperNearestTargetBluePolicy(self.config['blue_policy'], self.config['action'])
        self.controller_diagnostics = {}
        self.firing_geometry_diagnostics = {}

    def reset(self, seed=None):
        self.controller_diagnostics = {}
        self.firing_geometry_diagnostics = {}
        return super().reset(seed)

    def _advance(self, states, actions):
        phis = np.zeros(self.team_size, dtype=np.float64)
        samples = []
        for index, state in enumerate(states):
            if state.alive:
                control = action_to_control_v329(state, actions[index], self.config['action'], self.spec,
                                               self.dynamics.gravity)
                phis[index] = control.phi
                samples.append((control.nz, abs(control.phi), *np.abs(actions[index])))
                states[index] = self.integrator.step(state, control, self.dynamics, self.spec)
        side = 'red' if states is self.red else 'blue'
        # Sums and counts allow logging weighted means over living agents only.
        values = np.asarray(samples, dtype=float).reshape(-1, 5)
        self.controller_diagnostics[side] = dict(
            agent_steps=len(samples), nz_sum=float(values[:, 0].sum()),
            nz_saturated=int((values[:, 0] >= 9.9).sum()),
            abs_phi_sum=float(values[:, 1].sum()),
            abs_action_sum=values[:, 2:].sum(axis=0).tolist(),
            action_saturated=(values[:, 2:] >= .99).sum(axis=0).tolist())
        return phis

    def _entry_attempts(self, attackers, targets, fire_states, side):
        # Read-only geometry at the authoritative firing stage; no hit sampling.
        if side == 'red':
            eligible, _ = self.weapon.eligibility(attackers, targets, self.red_ammo)
            counts = dict(live=0, range=0, range_az=0, geometry=0, eligible=0)
            for i, own in enumerate(attackers):
                flags = dict(live=False, range=False, range_az=False, geometry=False)
                for target in targets:
                    if not own.alive or not target.alive:
                        continue
                    flags['live'] = True
                    g = engagement_geometry(own, target)
                    in_range = self.weapon.range_min <= g.distance <= self.weapon.range_max
                    az = in_range and angle_within(abs(g.boresight_azimuth_error), self.weapon.boresight_azimuth_max)
                    full = az and angle_within(abs(g.boresight_elevation_error), self.weapon.boresight_elevation_max)
                    for key, value in (('range', in_range), ('range_az', az), ('geometry', full)):
                        flags[key] |= bool(value)
                for key, value in flags.items():
                    counts[key] += int(value)
                counts['eligible'] += int(eligible[i].any())
            self.firing_geometry_diagnostics = counts
        return super()._entry_attempts(attackers, targets, fire_states, side)

    def step(self, red_actions, blue_actions=None):
        result = super().step(red_actions, blue_actions)
        result[4]['controller_diagnostics'] = {side: dict(row) for side, row in self.controller_diagnostics.items()}
        result[4]['red_firing_geometry_agent_steps'] = dict(self.firing_geometry_diagnostics)
        return result
