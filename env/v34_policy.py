"""Deterministic, current-sensor-only tiered Blue response."""
import numpy as np
from .v30_policy import SensorLimitedPursuitPolicy
from .sensor import sensor_geometry


class SensorLimitedTieredBluePolicy(SensorLimitedPursuitPolicy):
    def __init__(self, config, action_config, sensor_config, scenario_config):
        super().__init__(config, action_config, sensor_config, scenario_config)
        self.reset()

    def reset(self):
        self.evading = np.zeros(5, dtype=bool)
        self.last_modes = []
        self.counts = dict.fromkeys(('SEARCH', 'PURSUIT', 'EVADE', 'BOUNDARY_GUARD'), 0)
        self.evade_episode = False

    def team_actions(self, team, targets):
        visible, distances, _, _ = sensor_geometry(team, targets, self.sensor_config)
        actions, modes = [], []
        for i, own in enumerate(team):
            indices = np.flatnonzero(visible[i])
            guard = (np.hypot(own.x, own.y) >= self.config['guard_radius']
                     or own.altitude <= self.config['guard_altitude_min']
                     or own.altitude >= self.config['guard_altitude_max'])
            if not own.alive:
                self.evading[i] = False
                mode, action = 'DEAD', np.zeros(3, dtype=np.float32)
            elif guard or not indices.size:
                self.evading[i] = False
                mode = 'BOUNDARY_GUARD' if guard else 'SEARCH'
                action = self._action(own, targets, visible[i], distances[i])
            else:
                nearest = min(indices, key=lambda j: (distances[i, j], j))
                d = distances[i, nearest]
                if self.evading[i]:
                    if d >= self.config['evade_exit_distance']:
                        self.evading[i] = False
                else:
                    enemies = np.count_nonzero(visible[i] & (distances[i] <= self.config['threat_radius']))
                    friends = sum(s.alive and np.linalg.norm(
                        [s.x-own.x, s.y-own.y, s.z-own.z]) <= self.config['support_radius'] for s in team)
                    self.evading[i] = d <= self.config['evade_enter_distance'] and enemies > friends
                if self.evading[i]:
                    mode = 'EVADE'
                    threats = indices[distances[i, indices] <= self.config['evade_exit_distance']]
                    centroid = np.mean([[targets[j].x, targets[j].y] for j in threats], axis=0)
                    delta = np.array([own.x, own.y]) - centroid
                    if np.linalg.norm(delta) <= 1e-6:
                        target = targets[nearest]
                        delta = np.array([own.x-target.x, own.y-target.y])
                    heading = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-6 else own.psi + np.pi
                    action = self.action_toward(own, heading, 0., self.config['desired_speed'])
                    self.evade_episode = True
                else:
                    mode = 'PURSUIT'
                    action = self._action(own, targets, visible[i], distances[i])
            if mode != 'DEAD':
                self.counts[mode] += 1
            actions.append(action)
            modes.append(mode)
        self.last_modes = modes
        return np.stack(actions)

    def diagnostics(self):
        return dict(blue_policy_modes=list(self.last_modes),
            blue_search_agent_steps=self.counts['SEARCH'],
            blue_pursuit_agent_steps=self.counts['PURSUIT'],
            blue_evade_agent_steps=self.counts['EVADE'],
            blue_guard_agent_steps=self.counts['BOUNDARY_GUARD'],
            blue_evade_episode=self.evade_episode)
