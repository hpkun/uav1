"""Local-observation cooperative waypoint navigation with finite-horizon PBRS."""
import numpy as np
from env.models import AircraftState, AircraftSpec
from env.v311_kinematics import advance_aircraft
from .config import validate_config, observation_dim, IDENTITY
from .geometry import waypoint_geometry


class Navigation3DEnv:
    environment_id = IDENTITY
    action_dim = 3

    def __init__(self, config):
        self.config = validate_config(config)
        self.num_agents = config['num_agents']
        self.observation_dim = observation_dim(config)
        self.offsets = np.asarray(config['formation_offsets_ned'], dtype=float)
        self.route = np.asarray(config['waypoint_centers_ned'], dtype=float)
        self.assigned = np.array(config['active_agents'], dtype=bool)
        m = config['motion']
        self.spec = AircraftSpec(m['v_min'], m['v_max'], -m['pitch_limit_rad'], m['pitch_limit_rad'])
        self._done = True

    @property
    def positions(self):
        return np.array([[s.x, s.y, s.z] for s in self.states])

    @property
    def alive_mask(self):
        return np.array([s.alive for s in self.states], dtype=bool) & self.assigned

    @property
    def valid_mask(self):
        return self.alive_mask & (not self._done)

    @property
    def goals(self):
        return self.route[self.waypoint_index] + self.offsets

    @property
    def final_goals(self):
        return self.route[-1] + self.offsets

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        self.seed, self.steps, self.waypoint_index = int(seed), 0, 0
        self._done = False
        p = np.asarray(self.config['initial_center_ned']) + self.offsets
        p = p + rng.uniform(-self.config['initial_jitter_m'], self.config['initial_jitter_m'], p.shape)
        self.states = []
        for i, position in enumerate(p):
            displacement = self.goals[i] - position
            heading = np.arctan2(displacement[1], displacement[0])
            pitch = np.arctan2(-displacement[2], np.hypot(*displacement[:2]))
            self.states.append(AircraftState(*position, self.config['initial_speed'], float(pitch), float(heading), bool(self.assigned[i])))
        self.component_totals = {k: np.zeros(self.num_agents) for k in
            ('distance', 'alignment', 'formation', 'danger', 'failure', 'time', 'completion')}
        self.component_positive = {k: np.zeros(self.num_agents) for k in self.component_totals}
        self.component_negative = {k: np.zeros(self.num_agents) for k in self.component_totals}
        return self.observe(), self._info(False, 'running')

    def potential_components(self):
        """Fixed final destinations make Phi independent of waypoint switching.

        Intermediate waypoints define ordered task completion, not new reward
        baselines. Final destinations are also explicitly present in observation.
        All three potentials are nonnegative and bounded by their weights.
        """
        r = self.config['reward']
        result = {k: np.zeros(self.num_agents) for k in ('distance', 'alignment', 'formation')}
        positions, mask = self.positions, self.valid_mask
        for i in np.flatnonzero(mask):
            state = self.states[i]
            g = waypoint_geometry(positions[i], state.theta, state.psi, self.final_goals[i])
            result['distance'][i] = r['distance_weight'] * (1 - np.clip(g.distance/self.config['distance_scale_m'], 0, 1))
            result['alignment'][i] = r['alignment_weight'] * (1-g.angle/np.pi)
            peers = np.flatnonzero(mask & (np.arange(self.num_agents) != i))
            error = np.linalg.norm((positions[peers]-positions[i])-(self.offsets[peers]-self.offsets[i]), axis=-1).mean() if len(peers) else 0.
            result['formation'][i] = r['formation_weight'] * (1-np.clip(error/r['formation_scale_m'], 0, 1))
        return result

    def observe(self):
        c, n = self.config, self.num_agents
        obs = np.zeros((n, self.observation_dim), dtype=np.float32)
        p, mask = self.positions, self.valid_mask
        velocities = np.array([s.velocity_vector() for s in self.states])
        position_scale = np.array([c['bounds']['horizontal_limit_m']]*2 + [c['bounds']['max_altitude_m']])
        for i in np.flatnonzero(mask):
            s = self.states[i]
            g = waypoint_geometry(p[i], s.theta, s.psi, self.goals[i])
            obs[i, :20] = np.r_[p[i]/position_scale, s.v/c['motion']['v_max'],
                np.sin(s.psi), np.cos(s.psi), np.sin(s.theta), np.cos(s.theta),
                (self.goals[i]-p[i])/c['distance_scale_m'],
                (self.final_goals[i]-p[i])/c['distance_scale_m'],
                g.angle/np.pi, g.body_azimuth/np.pi, g.body_elevation/np.pi,
                self.waypoint_index/max(1, len(self.route)-1), 1., 1-self.steps/c['max_steps']]
            # Stable slot order: ascending teammate index, omitting self.
            for slot, j in enumerate(k for k in range(n) if k != i):
                if mask[j] and np.linalg.norm(p[j]-p[i]) <= c['visibility_radius_m']:
                    obs[i, 20+10*slot:30+10*slot] = np.r_[
                        (p[j]-p[i])/c['distance_scale_m'],
                        (velocities[j]-velocities[i])/(2*c['motion']['v_max']),
                        (self.offsets[j]-self.offsets[i])/c['distance_scale_m'], 1.]
        return np.clip(obs, -1, 1)

    def step(self, actions):
        if self._done:
            raise RuntimeError('reset required after navigation episode end')
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.num_agents, 3) or not np.isfinite(actions).all() or (np.abs(actions) > 1).any():
            raise ValueError('navigation actions must be finite bounded [agents,3]')
        before = self.potential_components()
        previous_positions, before_mask = self.positions, self.valid_mask.copy()
        self.states = [advance_aircraft(s, a, self.config['dt'], self.config['motion'], self.spec) for s, a in zip(self.states, actions)]
        self.steps += 1
        p, b, r = self.positions, self.config['bounds'], self.config['reward']
        # Swept closest approach on each linear integration interval prevents tunnelling.
        collision, danger = np.zeros(self.num_agents, bool), np.zeros(self.num_agents, bool)
        for i in np.flatnonzero(before_mask):
            for j in np.flatnonzero(before_mask & (np.arange(self.num_agents) > i)):
                start = previous_positions[i]-previous_positions[j]
                delta = (p[i]-p[j])-start
                t = np.clip(-np.dot(start, delta)/np.dot(delta, delta), 0, 1) if np.dot(delta, delta) > 0 else 0.
                separation = np.linalg.norm(start+t*delta)
                collision[[i, j]] |= separation <= r['collision_radius_m']
                danger[[i, j]] |= separation < r['danger_radius_m']
        outside = (np.abs(p[:, :2]) >= b['horizontal_limit_m']).any(-1) | (-p[:, 2] <= b['min_altitude_m']) | (-p[:, 2] >= b['max_altitude_m'])
        lost = before_mask & (outside | collision)
        for i in np.flatnonzero(lost):
            self.states[i].alive = False
        reached = bool((np.linalg.norm(p-self.goals, axis=-1)[self.assigned] <= self.config['goal_radius_m']).all())
        success = reached and self.waypoint_index == len(self.route)-1 and not lost.any()
        switched = reached and not success and not lost.any()
        if switched:
            self.waypoint_index += 1
        terminated = bool(success or lost.any())
        truncated = bool(not terminated and self.steps >= self.config['max_steps'])
        # Time limit is an observed finite-horizon terminal, not a continuing task.
        self._done = terminated or truncated
        after = self.potential_components()
        components = {k: (r['gamma']*after[k]-before[k])*before_mask for k in before}
        components.update(danger=-r['danger_penalty']*danger*before_mask,
            failure=-r['failure_penalty']*lost,
            time=-r['time_penalty']*before_mask,
            completion=r['completion_bonus']*success*before_mask)
        reward = np.sum(list(components.values()), axis=0)
        for key, value in components.items():
            self.component_totals[key] += value
            self.component_positive[key] += np.maximum(value, 0)
            self.component_negative[key] += np.minimum(value, 0)
        reason = 'success' if success else ('collision' if collision.any() else 'boundary') if lost.any() else 'time_limit' if truncated else 'running'
        info = self._info(success, reason)
        info.update(reward_components={k: v.copy() for k, v in components.items()},
            potential_before=before, potential_after=after, waypoint_switched=bool(switched),
            executed_actions=actions*before_mask[:, None], collision=collision, boundary_loss=outside & before_mask)
        return self.observe(), reward.astype(np.float32), terminated, truncated, info

    def _info(self, success, reason):
        return dict(environment_id=self.environment_id, seed=self.seed, episode_steps=self.steps,
            success=bool(success), termination_reason=reason, waypoint_index=self.waypoint_index,
            alive_mask=self.alive_mask.copy(), valid_mask=self.valid_mask.copy(),
            goal_distances=np.linalg.norm(self.positions-self.goals, axis=-1),
            component_totals={k: v.copy() for k, v in self.component_totals.items()},
            component_positive={k: v.copy() for k, v in self.component_positive.items()},
            component_negative={k: v.copy() for k, v in self.component_negative.items()},
            episode_return=float(sum(v.sum() for v in self.component_totals.values())))
