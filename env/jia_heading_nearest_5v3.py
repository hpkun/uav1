"""Isolated 5v3 radial speed-heading environment with fixed-ID full-entity observations and nearest-target shaping."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from gymnasium import spaces

from .jia_heading import rk4_pose, wrap_heading


@dataclass(frozen=True)
class JiaHeadingNearest5v3Config:
    scenario: str = 'jia_heading_nearest_5v3_radial'
    n_defenders: int = 5
    n_attackers: int = 3
    arena_half_extent: float = 200.0
    base_x: float = 0.0
    base_y: float = 0.0
    breach_radius: float = 30.0
    danger_radius: float = 45.0
    capture_radius: float = 5.0
    dt: float = 0.1
    episode_steps: int = 5000
    defender_speed_min: float = 6.0
    defender_speed_max: float = 10.0
    defender_acceleration_max: float = 2.0
    defender_turn_rate_max: float = 0.5
    attacker_speed_max: float = 6.0
    attacker_acceleration_max: float = 2.0
    defender_initial_speed: float = 6.0
    attacker_initial_speed: float = 6.0
    sensing_radius: float = 600.0
    communication_radius: float | None = None
    visible_neighbors: int = 4
    visible_attackers: int = 3
    observation_layout: str = 'relative_nearest_entities'
    assignment_method: str = 'none'
    spawn_min_spacing: float = 8.0
    attack_mode: str = 'radial_straight'
    episode_protocol: str = 'resolve_all_one_penalty_trade'
    capture_reward: float = 20.0
    boundary_penalty: float = 2.0
    defender_oob_retire: bool = True
    penetration_penalty: float = 50.0
    breach_penalty_mode: str = 'per_target'
    approach_weight: float = 0.8
    heading_weight: float = 0.03
    time_penalty_per_step: float = 0.015

    def __post_init__(self):
        c = self
        if (
            c.scenario != 'jia_heading_nearest_5v3_radial'
            or (c.n_defenders, c.n_attackers) != (5, 3)
            or (c.visible_neighbors, c.visible_attackers) != (4, 3)
            or c.observation_layout != 'relative_nearest_entities'
            or c.assignment_method != 'none'
            or c.attack_mode != 'radial_straight'
            or c.episode_protocol != 'resolve_all_one_penalty_trade'
            or c.breach_penalty_mode != 'per_target'
            or not c.defender_oob_retire
            or c.base_x != 0.0
            or c.base_y != 0.0
            or c.arena_half_extent < 180.0
            or c.breach_radius != 30.0
        ):
            raise ValueError('Jia heading nearest 5v3 scenario requires the isolated 5v3 radial nearest-entity interface')
        if type(c.episode_steps) is not int or c.episode_steps < 1:
            raise ValueError('episode_steps must be positive')
        positive_fields = (
            'arena_half_extent',
            'breach_radius',
            'danger_radius',
            'capture_radius',
            'dt',
            'defender_speed_min',
            'defender_speed_max',
            'defender_acceleration_max',
            'defender_turn_rate_max',
            'attacker_speed_max',
            'attacker_acceleration_max',
            'sensing_radius',
            'spawn_min_spacing',
        )
        if any(not math.isfinite(getattr(c, name)) or getattr(c, name) <= 0 for name in positive_fields):
            raise ValueError('Physical scales must be finite and positive')
        if c.communication_radius is not None and (
            not math.isfinite(c.communication_radius) or c.communication_radius <= 0
        ):
            raise ValueError('communication_radius must be None or a positive finite value')
        if (
            not (0 < c.defender_speed_min <= c.defender_speed_max)
            or not math.isfinite(c.defender_initial_speed)
            or not math.isfinite(c.attacker_initial_speed)
            or c.defender_initial_speed != c.defender_speed_min
            or c.attacker_initial_speed != c.attacker_speed_max
        ):
            raise ValueError('Initial speeds must match the isolated speed-heading interface')
        if c.breach_radius >= c.danger_radius or c.breach_radius > c.arena_half_extent:
            raise ValueError('Breach circle must fit within the arena and danger radius')
        reward_fields = (
            'capture_reward',
            'boundary_penalty',
            'penetration_penalty',
            'approach_weight',
            'heading_weight',
            'time_penalty_per_step',
        )
        if any(not math.isfinite(getattr(c, name)) or getattr(c, name) < 0 for name in reward_fields):
            raise ValueError('Reward weights must be finite and nonnegative')


class JiaHeadingNearest5v3Env:
    """Five controlled defenders against three radial attackers with 41D fixed-ID entity obs and nearest-target shaping."""

    def __init__(self, config: JiaHeadingNearest5v3Config):
        if not isinstance(config, JiaHeadingNearest5v3Config):
            raise ValueError('JiaHeadingNearest5v3Env requires JiaHeadingNearest5v3Config')
        self.config = config
        self.n_defenders = 5
        self.n_attackers = 3
        self.observation_dim = 41
        self.observation_core_dim = 41
        self.state_dim = 50
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.n_defenders, self.observation_dim), np.float32)
        self.state_space = spaces.Box(-np.inf, np.inf, (self.state_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_defenders, 2), np.float32)
        self.base = np.array([config.base_x, config.base_y], dtype=np.float64)
        self.defender_positions = np.zeros((self.n_defenders, 2), np.float64)
        self.attacker_positions = np.zeros((self.n_attackers, 2), np.float64)
        self.defender_velocities = np.zeros((self.n_defenders, 2), np.float64)
        self.attacker_velocities = np.zeros((self.n_attackers, 2), np.float64)
        self.defender_headings = np.zeros(self.n_defenders, np.float64)
        self.attacker_headings = np.zeros(self.n_attackers, np.float64)
        self.defender_physical_commands = np.zeros((self.n_defenders, 2), np.float64)
        self.attacker_physical_commands = np.zeros((self.n_attackers, 2), np.float64)
        self.defender_alive = np.ones(self.n_defenders, bool)
        self.attacker_alive = np.ones(self.n_attackers, bool)
        self.assignment = None
        self.rng = np.random.default_rng()
        self.seed = None
        self.finished = True

    @property
    def positions(self):
        return (self.defender_positions, self.attacker_positions)

    @property
    def velocities(self):
        return (self.defender_velocities, self.attacker_velocities)

    @property
    def headings(self):
        return (self.defender_headings, self.attacker_headings)

    @property
    def last_physical_commands(self):
        return (self.defender_physical_commands, self.attacker_physical_commands)

    @property
    def contract(self):
        c = self.config
        source = Path(__file__).read_bytes() + Path(__file__).with_name('jia_heading.py').read_bytes()
        reward_schema = (
            'shared_capture_and_per_target_breach+nearest_visible_approach'
            + ('+nearest_visible_heading' if c.heading_weight else '')
            + '+boundary_retirement'
            + '; team=global+sum(local_i)'
            + ('+fixed_shared_time_cost' if c.time_penalty_per_step else '')
        )
        return dict(
            schema='Interception/15',
            version='interception_15',
            n_agents=self.n_defenders,
            n_attackers=self.n_attackers,
            observation_dim=self.observation_dim,
            state_dim=self.state_dim,
            action_dim=2,
            action_schema='normalized_speed_turn_box',
            reward_schema=reward_schema,
            observation_schema='self_rel_base_xy_vxy_heading_cos_sin_fixed_id_4_ally_3_enemy_rel_xy_rel_vxy_valid',
            state_schema='all_5_defender_and_3_attacker_xy_vxy_alive;defender_heading_cos_sin',
            assignment_schema=None,
            implementation_sha256=hashlib.sha256(source).hexdigest(),
            config=asdict(c),
        )

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('Use an explicit JiaHeadingNearest5v3Config for scenario changes')
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        attacker_angles = (
            np.pi / 2
            + np.arange(self.n_attackers) * (2 * np.pi / self.n_attackers)
            + self.rng.uniform(-np.pi / 60, np.pi / 60, self.n_attackers)
        )
        attacker_radii = self.rng.uniform(160.0, 180.0, self.n_attackers)
        defender_angles = (
            np.pi / 2
            + np.arange(self.n_defenders) * (2 * np.pi / self.n_defenders)
            + np.pi / 9
            + self.rng.uniform(-np.pi / 60, np.pi / 60, self.n_defenders)
        )
        defender_radii = self.rng.uniform(50.0, 60.0, self.n_defenders)
        self.defender_positions = np.column_stack((
            defender_radii * np.cos(defender_angles),
            defender_radii * np.sin(defender_angles),
        ))
        self.attacker_positions = np.column_stack((
            attacker_radii * np.cos(attacker_angles),
            attacker_radii * np.sin(attacker_angles),
        ))
        self.defender_velocities.fill(0.0)
        self.attacker_velocities.fill(0.0)
        self.defender_alive.fill(True)
        self.attacker_alive.fill(True)
        self.defender_headings = wrap_heading(
            np.arctan2(self.defender_positions[:, 1], self.defender_positions[:, 0])
        )
        self.attacker_headings = wrap_heading(
            np.arctan2(self.attacker_positions[:, 1], self.attacker_positions[:, 0]) + np.pi
        )
        self.defender_velocities = self.config.defender_initial_speed * np.column_stack((
            np.cos(self.defender_headings),
            np.sin(self.defender_headings),
        ))
        self.attacker_velocities = self.config.attacker_speed_max * np.column_stack((
            np.cos(self.attacker_headings),
            np.sin(self.attacker_headings),
        ))
        self.defender_physical_commands.fill(0.0)
        self.attacker_physical_commands.fill(0.0)
        self.step_count = 0
        self.finished = False
        self.captured_count = 0
        self.penetrated_count = 0
        self.defender_oob_count = 0
        self.first_capture_time_s = None
        self.global_return = 0.0
        self.mean_local_return = 0.0
        self.team_return = 0.0
        self.event_history = []
        return self.observations(), self._info([], 0.0, np.zeros(self.n_defenders), None)

    def observations(self):
        c = self.config
        out = np.zeros((self.n_defenders, self.observation_dim), dtype=np.float32)
        pos_scale = c.arena_half_extent
        rel_pos_scale = 2.0 * c.arena_half_extent
        ally_vel_scale = 2.0 * c.defender_speed_max
        enemy_vel_scale = c.defender_speed_max + c.attacker_speed_max
        ally_radius = c.communication_radius if c.communication_radius is not None else c.sensing_radius
        for i in range(self.n_defenders):
            if not self.defender_alive[i]:
                continue
            own_pos = self.defender_positions[i]
            own_vel = self.defender_velocities[i]
            out[i, 0:2] = (own_pos - self.base) / pos_scale
            out[i, 2:4] = own_vel / c.defender_speed_max
            out[i, 4] = math.cos(self.defender_headings[i])
            out[i, 5] = math.sin(self.defender_headings[i])
            slot = 0
            for j in range(self.n_defenders):
                if j == i:
                    continue
                if self.defender_alive[j]:
                    delta_pos = self.defender_positions[j] - own_pos
                    if float(np.linalg.norm(delta_pos)) <= ally_radius:
                        k = 6 + slot * 5
                        out[i, k:k + 2] = delta_pos / rel_pos_scale
                        out[i, k + 2:k + 4] = (self.defender_velocities[j] - own_vel) / ally_vel_scale
                        out[i, k + 4] = 1.0
                slot += 1
            for j in range(self.n_attackers):
                if self.attacker_alive[j]:
                    delta_pos = self.attacker_positions[j] - own_pos
                    if float(np.linalg.norm(delta_pos)) <= c.sensing_radius:
                        k = 26 + j * 5
                        out[i, k:k + 2] = delta_pos / rel_pos_scale
                        out[i, k + 2:k + 4] = (self.attacker_velocities[j] - own_vel) / enemy_vel_scale
                        out[i, k + 4] = 1.0
        return out

    def central_state(self):
        c = self.config
        defender_rows = np.zeros((self.n_defenders, 5), dtype=np.float32)
        defender_rows[self.defender_alive, :2] = self.defender_positions[self.defender_alive] / c.arena_half_extent
        defender_rows[self.defender_alive, 2:4] = self.defender_velocities[self.defender_alive] / c.defender_speed_max
        defender_rows[:, 4] = self.defender_alive
        attacker_rows = np.zeros((self.n_attackers, 5), dtype=np.float32)
        attacker_rows[self.attacker_alive, :2] = self.attacker_positions[self.attacker_alive] / c.arena_half_extent
        attacker_rows[self.attacker_alive, 2:4] = self.attacker_velocities[self.attacker_alive] / c.attacker_speed_max
        attacker_rows[:, 4] = self.attacker_alive
        heading = np.column_stack((np.cos(self.defender_headings), np.sin(self.defender_headings))).astype(np.float32)
        heading[~self.defender_alive] = 0.0
        return np.concatenate((defender_rows.ravel(), attacker_rows.ravel(), heading.ravel()))

    def _validate_defender_actions(self, defender_actions):
        actions = np.asarray(defender_actions, dtype=np.float64)
        if actions.shape != (self.n_defenders, 2) or not np.isfinite(actions).all():
            raise ValueError('Expected finite (5,2) normalized speed-turn actions')
        if np.any(np.abs(actions[self.defender_alive]) > 1.0 + 1e-6):
            raise ValueError('Active normalized speed-turn action exceeds [-1,1]')
        return actions

    def _advance_motion(self, actions, defenders_before):
        c = self.config
        self.defender_physical_commands.fill(0.0)
        self.attacker_physical_commands.fill(0.0)
        for i in np.flatnonzero(defenders_before):
            speed = c.defender_speed_min + (actions[i, 0] + 1.0) * (c.defender_speed_max - c.defender_speed_min) / 2.0
            turn = c.defender_turn_rate_max * actions[i, 1]
            self.defender_physical_commands[i] = (speed, turn)
            self.defender_positions[i], self.defender_headings[i] = rk4_pose(
                self.defender_positions[i], self.defender_headings[i], speed, turn, c.dt
            )
            self.defender_velocities[i] = speed * np.array(
                [math.cos(self.defender_headings[i]), math.sin(self.defender_headings[i])]
            )
        for j in np.flatnonzero(self.attacker_alive):
            speed = c.attacker_speed_max
            self.attacker_physical_commands[j] = (speed, 0.0)
            self.attacker_positions[j], self.attacker_headings[j] = rk4_pose(
                self.attacker_positions[j], self.attacker_headings[j], speed, 0.0, c.dt
            )
            self.attacker_velocities[j] = speed * np.array(
                [math.cos(self.attacker_headings[j]), math.sin(self.attacker_headings[j])]
            )
        return (actions.copy(), np.zeros((self.n_attackers, 2), dtype=np.float64))

    def _heading_alignment(self, i, target):
        delta = self.attacker_positions[target] - self.defender_positions[i]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12:
            return 0.0
        unit_heading = np.array([math.cos(self.defender_headings[i]), math.sin(self.defender_headings[i])], dtype=np.float64)
        unit_target = delta / distance
        return float(np.clip(np.dot(unit_heading, unit_target), -1.0, 1.0))

    def step(self, defender_actions):
        if self.finished:
            raise RuntimeError('reset required after episode end')
        c = self.config
        actions = self._validate_defender_actions(defender_actions).copy()
        def_before = self.defender_positions.copy()
        att_before = self.attacker_positions.copy()
        alive_before = self.attacker_alive.copy()
        defenders_before = self.defender_alive.copy()
        actions[~defenders_before] = 0.0
        joint_actions = self._advance_motion(actions, defenders_before)
        self.step_count += 1
        time_s = self.step_count * c.dt
        events = []

        boundary = defenders_before & np.any(np.abs(self.defender_positions) > c.arena_half_extent, axis=1)
        for i in np.flatnonzero(boundary):
            for axis in range(2):
                if abs(self.defender_positions[i, axis]) > c.arena_half_extent:
                    self.defender_positions[i, axis] = np.sign(self.defender_positions[i, axis]) * c.arena_half_extent
                    self.defender_velocities[i, axis] = 0.0
            self.defender_alive[i] = False
            self.defender_velocities[i] = 0.0
            events.append(dict(
                type='defender_oob',
                step=self.step_count,
                time_s=time_s,
                defender=int(i),
                position=self.defender_positions[i].tolist(),
            ))
        self.defender_oob_count += int(boundary.sum())

        for j in np.flatnonzero(self.attacker_alive):
            for axis in range(2):
                if abs(self.attacker_positions[j, axis]) > c.arena_half_extent:
                    self.attacker_positions[j, axis] = np.sign(self.attacker_positions[j, axis]) * c.arena_half_extent
                    self.attacker_velocities[j, axis] = 0.0

        captured_by = np.zeros(self.n_defenders, np.int64)
        contacts = sorted(
            (float(np.linalg.norm(self.defender_positions[i] - self.attacker_positions[j])), int(i), int(j))
            for i in np.flatnonzero(self.defender_alive)
            for j in np.flatnonzero(self.attacker_alive)
        )
        matches = []
        for distance, i, j in contacts:
            if distance > c.capture_radius:
                break
            if self.defender_alive[i] and self.attacker_alive[j]:
                self.defender_alive[i] = False
                self.attacker_alive[j] = False
                self.defender_velocities[i] = 0.0
                self.attacker_velocities[j] = 0.0
                matches.append((i, j))
        for i, j in matches:
            captured_by[i] += 1
            self.captured_count += 1
            if self.first_capture_time_s is None:
                self.first_capture_time_s = time_s
            events.append(dict(
                type='capture',
                step=self.step_count,
                time_s=time_s,
                defender=i,
                attacker=j,
                position=self.attacker_positions[j].tolist(),
            ))

        breached = []
        for j in np.flatnonzero(self.attacker_alive):
            if np.linalg.norm(self.attacker_positions[j] - self.base) <= c.breach_radius:
                breached.append(int(j))
                events.append(dict(
                    type='penetration',
                    step=self.step_count,
                    time_s=time_s,
                    attacker=int(j),
                    position=self.attacker_positions[j].tolist(),
                ))
        if self.step_count >= c.episode_steps and self.attacker_alive.sum() > len(breached):
            events.append(dict(type='time_limit', step=self.step_count, time_s=time_s))
            for j in np.flatnonzero(self.attacker_alive):
                if j not in breached:
                    breached.append(int(j))
                    events.append(dict(
                        type='penetration',
                        step=self.step_count,
                        time_s=time_s,
                        attacker=int(j),
                        position=self.attacker_positions[j].tolist(),
                        reason='time_limit',
                    ))
        self.attacker_alive[breached] = False
        self.attacker_velocities[breached] = 0.0
        self.penetrated_count += len(breached)
        outcome = (
            ('penetrated' if self.penetrated_count else 'all_captured')
            if not self.attacker_alive.any()
            else None
        )
        self.finished = outcome is not None

        approach = np.zeros(self.n_defenders, dtype=np.float64)
        heading = np.zeros(self.n_defenders, dtype=np.float64)
        reward_target_ids = np.full(self.n_defenders, -1, dtype=np.int64)
        for i in range(self.n_defenders):
            if not defenders_before[i] or boundary[i]:
                continue
            candidates = [
                int(j)
                for j in np.flatnonzero(alive_before)
                if float(np.linalg.norm(att_before[j] - def_before[i])) <= c.sensing_radius
            ]
            if not candidates:
                continue
            d_before = min(float(np.linalg.norm(att_before[j] - def_before[i])) for j in candidates)
            d_after, nearest_after = min(
                (float(np.linalg.norm(self.attacker_positions[j] - self.defender_positions[i])), j)
                for j in candidates
            )
            approach[i] = d_before - d_after
            reward_target_ids[i] = nearest_after
            if c.heading_weight:
                heading[i] = self._heading_alignment(i, nearest_after)

        global_reward = c.capture_reward * int(captured_by.sum()) - c.penetration_penalty * len(breached)
        local = (
            -c.boundary_penalty * boundary.astype(float)
            + c.approach_weight * approach
            + c.heading_weight * heading
        )
        individual_rewards = global_reward + local
        team_reward = global_reward + float(local.sum())

        self.global_return += global_reward
        self.mean_local_return += float(local.mean())
        self.team_return += team_reward
        self.event_history.extend(events)

        info = self._info(events, global_reward, local, outcome)
        info['reward_target_ids'] = reward_target_ids
        info['reward_components']['approach'] = float(c.approach_weight * approach.sum())
        if c.heading_weight:
            info['reward_components']['heading'] = float(c.heading_weight * heading.sum())
        if c.time_penalty_per_step:
            penalty = c.time_penalty_per_step
            self.team_return -= penalty
            info['team_reward'] -= penalty
            info['reward_components']['time'] = -penalty
            info['individual_rewards'][defenders_before] -= penalty
            if 'episode_summary' in info:
                info['episode_summary']['team_return'] = self.team_return
            individual_rewards = info['individual_rewards']
        if not np.isclose(sum(info['reward_components'].values()), info['team_reward'], rtol=0, atol=1e-10):
            raise AssertionError('Jia heading nearest 5v3 reward ledger mismatch')
        info['executed_actions'] = (joint_actions[0].copy(), joint_actions[1].copy())
        return (
            self.observations(),
            individual_rewards.astype(np.float32),
            outcome in ('penetrated', 'all_captured'),
            outcome == 'time_limit',
            info,
        )

    def _info(self, events, global_reward, local_rewards, outcome):
        c = self.config
        local_arr = np.asarray(local_rewards, dtype=np.float64).copy()
        info = dict(
            version='interception_15',
            step=self.step_count,
            time_s=self.step_count * c.dt,
            attack_mode=c.attack_mode,
            central_state=self.central_state(),
            active_masks=self.defender_alive.astype(np.float32),
            defender_alive=self.defender_alive.copy(),
            attacker_alive=self.attacker_alive.copy(),
            assigned_target_ids=None,
            global_reward=float(global_reward),
            local_rewards=local_arr,
            individual_rewards=(global_reward + local_arr).copy(),
            team_reward=float(global_reward + local_arr.sum()),
            events=list(events),
            reward_components=dict(
                capture=float(c.capture_reward * sum(e['type'] == 'capture' for e in events)),
                boundary=-float(c.boundary_penalty * sum(e['type'] == 'defender_oob' for e in events)),
                penetration=-float(c.penetration_penalty * sum(e['type'] == 'penetration' for e in events)),
            ),
            physical_commands=(
                self.defender_physical_commands.copy(),
                self.attacker_physical_commands.copy(),
            ),
        )
        if outcome is not None:
            info['episode_summary'] = dict(
                attack_mode=c.attack_mode,
                outcome=outcome,
                seed=self.seed,
                steps=self.step_count,
                duration_s=info['time_s'],
                n_attackers=self.n_attackers,
                captured_count=self.captured_count,
                penetrated_count=self.penetrated_count,
                first_capture_time_s=self.first_capture_time_s,
                defender_oob_count=self.defender_oob_count,
                defender_retired_count=int((~self.defender_alive).sum()),
                global_return=self.global_return,
                mean_local_return=self.mean_local_return,
                team_return=self.team_return,
            )
        return info

    def close(self):
        pass
