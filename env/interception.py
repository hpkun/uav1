"""Two-dimensional airbase interception task."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class InterceptionConfig:
    n_defenders: int = 5
    n_attackers: int = 5
    arena_half_extent: float = 100.0
    base_x: float = 0.0
    base_y: float = -80.0
    breach_radius: float = 15.0
    danger_radius: float = 30.0
    capture_radius: float = 5.0
    dt: float = 0.1
    episode_steps: int = 5000
    defender_speed_max: float = 5.0
    defender_acceleration_max: float = 2.0
    attacker_speed_max: float = 8.0
    attacker_acceleration_max: float = 3.0
    sensing_radius: float = 80.0
    communication_radius: float = 60.0
    visible_attackers: int = 3
    visible_neighbors: int = 3
    spawn_min_spacing: float = 8.0
    attack_mode: str = 'direct_rush'
    split_waypoint_x: float = 70.0
    split_waypoint_y: float = 5.0
    split_switch_radius: float = 12.0
    danger_weight: float = 0.05
    capture_reward: float = 20.0
    boundary_penalty: float = 2.0
    penetration_penalty: float = 100.0
    approach_weight: float = 0.05

    def __post_init__(self):
        if (self.n_defenders, self.n_attackers, self.visible_attackers, self.visible_neighbors) != (5, 5, 3, 3):
            raise ValueError('Interception/1 requires five defenders, five attackers and three observation slots per role')
        if self.attack_mode not in ('direct_rush', 'split_attack'):
            raise ValueError('Unknown attack mode')
        if type(self.episode_steps) is not int or self.episode_steps < 1:
            raise ValueError('episode_steps must be positive')
        names = ('arena_half_extent','breach_radius','danger_radius','capture_radius','dt',
                 'defender_speed_max','defender_acceleration_max','attacker_speed_max',
                 'attacker_acceleration_max','sensing_radius','communication_radius','spawn_min_spacing',
                 'split_switch_radius')
        if any(not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0 for name in names):
            raise ValueError('Physical scales must be finite and positive')
        if any(not math.isfinite(getattr(self, name)) for name in
               ('base_x','base_y','split_waypoint_x','split_waypoint_y','danger_weight',
                'capture_reward','boundary_penalty','penetration_penalty','approach_weight')):
            raise ValueError('Parameters must be finite')
        if (self.breach_radius >= self.danger_radius or
                max(abs(self.base_x),abs(self.base_y))+self.breach_radius > self.arena_half_extent):
            raise ValueError('Breach circle must fit within the arena and danger radius')
        if any(getattr(self, name) < 0 for name in
               ('danger_weight','capture_reward','boundary_penalty','penetration_penalty','approach_weight')):
            raise ValueError('Reward weights must be nonnegative')


def _limit(vectors: np.ndarray, maximum: float) -> np.ndarray:
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors * np.minimum(1.0, maximum / np.maximum(lengths, 1e-12))


class InterceptionEnv:
    """Five controlled defenders against five scripted attackers."""
    observation_dim = 36
    state_dim = 180

    def __init__(self, config: InterceptionConfig | None = None):
        self.config = config or InterceptionConfig()
        self.observation_space = spaces.Box(-np.inf, np.inf, (5, 36), np.float32)
        self.action_space = spaces.Box(-self.config.defender_acceleration_max,
                                       self.config.defender_acceleration_max, (5, 2), np.float32)
        self.state_space = spaces.Box(-np.inf, np.inf, (180,), np.float32)
        self.base = np.array([self.config.base_x, self.config.base_y], dtype=np.float64)
        self.positions = np.zeros((2, 5, 2), np.float64)  # 0 defenders, 1 attackers
        self.velocities = np.zeros_like(self.positions)
        self.attacker_alive = np.ones(5, bool)
        self.rng = np.random.default_rng()
        self.finished = True

    @property
    def contract(self):
        return dict(schema='Interception/1', version='interception_1', n_agents=5,
                    n_attackers=5, observation_dim=36, state_dim=180, action_dim=2,
                    action_schema='world_acceleration_disk_si',
                    max_acceleration=self.config.defender_acceleration_max,
                    observation_schema='self_xy_vxy_base_relative_xy_nearest_3_defenders_nearest_3_attackers_relative_xy_vxy_valid',
                    state_schema='five_defender_observations_in_agent_order',
                    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    config=asdict(self.config))

    def _spawn(self, xlo, xhi, ylo, yhi):
        result = []
        for _ in range(5):
            for _attempt in range(1000):
                point = self.rng.uniform((xlo, ylo), (xhi, yhi))
                if all(np.linalg.norm(point-other) >= self.config.spawn_min_spacing for other in result):
                    result.append(point)
                    break
            else:
                raise RuntimeError('Could not sample a spaced deployment')
        return np.asarray(result)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('Use an explicit InterceptionConfig for scenario changes')
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.positions[0] = self._spawn(-35, 35, -95, -75)
        self.positions[1] = self._spawn(-30, 30, 75, 90)
        self.velocities.fill(0)
        self.attacker_alive.fill(True)
        self.split_flank = np.where(np.argsort(np.argsort(self.positions[1,:,0])) < 2, -1, 1)
        self.split_switched = np.zeros(5, bool)
        self.step_count = 0
        self.finished = False
        self.captured_count = 0
        self.defender_oob_count = 0
        self.first_capture_time_s = None
        self.global_return = 0.0
        self.mean_local_return = 0.0
        self.team_return = 0.0
        self.event_history = []
        return self.observations(), self._info([], 0.0, np.zeros(5), None)

    def observations(self):
        c = self.config
        out = np.zeros((5, 36), dtype=np.float32)
        for i in range(5):
            own = self.positions[0, i]
            out[i, :2] = own / c.arena_half_extent
            out[i, 2:4] = self.velocities[0, i] / c.defender_speed_max
            out[i, 4:6] = (self.base-own) / (2*c.arena_half_extent)
            groups = (([(j, self.positions[0,j], self.velocities[0,j]) for j in range(5) if j != i],
                       c.communication_radius, 6),
                      ([(j, self.positions[1,j], self.velocities[1,j]) for j in range(5) if self.attacker_alive[j]],
                       c.sensing_radius, 21))
            for candidates, radius, start in groups:
                nearest = sorted(((float(np.linalg.norm(p-own)),j,p,v) for j,p,v in candidates),
                                 key=lambda row:(row[0],row[1]))
                for slot, (distance, _, pos, vel) in enumerate(row for row in nearest if row[0] <= radius):
                    if slot == 3:
                        break
                    k = start + slot*5
                    out[i,k:k+2] = (pos-own)/(2*c.arena_half_extent)
                    out[i,k+2:k+4] = (vel-self.velocities[0,i])/c.attacker_speed_max
                    out[i,k+4] = 1.0
        return out

    def central_state(self):
        return self.observations().reshape(180).copy()

    def _attacker_actions(self):
        c = self.config
        actions = np.zeros((5,2), np.float64)
        for j in np.flatnonzero(self.attacker_alive):
            target = self.base.copy()
            if c.attack_mode == 'split_attack' and not self.split_switched[j]:
                waypoint = np.array([self.split_flank[j]*c.split_waypoint_x,c.split_waypoint_y])
                if np.linalg.norm(waypoint-self.positions[1,j]) <= c.split_switch_radius or self.positions[1,j,1] <= c.split_waypoint_y:
                    self.split_switched[j] = True
                else:
                    target = waypoint
            delta = target-self.positions[1,j]
            desired = c.attacker_speed_max*delta/max(np.linalg.norm(delta),1e-12)
            if c.attack_mode == 'direct_rush':
                peers = self.positions[1,self.attacker_alive]
                desired += 0.02*(peers.mean(axis=0)-self.positions[1,j])
                for other in peers:
                    gap = self.positions[1,j]-other
                    distance = np.linalg.norm(gap)
                    if 1e-9 < distance < c.spawn_min_spacing:
                        desired += 2*gap/distance
            actions[j] = _limit(((desired-self.velocities[1,j])/0.5)[None],c.attacker_acceleration_max)[0]
        return actions

    def step(self, defender_actions):
        if self.finished:
            raise RuntimeError('reset required after episode end')
        c = self.config
        actions = np.asarray(defender_actions,dtype=np.float64)
        if actions.shape != (5,2) or not np.isfinite(actions).all():
            raise ValueError('Expected finite (5,2) defender accelerations')
        if np.any(np.linalg.norm(actions,axis=1) > c.defender_acceleration_max+1e-6):
            raise ValueError('Defender acceleration exceeds disk')
        before = self.positions.copy()
        alive_before = self.attacker_alive.copy()
        attacker_actions = self._attacker_actions()
        joint_actions = np.stack((actions,attacker_actions))
        self.positions[0] += self.velocities[0]*c.dt
        self.positions[1,self.attacker_alive] += self.velocities[1,self.attacker_alive]*c.dt
        self.velocities[0] = _limit(self.velocities[0]+actions*c.dt,c.defender_speed_max)
        self.velocities[1,self.attacker_alive] = _limit(
            self.velocities[1,self.attacker_alive]+attacker_actions[self.attacker_alive]*c.dt,c.attacker_speed_max)
        self.step_count += 1
        time_s = self.step_count*c.dt
        events = []
        boundary = np.any(np.abs(self.positions[0]) > c.arena_half_extent,axis=1)
        for i in np.flatnonzero(boundary):
            for axis in range(2):
                if abs(self.positions[0,i,axis]) > c.arena_half_extent:
                    self.positions[0,i,axis] = np.sign(self.positions[0,i,axis])*c.arena_half_extent
                    self.velocities[0,i,axis] = 0
            events.append(dict(type='defender_oob',step=self.step_count,time_s=time_s,defender=int(i),
                               position=self.positions[0,i].tolist()))
        self.defender_oob_count += int(boundary.sum())
        # Scripted attackers are constrained to the arena; boundary handling never counts as a capture.
        for j in np.flatnonzero(self.attacker_alive):
            for axis in range(2):
                if abs(self.positions[1,j,axis]) > c.arena_half_extent:
                    self.positions[1,j,axis] = np.sign(self.positions[1,j,axis])*c.arena_half_extent
                    self.velocities[1,j,axis] = 0
        captured_by = np.zeros(5,np.int64)
        for j in np.flatnonzero(self.attacker_alive):
            distances = np.linalg.norm(self.positions[0]-self.positions[1,j],axis=1)
            i = int(np.argmin(distances))
            if distances[i] <= c.capture_radius:
                self.attacker_alive[j] = False
                self.velocities[1,j] = 0
                captured_by[i] += 1
                self.captured_count += 1
                if self.first_capture_time_s is None:
                    self.first_capture_time_s = time_s
                events.append(dict(type='capture',step=self.step_count,time_s=time_s,defender=i,attacker=int(j),
                                   position=self.positions[1,j].tolist()))
        breached = []
        for j in np.flatnonzero(self.attacker_alive):
            if np.linalg.norm(self.positions[1,j]-self.base) <= c.breach_radius:
                breached.append(int(j))
                events.append(dict(type='penetration',step=self.step_count,time_s=time_s,attacker=int(j),
                                   position=self.positions[1,j].tolist()))
        outcome = ('penetrated' if breached else 'all_captured' if not self.attacker_alive.any() else
                   'time_limit' if self.step_count >= c.episode_steps else None)
        self.finished = outcome is not None
        if outcome == 'time_limit':
            events.append(dict(type='time_limit',step=self.step_count,time_s=time_s))
        # Track the same pre-step target, including if it was captured this step.
        approach = np.zeros(5)
        for i in range(5):
            candidates = np.flatnonzero(alive_before)
            if len(candidates):
                target = min(candidates,key=lambda j:(np.linalg.norm(before[0,i]-before[1,j]),int(j)))
                approach[i] = np.linalg.norm(before[0,i]-before[1,target])-np.linalg.norm(self.positions[0,i]-self.positions[1,target])
        active_distance = np.linalg.norm(self.positions[1,self.attacker_alive]-self.base,axis=1)
        global_reward = -c.danger_weight*float(np.exp(-np.square(active_distance/c.danger_radius)).sum())
        local = (c.capture_reward*captured_by-c.boundary_penalty*boundary.astype(float)
                 -c.penetration_penalty*bool(breached)+c.approach_weight*approach)
        self.global_return += global_reward
        self.mean_local_return += float(local.mean())
        self.team_return += global_reward+float(local.mean())
        self.event_history.extend(events)
        info = self._info(events,global_reward,local,outcome)
        info['reward_components']['approach'] = float(c.approach_weight*approach.mean())
        if not np.isclose(sum(info['reward_components'].values()), info['team_reward'],rtol=0,atol=1e-10):
            raise AssertionError('Interception reward ledger mismatch')
        info['executed_actions'] = joint_actions.copy()
        return self.observations(), (global_reward+local).astype(np.float32), outcome in ('penetrated','all_captured'), outcome=='time_limit', info

    def _info(self, events, global_reward, local_rewards, outcome):
        info = dict(version='interception_1', step=self.step_count, time_s=self.step_count*self.config.dt,
                    attack_mode=self.config.attack_mode, central_state=self.central_state(),
                    active_masks=np.ones(5,dtype=np.float32), attacker_alive=self.attacker_alive.copy(),
                    global_reward=float(global_reward), local_rewards=np.asarray(local_rewards,dtype=np.float64).copy(),
                    individual_rewards=(global_reward+local_rewards).copy(),
                    team_reward=float(global_reward+np.mean(local_rewards)), events=list(events),
                    reward_components=dict(danger=float(global_reward),capture=float(self.config.capture_reward*sum(e['type']=='capture' for e in events))/5,
                                           boundary=-float(self.config.boundary_penalty*sum(e['type']=='defender_oob' for e in events))/5,
                                           penetration=-float(self.config.penetration_penalty*any(e['type']=='penetration' for e in events))))
        if outcome is not None:
            info['episode_summary'] = dict(attack_mode=self.config.attack_mode,outcome=outcome,
                seed=self.seed,steps=self.step_count,duration_s=info['time_s'],captured_count=self.captured_count,
                penetrated_count=sum(e['type']=='penetration' for e in events),
                first_capture_time_s=self.first_capture_time_s,defender_oob_count=self.defender_oob_count,
                global_return=self.global_return,mean_local_return=self.mean_local_return,team_return=self.team_return)
        return info

    def close(self):
        pass
