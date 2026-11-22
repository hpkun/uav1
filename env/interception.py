"""Two-dimensional airbase interception task."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from gymnasium import spaces
from assignment import DistanceAssignment, NearestVisibleAssignment


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
    communication_radius: float | None = None  # Historical relative-observation runs only.
    visible_attackers: int = 5
    visible_neighbors: int = 4
    observation_layout: str = 'absolute_no_base'
    assignment_period: int = 20
    assignment_method: str = 'distance_matching'
    spawn_min_spacing: float = 8.0
    attack_mode: str = 'direct_rush'
    episode_protocol: str = 'resolve_all_one_penalty_trade'
    split_waypoint_x: float = 70.0
    split_waypoint_y: float = 5.0
    split_switch_radius: float = 12.0
    danger_weight: float = 0.0  # Kept for loading historical runs.
    capture_reward: float = 20.0
    boundary_penalty: float = 2.0
    penetration_penalty: float = 100.0
    approach_weight: float = 0.05

    def __post_init__(self):
        if ((self.n_defenders,self.n_attackers)!=(5,5) or
                (self.visible_neighbors,self.visible_attackers) not in ((3,3),(4,5))):
            raise ValueError('Interception requires five agents per side and a supported observation layout')
        if self.observation_layout not in ('relative_with_base','absolute_no_base','relative_assignment'):
            raise ValueError('Unsupported observation layout')
        if self.observation_layout=='relative_assignment' and (
                (self.visible_neighbors,self.visible_attackers)!=(4,5) or
                self.episode_protocol!='resolve_all_one_penalty_trade'):
            raise ValueError('Assignment layout requires the five-on-five one-to-one scenario')
        if type(self.assignment_period) is not int or self.assignment_period < 1:
            raise ValueError('assignment_period must be positive')
        if self.assignment_method not in ('nearest_visible_legacy','independent_nearest','distance_matching'):
            raise ValueError('Unsupported assignment method')
        if self.observation_layout=='relative_with_base' and (
                self.communication_radius is None or not math.isfinite(self.communication_radius)
                or self.communication_radius<=0):
            raise ValueError('Historical relative observations require a communication radius')
        if self.attack_mode not in ('direct_rush', 'split_attack'):
            raise ValueError('Unknown attack mode')
        if self.episode_protocol not in ('resolve_all_one_penalty_trade', 'resolve_all_one_penalty', 'resolve_all', 'first_breach'):
            raise ValueError('Unknown episode protocol')
        if type(self.episode_steps) is not int or self.episode_steps < 1:
            raise ValueError('episode_steps must be positive')
        names = ('arena_half_extent','breach_radius','danger_radius','capture_radius','dt',
                 'defender_speed_max','defender_acceleration_max','attacker_speed_max',
                 'attacker_acceleration_max','sensing_radius','spawn_min_spacing',
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
    def __init__(self, config: InterceptionConfig | None = None):
        self.config = config or InterceptionConfig()
        self.assigned_layout = self.config.observation_layout == 'relative_assignment'
        own_features=4 if self.config.observation_layout!='relative_with_base' else 6
        self.observation_dim = own_features+5*(self.config.visible_neighbors+self.config.visible_attackers)+(5 if self.assigned_layout else 0)
        self.state_dim = 75 if self.assigned_layout else 5*self.observation_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (5, self.observation_dim), np.float32)
        self.action_space = spaces.Box(-self.config.defender_acceleration_max,
                                       self.config.defender_acceleration_max, (5, 2), np.float32)
        self.state_space = spaces.Box(-np.inf, np.inf, (self.state_dim,), np.float32)
        self.base = np.array([self.config.base_x, self.config.base_y], dtype=np.float64)
        self.positions = np.zeros((2, 5, 2), np.float64)  # 0 defenders, 1 attackers
        self.velocities = np.zeros_like(self.positions)
        self.attacker_alive = np.ones(5, bool)
        self.defender_alive = np.ones(5, bool)
        self.assignment = self._make_assignment() if self.assigned_layout else None
        self.rng = np.random.default_rng()
        self.finished = True

    @property
    def contract(self):
        return dict(schema='Interception/2' if self.assigned_layout else 'Interception/1',
                    version='interception_2' if self.assigned_layout else 'interception_1', n_agents=5,
                    n_attackers=5, observation_dim=self.observation_dim, state_dim=self.state_dim, action_dim=2,
                    action_schema='world_acceleration_disk_si',
                    max_acceleration=self.config.defender_acceleration_max,
                    reward_schema=('shared_capture_and_first_breach+assigned_target_approach+boundary; team=global+mean(local_i)'
                                   if self.assigned_layout else
                                   'shared_capture_and_first_breach+local_shaping; team=global+mean(local_i)'
                                   if self.config.episode_protocol in ('resolve_all_one_penalty_trade','resolve_all_one_penalty') else
                                   'shared_capture_and_breach+local_shaping; team=global+mean(local_i)'
                                   if self.config.episode_protocol=='resolve_all' else
                                   'r_i=global+local_i; team=global+mean(local_i)'),
                    observation_schema=('self_xy_vxy_visible_friend_and_attacker_relative_xy_relative_vxy_valid_assigned_target_relative_xy_relative_vxy_valid'
                                        if self.assigned_layout else
                                        'self_xy_vxy_base_relative_xy_nearest_3_defenders_nearest_3_attackers_relative_xy_vxy_valid'
                                        if self.observation_dim==36 else
                                        'self_xy_vxy_base_relative_xy_all_4_visible_defenders_all_5_visible_attackers_relative_xy_vxy_valid'
                                        if self.observation_dim==51 else
                                        'self_xy_vxy_all_4_visible_defenders_all_5_visible_attackers_absolute_xy_vxy_valid'),
                    state_schema=('all_defender_and_attacker_xy_vxy_alive_assignment_matrix'
                                  if self.assigned_layout else 'five_defender_observations_in_agent_order'),
                    assignment_schema=(f'{self.config.assignment_method}_period_{self.config.assignment_period}'
                                       if self.assigned_layout and self.config.assignment_method!='nearest_visible_legacy'
                                       else f'nearest_visible_periodic_or_event_{self.config.assignment_period}'
                                       if self.assigned_layout else None),
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

    def _make_assignment(self):
        if self.config.assignment_method == 'nearest_visible_legacy':
            return NearestVisibleAssignment(self.config.assignment_period)
        return DistanceAssignment(self.config.assignment_period, self.config.assignment_method)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('Use an explicit InterceptionConfig for scenario changes')
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.positions[0] = self._spawn(-35, 35, -95, -75)
        self.positions[1] = self._spawn(-30, 30, 75, 90)
        self.velocities.fill(0)
        self.attacker_alive.fill(True)
        self.defender_alive.fill(True)
        self.split_flank = np.where(np.argsort(np.argsort(self.positions[1,:,0])) < 2, -1, 1)
        self.split_switched = np.zeros(5, bool)
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
        if self.assignment is not None:
            self.assignment = self._make_assignment()
            self.assignment.update(self.positions[0],self.positions[1],self.defender_alive,
                                   self.attacker_alive,self.config.sensing_radius,0)
        return self.observations(), self._info([], 0.0, np.zeros(5), None)

    def observations(self):
        c = self.config
        out = np.zeros((5,self.observation_dim),dtype=np.float32)
        absolute=c.observation_layout=='absolute_no_base'
        own_features=4 if c.observation_layout!='relative_with_base' else 6
        for i in range(5):
            if not self.defender_alive[i]:
                continue
            own = self.positions[0, i]
            out[i, :2] = own / c.arena_half_extent
            out[i, 2:4] = self.velocities[0, i] / (c.defender_speed_max if self.assigned_layout else c.attacker_speed_max if absolute else c.defender_speed_max)
            if c.observation_layout=='relative_with_base':
                out[i, 4:6] = (self.base-own) / (2*c.arena_half_extent)
            groups = (([(j, self.positions[0,j], self.velocities[0,j]) for j in range(5)
                        if j != i and self.defender_alive[j]],
                       c.sensing_radius if c.observation_layout!='relative_with_base' else c.communication_radius, own_features, c.visible_neighbors),
                      ([(j, self.positions[1,j], self.velocities[1,j]) for j in range(5) if self.attacker_alive[j]],
                       c.sensing_radius, own_features+5*c.visible_neighbors, c.visible_attackers))
            for candidates, radius, start, slots in groups:
                nearest = sorted(((float(np.linalg.norm(p-own)),j,p,v) for j,p,v in candidates),
                                 key=lambda row:(row[0],row[1]))
                for slot, (distance, _, pos, vel) in enumerate(row for row in nearest if row[0] <= radius):
                    if slot == slots:
                        break
                    k = start + slot*5
                    out[i,k:k+2] = pos/c.arena_half_extent if absolute else (pos-own)/(c.sensing_radius if self.assigned_layout else 2*c.arena_half_extent)
                    out[i,k+2:k+4] = (vel-self.velocities[0,i])/(c.defender_speed_max+c.attacker_speed_max) if self.assigned_layout else vel/c.attacker_speed_max if absolute else (vel-self.velocities[0,i])/c.attacker_speed_max
                    out[i,k+4] = 1.0
            if self.assignment is not None:
                target = self.assignment.targets[i]
                if target >= 0 and (self.config.assignment_method == 'nearest_visible_legacy' or
                                    np.linalg.norm(self.positions[1,target]-own) <= c.sensing_radius):
                    out[i,-5:-3] = (self.positions[1,target]-own)/c.sensing_radius
                    out[i,-3:-1] = (self.velocities[1,target]-self.velocities[0,i])/(c.defender_speed_max+c.attacker_speed_max)
                    out[i,-1] = 1.0
        return out

    def central_state(self):
        if self.assignment is not None:
            physical = []
            for side,maximum,alive in ((0,self.config.defender_speed_max,self.defender_alive),
                                       (1,self.config.attacker_speed_max,self.attacker_alive)):
                rows=np.zeros((5,5),np.float32)
                rows[alive,:2]=self.positions[side,alive]/self.config.arena_half_extent
                rows[alive,2:4]=self.velocities[side,alive]/maximum
                rows[:,4]=alive
                physical.append(rows.ravel())
            matrix=np.zeros((5,5),np.float32)
            for i,j in enumerate(self.assignment.targets):
                if self.defender_alive[i] and j>=0:matrix[i,j]=1
            return np.concatenate((*physical,matrix.ravel()))
        return self.observations().reshape(self.state_dim).copy()

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
        defenders_before = self.defender_alive.copy()
        assigned_before = self.assignment.targets.copy() if self.assignment is not None else None
        actions = actions.copy()
        actions[~defenders_before] = 0
        attacker_actions = self._attacker_actions()
        joint_actions = np.stack((actions,attacker_actions))
        self.positions[0,defenders_before] += self.velocities[0,defenders_before]*c.dt
        self.positions[1,self.attacker_alive] += self.velocities[1,self.attacker_alive]*c.dt
        self.velocities[0,defenders_before] = _limit(
            self.velocities[0,defenders_before]+actions[defenders_before]*c.dt,c.defender_speed_max)
        self.velocities[1,self.attacker_alive] = _limit(
            self.velocities[1,self.attacker_alive]+attacker_actions[self.attacker_alive]*c.dt,c.attacker_speed_max)
        self.step_count += 1
        time_s = self.step_count*c.dt
        events = []
        boundary = defenders_before & np.any(np.abs(self.positions[0]) > c.arena_half_extent,axis=1)
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
        if c.episode_protocol == 'resolve_all_one_penalty_trade':
            # Match simultaneous contacts once, by distance then stable agent IDs.
            contacts = sorted((float(np.linalg.norm(self.positions[0,i]-self.positions[1,j])),int(i),int(j))
                              for i in np.flatnonzero(self.defender_alive)
                              for j in np.flatnonzero(self.attacker_alive))
            matches = []
            for distance,i,j in contacts:
                if distance > c.capture_radius:
                    break
                if self.defender_alive[i] and self.attacker_alive[j]:
                    self.defender_alive[i] = False
                    self.attacker_alive[j] = False
                    self.velocities[0,i] = 0
                    self.velocities[1,j] = 0
                    matches.append((i,j))
        else:
            matches = []
            for j in np.flatnonzero(self.attacker_alive):
                distances = np.linalg.norm(self.positions[0]-self.positions[1,j],axis=1)
                i = int(np.argmin(distances))
                if distances[i] <= c.capture_radius:
                    self.attacker_alive[j] = False
                    self.velocities[1,j] = 0
                    matches.append((i,int(j)))
        for i,j in matches:
            captured_by[i] += 1
            self.captured_count += 1
            if self.first_capture_time_s is None:
                self.first_capture_time_s = time_s
            events.append(dict(type='capture',step=self.step_count,time_s=time_s,defender=i,attacker=j,
                               position=self.positions[1,j].tolist()))
        breached = []
        for j in np.flatnonzero(self.attacker_alive):
            if np.linalg.norm(self.positions[1,j]-self.base) <= c.breach_radius:
                breached.append(int(j))
                events.append(dict(type='penetration',step=self.step_count,time_s=time_s,attacker=int(j),
                                   position=self.positions[1,j].tolist()))
        if c.episode_protocol!='first_breach':
            if self.step_count >= c.episode_steps and self.attacker_alive.sum() > len(breached):
                events.append(dict(type='time_limit',step=self.step_count,time_s=time_s))
                for j in np.flatnonzero(self.attacker_alive):
                    if j not in breached:
                        breached.append(int(j))
                        events.append(dict(type='penetration',step=self.step_count,time_s=time_s,
                                           attacker=int(j),position=self.positions[1,j].tolist(),reason='time_limit'))
            first_breach=bool(breached) and self.penetrated_count==0
            self.attacker_alive[breached] = False
            self.velocities[1,breached] = 0
            self.penetrated_count += len(breached)
            outcome = ('penetrated' if self.penetrated_count else 'all_captured') if not self.attacker_alive.any() else None
        else:
            first_breach=bool(breached)
            outcome = ('penetrated' if breached else 'all_captured' if not self.attacker_alive.any() else
                       'time_limit' if self.step_count >= c.episode_steps else None)
        self.finished = outcome is not None
        if outcome == 'time_limit' and c.episode_protocol=='first_breach':
            events.append(dict(type='time_limit',step=self.step_count,time_s=time_s))
        # Legacy reward definitions are retained for historical run configurations.
        approach = np.zeros(5)
        for i in range(5):
            if not defenders_before[i]:
                continue
            candidates = ([int(assigned_before[i])] if assigned_before is not None and assigned_before[i]>=0 else
                          [] if assigned_before is not None else
                          [j for j in np.flatnonzero(alive_before)
                           if c.danger_weight or np.linalg.norm(before[0,i]-before[1,j]) <= c.sensing_radius])
            if len(candidates):
                target = min(candidates,key=lambda j:(np.linalg.norm(before[0,i]-before[1,j]),int(j)))
                target_after = (self.positions[1,target] if c.episode_protocol!='first_breach' or c.danger_weight
                                else before[1,target])
                approach[i] = np.linalg.norm(before[0,i]-before[1,target])-np.linalg.norm(self.positions[0,i]-target_after)
        # Legacy configurations retain their original reward for historical checkpoints.
        global_reward = 0.0
        if c.episode_protocol in ('resolve_all_one_penalty_trade','resolve_all_one_penalty'):
            global_reward = c.capture_reward*int(captured_by.sum())-c.penetration_penalty*first_breach
        elif c.episode_protocol=='resolve_all':
            global_reward = c.capture_reward*int(captured_by.sum())-c.penetration_penalty*len(breached)
        elif c.danger_weight:
            active_distance = np.linalg.norm(self.positions[1,self.attacker_alive]-self.base,axis=1)
            global_reward = -c.danger_weight*float(np.exp(-np.square(active_distance/c.danger_radius)).sum())
        local = (-c.boundary_penalty*boundary.astype(float)+c.approach_weight*approach)
        if c.episode_protocol=='first_breach':
            local += c.capture_reward*captured_by-c.penetration_penalty*bool(breached)
        self.global_return += global_reward
        self.mean_local_return += float(local.mean())
        self.team_return += global_reward+float(local.mean())
        self.event_history.extend(events)
        if self.assignment is not None:
            self.assignment.update(self.positions[0],self.positions[1],self.defender_alive,
                                   self.attacker_alive,c.sensing_radius,self.step_count,events)
        info = self._info(events,global_reward,local,outcome,first_breach=first_breach)
        if assigned_before is not None:info['reward_target_ids']=assigned_before.copy()
        info['reward_components']['approach'] = float(c.approach_weight*approach.mean())
        if not np.isclose(sum(info['reward_components'].values()), info['team_reward'],rtol=0,atol=1e-10):
            raise AssertionError('Interception reward ledger mismatch')
        info['executed_actions'] = joint_actions.copy()
        return self.observations(), (global_reward+local).astype(np.float32), outcome in ('penetrated','all_captured'), outcome=='time_limit', info

    def _info(self, events, global_reward, local_rewards, outcome, *, first_breach=False):
        info = dict(version='interception_2' if self.assigned_layout else 'interception_1',
                    step=self.step_count, time_s=self.step_count*self.config.dt,
                    attack_mode=self.config.attack_mode, central_state=self.central_state(),
                    active_masks=self.defender_alive.astype(np.float32),
                    defender_alive=self.defender_alive.copy(), attacker_alive=self.attacker_alive.copy(),
                    assigned_target_ids=(self.assignment.targets.copy() if self.assignment is not None else None),
                    global_reward=float(global_reward), local_rewards=np.asarray(local_rewards,dtype=np.float64).copy(),
                    individual_rewards=(global_reward+local_rewards).copy(),
                    team_reward=float(global_reward+np.mean(local_rewards)), events=list(events),
                    reward_components=dict(capture=float(self.config.capture_reward*sum(e['type']=='capture' for e in events))
                                           /(5 if self.config.episode_protocol=='first_breach' else 1),
                                           boundary=-float(self.config.boundary_penalty*sum(e['type']=='defender_oob' for e in events))/5,
                                           penetration=-float(self.config.penetration_penalty*(
                                               int(first_breach) if self.config.episode_protocol in
                                               ('resolve_all_one_penalty_trade','resolve_all_one_penalty') else
                                               int(any(e['type']=='penetration' for e in events)) if self.config.episode_protocol=='first_breach'
                                               else sum(e['type']=='penetration' for e in events)))))
        if self.config.danger_weight and self.config.episode_protocol=='first_breach':
            info['reward_components']['danger'] = float(global_reward)
        if outcome is not None:
            info['episode_summary'] = dict(attack_mode=self.config.attack_mode,outcome=outcome,
                seed=self.seed,steps=self.step_count,duration_s=info['time_s'],captured_count=self.captured_count,
                penetrated_count=(self.penetrated_count if self.config.episode_protocol!='first_breach' else
                                  sum(e['type']=='penetration' for e in events)),
                first_capture_time_s=self.first_capture_time_s,defender_oob_count=self.defender_oob_count,
                defender_retired_count=int((~self.defender_alive).sum()),
                global_return=self.global_return,mean_local_return=self.mean_local_return,team_return=self.team_return)
        return info

    def close(self):
        pass
