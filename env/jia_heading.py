"""Isolated radial speed-heading dynamics with constant-command RK4 integration."""
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
from gymnasium import spaces

from .interception import InterceptionConfig, InterceptionEnv


def wrap_heading(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def rk4_pose(position, heading, speed, turn_rate, dt):
    """Integrate (x,y,psi) with constant speed and turn-rate during one step."""
    state = np.array([position[0], position[1], heading], dtype=np.float64)
    def derivative(value):
        return np.array([speed*np.cos(value[2]), speed*np.sin(value[2]), turn_rate])
    k1 = derivative(state)
    k2 = derivative(state + dt*k1/2)
    k3 = derivative(state + dt*k2/2)
    k4 = derivative(state + dt*k3)
    result = state + dt*(k1+2*k2+2*k3+k4)/6
    return result[:2], float(wrap_heading(result[2]))


@dataclass(frozen=True)
class JiaHeadingConfig(InterceptionConfig):
    scenario: str = 'jia_heading_radial'
    defender_speed_min: float = 6.0
    defender_turn_rate_max: float = 0.5
    time_penalty_per_step: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        c = self
        if (c.scenario != 'jia_heading_radial' or c.attack_mode != 'radial_straight' or
                c.assignment_method != 'ideal_global_distance_matching' or
                c.observation_layout != 'relative_assignment_target_velocity_no_defenders' or
                c.episode_protocol != 'resolve_all_one_penalty_trade' or
                c.breach_penalty_mode != 'per_target' or not c.defender_oob_retire or
                not math.isfinite(c.defender_speed_min) or not math.isfinite(c.defender_turn_rate_max) or
                not math.isfinite(c.time_penalty_per_step) or c.time_penalty_per_step < 0 or
                not 0 < c.defender_speed_min <= c.defender_speed_max or
                c.defender_initial_speed != c.defender_speed_min or c.defender_turn_rate_max <= 0 or
                c.attacker_initial_speed != c.attacker_speed_max):
            raise ValueError('Jia heading scenario requires the isolated radial speed-heading interface')


class JiaHeadingEnv(InterceptionEnv):
    def __init__(self, config: JiaHeadingConfig):
        if not isinstance(config, JiaHeadingConfig):
            raise ValueError('JiaHeadingEnv requires JiaHeadingConfig')
        self.headings = np.zeros((2,5), np.float64)
        self.last_physical_commands = np.zeros((2,5,2), np.float64)
        super().__init__(config)
        self.observation_dim = self.observation_core_dim + 2
        self.state_dim = 85
        self.observation_space = spaces.Box(-np.inf,np.inf,(5,self.observation_dim),np.float32)
        self.state_space = spaces.Box(-np.inf,np.inf,(self.state_dim,),np.float32)
        self.action_space = spaces.Box(-1.0,1.0,(5,2),np.float32)

    @property
    def contract(self):
        result = super().contract
        source = Path(__file__).read_bytes() + Path(__file__).with_name('interception.py').read_bytes()
        result.update(schema='Interception/13',version='interception_13',
                      action_schema='normalized_speed_turn_box',action_dim=2,
                      observation_schema=result['observation_schema']+';self_heading_cos_sin',
                      state_schema=result['state_schema']+';defender_heading_cos_sin',
                      implementation_sha256=hashlib.sha256(source).hexdigest(),config=asdict(self.config))
        if self.config.time_penalty_per_step:
            result['reward_schema'] += '+fixed_shared_time_cost'
        result.pop('max_acceleration',None)
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed,options=options)
        for side in range(2):
            self.headings[side] = wrap_heading(np.arctan2(self.positions[side,:,1],self.positions[side,:,0])+
                                               (np.pi if side else 0))
        self.velocities[0] = self.config.defender_initial_speed*np.column_stack((np.cos(self.headings[0]),np.sin(self.headings[0])))
        self.velocities[1] = self.config.attacker_speed_max*np.column_stack((np.cos(self.headings[1]),np.sin(self.headings[1])))
        self.last_physical_commands.fill(0)
        return self.observations(),self._info([],0.0,np.zeros(5),None)

    def observations(self):
        base = super().observations()
        heading = np.column_stack((np.cos(self.headings[0]),np.sin(self.headings[0]))).astype(np.float32)
        heading[~self.defender_alive] = 0
        return np.concatenate((base,heading),axis=1)

    def central_state(self):
        base = super().central_state()
        heading = np.column_stack((np.cos(self.headings[0]),np.sin(self.headings[0]))).astype(np.float32)
        heading[~self.defender_alive] = 0
        return np.concatenate((base,heading.ravel()))

    def _info(self, events, global_reward, local_rewards, outcome, *, first_breach=False):
        info = super()._info(events,global_reward,local_rewards,outcome,first_breach=first_breach)
        info['version'] = 'interception_13'
        info['physical_commands'] = self.last_physical_commands.copy()
        return info

    def step(self, defender_actions):
        if not self.config.time_penalty_per_step:
            return super().step(defender_actions)
        active_before = self.defender_alive.copy()
        observations, rewards, terminated, truncated, info = super().step(defender_actions)
        penalty = self.config.time_penalty_per_step
        self.team_return -= penalty
        info['team_reward'] -= penalty
        info['reward_components']['time'] = -penalty
        info['individual_rewards'][active_before] -= penalty
        if 'episode_summary' in info:
            info['episode_summary']['team_return'] = self.team_return
        if not np.isclose(sum(info['reward_components'].values()), info['team_reward'], rtol=0, atol=1e-10):
            raise AssertionError('Jia heading time reward ledger mismatch')
        return observations, info['individual_rewards'].astype(np.float32), terminated, truncated, info

    def _validate_defender_actions(self, defender_actions):
        actions = np.asarray(defender_actions,dtype=np.float64)
        if actions.shape != (5,2) or not np.isfinite(actions).all():
            raise ValueError('Expected finite (5,2) normalized speed-turn actions')
        if np.any(np.abs(actions[self.defender_alive]) > 1+1e-6):
            raise ValueError('Active normalized speed-turn action exceeds [-1,1]')
        return actions

    def _advance_motion(self, actions, defenders_before):
        c = self.config
        self.last_physical_commands.fill(0)
        for i in np.flatnonzero(defenders_before):
            speed = c.defender_speed_min+(actions[i,0]+1)*(c.defender_speed_max-c.defender_speed_min)/2
            turn = c.defender_turn_rate_max*actions[i,1]
            self.last_physical_commands[0,i] = (speed,turn)
            self.positions[0,i],self.headings[0,i] = rk4_pose(self.positions[0,i],self.headings[0,i],speed,turn,c.dt)
            self.velocities[0,i] = speed*np.array([np.cos(self.headings[0,i]),np.sin(self.headings[0,i])])
        for j in np.flatnonzero(self.attacker_alive):
            speed = c.attacker_speed_max
            self.last_physical_commands[1,j] = (speed,0)
            self.positions[1,j],self.headings[1,j] = rk4_pose(self.positions[1,j],self.headings[1,j],speed,0,c.dt)
            self.velocities[1,j] = speed*np.array([np.cos(self.headings[1,j]),np.sin(self.headings[1,j])])
        # The first slice is exactly the normalized defender action stored in Replay.
        return np.stack((actions,np.zeros_like(actions))),self.velocities[0].copy()

    def _heading_alignment(self, i, target, before, velocity_after):
        delta = self.positions[1,target]-self.positions[0,i]
        if np.linalg.norm(delta) <= 1e-12:
            return 0.0
        bearing = np.arctan2(delta[1],delta[0])
        return float(np.cos(self.headings[0,i]-bearing))
