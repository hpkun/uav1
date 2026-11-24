"""Isolated one-versus-one Jia interception using the existing five-slot learner."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

from .interception import InterceptionConfig, InterceptionEnv


@dataclass(frozen=True)
class OneVsOneConfig(InterceptionConfig):
    scenario: str = 'one_vs_one'

    def __post_init__(self):
        super().__post_init__()
        if (self.scenario != 'one_vs_one' or self.attack_mode != 'jia_straight' or
                self.assignment_method != 'ideal_global_distance_matching' or
                self.observation_layout != 'relative_assignment_target_velocity_no_defenders' or
                not self.defender_oob_retire):
            raise ValueError('One-versus-one requires the isolated Jia low-level configuration')


class OneVsOneEnv(InterceptionEnv):
    """One active pair; unused slots stay masked for the existing MADDPG loop."""

    def __init__(self, config: OneVsOneConfig):
        if not isinstance(config, OneVsOneConfig):
            raise ValueError('OneVsOneEnv requires OneVsOneConfig')
        super().__init__(config)

    @property
    def contract(self):
        result = super().contract
        source = Path(__file__).read_bytes() + Path(__file__).with_name('interception.py').read_bytes()
        result.update(schema='Interception/10', version='interception_10',
                      n_attackers=1, active_defenders=1, active_attackers=1,
                      reward_schema=('single_capture_or_first_breach+assigned_target_approach+'
                                     + ('assigned_target_heading+' if self.config.heading_weight else '')
                                     + 'boundary_retirement; team=global+local_active'),
                      state_schema='five_slot_padded_defender_and_attacker_xy_vxy_alive_assignment_matrix',
                      implementation_sha256=hashlib.sha256(source).hexdigest(),
                      config=asdict(self.config))
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        c = self.config
        self.defender_alive[1:] = False
        self.attacker_alive[1:] = False
        self.velocities[0, 1:] = 0.0
        self.velocities[1, 1:] = 0.0
        self.positions[0, 0] = (c.base_x + self.rng.uniform(-10.0, 10.0),
                                self.rng.uniform(-95.0, -85.0))
        self.positions[1, 0] = (c.base_x + self.rng.uniform(-4.0, 4.0),
                                self.rng.uniform(75.0, 90.0))
        self.velocities[0, 0] = (0.0, c.defender_initial_speed)
        target = np.array([c.base_x, c.base_y + c.breach_radius], dtype=np.float64)
        direction = target - self.positions[1, 0]
        direction /= np.linalg.norm(direction)
        self.attacker_course_directions[0] = direction
        self.velocities[1, 0] = c.attacker_initial_speed * direction
        self.assignment = self._make_assignment()
        self._update_assignment(0)
        return self.observations(), self._info([], 0.0, np.zeros(5), None)

    def _info(self, events, global_reward, local_rewards, outcome, *, first_breach=False):
        info = super()._info(events, global_reward, local_rewards, outcome,
                             first_breach=first_breach)
        info['version'] = 'interception_10'
        if 'episode_summary' in info:
            info['episode_summary']['n_attackers'] = 1
            info['episode_summary']['defender_retired_count'] = int(not self.defender_alive[0])
        return info

    def step(self, defender_actions):
        obs, individual, terminated, truncated, info = super().step(defender_actions)
        local = info['local_rewards']
        adjustment = float(local[0] - local.mean())
        self.mean_local_return += adjustment
        self.team_return += adjustment
        info['team_reward'] += adjustment
        for key in ('approach', 'heading', 'boundary'):
            if key in info['reward_components']:
                info['reward_components'][key] *= 5
        if 'episode_summary' in info:
            info['episode_summary']['mean_local_return'] = self.mean_local_return
            info['episode_summary']['team_return'] = self.team_return
        if not np.isclose(sum(info['reward_components'].values()), info['team_reward'],
                          rtol=0, atol=1e-9):
            raise AssertionError('One-versus-one reward ledger mismatch')
        return obs, individual, terminated, truncated, info
