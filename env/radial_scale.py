"""One- and two-pair central radial scenarios on the existing five-slot interface."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np

from .interception import InterceptionConfig, InterceptionEnv


@dataclass(frozen=True)
class RadialScaleConfig(InterceptionConfig):
    scenario: str = 'radial_scale'
    active_pairs: int = 1

    def __post_init__(self):
        super().__post_init__()
        if (self.scenario != 'radial_scale' or self.active_pairs not in (1, 2) or
                self.attack_mode != 'radial_straight' or
                self.assignment_method != 'ideal_global_distance_matching' or
                self.observation_layout != 'relative_assignment_target_velocity_no_defenders' or
                self.episode_protocol != 'resolve_all_one_penalty_trade' or
                self.breach_penalty_mode != 'per_target' or not self.defender_oob_retire):
            raise ValueError('Radial scale requires one or two pairs and the central ideal interface')


class RadialScaleEnv(InterceptionEnv):
    """Active pairs are equally spaced; unused learner slots remain masked."""

    def __init__(self, config: RadialScaleConfig):
        if not isinstance(config, RadialScaleConfig):
            raise ValueError('RadialScaleEnv requires RadialScaleConfig')
        super().__init__(config)

    @property
    def contract(self):
        result = super().contract
        source = Path(__file__).read_bytes() + Path(__file__).with_name('interception.py').read_bytes()
        count = self.config.active_pairs
        result.update(schema='Interception/12', version='interception_12',
                      n_attackers=count, active_defenders=count, active_attackers=count,
                      state_schema='five_slot_padded_defender_and_attacker_xy_vxy_alive_assignment_matrix',
                      implementation_sha256=hashlib.sha256(source).hexdigest(),
                      config=asdict(self.config))
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        count = self.config.active_pairs
        angles = np.pi/2 + np.arange(count)*2*np.pi/count + self.rng.uniform(-np.pi/60, np.pi/60, count)
        attacker_radii = self.rng.uniform(160, 180, count)
        defender_radii = self.rng.uniform(50, 60, count)
        defender_angles = angles + np.pi/9
        self.positions.fill(0)
        self.velocities.fill(0)
        self.attacker_course_directions.fill(0)
        self.defender_alive[:] = np.arange(5) < count
        self.attacker_alive[:] = np.arange(5) < count
        self.positions[0, :count] = np.column_stack((defender_radii*np.cos(defender_angles),
                                                      defender_radii*np.sin(defender_angles)))
        self.positions[1, :count] = np.column_stack((attacker_radii*np.cos(angles),
                                                      attacker_radii*np.sin(angles)))
        self.attacker_course_directions[:count] = -self.positions[1, :count]/attacker_radii[:, None]
        self.velocities[1, :count] = self.config.attacker_initial_speed*self.attacker_course_directions[:count]
        self.assignment = self._make_assignment()
        self._update_assignment(0)
        return self.observations(), self._info([], 0.0, np.zeros(5), None)

    def _info(self, events, global_reward, local_rewards, outcome, *, first_breach=False):
        info = super()._info(events, global_reward, local_rewards, outcome,
                             first_breach=first_breach)
        info['version'] = 'interception_12'
        if 'episode_summary' in info:
            info['episode_summary']['n_attackers'] = self.config.active_pairs
            info['episode_summary']['defender_retired_count'] = int((~self.defender_alive[:self.config.active_pairs]).sum())
        return info
