"""v3.48 mechanics with Li (2023) Eq. (25) reward and task-success outcome.

This is reward/outcome alignment only, not a full paper environment replica.
"""
import numpy as np

from .combat_v348 import CombatEnvironmentV348
from .combat_v330 import CombatEnvironmentV330
from .combat_v36 import CombatEnvironmentV36
from .combat_v37 import CombatEnvironmentV37
from .combat_v39 import CombatEnvironmentV39
from .v350_reward import nearest_target_rewards


REWARD_SEMANTICS = 'li2023_eq25_nearest_alive_target_no_pbrs'
OUTCOME_SEMANTICS = 'destroy_all_blue_else_blue_win_at_episode_end'


class CombatEnvironmentV350(CombatEnvironmentV348):
    environment_version = '3.50'
    # Reuse the existing zero potential and non-shaping reward hooks. Neither
    # state_potential nor the PBRS difference/terminal compensation is called.
    potential = CombatEnvironmentV330.potential
    advantage_reward = CombatEnvironmentV36.advantage_reward
    reward_aliases = CombatEnvironmentV37.reward_aliases
    _reward_diagnostics = CombatEnvironmentV39._reward_diagnostics

    def shaping_next_potential(self, actual_phi_next, terminated, truncated):
        return actual_phi_next

    def dense_combat_reward(self):
        diagnostic, self.reward_target_indices = nearest_target_rewards(
            self.red, self.blue, self.config['reward'])
        return diagnostic['tactical'], diagnostic

    def _outcome(self, truncated):
        if not self.blue_alive_mask.any():
            reason = ('red_win_mutual_destruction' if not self.red_alive_mask.any()
                      else 'red_win_elimination')
            return True, False, False, reason
        if not self.red_alive_mask.any():
            return False, True, False, 'blue_win_elimination'
        if truncated:
            return False, True, False, 'blue_win_timeout'
        return False, False, False, 'ongoing'

    def _paper_info(self):
        return dict(reward_semantics=REWARD_SEMANTICS, outcome_semantics=OUTCOME_SEMANTICS,
                    pbrs_enabled=False, full_paper_environment_replication=False,
                    reward_target_indices=self.reward_target_indices.copy(),
                    reward_target_selection='nearest_alive_blue_distance_then_index',
                    reward_sampling_phase='post_motion_noncombat_pre_combat_settlement')

    def _info(self, *args, **kwargs):
        info = super()._info(*args, **kwargs)
        info.update(self._paper_info())
        return info

    def reset(self, seed=None):
        self.reward_target_indices = np.full(self.team_size, -1, dtype=np.int64)
        obs, info = super().reset(seed)
        info.update(self._paper_info())
        return obs, info

    def step(self, red_actions, blue_actions=None):
        result = super().step(red_actions, blue_actions)
        # Historical wrappers set the old research flag after _info returns.
        result[4].update(self._paper_info())
        return result
