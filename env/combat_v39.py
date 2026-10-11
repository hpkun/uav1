"""v3.8 initialization/combat with independent paper-range reward diagnostics."""
import numpy as np
from .combat_v38 import CombatEnvironmentV38
from .v39_reward import tactical_rewards


class CombatEnvironmentV39(CombatEnvironmentV38):
    environment_version = '3.9'

    def _reset_metrics(self):
        super()._reset_metrics()
        self.episode_paper_rewards = np.zeros(self.team_size, dtype=float)
        self.episode_extended_rewards = np.zeros(self.team_size, dtype=float)

    def dense_combat_reward(self):
        diagnostics = tactical_rewards(self.red, self.blue, self.config['sensor'], self.config['reward'])
        return diagnostics['tactical'], diagnostics

    def _reward_diagnostics(self, info, paper, extended):
        info.update(paper_rewards=paper.copy(), extended_rewards=extended.copy(),
            paper_reward=paper.copy(), extended_reward=extended.copy(),
            episode_paper_reward_rewards=self.episode_paper_rewards.copy(),
            episode_extended_reward_rewards=self.episode_extended_rewards.copy(),
            episode_paper_reward_total=float(self.episode_paper_rewards.sum()),
            episode_safety_total=float(self.episode_reward_components['safe'].sum()),
            episode_extended_reward_total=float(self.episode_extended_rewards.sum()))

    def reset(self, seed=None):
        obs, info = super().reset(seed)
        zero = np.zeros(self.team_size, dtype=float)
        self._reward_diagnostics(info, zero, zero)
        return obs, info

    def step(self, red_actions, blue_actions=None):
        obs, _, terminated, truncated, info = super().step(red_actions, blue_actions)
        paper = sum((info[f'r{i}_rewards'] for i in range(1, 5)), np.zeros(self.team_size, dtype=float))
        extended = paper + info['safety_rewards']
        # Return the authoritative decomposition directly, retaining float64
        # until the algorithm's existing tensor conversion. This avoids repeated
        # float32 rounding obscuring exact episode-level decomposition.
        self.episode_paper_rewards += paper
        self.episode_extended_rewards += extended
        self._reward_diagnostics(info, paper, extended)
        info['local_rewards'] = extended.copy()
        return obs, extended, terminated, truncated, info
