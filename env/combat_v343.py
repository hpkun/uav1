"""v3.42 mechanics + terminal-safe PBRS/R4 + appended 49D observation."""
import numpy as np
from .combat_v342 import CombatEnvironmentV342
from .v343_reward import state_potential, continuous_tactical_rewards
from .v343_observation import build_team_observations_v343


class CombatEnvironmentV343(CombatEnvironmentV342):
    environment_version = '3.43'

    def _observations(self):
        return build_team_observations_v343(self.red,self.blue,self.config['observation'])

    def potential(self):
        return state_potential(self.red,self.blue,self.config['research_reward'])

    def dense_combat_reward(self):
        diagnostic = continuous_tactical_rewards(self.red,self.blue,self.config['research_reward'])
        return diagnostic['tactical'],diagnostic

    def shaping_next_potential(self, actual_phi_next, terminated, truncated):
        # This finite-horizon experiment treats truncation as episode end.
        return np.zeros_like(actual_phi_next) if terminated or truncated else actual_phi_next

    def advantage_reward(self, current, next_, dense):
        self._step_r3 = self.config['research_reward']['gamma']*next_-current
        return self._step_r3+dense

    def reward_aliases(self, components, combat, boundary, dense_diagnostics):
        # Called exactly once after advantage_reward, before all accumulators.
        # guide is a log alias; it is never added to dense a second time.
        dense_diagnostics['guide'] = self._step_r3.copy()
        dense_diagnostics['tactical'] = components['adv'].copy()
        return dict(r1=combat, r2=boundary, r3=self._step_r3.copy(),
                    r4=dense_diagnostics['tactical_offense']+dense_diagnostics['tactical_defense'])

    def _reward_diagnostics(self, info, paper, extended):
        # Keep the parent storage private, expose accurate research names only.
        info.update(research_rewards=extended.copy(), extended_rewards=extended.copy(),
            episode_research_reward_rewards=self.episode_extended_rewards.copy(),
            episode_research_reward_total=float(self.episode_extended_rewards.sum()),
            episode_extended_reward_rewards=self.episode_extended_rewards.copy(),
            episode_extended_reward_total=float(self.episode_extended_rewards.sum()),
            episode_safety_total=float(self.episode_reward_components['safe'].sum()),
            reward_semantics='candidate research PBRS/R4; not full paper replication')

    def step(self, red_actions, blue_actions=None):
        result = super().step(red_actions,blue_actions)
        info = result[4]
        info['pbrs_enabled'] = True
        for key in list(info):
            if key.startswith('madsac_r'):
                info.pop(key)
        return result

    def reset(self, seed=None):
        obs,info = super().reset(seed)
        info['pbrs_enabled'] = True
        return obs,info
