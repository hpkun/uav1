"""Hypothesis only: replace R4 by continuous 3D alignment/threat, keep R1-R3."""
from .combat_v348 import CombatEnvironmentV348
from .v343_reward import continuous_tactical_rewards


class CombatEnvironmentV349(CombatEnvironmentV348):
    environment_version='3.49'

    def dense_combat_reward(self):
        diagnostic=continuous_tactical_rewards(self.red,self.blue,self.config['candidate_reward'])
        return diagnostic['tactical'],diagnostic

    def _info(self,*args,**kwargs):
        info=super()._info(*args,**kwargs)
        info.update(reward_semantics='experimental continuous 3D R4; unchanged v348 R1/R2/PBRS',
            candidate_reward_hypothesis=True,
            episode_candidate_alignment_offense_total=float(self.episode_reward_components['tactical_offense'].sum()),
            episode_candidate_alignment_threat_total=float(self.episode_reward_components['tactical_defense'].sum()))
        return info
