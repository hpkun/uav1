"""v3.5 Blue/base mechanisms with shared 4km radar, 2km ATA/HA weapon and tactical reward."""
import numpy as np
from .combat_v35 import CombatEnvironmentV35
from .v36_weapon import MADSACStyleFiniteAmmoWeapon
from .v36_reward import tactical_rewards

class CombatEnvironmentV36(CombatEnvironmentV35):
    environment_version='3.6'

    def build_weapon(self):return MADSACStyleFiniteAmmoWeapon(**self.config['weapon'])

    def _reset_metrics(self):
        super()._reset_metrics()
        for name in ('guide','tactical_offense','tactical_defense','tactical'):
            self.episode_reward_components[name]=np.zeros(5,dtype=float)

    def potential(self):
        zero=np.zeros(5,dtype=float)
        return zero,dict(distance=zero.copy(),angle=zero.copy())

    def dense_combat_reward(self):
        diagnostic=tactical_rewards(self.red,self.blue,self.config['sensor'],self.config['weapon'],self.config['reward'])
        return diagnostic['tactical'],diagnostic

    def advantage_reward(self,current,next_,dense):return dense

    def step(self,red_actions,blue_actions=None):
        result=super().step(red_actions,blue_actions)
        result[4]['tactical_total_rewards']=result[4].pop('tactical_rewards')
        result[4]['pbrs_enabled']=False
        return result
