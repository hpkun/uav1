"""v3.6 mechanisms with Gaussian hits and faithful, explicit R1–R4 aliases."""
from .combat_v36 import CombatEnvironmentV36
from .v37_weapon import MADSACGaussianFiniteAmmoWeapon
from .v36_geometry import combat_geometry

class CombatEnvironmentV37(CombatEnvironmentV36):
    environment_version='3.7'

    def build_weapon(self):return MADSACGaussianFiniteAmmoWeapon(**self.config['weapon'])

    def attempt_hit_for_pair(self,attacker,target):
        g=combat_geometry([attacker],[target])
        return self.weapon.attempt_hit(self.rng,g['distance'][0,0],g['ata'][0,0],g['ha'][0,0])

    def reward_aliases(self,components,combat,boundary,dense_diagnostics):
        return dict(r1=combat,r2=boundary,r3=dense_diagnostics['guide'],
                    r4=dense_diagnostics['tactical_offense']+dense_diagnostics['tactical_defense'])

    def step(self,red_actions,blue_actions=None):
        result=super().step(red_actions,blue_actions)
        info=result[4]
        # Individual combat excludes boundary; legacy event remains R1+R2.
        info['individual_event_rewards']=info['r1_rewards'].copy()
        info['boundary_rewards']=info['r2_rewards'].copy()
        info['tactical_rewards']=info['r4_rewards'].copy()
        info['safety_rewards']=info['safe_rewards'].copy()
        for name,source in (('individual_event','r1'),('boundary','r2'),('tactical','r4'),('safety','safe')):
            info[f'episode_{name}_rewards']=info[f'episode_{source}_rewards'].copy()
            info[f'episode_{name}_total']=info[f'episode_{source}_total']
        for i in range(1,5):info[f'madsac_r{i}_total']=info[f'episode_r{i}_total']
        return result
