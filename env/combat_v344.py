"""v3.43 rewards/motion/49D observations with deterministic continuous dwell."""
from .combat_v343 import CombatEnvironmentV343
from .v344_dwell import DwellEnvelope,DwellState,required_dwell_steps,EVENT_SEMANTICS


class CombatEnvironmentV344(CombatEnvironmentV343):
    environment_version='3.44'
    event_semantics=EVENT_SEMANTICS

    def build_weapon(self):
        return DwellEnvelope(**self.config['weapon'])

    def _reset_metrics(self):
        super()._reset_metrics()
        self.dwell_steps=required_dwell_steps(self.config['fire_control']['dwell_seconds'],self.dt)
        self.dwell={side:DwellState(self.team_size,self.dwell_steps) for side in ('red','blue')}

    def attempt_hit_for_pair(self, attacker, target):
        raise RuntimeError('probability hit path is forbidden in continuous dwell versions')

    def _update_fire_window_metrics(self, red_pairs, blue_pairs):
        # Old probabilistic fire metrics stay zero/None, not renamed dwell events.
        pass

    def _entry_attempts(self, attackers, targets, fire_states, side):
        eligible,distance=self.weapon.eligibility(attackers,targets)
        selected=self.blue_step_target_indices if side=='blue' else None
        return self.dwell[side].update(eligible,distance,self.steps,selected)

    def _resolve_combat(self, red_attempts, blue_attempts):
        # Both lists have already been built from the same pre-event living set.
        # Existing resolver deduplicates victims; existing R1 shares per victim.
        result=super()._resolve_combat(red_attempts,blue_attempts)
        for side,team,targets,kills in (('red',self.red,self.blue,result[0]),('blue',self.blue,self.red,result[1])):
            self.dwell[side].metrics['dwell_neutralizations']+=len(kills)
            self.dwell[side].invalidate(team,targets)
        return result

    def _info(self,rewards,components,actions,executed_phi,truncated,red_fire_pairs,blue_fire_pairs,
              red_step_attempts,blue_step_attempts,red_step_hits,blue_step_hits,red_step_kills,blue_step_kills):
        ended=bool(truncated or not self.red_alive_mask.any() or not self.blue_alive_mask.any())
        for side,team,targets in (('red',self.red,self.blue),('blue',self.blue,self.red)):
            self.dwell[side].invalidate(team,targets,episode_end=ended)
        info=super()._info(rewards,components,actions,executed_phi,truncated,0,0,0,0,0,0,red_step_kills,blue_step_kills)
        info.update(event_semantics=EVENT_SEMANTICS,dwell_steps=self.dwell_steps,
            dwell_seconds=self.config['fire_control']['dwell_seconds'],
            legacy_fire_metrics_applicable=False,ammo_cooldown_applicable=False,
            legacy_fire_metrics_policy='zero/None compatibility placeholders; not physical fire/hit counts')
        for side in ('red','blue'):
            state=self.dwell[side]
            info.update({f'{side}_{k}':v for k,v in state.diagnostics().items()})
            for name,value in (('targets',state.targets),('counts',state.counts),('eligible',state.eligible),
                               ('sampled_targets',state.sampled_targets),('sampled_counts',state.sampled_counts),
                               ('completed_this_step',state.completed_this_step)):
                info[f'{side}_dwell_{name}']=value.copy()
        return info

    def step(self, red_actions, blue_actions=None):
        result=super().step(red_actions,blue_actions)
        result[4]['firing_geometry_semantics']='3D dwell envelope; legacy fire counters not applicable'
        return result

    def reset(self, seed=None):
        obs,info=super().reset(seed)
        for side,team,targets in (('red',self.red,self.blue),('blue',self.blue,self.red)):
            info[f'initial_{side}_dwell_eligible_pairs']=int(self.weapon.eligibility(team,targets)[0].sum())
            info[f'initial_{side}_fire_eligible_pairs']=0
        return obs,info
