"""Frozen v3.0 core transition order with agent-size allocations only generalized."""
import numpy as np
from .v30_reward import safety_rewards

def step_core(self,red_actions,blue_actions=None):
    red_alive_before=self.red_alive_mask.copy()
    before,diag_before=self.potential()
    if blue_actions is None:blue_actions=self.fixed_policy.team_actions(self.blue,self.red)
    arrays=[]
    for name,actions,mask in (('red',red_actions,self.red_alive_mask),('blue',blue_actions,self.blue_alive_mask)):
        actions=np.asarray(actions,dtype=np.float32)
        if actions.shape!=(self.team_size,3) or not np.isfinite(actions).all():raise ValueError(f'{name}_actions must be finite shape ({self.team_size},3)')
        arrays.append(np.clip(actions,-1,1)*mask[:,None])
    red_action,blue_action=arrays
    self.red_last_executed_phi=self._advance(self.red,red_action)
    self.blue_last_executed_phi=self._advance(self.blue,blue_action);self.steps+=1
    red_exits,_,red_ground,_=self._resolve_noncombat_losses()
    dense,dense_diagnostics=self.dense_combat_reward()
    red_pairs=int(self.weapon.eligibility(self.red,self.blue,self.red_ammo)[0].sum())
    blue_pairs=int(self.weapon.eligibility(self.blue,self.red,self.blue_ammo)[0].sum())
    self._update_fire_window_metrics(red_pairs,blue_pairs)
    red_attempts=self._entry_attempts(self.red,self.blue,self.red_fire_states,'red')
    blue_attempts=self._entry_attempts(self.blue,self.red,self.blue_fire_states,'blue')
    red_kills,blue_kills=self._resolve_combat(red_attempts,blue_attempts)
    combat,boundary=self._event_reward_components(red_exits,red_ground,red_kills,blue_kills)
    individual_event=combat+boundary
    event,event_diagnostics=self.event_reward_extension(individual_event,red_alive_before,red_exits,red_ground,blue_kills)
    terminated=not self.red_alive_mask.any() or not self.blue_alive_mask.any()
    truncated=not terminated and self.steps>=self.max_steps
    win,loss,draw,_=self._outcome(truncated)
    rcfg=self.config['reward']
    outcome=np.full(self.team_size,rcfg['win_reward'] if win else rcfg['lose_penalty'] if loss else rcfg['draw_reward'] if draw else 0.,dtype=float)
    after,diag_after=self.potential()
    shaping_after=self.shaping_next_potential(after,terminated,truncated)
    adv=self.advantage_reward(before,shaping_after,dense)
    safe=safety_rewards(self.red,self.blue,rcfg)
    components=dict(event=event,outcome=outcome,adv=adv,safe=safe)
    rewards=sum(components.values(),np.zeros(self.team_size,dtype=float)).astype(np.float32)
    # Four aliases keep existing evaluator infrastructure readable; the new
    # explicit fields are authoritative and are not legacy state rewards.
    aliases=self.reward_aliases(components,combat,boundary,dense_diagnostics)
    for name,value in {**components,**aliases}.items():self.episode_reward_components[name]+=value
    for name,value in event_diagnostics.items():self.episode_reward_components[name]+=value
    for name,value in dense_diagnostics.items():self.episode_reward_components[name]+=value
    info=self._info(rewards,aliases,red_action,self.red_last_executed_phi,truncated,red_pairs,blue_pairs,
        len(red_attempts),len(blue_attempts),sum(x[2] for x in red_attempts),sum(x[2] for x in blue_attempts),len(red_kills),len(blue_kills))
    info.update(timeout=bool(truncated),red_ammo=self.red_ammo.copy(),blue_ammo=self.blue_ammo.copy(),
                executed_blue_actions=blue_action.copy(),phi_current=before.copy(),phi_next=after.copy(),
                phi_next_actual=after.copy(),phi_next_for_shaping=shaping_after.copy())
    for name,value in components.items():info[f'{name}_rewards']=value.copy()
    for name,value in event_diagnostics.items():info[f'{name}_rewards']=value.copy()
    for name,value in dense_diagnostics.items():info[f'{name}_rewards']=value.copy()
    for label,phi,diagnostic in (('current',before,diag_before),('next',after,diag_after)):
        info[f'mean_potential_{label}']=float(phi.mean())
        for name in ('distance','angle'):info[f'mean_{name}_component_{label}']=float(diagnostic[name].mean())
    return self._observations(),rewards,bool(terminated),bool(truncated),info
