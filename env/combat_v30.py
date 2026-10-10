"""Isolated partial-information finite-ammunition combat family."""
from dataclasses import dataclass, field
import numpy as np
from .combat_env import MultiUAVCombatEnv
from .config import load_config,validate_config,aircraft_spec,environment_dimensions
from .dynamics import PointMassDynamics
from .integrator import RK4Integrator
from .v30_policy import SensorLimitedPursuitPolicy
from .v30_weapon import FiniteAmmoWeapon
from .v30_observation import build_v30_observations
from .v30_reward import potentials,potential_shaping,safety_rewards


@dataclass
class StrictPairEntryState:
    previous_eligible: np.ndarray = field(default_factory=lambda:np.zeros((5,5),dtype=bool))


class CombatEnvironmentV30(MultiUAVCombatEnv):
    environment_version = '3.0'
    def __init__(self,config):
        self.config=validate_config(config) if isinstance(config,dict) else load_config(config)
        if str(self.config['environment_version'])!=self.environment_version:
            raise ValueError(f'v3.0 family requires version {self.environment_version}')
        self.observation_dim,self.action_dim,self.team_size=environment_dimensions(self.config)
        self.spec=aircraft_spec(self.config)
        self.dt=float(self.config['simulation']['dt']);self.max_steps=int(self.config['simulation']['max_steps'])
        self.arena_radius=float(self.config['arena']['radius'])
        self.dynamics=PointMassDynamics();self.integrator=RK4Integrator(self.dt)
        self.fixed_policy=SensorLimitedPursuitPolicy(self.config['blue_policy'],self.config['action'],self.config['sensor'],self.config['scenario'])
        self.weapon=self.build_weapon()
        self.red_ammo=np.full(5,self.weapon.ammo_per_aircraft,dtype=np.int64)
        self.blue_ammo=self.red_ammo.copy()
        self.red=[];self.blue=[];self.rng=np.random.default_rng();self.steps=0
        self.red_last_executed_phi=np.zeros(5,dtype=np.float32);self.blue_last_executed_phi=self.red_last_executed_phi.copy()
        self.red_fire_states=self._new_fire_states();self.blue_fire_states=self._new_fire_states()
        self._reset_metrics()

    def _new_fire_states(self): return StrictPairEntryState()

    def build_weapon(self):
        return FiniteAmmoWeapon(**self.config['weapon'])

    def attempt_hit_for_pair(self,attacker,target):
        """Historical sampling order and RNG consumption remain unchanged."""
        return self.weapon.attempt_hit(self.rng)

    def reward_aliases(self,components,combat,boundary,dense_diagnostics):
        return dict(zip(('r1','r2','r3','r4'),components.values()))

    def dense_combat_reward(self):
        """No arithmetic or diagnostic changes in historical environments."""
        return None, {}

    def advantage_reward(self,current,next_,dense):
        return potential_shaping(current,next_,self.config['reward'])

    def _reset_metrics(self):
        super()._reset_metrics()
        for side in ('red','blue'):
            self.combat_counts[side].update(ceiling_losses=0,ammo_used=0)
        self.episode_reward_components.update({n:np.zeros(5,dtype=float) for n in ('event','outcome','adv','safe')})

    def reset(self,seed=None):
        self.red_ammo.fill(self.weapon.ammo_per_aircraft);self.blue_ammo.fill(self.weapon.ammo_per_aircraft)
        return super().reset(seed)

    def _observations(self):
        return build_v30_observations(self.red,self.blue,self.config['observation'],self.config['sensor'],
            self.red_ammo,self.weapon.ammo_per_aircraft,self.red_last_executed_phi)

    def potential(self):
        return potentials(self.red,self.blue,self.config['sensor'],self.config['weapon'],self.config['reward'])

    def shaping_next_potential(self, actual_phi_next, terminated, truncated):
        """Historical v3.0/v3.1 semantics, including terminal transitions."""
        return actual_phi_next

    def event_reward_extension(self, individual_event, red_alive_before, red_exits, red_ground, blue_kills):
        """Identity extension preserves old rewards and complete info byte-for-byte."""
        return individual_event, {}

    def _resolve_noncombat_losses(self):
        lists=[[],[],[],[]]
        for side_index,(side,states) in enumerate((('red',self.red),('blue',self.blue))):
            for i,state in enumerate(states):
                if not state.alive: continue
                # Ground is a separate cause; a ceiling loss is also a boundary loss.
                if state.altitude<=self.config['arena']['altitude_min']:
                    state.alive=False;lists[side_index+2].append(i)
                elif np.hypot(state.x,state.y)>self.arena_radius or state.altitude>self.config['arena']['altitude_max']:
                    state.alive=False;lists[side_index].append(i)
                    if state.altitude>self.config['arena']['altitude_max']:
                        self.combat_counts[side]['ceiling_losses']+=1
            self.combat_counts[side]['boundary_exits']+=len(lists[side_index])
            self.combat_counts[side]['ground_losses']+=len(lists[side_index+2])
        return tuple(lists)

    def _in_fire_window(self,attacker,target,ammo=1):
        return self.weapon.qualifies(attacker,target,ammo)

    def _entry_attempts(self,attackers,targets,fire_states,side):
        ammo=getattr(self,f'{side}_ammo')
        current,distance=self.weapon.eligibility(attackers,targets,ammo)
        entries=current & ~fire_states.previous_eligible
        # Consume every simultaneous entry, including pairs not selected to fire.
        fire_states.previous_eligible=current.copy()
        attempts=[]
        for i in range(5):
            indices=np.flatnonzero(entries[i])
            if indices.size and ammo[i]>0:
                j=int(min(indices,key=lambda j:(distance[i,j],j)))
                ammo[i]-=1
                attempts.append((i,j,self.attempt_hit_for_pair(attackers[i],targets[j])))
        self.combat_counts[side]['ammo_used']+=len(attempts)
        self.combat_counts[side]['fire_attempts']+=len(attempts)
        hits=sum(hit for _,_,hit in attempts);self.combat_counts[side]['weapon_hits']+=hits
        for key,yes in (('attempt',bool(attempts)),('hit',bool(hits))):
            if yes and self.first_steps[side][key] is None:self.first_steps[side][key]=self.steps
        return attempts

    def _outcome(self,truncated):
        red,blue=int(self.red_alive_mask.sum()),int(self.blue_alive_mask.sum())
        if red==blue==0:return False,False,True,'draw_mutual_destruction'
        if blue==0:return True,False,False,'red_win_elimination'
        if red==0:return False,True,False,'blue_win_elimination'
        if truncated:
            if red>blue:return True,False,False,'red_win_timeout_survivors'
            if red<blue:return False,True,False,'blue_win_timeout_survivors'
            return False,False,True,'draw_timeout_equal_survivors'
        return False,False,False,'ongoing'

    def step(self,red_actions,blue_actions=None):
        red_alive_before=self.red_alive_mask.copy()
        before,diag_before=self.potential()
        if blue_actions is None:blue_actions=self.fixed_policy.team_actions(self.blue,self.red)
        arrays=[]
        for name,actions,mask in (('red',red_actions,self.red_alive_mask),('blue',blue_actions,self.blue_alive_mask)):
            actions=np.asarray(actions,dtype=np.float32)
            if actions.shape!=(5,3) or not np.isfinite(actions).all():raise ValueError(f'{name}_actions must be finite shape (5,3)')
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
        outcome=np.full(5,rcfg['win_reward'] if win else rcfg['lose_penalty'] if loss else rcfg['draw_reward'] if draw else 0.,dtype=float)
        after,diag_after=self.potential()
        shaping_after=self.shaping_next_potential(after,terminated,truncated)
        adv=self.advantage_reward(before,shaping_after,dense)
        safe=safety_rewards(self.red,self.blue,rcfg)
        components=dict(event=event,outcome=outcome,adv=adv,safe=safe)
        rewards=sum(components.values(),np.zeros(5,dtype=float)).astype(np.float32)
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
