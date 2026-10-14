"""v3.29 mechanics with real four-aircraft state and a 45D observation."""
import numpy as np
from .combat_v329 import CombatEnvironmentV329
from .combat_v31 import CombatEnvironmentV31
from .combat_env import MultiUAVCombatEnv
from .config import validate_config,load_config,aircraft_spec,environment_dimensions
from .dynamics import PointMassDynamics
from .integrator import RK4Integrator
from .v310_policy import PaperNearestTargetBluePolicy
from .v330_observation import build_team_observations_v330
from .v330_transition import step_core
from .sensor import sensor_geometry


class CombatEnvironmentV330(CombatEnvironmentV329):
    environment_version='3.30'

    def __init__(self,config):
        # Bypass the old v3.11 constructor's five-element allocations entirely.
        self.config=validate_config(config) if isinstance(config,dict) else load_config(config)
        self.observation_dim,self.action_dim,self.team_size=environment_dimensions(self.config)
        self.spec=aircraft_spec(self.config)
        self.dt=float(self.config['simulation']['dt']);self.max_steps=int(self.config['simulation']['max_steps'])
        self.arena_radius=float(self.config['arena']['radius'])
        self.dynamics=PointMassDynamics();self.integrator=RK4Integrator(self.dt)
        self.fixed_policy=PaperNearestTargetBluePolicy(self.config['blue_policy'],self.config['action'])
        self.weapon=self.build_weapon()
        self.red_ammo=np.full(self.team_size,self.weapon.ammo_per_aircraft,dtype=np.int64)
        self.blue_ammo=self.red_ammo.copy()
        self.red_fire_cooldown_steps=np.zeros(self.team_size,dtype=np.int64)
        self.blue_fire_cooldown_steps=self.red_fire_cooldown_steps.copy()
        self.cooldown_steps=round(self.config['fire_control']['cooldown_seconds']/self.dt)
        self.blue_step_target_indices=np.full(self.team_size,-1,dtype=np.int64)
        self.red_last_executed_phi=np.zeros(self.team_size,dtype=np.float32)
        self.blue_last_executed_phi=self.red_last_executed_phi.copy()
        self.red_fire_states=self.blue_fire_states=None
        self.red=[];self.blue=[];self.steps=0;self.rng=np.random.default_rng()
        self.controller_diagnostics={};self.firing_geometry_diagnostics={}
        self._reset_metrics()

    def _reset_metrics(self):
        MultiUAVCombatEnv._reset_metrics(self)
        for side in ('red','blue'):self.combat_counts[side].update(ceiling_losses=0,ammo_used=0)
        for name in ('event','outcome','adv','safe','individual_event','team_casualty',
                     'guide','tactical_offense','tactical_defense','tactical'):
            self.episode_reward_components[name]=np.zeros(self.team_size,dtype=float)
        self.blue_returns=dict.fromkeys(('boundary','horizontal','ground','ceiling'),0)
        self.episode_paper_rewards=np.zeros(self.team_size,dtype=float)
        self.episode_extended_rewards=np.zeros(self.team_size,dtype=float)

    def potential(self):
        zero=np.zeros(self.team_size,dtype=float)
        return zero,dict(distance=zero.copy(),angle=zero.copy())

    def _observations(self):
        return build_team_observations_v330(self.red,self.blue,self.config['observation'])

    def reset(self,seed=None):
        self.controller_diagnostics={};self.firing_geometry_diagnostics={}
        obs,info=CombatEnvironmentV31.reset(self,seed)
        self.fixed_policy.reset(seed)
        self.red_fire_cooldown_steps.fill(0);self.blue_fire_cooldown_steps.fill(0)
        self.blue_step_target_indices.fill(-1)
        alpha=info['radial_angle'];u=np.array([np.cos(alpha),np.sin(alpha)])
        radius=self.config['scenario']['center_radius']
        distance=np.linalg.norm(np.array([s.as_array()[:3] for s in self.red])[:,None]-
                                np.array([s.as_array()[:3] for s in self.blue])[None],axis=-1)
        info.update(red_nominal_anchor=-radius*u,blue_nominal_anchor=radius*u,
            nominal_anchor_separation=2*radius,initialization_protocol='random_diameter_compact_4v4',
            minimum_inter_team_distance=float(distance.min()),mean_inter_team_distance=float(distance.mean()),
            maximum_inter_team_distance=float(distance.max()),
            initial_red_visible_pairs=int(sensor_geometry(self.red,self.blue,self.config['sensor'])[0].sum()),
            initial_blue_visible_pairs=int(sensor_geometry(self.blue,self.red,self.config['sensor'])[0].sum()),
            initial_red_fire_eligible_pairs=int(self.weapon.eligibility(self.red,self.blue,self.red_ammo)[0].sum()),
            initial_blue_fire_eligible_pairs=int(self.weapon.eligibility(self.blue,self.red,self.blue_ammo)[0].sum()))
        zero=np.zeros(self.team_size,dtype=float);self._reward_diagnostics(info,zero,zero)
        return obs,info

    def step(self,red_actions,blue_actions=None):
        # Same wrapper order as v3.26 -> v3.9/37/36/35 -> frozen v3.0 core.
        for i,own in enumerate(self.blue):
            target=self.fixed_policy.nearest_target_index(own,self.red) if own.alive else None
            self.blue_step_target_indices[i]=-1 if target is None else target
        automatic=blue_actions is None
        if automatic:blue_actions=self.fixed_policy.team_actions(self.blue,self.red)
        obs,_,terminated,truncated,info=step_core(self,red_actions,blue_actions)
        info.update({f'blue_{name}_returns':v for name,v in self.blue_returns.items()},
                    blue_boundary_return_episode=self.blue_returns['boundary']>0)
        if automatic:info.update(self.fixed_policy.diagnostics())
        info['tactical_total_rewards']=info.pop('tactical_rewards');info['pbrs_enabled']=False
        info['individual_event_rewards']=info['r1_rewards'].copy()
        info['boundary_rewards']=info['r2_rewards'].copy()
        info['tactical_rewards']=info['r4_rewards'].copy()
        info['safety_rewards']=info['safe_rewards'].copy()
        for name,source in (('individual_event','r1'),('boundary','r2'),('tactical','r4'),('safety','safe')):
            info[f'episode_{name}_rewards']=info[f'episode_{source}_rewards'].copy()
            info[f'episode_{name}_total']=info[f'episode_{source}_total']
        for i in range(1,5):info[f'madsac_r{i}_total']=info[f'episode_r{i}_total']
        paper=sum((info[f'r{i}_rewards'] for i in range(1,5)),np.zeros(self.team_size,dtype=float))
        extended=paper+info['safety_rewards']
        self.episode_paper_rewards+=paper;self.episode_extended_rewards+=extended
        self._reward_diagnostics(info,paper,extended);info['local_rewards']=extended.copy()
        info['blue_pursuit_target_indices']=self.blue_step_target_indices.copy()
        info['controller_diagnostics']={side:dict(row) for side,row in self.controller_diagnostics.items()}
        info['red_firing_geometry_agent_steps']=dict(self.firing_geometry_diagnostics)
        return obs,extended,terminated,truncated,info
