"""v3.3 reward/mechanisms with paper Blue decisions and forced boundary return."""
import numpy as np
from .combat_v33 import CombatEnvironmentV33
from .v35_policy import PaperTieredBluePolicy
from .math_utils import wrap_angle

class CombatEnvironmentV35(CombatEnvironmentV33):
    environment_version='3.5'

    def __init__(self,config):
        super().__init__(config)
        self.fixed_policy=PaperTieredBluePolicy(self.config['blue_policy'],self.config['action'],self.config['sensor'])

    def _reset_metrics(self):
        super()._reset_metrics()
        self.blue_returns=dict.fromkeys(('boundary','horizontal','ground','ceiling'),0)

    def reset(self,seed=None):
        result=super().reset(seed)
        self.fixed_policy.reset(seed)
        return result

    def _resolve_noncombat_losses(self):
        # The inherited Red precedence and counters remain exact. Blue no longer
        # participates in loss accounting; each live Blue is corrected before fire.
        exits=[];ground=[]
        for i,state in enumerate(self.red):
            if not state.alive:continue
            if state.altitude<=self.config['arena']['altitude_min']:
                state.alive=False;ground.append(i)
            elif np.hypot(state.x,state.y)>self.arena_radius or state.altitude>self.config['arena']['altitude_max']:
                state.alive=False;exits.append(i)
                if state.altitude>self.config['arena']['altitude_max']:
                    self.combat_counts['red']['ceiling_losses']+=1
        self.combat_counts['red']['boundary_exits']+=len(exits)
        self.combat_counts['red']['ground_losses']+=len(ground)
        for state in self.blue:
            if not state.alive:continue
            radius=np.hypot(state.x,state.y);returned=False
            if radius>self.arena_radius:
                scale=(self.arena_radius-1e-6)/radius
                state.x*=scale;state.y*=scale;state.psi=wrap_angle(state.psi+np.pi)
                self.blue_returns['horizontal']+=1;returned=True
            if state.altitude<=self.config['arena']['altitude_min']:
                state.z=-(self.config['arena']['altitude_min']+1e-6);state.theta=abs(state.theta)
                self.blue_returns['ground']+=1;returned=True
            elif state.altitude>self.config['arena']['altitude_max']:
                state.z=-(self.config['arena']['altitude_max']-1e-6);state.theta=-abs(state.theta)
                self.blue_returns['ceiling']+=1;returned=True
            if returned:self.blue_returns['boundary']+=1
        return exits,[],ground,[]

    def step(self,red_actions,blue_actions=None):
        automatic=blue_actions is None
        if automatic:blue_actions=self.fixed_policy.team_actions(self.blue,self.red)
        result=super().step(red_actions,blue_actions)
        result[4].update({f'blue_{name}_returns':value for name,value in self.blue_returns.items()},
            blue_boundary_return_episode=self.blue_returns['boundary']>0)
        if automatic:result[4].update(self.fixed_policy.diagnostics())
        return result
