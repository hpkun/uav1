"""v3.2 physics/PBRS with team casualty externality and stronger outcome reward."""
import numpy as np
from .combat_v32 import CombatEnvironmentV32

class CombatEnvironmentV33(CombatEnvironmentV32):
    environment_version = '3.3'

    def _reset_metrics(self):
        super()._reset_metrics()
        self.episode_reward_components.update({name:np.zeros(self.team_size,dtype=float)
            for name in ('individual_event','team_casualty')})

    def event_reward_extension(self, individual_event, red_alive_before, red_exits, red_ground, blue_kills):
        lost=set(red_exits)|set(red_ground)|set(blue_kills)
        new_losses=sum(bool(red_alive_before[i]) for i in lost)
        casualty=self.config['reward']['team_casualty_penalty']*new_losses*red_alive_before.astype(float)
        return individual_event+casualty, dict(individual_event=individual_event,team_casualty=casualty)
