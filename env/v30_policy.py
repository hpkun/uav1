"""Deterministic sensor-limited LOS pursuit, center search and arena guard."""
import numpy as np
from .fixed_policy import NearestTargetPursuitPolicy
from .sensor import sensor_geometry


class SensorLimitedPursuitPolicy(NearestTargetPursuitPolicy):
    def __init__(self,config,action_config,sensor_config,scenario_config):
        super().__init__(config,action_config)
        self.sensor_config=sensor_config
        self.center_altitude=float(scenario_config['altitude_center'])

    def _action(self,own,targets,visible,distances):
        if not own.alive: return np.zeros(3,dtype=np.float32)
        guard=(np.hypot(own.x,own.y)>=self.config['guard_radius']
               or own.altitude<=self.config['guard_altitude_min']
               or own.altitude>=self.config['guard_altitude_max'])
        indices=np.flatnonzero(visible)
        if not guard and indices.size:
            j=min(indices,key=lambda i:(distances[i],i));target=targets[j]
            x,y,z=target.x,target.y,target.z
        else:
            x,y,z=0.,0.,-self.center_altitude
        dx,dy=x-own.x,y-own.y
        return self.action_toward(own,float(np.arctan2(dy,dx)),
            float(np.arctan2(own.z-z,np.hypot(dx,dy))),float(self.config['desired_speed']))

    def action(self,own,targets):
        visible,distances,_,_=sensor_geometry([own],targets,self.sensor_config)
        return self._action(own,targets,visible[0],distances[0])

    def team_actions(self,team,targets):
        visible,distances,_,_=sensor_geometry(team,targets,self.sensor_config)
        return np.stack([self._action(s,targets,visible[i],distances[i]) for i,s in enumerate(team)])
