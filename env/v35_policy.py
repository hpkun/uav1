"""Paper-structured 3D Blue; global alive-Red information is intentional."""
import numpy as np
from .fixed_policy import NearestTargetPursuitPolicy
from .sensor import sensor_geometry

MODES=('DETECTED_PURSUIT','DETECTED_ESCAPE','NO_DETECTION_GUERRILLA','NO_DETECTION_CENTRIPETAL')

class PaperTieredBluePolicy(NearestTargetPursuitPolicy):
    def __init__(self,config,action_config,sensor_config):
        super().__init__(config,action_config)
        self.sensor_config=sensor_config
        self.reset()

    def reset(self,seed=None):
        # Domain-separated stream; never draws from initialization/hit RNG.
        self.rng=np.random.default_rng(np.random.SeedSequence(seed,spawn_key=(35,)))
        self.last_modes=[];self.last_targets=[]
        self.counts=dict.fromkeys(MODES,0)

    @staticmethod
    def cluster(targets):
        points=np.asarray([[s.x,s.y,s.z] for s in targets if s.alive],dtype=float)
        if not len(points):return None,0.
        center=points.mean(axis=0)
        return center,float(np.sqrt(np.mean(np.sum((points-center)**2,axis=1))))

    def toward_vector(self,own,vector):
        return self.action_toward(own,float(np.arctan2(vector[1],vector[0])),
            float(np.arctan2(-vector[2],np.hypot(vector[0],vector[1]))),self.config['desired_speed'])

    def team_actions(self,team,targets):
        visible,distances,_,_=sensor_geometry(team,targets,self.sensor_config)
        alive=np.flatnonzero([s.alive for s in targets]);center,spread=self.cluster(targets)
        friends=sum(s.alive for s in team)
        actions=[];modes=[];selected=[]
        for i,own in enumerate(team):
            position=np.array([own.x,own.y,own.z]);target_index=None
            if not own.alive or not len(alive):
                mode='DEAD' if not own.alive else 'NO_TARGETS_SAFE'
                action=np.zeros(3,dtype=np.float32)
            else:
                indices=np.flatnonzero(visible[i])
                if len(indices):
                    nearest=min(indices,key=lambda j:(distances[i,j],j))
                    if distances[i,nearest]<=self.config['close_distance'] and len(alive)>friends:
                        mode='DETECTED_ESCAPE';vector=position-center
                        if np.linalg.norm(vector)<=1e-12:
                            t=targets[nearest];vector=position-np.array([t.x,t.y,t.z])
                        action=(self.toward_vector(own,vector) if np.linalg.norm(vector)>1e-12 else
                            self.action_toward(own,own.psi+np.pi,0.,self.config['desired_speed']))
                    else:
                        mode='DETECTED_PURSUIT';target_index=int(nearest)
                elif spread>self.config['dispersion_threshold']:
                    mode='NO_DETECTION_GUERRILLA';target_index=int(self.rng.choice(alive))
                else:
                    mode='NO_DETECTION_CENTRIPETAL';action=self.toward_vector(own,center-position)
                if target_index is not None:
                    t=targets[target_index];action=self.toward_vector(own,np.array([t.x,t.y,t.z])-position)
                self.counts[mode]+=1
            actions.append(action);modes.append(mode);selected.append(target_index)
        self.last_modes=modes;self.last_targets=selected
        return np.stack(actions)

    def diagnostics(self):
        return dict(blue_policy_modes=list(self.last_modes),
            **{f'blue_{name}_agent_steps':self.counts[mode] for name,mode in zip(
                ('detected_pursuit','detected_escape','guerrilla','centripetal'),MODES)},
            **{f'blue_{name}_episode':self.counts[mode]>0 for name,mode in zip(
                ('escape','guerrilla','centripetal'),MODES[1:])})
