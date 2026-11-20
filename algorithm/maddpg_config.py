"""Independent deterministic multi-agent DDPG configuration."""
from dataclasses import dataclass, fields
import math


@dataclass
class MADDPGConfig:
    hidden_dim:int=128
    actor_learning_rate:float=.0003
    critic_learning_rate:float=.001
    gamma:float=.99
    tau:float=.005
    batch_size:int=512
    buffer_capacity:int=200000
    learning_starts:int=2048
    update_interval_steps:int=100
    updates_per_interval:int=4
    exploration_std_mps2:float=1.5
    actor_max_grad_norm:float=.5
    critic_max_grad_norm:float=10.

    def __post_init__(self):
        for name in ('hidden_dim','batch_size','buffer_capacity','learning_starts','update_interval_steps','updates_per_interval'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError(name+' must be a positive integer')
        for field in fields(self):
            if field.type is float and (type(getattr(self,field.name)) not in (float,int) or
                    not math.isfinite(getattr(self,field.name))):raise ValueError(field.name+' must be finite')
        if not 0<=self.gamma<=1 or not 0<self.tau<=1:raise ValueError('Invalid discount/target update')
        if min(self.actor_learning_rate,self.critic_learning_rate,self.actor_max_grad_norm,self.critic_max_grad_norm)<=0:
            raise ValueError('Invalid learning rate or gradient bound')
        if self.exploration_std_mps2<0 or self.buffer_capacity<max(self.batch_size,self.learning_starts):
            raise ValueError('Invalid exploration or replay capacity')

    def warmup(self,episode_steps):
        return max(self.batch_size,self.learning_starts)
