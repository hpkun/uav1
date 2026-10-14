"""Independent joint replay; lazy chunks avoid eagerly allocating over 2GB."""
from dataclasses import dataclass
import numpy as np


@dataclass
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    alive_masks: np.ndarray
    next_alive_masks: np.ndarray


class JointReplayBuffer:
    shapes=dict(observations=(5,56),actions=(5,3),rewards=(5,),next_observations=(5,56),
                dones=(),alive_masks=(5,),next_alive_masks=(5,))

    def __init__(self,capacity=1000000,seed=0,chunk_size=4096,observation_dim=56,num_agents=5,action_dim=3):
        if capacity<=0 or chunk_size<=0:raise ValueError('positive replay/chunk size required')
        self.capacity,self.chunk_size=int(capacity),int(chunk_size)
        self.shapes=dict(observations=(num_agents,observation_dim),actions=(num_agents,action_dim),rewards=(num_agents,),
            next_observations=(num_agents,observation_dim),dones=(),alive_masks=(num_agents,),next_alive_masks=(num_agents,))
        self.size=self.position=0;self.chunks={};self.rng=np.random.default_rng(seed)

    def __len__(self):return self.size

    def add_batch(self,observations,actions,rewards,next_observations,dones,alive_masks,next_alive_masks):
        batch=ReplayBatch(observations,actions,rewards,next_observations,dones,alive_masks,next_alive_masks)
        arrays={k:np.asarray(v,dtype=np.float32) for k,v in vars(batch).items()}
        n=len(arrays['observations'])
        for key,shape in self.shapes.items():
            if arrays[key].shape!=(n,*shape) or not np.isfinite(arrays[key]).all():
                raise ValueError(f'invalid replay {key}')
        offset=0
        while offset<n:
            block,local=divmod(self.position,self.chunk_size)
            length=min(self.chunk_size,self.capacity-block*self.chunk_size)
            if block not in self.chunks:
                self.chunks[block]={k:np.empty((length,*shape),dtype=np.float32) for k,shape in self.shapes.items()}
            count=min(n-offset,length-local)
            for key in self.shapes:self.chunks[block][key][local:local+count]=arrays[key][offset:offset+count]
            offset+=count;self.position=(self.position+count)%self.capacity;self.size=min(self.capacity,self.size+count)

    def sample(self,batch_size):
        if not 0<batch_size<=self.size:raise ValueError('replay has insufficient samples')
        indices=self.rng.choice(self.size,batch_size,replace=False)
        data={k:np.empty((batch_size,*s),dtype=np.float32) for k,s in self.shapes.items()}
        blocks=indices//self.chunk_size
        for block in np.unique(blocks):
            selected=np.flatnonzero(blocks==block);local=indices[selected]%self.chunk_size
            for key in self.shapes:data[key][selected]=self.chunks[int(block)][key][local]
        return ReplayBatch(**data)
