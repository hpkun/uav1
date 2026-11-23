"""Joint-transition prioritized replay for MADDPG training only."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ReplayConfig:
    mode: str = 'uniform'
    prioritized_fraction: float = 0.5
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    epsilon: float = 1e-4

    def __post_init__(self):
        if self.mode not in ('uniform', 'critic_per_with_uniform'):
            raise ValueError('Unsupported replay mode')
        values=(self.prioritized_fraction,self.alpha,self.beta_start,self.beta_end,self.epsilon)
        if any(type(value) not in (int,float) or not math.isfinite(value) for value in values):
            raise ValueError('Replay parameters must be finite numbers')
        if not (0 < self.prioritized_fraction < 1 and 0 < self.alpha <= 1 and
                0 <= self.beta_start <= self.beta_end <= 1 and self.epsilon > 0):
            raise ValueError('Invalid prioritized replay parameters')

    @classmethod
    def from_mapping(cls, value):
        if value is None:
            return cls()
        if (not isinstance(value,dict) or 'mode' not in value or
                not set(value)<=set(cls.__dataclass_fields__)):
            raise ValueError('Invalid training.replay configuration')
        return cls(**value)

    def beta(self,step,total_steps):
        return self.beta_start+(self.beta_end-self.beta_start)*min(1,max(0,step/total_steps))


class PrioritizedReplayBuffer:
    """One priority per joint transition; separate event metadata for diagnostics."""
    event_names=('capture','first_breach','boundary')

    def __init__(self,capacity,seed,*,alpha,epsilon):
        if type(capacity) is not int or capacity<1 or not 0<alpha<=1 or epsilon<=0:
            raise ValueError('Invalid prioritized replay capacity or parameters')
        self.capacity=capacity
        self.alpha=alpha
        self.epsilon=epsilon
        self.rows=[]
        self.cursor=0
        self.rng=np.random.default_rng(seed)
        self.leaves=1<<(capacity-1).bit_length()
        self.sums=np.zeros(2*self.leaves,dtype=np.float64)
        self.maximums=np.zeros(2*self.leaves,dtype=np.float64)
        self.raw_priorities=np.zeros(capacity,dtype=np.float64)
        self.event_counts=np.zeros((capacity,3),dtype=np.uint8)
        self.buffer_event_counts=np.zeros(3,dtype=np.int64)
        self.buffer_ordinary_count=0

    def __len__(self):
        return len(self.rows)

    def _set_priority(self,index,priority):
        self.raw_priorities[index]=priority
        node=self.leaves+index
        self.sums[node]=priority**self.alpha
        self.maximums[node]=priority
        while node>1:
            node//=2
            self.sums[node]=self.sums[2*node]+self.sums[2*node+1]
            self.maximums[node]=max(self.maximums[2*node],self.maximums[2*node+1])

    def add(self,*,event_metadata,**row):
        if not isinstance(event_metadata,dict) or set(event_metadata)!=set(self.event_names):
            raise ValueError('Expected capture, first_breach and boundary counts')
        counts=[]
        for name in self.event_names:
            value=event_metadata[name]
            maximum=1 if name=='first_breach' else 5
            if type(value) not in (int,bool) or not 0<=value<=maximum:
                raise ValueError('Invalid replay event count')
            counts.append(value)
        copied={key:np.asarray(value,dtype=np.float32).copy() for key,value in row.items()}
        index=self.cursor
        if index<len(self.rows):
            self.rows[index]=copied
            self.buffer_event_counts-=self.event_counts[index]>0
            self.buffer_ordinary_count-=int(not self.event_counts[index].any())
        else:
            self.rows.append(copied)
        self.event_counts[index]=counts
        self.buffer_event_counts+=self.event_counts[index]>0
        self.buffer_ordinary_count+=int(not self.event_counts[index].any())
        priority=float(self.maximums[1]) if self.maximums[1]>0 else 1.0
        self._set_priority(index,priority)
        self.cursor=(index+1)%self.capacity

    def _stack(self,indices):
        return {key:np.stack([self.rows[int(i)][key] for i in indices]) for key in self.rows[0]}

    def _uniform(self,count):
        return self.rng.integers(0,len(self),size=count,dtype=np.int64)

    def _prioritized(self,count):
        total=self.sums[1]
        if not total>0 or not math.isfinite(total):
            raise FloatingPointError('Invalid replay priority total')
        indices=np.empty(count,dtype=np.int64)
        for k in range(count):
            value=(k+self.rng.random())*total/count
            node=1
            while node<self.leaves:
                left=2*node
                if value<self.sums[left]:
                    node=left
                else:
                    value-=self.sums[left]
                    node=left+1
            indices[k]=node-self.leaves
            if indices[k]>=len(self):
                raise AssertionError('Priority tree selected an empty slot')
        return indices

    def sample_update(self,batch_size,*,prioritized_fraction,beta):
        if len(self)<1 or type(batch_size) is not int or batch_size<2 or not 0<=beta<=1:
            raise ValueError('Invalid prioritized replay sample request')
        n_priority=round(batch_size*prioritized_fraction)
        n_shared=batch_size-n_priority
        if not 0<n_priority<batch_size:
            raise ValueError('Replay batch needs uniform and prioritized samples')
        shared=self._uniform(n_shared)
        priority=self._prioritized(n_priority)
        actor_extra=self._uniform(n_priority)
        critic_indices=np.concatenate((shared,priority))
        actor_indices=np.concatenate((shared,actor_extra))
        fraction=n_priority/batch_size
        per_probability=self.sums[self.leaves+critic_indices]/self.sums[1]
        probability=(1-fraction)/len(self)+fraction*per_probability
        weights=np.power(len(self)*probability,-beta)
        weights/=weights.max()
        flags=self.event_counts[critic_indices]>0
        buffer_fractions=self.buffer_event_counts/len(self)
        sampled_fractions=flags.mean(axis=0)
        stats=dict(per_beta=float(beta),per_is_weight_mean=float(weights.mean()),
                   per_is_weight_min=float(weights.min()),
                   per_is_ess=float(weights.sum()**2/(weights@weights)/batch_size),
                   per_priority_mean=float(self.raw_priorities[critic_indices].mean()),
                   per_priority_max=float(self.maximums[1]))
        for k,name in enumerate(self.event_names):
            stats[f'per_buffer_{name}_fraction']=float(buffer_fractions[k])
            stats[f'per_sample_{name}_fraction']=float(sampled_fractions[k])
        stats['per_buffer_ordinary_fraction']=self.buffer_ordinary_count/len(self)
        stats['per_sample_ordinary_fraction']=float(np.mean(~np.any(flags,axis=1)))
        return self._stack(critic_indices),self._stack(actor_indices),critic_indices,weights.astype(np.float32),stats

    def update_priorities(self,indices,td_errors):
        if len(indices)!=len(td_errors):
            raise ValueError('Priority index/error length mismatch')
        combined={}
        for index,error in zip(indices,td_errors):
            index=int(index)
            error=float(error)
            if not 0<=index<len(self) or not math.isfinite(error) or error<0:
                raise ValueError('Invalid replay priority update')
            combined[index]=max(combined.get(index,0.0),error+self.epsilon)
        for index,priority in combined.items():
            self._set_priority(index,priority)
