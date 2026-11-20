"""Shared deterministic defender policy with global and local centralized critics."""
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class DualCriticConfig:
    hidden_dim: int = 256
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_capacity: int = 1_000_000
    learning_starts: int = 1024
    update_interval_steps: int = 1
    updates_per_interval: int = 1
    exploration_std_mps2: float = 0.2
    target_noise_std_mps2: float = 0.4
    target_noise_clip_mps2: float = 1.0
    policy_delay: int = 2

    def __post_init__(self):
        if min(self.hidden_dim,self.batch_size,self.buffer_capacity,self.learning_starts,
               self.update_interval_steps,self.updates_per_interval,self.policy_delay)<1:
            raise ValueError('Invalid dual-critic count')
        if not 0<self.gamma<=1 or not 0<self.tau<=1:
            raise ValueError('Invalid discount or target rate')
        if min(self.actor_learning_rate,self.critic_learning_rate)<=0 or min(
                self.exploration_std_mps2,self.target_noise_std_mps2,self.target_noise_clip_mps2)<0:
            raise ValueError('Invalid dual-critic optimization parameter')
        if self.buffer_capacity<max(self.batch_size,self.learning_starts):
            raise ValueError('Replay capacity below warmup')

    def warmup(self,_episode_steps):
        return max(self.batch_size,self.learning_starts)


class Actor(nn.Module):
    def __init__(self,obs_dim,hidden,radius):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(obs_dim,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,2))
        self.radius=radius

    def forward(self,obs):
        raw=self.net(obs)
        return self.radius*raw/torch.sqrt(1+raw.square().sum(-1,keepdim=True))


class Critic(nn.Module):
    def __init__(self,state_dim,hidden):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(state_dim+10,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))

    def forward(self,state,actions):
        return self.net(torch.cat((state,actions.reshape(len(state),10)),-1)).squeeze(-1)


class DualCritic:
    algorithm='dual_critic'
    schema='interception_dual_critic_1'

    def __init__(self,config,device,seed,*,environment_config,environment_contract):
        if environment_contract.get('version')!='interception_1' or environment_contract.get('observation_dim')!=36:
            raise ValueError('Dual critic requires Interception/1')
        self.config=config
        self.device=torch.device(device)
        self.seed=seed
        self.environment_config=deepcopy(environment_config)
        self.environment_contract=deepcopy(environment_contract)
        self.rng=np.random.default_rng(seed)
        torch.manual_seed(seed)
        radius=environment_contract['max_acceleration']
        self.actor=Actor(36,config.hidden_dim,radius).to(self.device)
        self.critics=nn.ModuleList([Critic(180,config.hidden_dim) for _ in range(2)]).to(self.device)
        self.target_actor=deepcopy(self.actor).requires_grad_(False).eval()
        self.target_critics=deepcopy(self.critics).requires_grad_(False).eval()
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=config.actor_learning_rate)
        self.critic_optimizer=torch.optim.Adam(self.critics.parameters(),lr=config.critic_learning_rate)
        self.critic_updates=0
        self.actor_updates=0

    def _joint(self,observations,target=False):
        actor=self.target_actor if target else self.actor
        return actor(observations)

    @torch.no_grad()
    def policy_actions(self,observations,active_masks,*,deterministic=True):
        obs=torch.as_tensor(observations,dtype=torch.float32,device=self.device)
        actions=self.actor(obs).cpu().numpy()
        if not deterministic:
            actions+=self.rng.normal(0,self.config.exploration_std_mps2,actions.shape)
            radius=self.environment_contract['max_acceleration']
            norms=np.linalg.norm(actions,axis=-1,keepdims=True)
            actions*=np.minimum(1,radius/np.maximum(norms,1e-12))
        return (actions*np.asarray(active_masks)[...,None]).astype(np.float32)

    def update(self,batch):
        b={key:torch.as_tensor(value,dtype=torch.float32,device=self.device) for key,value in batch.items()}
        size=len(b['states'])
        if any(b[key].shape!=shape for key,shape in dict(
                observations=(size,5,36),states=(size,180),actions=(size,5,2),
                global_rewards=(size,),local_rewards=(size,),next_observations=(size,5,36),
                next_states=(size,180),dones=(size,),active_masks=(size,5),next_active_masks=(size,5)).items()):
            raise ValueError('Invalid dual-critic replay batch')
        with torch.no_grad():
            target_actions=self.target_actor(b['next_observations'])
            noise=torch.randn_like(target_actions)*self.config.target_noise_std_mps2
            noise=noise.clamp(-self.config.target_noise_clip_mps2,self.config.target_noise_clip_mps2)
            target_actions=target_actions+noise
            radius=self.environment_contract['max_acceleration']
            target_actions=target_actions*torch.clamp(radius/target_actions.norm(dim=-1,keepdim=True).clamp_min(1e-12),max=1)
            target_actions*=b['next_active_masks'][...,None]
            targets=[reward+self.config.gamma*(1-b['dones'])*critic(b['next_states'],target_actions)
                     for reward,critic in zip((b['global_rewards'],b['local_rewards']),self.target_critics)]
        predictions=[critic(b['states'],b['actions']) for critic in self.critics]
        loss=sum((prediction-target).square().mean() for prediction,target in zip(predictions,targets))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optimizer.step()
        self.critic_updates+=1
        actor_loss=None
        if self.critic_updates%self.config.policy_delay==0:
            joint=self.actor(b['observations'])*b['active_masks'][...,None]
            actor_loss=-(self.critics[0](b['states'],joint)+self.critics[1](b['states'],joint)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_updates+=1
            with torch.no_grad():
                for source,target in ((self.actor,self.target_actor),(self.critics,self.target_critics)):
                    for online,slow in zip(source.parameters(),target.parameters()):
                        slow.mul_(1-self.config.tau).add_(online,alpha=self.config.tau)
        return dict(critic_loss=float(loss.detach()),actor_loss=None if actor_loss is None else float(actor_loss.detach()),
                    critic_updates=self.critic_updates,actor_updates=self.actor_updates)

    def save(self,path,*,global_step,environment_config):
        if environment_config!=self.environment_config:raise ValueError('Environment config mismatch')
        torch.save(dict(schema=self.schema,global_step=global_step,environment_config=self.environment_config,
                        environment_contract=self.environment_contract,config=asdict(self.config),
                        actor=self.actor.state_dict(),critics=self.critics.state_dict(),
                        target_actor=self.target_actor.state_dict(),target_critics=self.target_critics.state_dict(),
                        actor_optimizer=self.actor_optimizer.state_dict(),critic_optimizer=self.critic_optimizer.state_dict(),
                        critic_updates=self.critic_updates,actor_updates=self.actor_updates),Path(path))

    def load(self,path,*,load_optimizers=True):
        data=torch.load(path,map_location=self.device,weights_only=False)
        if any(data.get(key)!=expected for key,expected in dict(
                schema=self.schema,environment_config=self.environment_config,
                environment_contract=self.environment_contract,config=asdict(self.config)).items()):
            raise ValueError('Dual-critic checkpoint contract mismatch')
        for name in ('actor','critics','target_actor','target_critics'):
            getattr(self,name).load_state_dict(data[name],strict=True)
        if load_optimizers:
            self.actor_optimizer.load_state_dict(data['actor_optimizer'])
            self.critic_optimizer.load_state_dict(data['critic_optimizer'])
            self.critic_updates=data['critic_updates']
            self.actor_updates=data['actor_updates']
        return data
