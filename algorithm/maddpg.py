"""Five local deterministic defenders with centralized joint critics."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from .maddpg_config import MADDPGConfig
from .state import (validate_model_state,validate_optimizer_state,validate_counter,validate_torch_rng,
    validate_cuda_rng,validate_numpy_rng,validate_python_rng,validate_generator_rng,loading_transaction)


class DiskActor(nn.Module):
    def __init__(self,observation_dim,hidden_dim,radius):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(observation_dim,hidden_dim),nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,2))
        self.radius=radius

    def forward(self,observation):
        raw=self.net(observation)
        return self.radius*raw/torch.sqrt(1+raw.square().sum(-1,keepdim=True))


class JointCritic(nn.Module):
    def __init__(self,state_dim,hidden_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(state_dim+10,hidden_dim),nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,1))

    def forward(self,state,joint_actions):
        return self.net(torch.cat((state,joint_actions.reshape(len(state),10)),-1)).squeeze(-1)


class MADDPG:
    algorithm='maddpg'
    schema='interception_maddpg_2'

    def __init__(self,config=None,device='cpu',seed=0,*,environment_config,environment_contract):
        self.config=config or MADDPGConfig()
        if not isinstance(self.config,MADDPGConfig):raise ValueError('MADDPG requires MADDPGConfig')
        self.environment_config=deepcopy(environment_config)
        self.environment_spec=self.validate_environment_spec(environment_contract)
        self.device=torch.device(device);self.seed=seed
        self.rng=np.random.default_rng(seed)
        torch.manual_seed(seed)
        h=self.config.hidden_dim;radius=float(self.environment_spec['max_acceleration'])
        self.actors=nn.ModuleList([DiskActor(self.environment_spec['observation_dim'],h,radius) for _ in range(5)]).to(self.device)
        self.critics=nn.ModuleList([JointCritic(self.environment_spec['state_dim'],h) for _ in range(5)]).to(self.device)
        self.target_actors=deepcopy(self.actors).requires_grad_(False).eval()
        self.target_critics=deepcopy(self.critics).requires_grad_(False).eval()
        self.actor_optimizers=[torch.optim.Adam(a.parameters(),lr=self.config.actor_learning_rate) for a in self.actors]
        self.critic_optimizer=torch.optim.Adam(self.critics.parameters(),lr=self.config.critic_learning_rate)
        self.actor_update_count=self.critic_update_count=self.target_update_count=0
        self.actor_optimizer_steps=[0]*5

    @staticmethod
    def validate_environment_spec(contract):
        if (not isinstance(contract,dict) or
                (contract.get('version'),contract.get('observation_dim'),contract.get('state_dim')) not in
                (('interception_1',36,180),('interception_1',51,255),('interception_1',49,245),
                 ('interception_2',54,75),('interception_3',36,75)) or
                any(contract.get(k)!=v for k,v in dict(n_agents=5,action_dim=2).items()) or
                contract.get('action_schema')!='world_acceleration_disk_si' or
                not isinstance(contract.get('max_acceleration'),(int,float)) or contract['max_acceleration']<=0):
            raise ValueError('MADDPG requires a supported interception interface')
        return deepcopy(contract)

    def _joint(self,observations,masks,*,target=False):
        actors=self.target_actors if target else self.actors
        return torch.stack([actor(observations[:,i]) for i,actor in enumerate(actors)],1)*masks[...,None]

    @torch.no_grad()
    def policy_actions(self,observations,active_masks,*,deterministic=True):
        obs=torch.as_tensor(observations,dtype=torch.float32,device=self.device)
        mask=torch.as_tensor(active_masks,dtype=torch.float32,device=self.device)
        if obs.ndim!=3 or obs.shape[1:]!=(5,self.environment_spec['observation_dim']) or mask.shape!=obs.shape[:2]:raise ValueError('MADDPG observation/mask shape')
        actions=self._joint(obs,mask).cpu().numpy()
        if not deterministic and self.config.exploration_std_mps2:
            actions+=self.rng.normal(0,self.config.exploration_std_mps2,actions.shape).astype(np.float32)
            radius=self.environment_spec['max_acceleration']
            norms=np.linalg.norm(actions,axis=-1,keepdims=True)
            actions*=np.minimum(1.,radius/np.maximum(norms,1e-12))
            actions*=mask.cpu().numpy()[...,None]
        return actions.astype(np.float32,copy=False)

    def _batch(self,batch):
        b={k:torch.as_tensor(v,dtype=torch.float32,device=self.device) for k,v in batch.items()}
        size=b['rewards'].shape[0] if 'rewards' in b and b['rewards'].ndim==2 else 0
        shapes=dict(observations=(size,5,self.environment_spec['observation_dim']),next_observations=(size,5,self.environment_spec['observation_dim']),states=(size,self.environment_spec['state_dim']),
            next_states=(size,self.environment_spec['state_dim']),actions=(size,5,2),rewards=(size,5),dones=(size,),
            active_masks=(size,5),next_active_masks=(size,5))
        if size<1 or set(b)!=set(shapes) or any(b[k].shape!=shape or not torch.isfinite(b[k]).all() for k,shape in shapes.items()):
            raise ValueError('Invalid MADDPG joint replay batch')
        if any(not ((b[k]==0)|(b[k]==1)).all() for k in ('dones','active_masks','next_active_masks')):
            raise ValueError('MADDPG masks/dones must be binary')
        if (b['actions'].norm(dim=-1)>self.environment_spec['max_acceleration']+1e-4).any():
            raise ValueError('MADDPG actions exceed physical disk')
        return b

    def update(self,batch,*,actor_batch=None,critic_weights=None,return_td_errors=False):
        b=self._batch(batch);mask=b['active_masks']
        actor_data=b if actor_batch is None else self._batch(actor_batch)
        actor_mask=actor_data['active_masks']
        if critic_weights is not None:
            weights=torch.as_tensor(critic_weights,dtype=torch.float32,device=self.device)
            if weights.shape!=(len(mask),) or not torch.isfinite(weights).all() or (weights<=0).any():
                raise ValueError('Invalid MADDPG importance weights')
        with torch.no_grad():
            next_actions=self._joint(b['next_observations'],b['next_active_masks'],target=True)
            targets=torch.stack([b['rewards'][:,i]+self.config.gamma*(1-b['dones'])*b['next_active_masks'][:,i]*
                critic(b['next_states'],next_actions) for i,critic in enumerate(self.target_critics)],1)
        q=torch.stack([critic(b['states'],b['actions']) for critic in self.critics],1)
        active_q=q.detach()[mask.bool()]
        active_targets=targets[mask.bool()]
        td=q-targets
        td_errors=(td.detach().abs()*mask).sum(dim=1)/mask.sum(dim=1).clamp_min(1)
        squared=td.square()*mask
        if critic_weights is not None:squared=squared*weights[:,None]
        critic_loss=squared.sum()/mask.sum().clamp_min(1)
        if not torch.isfinite(critic_loss):raise FloatingPointError('Nonfinite MADDPG critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True);critic_loss.backward()
        critic_grad=nn.utils.clip_grad_norm_(self.critics.parameters(),self.config.critic_max_grad_norm)
        self.critic_optimizer.step();self.critic_update_count+=1
        with torch.no_grad():joint=self._joint(actor_data['observations'],actor_mask).detach()
        actor_losses=[];actor_grads=[]
        self.critics.requires_grad_(False)
        try:
            for i,(actor,opt) in enumerate(zip(self.actors,self.actor_optimizers)):
                active=actor_mask[:,i].bool()
                if not active.any():continue
                candidate=joint[active].clone();candidate[:,i]=actor(actor_data['observations'][active,i])
                loss=-self.critics[i](actor_data['states'][active],candidate).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MADDPG actor loss')
                opt.zero_grad(set_to_none=True);loss.backward()
                grad=nn.utils.clip_grad_norm_(actor.parameters(),self.config.actor_max_grad_norm)
                opt.step();self.actor_optimizer_steps[i]+=1
                actor_losses.append(float(loss.detach()));actor_grads.append(float(grad))
        finally:self.critics.requires_grad_(True)
        self.actor_update_count+=1
        with torch.no_grad():
            for online,target in ((self.actors,self.target_actors),(self.critics,self.target_critics)):
                for source,destination in zip(online.parameters(),target.parameters()):
                    destination.mul_(1-self.config.tau).add_(source,alpha=self.config.tau)
        self.target_update_count+=1
        metrics=dict(actor_loss=float(np.mean(actor_losses)) if actor_losses else None,
            critic_loss=float(critic_loss.detach()),actor_grad_norm=float(np.mean(actor_grads)) if actor_grads else None,
            critic_grad_norm=float(critic_grad),td_abs=float(((q.detach()-targets).abs()*mask).sum()/mask.sum().clamp_min(1)),
            q_mean=float(active_q.mean()) if active_q.numel() else 0.0,
            q_std=float(active_q.std(unbiased=False)) if active_q.numel() else 0.0,
            target_q_mean=float(active_targets.mean()) if active_targets.numel() else 0.0,
            actor_updates=self.actor_update_count,critic_updates=self.critic_update_count,
            target_updates=self.target_update_count,actor_optimizer_steps=sum(self.actor_optimizer_steps))
        if not np.isfinite([v for v in metrics.values() if v is not None]).all():raise FloatingPointError('Nonfinite MADDPG metrics')
        return (metrics,td_errors.cpu().numpy()) if return_td_errors else metrics

    def save(self,path,*,global_step,environment_config):
        if environment_config!=self.environment_config:raise ValueError('MADDPG environment mismatch at save')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(algorithm=self.algorithm,checkpoint_schema=self.schema,environment_spec=self.environment_spec,
            environment_config=self.environment_config,maddpg_config=asdict(self.config),global_step=int(global_step),
            provenance=dict(getattr(self,'experiment_provenance',{}),seed=self.seed,output_directory=str(path.parent.resolve())),
            checkpoint_scope='inference/evaluation and learner state only; no replay, workers or exact training resume',
            actors=self.actors.state_dict(),critics=self.critics.state_dict(),target_actors=self.target_actors.state_dict(),
            target_critics=self.target_critics.state_dict(),actor_optimizers=[o.state_dict() for o in self.actor_optimizers],
            critic_optimizer=self.critic_optimizer.state_dict(),actor_update_count=self.actor_update_count,
            critic_update_count=self.critic_update_count,target_update_count=self.target_update_count,
            actor_optimizer_steps=self.actor_optimizer_steps,exploration_rng_state=deepcopy(self.rng.bit_generator.state),
            torch_rng_state=torch.random.get_rng_state(),cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            numpy_rng_state=np.random.get_state(),python_rng_state=random.getstate()),path)

    def load(self,path,*,load_optimizers=True):
        networks=('actors','critics','target_actors','target_critics')
        counters=('actor_update_count','critic_update_count','target_update_count','actor_optimizer_steps')
        with loading_transaction(tuple(getattr(self,n) for n in networks),(*self.actor_optimizers,self.critic_optimizer),
                owner=self,counters=(*counters,'global_step'),generator_attr='rng',inference=not load_optimizers):
            payload=torch.load(path,map_location=self.device,weights_only=False)
            expected=dict(algorithm=self.algorithm,checkpoint_schema=self.schema,environment_spec=self.environment_spec,
                environment_config=self.environment_config,maddpg_config=asdict(self.config))
            required=set(expected)|set(networks)|{'global_step','provenance','checkpoint_scope'}
            if not isinstance(payload,dict) or not required<=payload.keys() or any(payload.get(k)!=v for k,v in expected.items()):
                raise ValueError('Incompatible MADDPG schema/config/environment contract')
            validate_counter(payload['global_step'],'global_step')
            for name in networks:validate_model_state(getattr(self,name),payload[name])
            if load_optimizers:
                training=set(counters)|{'actor_optimizers','critic_optimizer','exploration_rng_state','torch_rng_state',
                    'cuda_rng_states','numpy_rng_state','python_rng_state'}
                if not training<=payload.keys() or len(payload['actor_optimizers'])!=5 or len(payload['actor_optimizer_steps'])!=5:
                    raise ValueError('Incomplete MADDPG training state')
                for name in counters[:-1]:validate_counter(payload[name],name)
                for value in payload['actor_optimizer_steps']:validate_counter(value,'actor_optimizer_steps')
                for optimizer,state in zip(self.actor_optimizers,payload['actor_optimizers']):validate_optimizer_state(optimizer,state)
                validate_optimizer_state(self.critic_optimizer,payload['critic_optimizer'])
                validate_generator_rng(self.rng,payload['exploration_rng_state'])
                validate_torch_rng(payload['torch_rng_state']);validate_cuda_rng(payload['cuda_rng_states'],self.device)
                validate_numpy_rng(payload['numpy_rng_state']);validate_python_rng(payload['python_rng_state'])
            for name in networks:getattr(self,name).load_state_dict(payload[name],strict=True)
            if load_optimizers:
                for opt,state in zip(self.actor_optimizers,payload['actor_optimizers']):opt.load_state_dict(state)
                self.critic_optimizer.load_state_dict(payload['critic_optimizer'])
                self.rng.bit_generator.state=deepcopy(payload['exploration_rng_state'])
                torch.random.set_rng_state(payload['torch_rng_state'].cpu())
                if payload['cuda_rng_states'] and self.device.type=='cuda':torch.cuda.set_rng_state_all([s.cpu() for s in payload['cuda_rng_states']])
                np.random.set_state(payload['numpy_rng_state']);random.setstate(payload['python_rng_state'])
                for name in counters:setattr(self,name,payload[name])
            return payload
