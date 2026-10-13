"""Independent MAAC Eq.(7)-(9), stochastic score gradient and target networks."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from .networks import GaussianLocalActor,CentralizedAttentionCritic
from .protocol import IMPLEMENTATION_VERSION,validate_config_pair,validate_checkpoint,identity


def score_function_loss(log_prob,q_value,baseline,alpha,alive):
    weight=(alpha*log_prob-(q_value-baseline)).detach()
    return (log_prob*weight*alive).sum()/alive.sum().clamp_min(1)


class MAACTrainer:
    def __init__(self,env_config,config,device='cuda',seed=0,smoke=False):
        validate_config_pair(env_config,config)
        if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
        self.env_config,self.config,self.device,self.smoke=env_config,config,torch.device(device),bool(smoke)
        torch.manual_seed(seed);self.seed=seed
        n,t,impl=config['network'],config['training'],config['implementation']
        width=32 if smoke else n['hidden_dim']
        self.actors=nn.ModuleList([GaussianLocalActor(hidden_dim=width,log_std_min=impl['log_std_min'],
            log_std_max=impl['log_std_max']) for _ in range(5)]).to(self.device)
        self.critic=CentralizedAttentionCritic(hidden_dim=width,attention_heads=n['attention_heads']).to(self.device)
        self.target_actors=deepcopy(self.actors).requires_grad_(False).eval()
        self.target_critic=deepcopy(self.critic).requires_grad_(False).eval()
        self.actor_optimizers=[torch.optim.Adam(a.parameters(),lr=t['lr_actor']) for a in self.actors]
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=t['lr_critic'])
        self.gamma,self.alpha,self.tau=t['gamma'],t['alpha'],t['tau']
        self.baseline_samples=min(4,t['baseline_samples']) if smoke else t['baseline_samples']
        self.sampled_steps=self.actor_updates=self.critic_updates=self.policy_updates=self.target_updates=0

    def tensor(self,x):return torch.as_tensor(x,dtype=torch.float32,device=self.device)

    def batch_tensors(self,batch):
        data={k:self.tensor(v) for k,v in vars(batch).items()}
        b=len(data['observations'])
        expected=dict(observations=(b,5,56),actions=(b,5,3),rewards=(b,5),next_observations=(b,5,56),
            dones=(b,),alive_masks=(b,5),next_alive_masks=(b,5))
        for key,shape in expected.items():
            if data[key].shape!=shape or not torch.isfinite(data[key]).all():raise ValueError(f'invalid replay {key}')
        for key in ('dones','alive_masks','next_alive_masks'):
            if not ((data[key]==0)|(data[key]==1)).all():raise ValueError(f'{key} must be binary')
        return data

    @torch.no_grad()
    def act(self,observations,alive_masks=None,deterministic=True):
        obs=self.tensor(observations)
        if obs.ndim not in (2,3) or obs.shape[-2:]!=(5,56) or not torch.isfinite(obs).all():
            raise ValueError('actor team observations must be finite [5,56] or [B,5,56]')
        batched=obs.ndim==3
        if not batched:obs=obs[None]
        alive=torch.ones(obs.shape[:2],device=self.device) if alive_masks is None else self.tensor(alive_masks).reshape(obs.shape[:2])
        actions=torch.stack([a.sample(obs[:,i],deterministic)[0] for i,a in enumerate(self.actors)],1)*alive[...,None]
        result=actions.cpu().numpy()
        return result if batched else result[0]

    @torch.no_grad()
    def td_targets(self,rewards,next_obs,dones,next_alive):
        samples=[actor.sample(next_obs[:,i]) for i,actor in enumerate(self.target_actors)]
        actions=torch.stack([r[0] for r in samples],1)*next_alive[...,None]
        log_probs=torch.stack([r[1] for r in samples],1)
        q=self.target_critic(next_obs,actions,next_alive).squeeze(-1)
        return rewards+self.gamma*(1-dones[:,None])*next_alive*(q-self.alpha*log_probs)

    @torch.no_grad()
    def counterfactual_baseline(self,observations,joint_actions,alive,index,samples=None):
        # Independent samples from pi_i; all other a_j are held exactly fixed.
        k=self.baseline_samples if samples is None else samples
        b=len(observations)
        alternative=self.actors[index].sample(observations[:,index].unsqueeze(0).expand(k,-1,-1))[0]
        joint=joint_actions.unsqueeze(0).expand(k,-1,-1,-1).clone()
        joint[:,:,index]=alternative*alive[:,index][None,:,None]
        o=observations.unsqueeze(0).expand(k,-1,-1,-1).reshape(k*b,5,56)
        m=alive.unsqueeze(0).expand(k,-1,-1).reshape(k*b,5)
        return self.critic(o,joint.reshape(k*b,5,3),m)[:,index,0].reshape(k,b).mean(0)

    def update_critic(self,batch):
        d=self.batch_tensors(batch);alive=d['alive_masks']
        if alive.sum()==0:return {}
        y=self.td_targets(d['rewards'],d['next_observations'],d['dones'],d['next_alive_masks'])
        q=self.critic(d['observations'],d['actions'],alive).squeeze(-1)
        # Eq.(7): sum over agents, mean over replay transitions.
        loss=((q-y).square()*alive).sum()/len(q)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite MAAC critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True);loss.backward()
        # Shared attention parameters receive five tasks' gradients.
        for p in self.critic.shared_parameters():
            if p.grad is not None:p.grad.div_(5)
        norm=self.check_and_clip(self.critic,self.config['training']['critic_grad_clip'])
        self.critic_optimizer.step();self.critic_optimizer.zero_grad(set_to_none=True)
        self.critic_updates+=1
        return dict(critic_loss=float(loss.detach()),critic_grad_norm=norm)

    def update_policies(self,batch):
        d=self.batch_tensors(batch);o,alive=d['observations'],d['alive_masks']
        if alive.sum()==0:return {}
        # All current policy actions and MC baselines are sampled before any actor update.
        with torch.no_grad():
            samples=[a.sample(o[:,i]) for i,a in enumerate(self.actors)]
            joint=torch.stack([r[0] for r in samples],1)*alive[...,None]
            raws=[r[2].detach() for r in samples]
            qs=self.critic(o,joint,alive).squeeze(-1)
            baselines=[self.counterfactual_baseline(o,joint,alive,i) for i in range(5)]
        self.critic.requires_grad_(False)
        losses,norms,advantages=[],[],[]
        try:
            for i,(actor,optimizer) in enumerate(zip(self.actors,self.actor_optimizers)):
                if alive[:,i].sum()==0:continue
                log_prob=actor.log_prob(o[:,i],raws[i])
                loss=score_function_loss(log_prob,qs[:,i],baselines[i],self.alpha,alive[:,i])
                if not torch.isfinite(loss):raise RuntimeError('nonfinite MAAC actor loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=self.check_and_clip(actor,self.config['training']['actor_grad_clip'])
                optimizer.step();optimizer.zero_grad(set_to_none=True)
                self.actor_updates+=1;losses.append(float(loss.detach()));norms.append(norm)
                advantages.append(float((qs[:,i]-baselines[i]).abs().mul(alive[:,i]).sum()/alive[:,i].sum()))
        finally:self.critic.requires_grad_(True)
        self.policy_updates+=1
        return dict(actor_loss=float(np.mean(losses)),actor_grad_norm=float(np.mean(norms)),
            mean_abs_counterfactual_advantage=float(np.mean(advantages)))

    @staticmethod
    def check_and_clip(network,limit):
        grads=[p.grad for p in network.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('invalid MAAC gradient')
        return float(torch.nn.utils.clip_grad_norm_(network.parameters(),limit,error_if_nonfinite=True))

    @torch.no_grad()
    def update_targets(self):
        # tau is the ONLINE weight, matching the supplementary prose convention.
        for target,online in ((self.target_actors,self.actors),(self.target_critic,self.critic)):
            for dst,src in zip(target.parameters(),online.parameters()):dst.lerp_(src,self.tau)
        self.target_updates+=1

    def update(self,batch):
        result=self.update_critic(batch);result.update(self.update_policies(batch));self.update_targets()
        return result

    def save(self,path):
        torch.save(dict(algorithm='MAAC',implementation_version=IMPLEMENTATION_VERSION,
            extra=identity(self.env_config,self.config,self.smoke),actors=self.actors.state_dict(),
            critic=self.critic.state_dict(),target_actors=self.target_actors.state_dict(),
            target_critic=self.target_critic.state_dict(),actor_optimizers=[o.state_dict() for o in self.actor_optimizers],
            critic_optimizer=self.critic_optimizer.state_dict(),sampled_steps=self.sampled_steps,
            actor_updates=self.actor_updates,critic_updates=self.critic_updates,policy_updates=self.policy_updates,
            target_updates=self.target_updates,training_seed=self.seed),path)

    def load(self,path):
        # Evaluation load only: replay/environment/RNG continuation is not claimed.
        state=torch.load(path,map_location=self.device,weights_only=False)
        validate_checkpoint(state,self.env_config,self.config)
        if state['extra']['training_smoke']!=self.smoke:raise RuntimeError('MAAC checkpoint smoke/formal mismatch')
        for name in ('actors','critic','target_actors','target_critic'):
            getattr(self,name).load_state_dict(state[name])
        self.sampled_steps=state['sampled_steps']
        return state
