"""Li et al. MAAC baseline: shared policy, Eq.(14)-(15), single attention Q."""
from copy import deepcopy
import torch
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
        self.observation_dim,self.action_dim,self.num_agents=(n[k] for k in ('observation_dim','action_dim','num_agents'))
        width=32 if smoke else n['hidden_dim']
        self.actor=GaussianLocalActor(self.observation_dim,self.action_dim,hidden_dim=width,log_std_min=impl['log_std_min'],
            log_std_max=impl['log_std_max']).to(self.device)
        self.critic=CentralizedAttentionCritic(self.observation_dim,self.action_dim,self.num_agents,hidden_dim=width,attention_heads=n['attention_heads']).to(self.device)
        self.target_actor=deepcopy(self.actor).requires_grad_(False).eval()
        self.target_critic=deepcopy(self.critic).requires_grad_(False).eval()
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=t['lr_actor'])
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=t['lr_critic'])
        self.gamma,self.alpha,self.tau=t['gamma'],t['alpha'],t['tau']
        self.baseline_samples=min(4,t['baseline_samples']) if smoke else t['baseline_samples']
        self.sampled_steps=self.actor_updates=self.critic_updates=self.policy_updates=self.target_updates=0

    def tensor(self,x):return torch.as_tensor(x,dtype=torch.float32,device=self.device)

    def batch_tensors(self,batch):
        data={k:self.tensor(v) for k,v in vars(batch).items()}
        b=len(data['observations'])
        expected=dict(observations=(b,self.num_agents,self.observation_dim),actions=(b,self.num_agents,self.action_dim),rewards=(b,self.num_agents),next_observations=(b,self.num_agents,self.observation_dim),
            dones=(b,),alive_masks=(b,self.num_agents),next_alive_masks=(b,self.num_agents))
        for key,shape in expected.items():
            if data[key].shape!=shape or not torch.isfinite(data[key]).all():raise ValueError(f'invalid replay {key}')
        for key in ('dones','alive_masks','next_alive_masks'):
            if not ((data[key]==0)|(data[key]==1)).all():raise ValueError(f'{key} must be binary')
        return data

    @torch.no_grad()
    def act(self,observations,alive_masks=None,deterministic=True):
        obs=self.tensor(observations)
        if obs.ndim not in (2,3) or obs.shape[-2:]!=(self.num_agents,self.observation_dim) or not torch.isfinite(obs).all():
            raise ValueError(f'actor observations must be finite [N,D] or [B,N,D], N={self.num_agents}, D={self.observation_dim}')
        batched=obs.ndim==3
        if not batched:obs=obs[None]
        alive=torch.ones(obs.shape[:2],device=self.device) if alive_masks is None else self.tensor(alive_masks).reshape(obs.shape[:2])
        actions=self.actor.sample(obs,deterministic)[0]*alive[...,None]
        result=actions.cpu().numpy()
        return result if batched else result[0]

    @torch.no_grad()
    def td_targets(self,rewards,next_obs,dones,next_alive):
        actions,log_probs,_=self.target_actor.sample(next_obs)
        actions=actions*next_alive[...,None]
        q=self.target_critic(next_obs,actions,next_alive).squeeze(-1)
        return rewards+self.gamma*(1-dones[:,None])*next_alive*(q-self.alpha*log_probs)

    @torch.no_grad()
    def counterfactual_baseline(self,observations,joint_actions,alive,index,samples=None):
        # Independent samples from pi_i; all other a_j are held exactly fixed.
        k=self.baseline_samples if samples is None else samples
        b=len(observations)
        alternative=self.actor.sample(observations[:,index].unsqueeze(0).expand(k,-1,-1))[0]
        joint=joint_actions.unsqueeze(0).expand(k,-1,-1,-1).clone()
        joint[:,:,index]=alternative*alive[:,index][None,:,None]
        o=observations.unsqueeze(0).expand(k,-1,-1,-1).reshape(k*b,self.num_agents,self.observation_dim)
        m=alive.unsqueeze(0).expand(k,-1,-1).reshape(k*b,self.num_agents)
        return self.critic(o,joint.reshape(k*b,self.num_agents,self.action_dim),m)[:,index,0].reshape(k,b).mean(0)

    def update_critic(self,batch):
        d=self.batch_tensors(batch);alive=d['alive_masks']
        if alive.sum()==0:return {}
        y=self.td_targets(d['rewards'],d['next_observations'],d['dones'],d['next_alive_masks'])
        q=self.critic(d['observations'],d['actions'],alive).squeeze(-1)
        # Li Eq.(14), original MAAC Eq.(7): sum agents, mean replay transitions.
        loss=((q-y).square()*alive).sum()/len(q)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite MAAC critic loss')
        self.critic_optimizer.zero_grad(set_to_none=True);loss.backward()
        # Shared attention parameters receive five tasks' gradients.
        for p in self.critic.shared_parameters():
            if p.grad is not None:p.grad.div_(self.num_agents)
        norm=self.check_and_clip(self.critic,self.config['training']['critic_grad_clip'])
        self.critic_optimizer.step();self.critic_optimizer.zero_grad(set_to_none=True)
        self.critic_updates+=1
        return dict(critic_loss=float(loss.detach()),critic_grad_norm=norm)

    def update_policies(self,batch):
        d=self.batch_tensors(batch);o,alive=d['observations'],d['alive_masks']
        if alive.sum()==0:return {}
        # All live-agent contributions share one parameter snapshot and one step.
        with torch.no_grad():
            actions,_,raw=self.actor.sample(o)
            joint=actions*alive[...,None]
            qs=self.critic(o,joint,alive).squeeze(-1)
            baselines=torch.stack([self.counterfactual_baseline(o,joint,alive,i) for i in range(self.num_agents)],1)
        self.critic.requires_grad_(False)
        try:
            log_prob=self.actor.log_prob(o,raw)
            loss=score_function_loss(log_prob,qs,baselines,self.alpha,alive)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite MAAC actor loss')
            self.actor_optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=self.check_and_clip(self.actor,self.config['training']['actor_grad_clip'])
            self.actor_optimizer.step();self.actor_optimizer.zero_grad(set_to_none=True)
            self.actor_updates+=1
        finally:self.critic.requires_grad_(True)
        self.policy_updates+=1
        return dict(actor_loss=float(loss.detach()),actor_grad_norm=norm,
            mean_abs_counterfactual_advantage=float((qs-baselines).abs().mul(alive).sum()/alive.sum()))

    @staticmethod
    def check_and_clip(network,limit):
        grads=[p.grad for p in network.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('invalid MAAC gradient')
        return float(torch.nn.utils.clip_grad_norm_(network.parameters(),limit,error_if_nonfinite=True))

    @torch.no_grad()
    def update_targets(self):
        # tau is the ONLINE weight, matching the supplementary prose convention.
        for target,online in ((self.target_actor,self.actor),(self.target_critic,self.critic)):
            for dst,src in zip(target.parameters(),online.parameters()):dst.lerp_(src,self.tau)
        self.target_updates+=1

    def update(self,batch):
        result=self.update_critic(batch);result.update(self.update_policies(batch));self.update_targets()
        return result

    def save(self,path):
        torch.save(dict(algorithm='MAAC',implementation_version=IMPLEMENTATION_VERSION,
            extra=identity(self.env_config,self.config,self.smoke),actor=self.actor.state_dict(),
            critic=self.critic.state_dict(),target_actor=self.target_actor.state_dict(),
            target_critic=self.target_critic.state_dict(),actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),sampled_steps=self.sampled_steps,
            actor_updates=self.actor_updates,critic_updates=self.critic_updates,policy_updates=self.policy_updates,
            target_updates=self.target_updates,training_seed=self.seed),path)

    def load(self,path):
        # Evaluation load only: replay/environment/RNG continuation is not claimed.
        state=torch.load(path,map_location=self.device,weights_only=False)
        validate_checkpoint(state,self.env_config,self.config)
        if state['extra']['training_smoke']!=self.smoke:raise RuntimeError('MAAC checkpoint smoke/formal mismatch')
        for name in ('actor','critic','target_actor','target_critic'):
            getattr(self,name).load_state_dict(state[name])
        self.sampled_steps=state['sampled_steps']
        return state
