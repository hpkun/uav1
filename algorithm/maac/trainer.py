"""Fresh-rollout TD + own-action deterministic policy gradient; no target nets."""
from dataclasses import dataclass
import numpy as np
import torch
from .networks import SharedLocalActor, CentralizedAttentionCritic
from .protocol import IMPLEMENTATION_VERSION, validate_config_pair, validate_checkpoint, identity


@dataclass
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray
    alive_masks: np.ndarray
    next_alive_masks: np.ndarray


def own_action_joint(actions, index):
    return torch.cat([actions[:,j:j+1] if j==index else actions[:,j:j+1].detach()
                      for j in range(actions.shape[1])],dim=1)


class MAACTrainer:
    def __init__(self, env_config, config, device='cuda', seed=0, smoke=False):
        validate_config_pair(env_config,config)
        if device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; no CPU fallback')
        self.env_config,self.config,self.device,self.smoke=env_config,config,torch.device(device),bool(smoke)
        torch.manual_seed(seed)
        self.rng=np.random.default_rng(seed)
        self.seed=seed
        width=32 if smoke else config['network']['hidden_dim']
        self.actor=SharedLocalActor(hidden_dim=width).to(self.device)
        self.critic=CentralizedAttentionCritic(hidden_dim=width).to(self.device)
        t=config['training'];self.gamma=t['gamma']
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=t['lr_actor'])
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=t['lr_critic'])
        self.minibatch_size=min(16,t['minibatch_size']) if smoke else t['minibatch_size']
        self.sampled_steps=self.actor_updates=self.critic_updates=0

    def tensor(self,x):
        return torch.as_tensor(x,dtype=torch.float32,device=self.device)

    @torch.no_grad()
    def act(self, observations, alive_masks=None, deterministic=True):
        obs=self.tensor(observations)
        if obs.shape[-1]!=56 or not torch.isfinite(obs).all():
            raise ValueError('actor observations must be finite with last dimension 56')
        action=self.actor(obs)
        if alive_masks is not None:action=action*self.tensor(alive_masks)[...,None]
        return action.cpu().numpy()

    @torch.no_grad()
    def td_targets(self, rewards, next_obs, dones, next_alive):
        next_actions=self.actor(next_obs)*next_alive[...,None]
        q=self.critic(next_obs,next_actions,next_alive).squeeze(-1)
        return rewards+self.gamma*(1-dones[:,None])*next_alive*q

    def update(self,batch):
        arrays={k:self.tensor(v) for k,v in vars(batch).items()}
        # Flatten time/environment only, preserving the five-agent joint input.
        obs=arrays['observations'].reshape(-1,5,56)
        act=arrays['actions'].reshape(-1,5,3);rewards=arrays['rewards'].reshape(-1,5)
        next_obs=arrays['next_observations'].reshape(-1,5,56)
        dones=arrays['dones'].reshape(-1);alive=arrays['alive_masks'].reshape(-1,5)
        next_alive=arrays['next_alive_masks'].reshape(-1,5)
        if not all(torch.isfinite(x).all() for x in arrays.values()):raise ValueError('nonfinite rollout')
        targets=self.td_targets(rewards,next_obs,dones,next_alive)
        rows=[]
        for start in range(0,len(obs),self.minibatch_size):
            # A single pass through each fresh rollout; no replay or PPO epochs.
            if start==0:order=self.rng.permutation(len(obs))
            idx=order[start:start+self.minibatch_size]
            o,a,m,y=obs[idx],act[idx],alive[idx],targets[idx]
            count=m.sum()
            if count==0:continue
            q=self.critic(o,a,m).squeeze(-1)
            critic_loss=((q-y).square()*m).sum()/count
            self.critic_optimizer.zero_grad(set_to_none=True);critic_loss.backward()
            critic_norm=self._check_gradients(self.critic)
            self.critic_optimizer.step();self.critic_updates+=1
            self.critic_optimizer.zero_grad(set_to_none=True)
            for p in self.critic.parameters():p.requires_grad_(False)
            try:
                policy_actions=self.actor(o)*m[...,None]
                loss=0.
                for i in range(5):
                    values=self.critic(o,own_action_joint(policy_actions,i),m)[:,i,0]
                    loss=loss-(values*m[:,i]).sum()/count
                self.actor_optimizer.zero_grad(set_to_none=True);loss.backward()
                actor_norm=self._check_gradients(self.actor)
                self.actor_optimizer.step();self.actor_updates+=1
            finally:
                for p in self.critic.parameters():p.requires_grad_(True)
            if not torch.isfinite(critic_loss) or not torch.isfinite(loss):raise RuntimeError('nonfinite MAAC loss')
            rows.append(dict(critic_loss=float(critic_loss.detach()),actor_loss=float(loss.detach()),
                critic_grad_norm=critic_norm,actor_grad_norm=actor_norm,live_count=float(count)))
        return {key:sum(r[key]*r['live_count'] for r in rows)/sum(r['live_count'] for r in rows)
                for key in ('critic_loss','actor_loss','critic_grad_norm','actor_grad_norm')} if rows else {}

    @staticmethod
    def _check_gradients(network):
        grads=[p.grad for p in network.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('invalid MAAC gradient')
        return float(torch.sqrt(sum(g.square().sum() for g in grads)))

    def save(self,path):
        torch.save(dict(algorithm='MAAC',implementation_version=IMPLEMENTATION_VERSION,
            extra=identity(self.env_config,self.config,self.smoke),actor=self.actor.state_dict(),critic=self.critic.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),critic_optimizer=self.critic_optimizer.state_dict(),
            sampled_steps=self.sampled_steps,actor_updates=self.actor_updates,critic_updates=self.critic_updates,
            training_seed=self.seed),path)

    def load(self,path):
        state=torch.load(path,map_location=self.device,weights_only=False)
        validate_checkpoint(state,self.env_config,self.config)
        if state['extra']['training_smoke']!=self.smoke:raise RuntimeError('MAAC checkpoint smoke/formal mismatch')
        self.actor.load_state_dict(state['actor']);self.critic.load_state_dict(state['critic'])
        self.sampled_steps=state['sampled_steps']
        return state
