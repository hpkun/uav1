"""Classic independent-agent MADDPG: deterministic actors, one Q per agent."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .networks import DeterministicActor, CentralizedQCritic

MADDPG_IMPL_VERSION = 1


def masked_mean(values, mask):
    selected = values[mask.bool()]
    return selected.mean() if selected.numel() else selected.sum()


def critic_target(rewards, dones, next_alive, next_q, gamma):
    return rewards+gamma*(1-dones)*next_alive*next_q


def actor_joint_actions(replay_actions, own_action, agent):
    return torch.stack([own_action if j == agent else replay_actions[..., j, :].detach()
                        for j in range(replay_actions.shape[-2])], dim=-2)


@torch.no_grad()
def polyak_update(online, target, tau):
    for source, destination in zip(online.parameters(), target.parameters()):
        destination.mul_(1-tau).add_(source, alpha=tau)


def finite_grad_norm(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if any(not torch.isfinite(g).all() for g in grads):
        raise FloatingPointError('non-finite MADDPG gradient')
    return float(torch.sqrt(sum(g.square().sum() for g in grads))) if grads else 0.


class MADDPGTrainer:
    def __init__(self, observation_dim, action_dim, num_agents, actor_learning_rate=1e-4,
                 critic_learning_rate=1e-4, gamma=.99, tau=.001, device='cuda', seed=0):
        if observation_dim != 13*num_agents or num_agents not in (5, 8) or action_dim != 3:
            raise ValueError('MADDPG requires 65/3/5 or 104/3/8')
        if not 0 <= gamma <= 1 or not 0 < tau <= 1:
            raise ValueError('invalid gamma/tau')
        self.device = torch.device(device)
        self.observation_dim, self.action_dim, self.num_agents = observation_dim, action_dim, num_agents
        self.gamma, self.tau, self.seed = float(gamma), float(tau), int(seed)
        torch.manual_seed(seed)
        self.actors = nn.ModuleList([DeterministicActor(observation_dim, action_dim) for _ in range(num_agents)]).to(self.device)
        self.critics = nn.ModuleList([CentralizedQCritic(observation_dim, action_dim, num_agents) for _ in range(num_agents)]).to(self.device)
        self.target_actors = deepcopy(self.actors).requires_grad_(False)
        self.target_critics = deepcopy(self.critics).requires_grad_(False)
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=actor_learning_rate) for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=critic_learning_rate) for critic in self.critics]
        self.sampled_steps = self.vector_steps = self.actor_update_count = self.critic_update_count = 0
        self.actor_updates_per_agent = [0]*num_agents
        self.critic_updates_per_agent = [0]*num_agents

    @property
    def network_architecture(self):
        return {'observation_dim': self.observation_dim, 'action_dim': self.action_dim, 'num_agents': self.num_agents,
            'actor_layers': [self.observation_dim, 256, 256, self.action_dim],
            'critic_layers': [self.num_agents*(self.observation_dim+self.action_dim), 256, 256, 1],
            'actor_activation': 'relu', 'actor_output': 'tanh', 'critic_activation': 'relu',
            'independent_actors': self.num_agents, 'independent_critics': self.num_agents}

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, observations, alive_masks, deterministic=True):
        if not deterministic:
            raise ValueError('MADDPG actors are deterministic; runner adds OU behavior exploration')
        obs, alive = self.tensor(observations), self.tensor(alive_masks)
        obs = torch.where(alive.unsqueeze(-1).bool(), obs, torch.zeros_like(obs))
        actions = torch.stack([actor(obs[..., i, :]) for i, actor in enumerate(self.actors)], dim=-2)
        return torch.where(alive.unsqueeze(-1).bool(), actions, torch.zeros_like(actions)).cpu().numpy()

    def tensors(self, batch):
        return {key: self.tensor(value) for key, value in vars(batch).items()}

    @torch.no_grad()
    def targets(self, batch):
        obs, alive = batch['next_observations'], batch['next_alive_masks']
        obs = torch.where(alive.unsqueeze(-1).bool(), obs, torch.zeros_like(obs))
        actions = torch.stack([actor(obs[:, i]) for i, actor in enumerate(self.target_actors)], dim=1)
        actions = torch.where(alive.unsqueeze(-1).bool(), actions, torch.zeros_like(actions))
        q = torch.stack([critic(obs, actions, alive) for critic in self.target_critics], dim=1)
        return critic_target(batch['rewards'], batch['dones'][:, None], alive, q, self.gamma)

    def update_actor(self, agent, batch):
        for optimizer in self.actor_optimizers+self.critic_optimizers:
            optimizer.zero_grad(set_to_none=True)
        alive = batch['alive_masks'][:, agent]
        if not alive.bool().any():
            return 0., 0.
        critic = self.critics[agent]
        critic.requires_grad_(False)
        try:
            own_obs = torch.where(alive[:, None].bool(), batch['observations'][:, agent],
                                  torch.zeros_like(batch['observations'][:, agent]))
            own = self.actors[agent](own_obs)
            actions = actor_joint_actions(batch['actions'], own, agent)
            loss = -masked_mean(critic(batch['observations'], actions, batch['alive_masks']), alive)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite actor loss')
            loss.backward()
            norm = finite_grad_norm(self.actors[agent])
            self.actor_optimizers[agent].step()
            self.actor_updates_per_agent[agent] += 1
            return float(loss.detach()), norm
        finally:
            critic.requires_grad_(True)

    def update(self, replay_batch):
        batch = self.tensors(replay_batch)
        targets = self.targets(batch)
        if not torch.isfinite(targets).all():
            raise FloatingPointError('non-finite critic target')
        critic_losses, actor_losses, critic_grads, actor_grads, q_means = [], [], [], [], []
        for i, (critic, optimizer) in enumerate(zip(self.critics, self.critic_optimizers)):
            optimizer.zero_grad(set_to_none=True)
            alive = batch['alive_masks'][:, i]
            q = critic(batch['observations'], batch['actions'], batch['alive_masks'])
            if not torch.isfinite(q).all():
                raise FloatingPointError('non-finite Q')
            loss = masked_mean((q-targets[:, i]).square(), alive)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite critic loss')
            norm = 0.
            if alive.bool().any():
                loss.backward(); norm = finite_grad_norm(critic); optimizer.step()
                self.critic_updates_per_agent[i] += 1
            critic_losses.append(float(loss.detach())); critic_grads.append(norm); q_means.append(float(q.detach().mean()))
        for i in range(self.num_agents):
            loss, norm = self.update_actor(i, batch)
            actor_losses.append(loss); actor_grads.append(norm)
        for online, target in ((self.actors, self.target_actors), (self.critics, self.target_critics)):
            polyak_update(online, target, self.tau)
        self.actor_update_count += 1
        self.critic_update_count += 1
        return {'actor_loss': float(np.mean(actor_losses)), 'critic_loss': float(np.mean(critic_losses)),
            'actor_losses': actor_losses, 'critic_losses': critic_losses, 'actor_grad_norms': actor_grads,
            'critic_grad_norms': critic_grads, 'q_means': q_means, 'target_mean': float(targets.mean()),
            'target_abs_max': float(targets.abs().max()), 'actor_update_count': self.actor_update_count,
            'critic_update_count': self.critic_update_count}

    def parameter_counts(self):
        a = sum(p.numel() for p in self.actors[0].parameters())
        q = sum(p.numel() for p in self.critics[0].parameters())
        return {'actor_per_agent': a, 'actors_trainable': a*self.num_agents,
                'critic_per_agent': q, 'critics_trainable': q*self.num_agents,
                'trainable_total': (a+q)*self.num_agents, 'target_total_nontrainable': (a+q)*self.num_agents}

    def checkpoint_state(self, extra):
        return {'algorithm': 'maddpg', 'implementation_version': MADDPG_IMPL_VERSION,
            'sampled_steps': self.sampled_steps, 'vector_steps': self.vector_steps,
            'training_seed': self.seed, 'gamma': self.gamma, 'tau': self.tau,
            'network_architecture': self.network_architecture,
            **{name: [m.state_dict() for m in getattr(self, name)] for name in
                ('actors', 'critics', 'target_actors', 'target_critics', 'actor_optimizers', 'critic_optimizers')},
            'actor_update_count': self.actor_update_count, 'critic_update_count': self.critic_update_count,
            'actor_updates_per_agent': list(self.actor_updates_per_agent), 'critic_updates_per_agent': list(self.critic_updates_per_agent),
            'torch_rng_state': torch.get_rng_state(),
            'torch_cuda_rng_state_all': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'formal_exact_resume_supported': False, 'replay_buffer_included': False, 'extra': deepcopy(extra)}

    def save(self, path, extra):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(extra), path)

    def load(self, path, env_config, algorithm_config):
        from .protocol import validate_checkpoint, require_cuda
        require_cuda(self.device)
        state = torch.load(path, map_location=self.device, weights_only=False)
        validate_checkpoint(state, env_config, algorithm_config)
        if state['network_architecture'] != self.network_architecture or (state['gamma'], state['tau'], state['training_seed']) != (self.gamma, self.tau, self.seed):
            raise RuntimeError('MADDPG trainer/checkpoint architecture or hyperparameter mismatch')
        for name in ('actors', 'critics', 'target_actors', 'target_critics'):
            for module, saved in zip(getattr(self, name), state[name]):
                module.load_state_dict(saved, strict=True)
        for name in ('actor_optimizers', 'critic_optimizers'):
            for optimizer, saved in zip(getattr(self, name), state[name]):
                optimizer.load_state_dict(saved)
        for name in ('sampled_steps', 'vector_steps', 'actor_update_count', 'critic_update_count',
                     'actor_updates_per_agent', 'critic_updates_per_agent'):
            setattr(self, name, deepcopy(state[name]))
        return state['extra']

    def resume(self, *args, **kwargs):
        raise RuntimeError('formal exact resume unsupported: replay and environment state are not checkpointed')
