"""Independent per-agent twin centralized Qs and delayed deterministic policies."""
from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from .networks import DeterministicActor, CentralizedQCritic
from .protocol import MATD3_IMPL_VERSION, validate_dimensions, require_cuda
from .diagnostics import clip_gradients, scalar_statistics, action_statistics, parameter_distances


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


class MATD3Trainer:
    def __init__(self, observation_dim, action_dim, num_agents, actor_learning_rate=1e-4,
                 critic_learning_rate=1e-4, gamma=.99, tau=.001, device='cuda', seed=0,
                 gradient_clip_norm=.5, diagnostics_interval_updates=100,
                 policy_delay=2, target_policy_noise=.2, target_noise_clip=.5):
        validate_dimensions(observation_dim, action_dim, num_agents)
        if not 0 <= gamma <= 1 or not 0 < tau <= 1:
            raise ValueError('invalid gamma/tau')
        self.device = torch.device(device)
        require_cuda(self.device)
        self.observation_dim, self.action_dim, self.num_agents = observation_dim, action_dim, num_agents
        self.gamma, self.tau, self.seed = float(gamma), float(tau), int(seed)
        if not np.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0 or type(diagnostics_interval_updates) is not int or diagnostics_interval_updates <= 0:
            raise ValueError('positive gradient clipping norm and diagnostics interval required')
        self.gradient_clip_norm = float(gradient_clip_norm)
        self.diagnostics_interval_updates = diagnostics_interval_updates
        if type(policy_delay) is not int or policy_delay <= 0:
            raise ValueError('positive integer policy_delay required')
        if any(not np.isfinite(v) or v < 0 for v in (target_policy_noise, target_noise_clip)):
            raise ValueError('finite nonnegative target noise required')
        self.policy_delay = policy_delay
        self.target_policy_noise, self.target_noise_clip = float(target_policy_noise), float(target_noise_clip)
        self.target_smoothing_rng = torch.Generator(device=self.device)
        self.target_smoothing_rng.manual_seed(self.seed ^ 0x7D3001)
        torch.manual_seed(seed)
        self.actors = nn.ModuleList([DeterministicActor(observation_dim, action_dim) for _ in range(num_agents)]).to(self.device)
        self.critics = nn.ModuleList([CentralizedQCritic(observation_dim, action_dim, num_agents) for _ in range(num_agents)]).to(self.device)
        self.critics2 = nn.ModuleList([CentralizedQCritic(observation_dim, action_dim, num_agents) for _ in range(num_agents)]).to(self.device)
        self.target_critics2 = deepcopy(self.critics2).requires_grad_(False)
        self.initial_critic2_parameters = tuple(torch.nn.utils.parameters_to_vector(q.parameters()).detach().clone() for q in self.critics2)
        self.critic2_optimizers = [torch.optim.Adam(q.parameters(), lr=critic_learning_rate) for q in self.critics2]
        self.critic2_updates_per_agent = [0]*num_agents
        self.target_actors = deepcopy(self.actors).requires_grad_(False)
        self.target_critics = deepcopy(self.critics).requires_grad_(False)
        self.initial_actor_parameters = tuple(torch.nn.utils.parameters_to_vector(a.parameters()).detach().clone() for a in self.actors)
        self.initial_critic_parameters = tuple(torch.nn.utils.parameters_to_vector(q.parameters()).detach().clone() for q in self.critics)
        self._actor_diagnostics = [{} for _ in range(num_agents)]
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
            'independent_actors': self.num_agents, 'independent_critics': 2*self.num_agents, 'critics_per_agent': 2}

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, observations, alive_masks, deterministic=True):
        if not deterministic:
            raise ValueError('MATD3 actors are deterministic; runner adds OU behavior exploration')
        obs, alive = self.tensor(observations), self.tensor(alive_masks)
        obs = torch.where(alive.unsqueeze(-1).bool(), obs, torch.zeros_like(obs))
        actions = torch.stack([actor(obs[..., i, :]) for i, actor in enumerate(self.actors)], dim=-2)
        return torch.where(alive.unsqueeze(-1).bool(), actions, torch.zeros_like(actions)).cpu().numpy()

    def tensors(self, batch):
        return {key: self.tensor(value) for key, value in vars(batch).items()}

    @torch.no_grad()
    def smoothed_target_actions(self, observations, alive):
        obs = torch.where(alive.unsqueeze(-1).bool(), observations, torch.zeros_like(observations))
        means = torch.stack([actor(obs[:, i]) for i, actor in enumerate(self.target_actors)], dim=1)
        raw = torch.randn(means.shape, generator=self.target_smoothing_rng, device=self.device)*self.target_policy_noise
        noise = raw.clamp(-self.target_noise_clip, self.target_noise_clip)
        actions = (means+noise).clamp(-1, 1)
        live = alive.unsqueeze(-1).bool().expand_as(noise)
        selected = noise[live]
        self._smoothing_metrics = {
            'target_noise_rms': float(selected.square().mean().sqrt()) if selected.numel() else 0.,
            'target_noise_clip_fraction': float((raw[live].abs()>self.target_noise_clip).float().mean()) if selected.numel() else 0.,
            'target_action_clip_fraction': float(((means+noise)[live].abs()>1).float().mean()) if selected.numel() else 0.,
            'target_noise_alive_coordinates': int(live.sum())}
        return torch.where(live, actions, torch.zeros_like(actions))

    @torch.no_grad()
    def targets(self, batch):
        obs, alive = batch['next_observations'], batch['next_alive_masks']
        actions = self.smoothed_target_actions(obs, alive)
        q1 = torch.stack([q(obs, actions, alive) for q in self.target_critics], dim=1)
        q2 = torch.stack([q(obs, actions, alive) for q in self.target_critics2], dim=1)
        return critic_target(batch['rewards'], batch['dones'][:, None], alive, torch.minimum(q1, q2), self.gamma)

    def update_actor(self, agent, batch):
        for optimizer in self.actor_optimizers+self.critic_optimizers+self.critic2_optimizers:
            optimizer.zero_grad(set_to_none=True)
        alive = batch['alive_masks'][:, agent]
        if not alive.bool().any():
            self._actor_diagnostics[agent] = {'pre':0., 'post':0., 'clipped':False,
                'actions': torch.zeros_like(batch['actions'][:,agent])}
            return 0., 0.
        critic = self.critics[agent]
        critic_parameters = list(critic.parameters())+list(self.critics2[agent].parameters())
        original_requires_grad = [p.requires_grad for p in critic_parameters]
        for parameter in critic_parameters:
            parameter.requires_grad_(False)
        try:
            own_obs = torch.where(alive[:, None].bool(), batch['observations'][:, agent],
                                  torch.zeros_like(batch['observations'][:, agent]))
            own = self.actors[agent](own_obs)
            actions = actor_joint_actions(batch['actions'], own, agent)
            loss = -masked_mean(critic(batch['observations'], actions, batch['alive_masks']), alive)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite actor loss')
            loss.backward()
            norm, post, clipped = clip_gradients(self.actors[agent], self.gradient_clip_norm)
            self._actor_diagnostics[agent] = {'pre':norm,'post':post,'clipped':clipped,'actions':own.detach()}
            self.actor_optimizers[agent].step()
            self.actor_updates_per_agent[agent] += 1
            return float(loss.detach()), norm
        finally:
            for parameter, flag in zip(critic_parameters,original_requires_grad):parameter.requires_grad_(flag)

    def update_critic(self, agent, batch, targets, twin=1):
        if twin not in (1, 2):
            raise ValueError('twin must be 1 or 2')
        modules, optimizers, counts = (self.critics, self.critic_optimizers, self.critic_updates_per_agent) if twin == 1 else (self.critics2, self.critic2_optimizers, self.critic2_updates_per_agent)
        critic, optimizer = modules[agent], optimizers[agent]
        optimizer.zero_grad(set_to_none=True)
        alive = batch['alive_masks'][:,agent]
        q = critic(batch['observations'],batch['actions'],batch['alive_masks'])
        if not torch.isfinite(q).all() or not torch.isfinite(targets[:,agent]).all():
            raise FloatingPointError('non-finite Q or critic target')
        loss = masked_mean((q-targets[:,agent].detach()).square(),alive)
        if not torch.isfinite(loss):raise FloatingPointError('non-finite critic loss')
        pre, post, clipped = 0., 0., False
        if alive.bool().any():
            loss.backward()
            pre,post,clipped = clip_gradients(critic,self.gradient_clip_norm)
            optimizer.step();counts[agent]+=1
        return {'loss':float(loss.detach()),'pre':pre,'post':post,'clipped':clipped,'q':q.detach()}

    def update(self, replay_batch):
        require_cuda(self.device)
        batch = self.tensors(replay_batch)
        targets = self.targets(batch)
        if not torch.isfinite(targets).all():
            raise FloatingPointError('non-finite Bellman targets')
        alive = batch['alive_masks']
        metrics = {'statistics_scope': 'current-alive replay samples; population std; pre-step Q/TD; null for empty; gaps are disagreement, not overestimation',
            'alive_samples': alive.sum(0).int().cpu().tolist(),
            'alive_fraction': alive.mean(0).cpu().tolist(), **self._smoothing_metrics}
        results = []
        for twin in (1, 2):
            rows = [self.update_critic(i, batch, targets, twin) for i in range(self.num_agents)]
            results.append(rows)
            prefix = f'q{twin}'
            metrics[prefix+'_losses'] = [r['loss'] for r in rows]
            metrics[prefix+'_loss'] = float(np.mean(metrics[prefix+'_losses']))
            for key, source in (('grad_norm_preclip','pre'), ('grad_norm_postclip','post'), ('grad_clipped','clipped')):
                metrics[prefix+'_'+key] = [r[source] for r in rows]
            stats = [scalar_statistics(r['q'], alive[:, i]) for i, r in enumerate(rows)]
            td = [scalar_statistics(r['q']-targets[:, i], alive[:, i]) for i, r in enumerate(rows)]
            for name in ('mean','std','min','max'):
                metrics[prefix+'_'+name+'_per_agent'] = [r[name] for r in stats]
            for name in ('mean','abs_mean','abs_max'):
                metrics[prefix+'_td_error_'+name] = [r[name] for r in td]
        metrics['critic_loss'] = (metrics['q1_loss']+metrics['q2_loss'])/2
        gap = [scalar_statistics(a['q']-b['q'], alive[:, i]) for i, (a,b) in enumerate(zip(*results))]
        metrics['twin_q_signed_gap'] = [r['mean'] for r in gap]
        metrics['twin_q_absolute_gap'] = [r['abs_mean'] for r in gap]
        target_stats = [scalar_statistics(targets[:, i], alive[:, i]) for i in range(self.num_agents)]
        for name in ('mean','std','min','max'):
            metrics['target_'+name+'_per_agent'] = [r[name] for r in target_stats]
        metrics['target_variance_per_agent'] = [None if r['std'] is None else r['std']**2 for r in target_stats]
        self.critic_update_count += 1
        delayed = self.critic_update_count % self.policy_delay == 0
        metrics['actor_updated'] = delayed
        metrics['actor_loss'] = None
        if delayed:
            actor_results = [self.update_actor(i, batch) for i in range(self.num_agents)]
            metrics['actor_losses'] = [r[0] for r in actor_results]
            metrics['actor_loss'] = float(np.mean(metrics['actor_losses']))
            metrics['actor_grad_norm_preclip'] = [d['pre'] for d in self._actor_diagnostics]
            metrics['actor_grad_norm_postclip'] = [d['post'] for d in self._actor_diagnostics]
            metrics['actor_grad_clipped'] = [d['clipped'] for d in self._actor_diagnostics]
            for online, target in ((self.actors,self.target_actors), (self.critics,self.target_critics), (self.critics2,self.target_critics2)):
                polyak_update(online, target, self.tau)
            self.actor_update_count += 1
        with torch.no_grad():
            obs = torch.where(alive.unsqueeze(-1).bool(), batch['observations'], torch.zeros_like(batch['observations']))
            actions = torch.stack([a(obs[:, i]) for i,a in enumerate(self.actors)], dim=1)
            metrics.update(action_statistics(actions, alive))
            metrics['actor_action_statistics_per_agent'] = [action_statistics(actions[:,i],alive[:,i]) for i in range(self.num_agents)]
        metrics.update(actor_update_count=self.actor_update_count, critic_update_count=self.critic_update_count,
            actor_updates_per_agent=list(self.actor_updates_per_agent), critic_updates_per_agent=list(self.critic_updates_per_agent),
            critic2_updates_per_agent=list(self.critic2_updates_per_agent),
            actor_optimizer_steps=sum(self.actor_updates_per_agent),
            critic_optimizer_steps=sum(self.critic_updates_per_agent)+sum(self.critic2_updates_per_agent))
        metrics['utd_scope'] = 'rounds and actual per-network optimizer steps / sampled joint environment transitions'
        for name, counts in (('actor',self.actor_updates_per_agent),('q1',self.critic_updates_per_agent),('q2',self.critic2_updates_per_agent)):
            metrics[name+'_optimizer_utd_per_agent'] = [v/max(1,self.sampled_steps) for v in counts]
        if self.critic_update_count <= 2 or self.critic_update_count % self.diagnostics_interval_updates == 0:
            metrics.update(self.parameter_diagnostics())
        return metrics

    @torch.no_grad()
    def parameter_diagnostics(self):
        actors, actor_lag = parameter_distances(self.actors,self.initial_actor_parameters,self.target_actors)
        critics, critic_lag = parameter_distances(self.critics,self.initial_critic_parameters,self.target_critics)
        critics2, critic2_lag = parameter_distances(self.critics2,self.initial_critic2_parameters,self.target_critics2)
        result = {'critic2_parameter_drift_l2':critics2, 'critic2_online_target_distance':critic2_lag,
            'actor_parameter_drift_l2':actors, 'critic_parameter_drift_l2':critics,
            'actor_online_target_distance':actor_lag, 'critic_online_target_distance':critic_lag,
            'parameter_diagnostics_update': self.actor_update_count}
        for i in range(self.num_agents):
            result[f'actor_parameter_drift_l2_{i}'] = actors[i]
            result[f'critic_parameter_drift_l2_{i}'] = critics[i]
        return result

    def parameter_counts(self):
        a = sum(p.numel() for p in self.actors[0].parameters())
        q = sum(p.numel() for p in self.critics[0].parameters())
        return {'actor_per_agent': a, 'actors_trainable': a*self.num_agents,
                'critic_per_agent': q, 'critics_trainable': 2*q*self.num_agents,
                'trainable_total': (a+2*q)*self.num_agents, 'target_total_nontrainable': (a+2*q)*self.num_agents}

    def checkpoint_state(self, extra):
        from .protocol import schedule_contract
        contract = schedule_contract(extra['algorithm_config'])
        return {'algorithm': 'matd3', 'implementation_version': MATD3_IMPL_VERSION,
            **contract, 'transitions_since_update': extra['transitions_since_update'],
            'update_event_count': extra['update_event_count'],
            'effective_utd': {'actor':self.actor_update_count/max(1,self.sampled_steps),'critic':self.critic_update_count/max(1,self.sampled_steps)},
            'initial_parameter_references': {'actors':list(self.initial_actor_parameters),'critics':list(self.initial_critic_parameters),'critics2':list(self.initial_critic2_parameters)},
            'sampled_steps': self.sampled_steps, 'vector_steps': self.vector_steps,
            'training_seed': self.seed, 'gamma': self.gamma, 'tau': self.tau,
            'network_architecture': self.network_architecture, 'parameter_counts':self.parameter_counts(),
            'target_smoothing_rng_state':self.target_smoothing_rng.get_state(),
            **{name: [m.state_dict() for m in getattr(self, name)] for name in
                ('actors', 'critics', 'target_actors', 'target_critics', 'critics2', 'target_critics2', 'actor_optimizers', 'critic_optimizers', 'critic2_optimizers')},
            'actor_update_count': self.actor_update_count, 'critic_update_count': self.critic_update_count,
            'actor_updates_per_agent': list(self.actor_updates_per_agent), 'critic_updates_per_agent': list(self.critic_updates_per_agent),
            'critic2_updates_per_agent':list(self.critic2_updates_per_agent),
            'torch_rng_state': torch.get_rng_state(),
            'python_rng_state': random.getstate(), 'numpy_rng_state': np.random.get_state(),
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
        if state['network_architecture'] != self.network_architecture or (state['gamma'], state['tau'], state['training_seed'],state['gradient_clip_norm']) != (self.gamma, self.tau, self.seed,self.gradient_clip_norm):
            raise RuntimeError('MATD3 trainer/checkpoint architecture or hyperparameter mismatch')
        for key in ('policy_delay', 'target_policy_noise', 'target_noise_clip'):
            if state[key] != getattr(self,key):
                raise RuntimeError(f'MATD3 trainer/checkpoint {key} mismatch')
        for name in ('actors', 'critics', 'target_actors', 'target_critics', 'critics2', 'target_critics2'):
            for module, saved in zip(getattr(self, name), state[name]):
                module.load_state_dict(saved, strict=True)
        for name in ('actor_optimizers', 'critic_optimizers', 'critic2_optimizers'):
            for optimizer, saved in zip(getattr(self, name), state[name]):
                optimizer.load_state_dict(saved)
        for name in ('sampled_steps', 'vector_steps', 'actor_update_count', 'critic_update_count',
                     'actor_updates_per_agent', 'critic_updates_per_agent', 'critic2_updates_per_agent'):
            setattr(self, name, deepcopy(state[name]))
        references = state['initial_parameter_references']
        self.initial_actor_parameters = tuple(v.detach().clone().to(self.device) for v in references['actors'])
        self.initial_critic_parameters = tuple(v.detach().clone().to(self.device) for v in references['critics'])
        self.initial_critic2_parameters = tuple(v.detach().clone().to(self.device) for v in references['critics2'])
        self.target_smoothing_rng.set_state(state['target_smoothing_rng_state'].cpu())
        return state['extra']

    def resume(self, *args, **kwargs):
        raise RuntimeError('formal exact resume unsupported: replay and environment state are not checkpointed')
