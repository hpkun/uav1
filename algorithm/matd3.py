"""Five local deterministic defenders with twin centralized joint critics (MATD3)."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from .maddpg import BoxActor, DiskActor, JointCritic, MADDPG
from .matd3_config import MATD3Config, NormalizedMATD3Config
from .state import (
    loading_transaction,
    validate_counter,
    validate_cuda_rng,
    validate_generator_rng,
    validate_model_state,
    validate_numpy_rng,
    validate_optimizer_state,
    validate_python_rng,
    validate_torch_rng,
)


class MATD3:
    algorithm = 'matd3'
    schema = 'interception_matd3_1'

    def __init__(self, config=None, device='cpu', seed=0, *, environment_config, environment_contract):
        self.config = config or MATD3Config()
        if not isinstance(self.config, MATD3Config):
            raise ValueError('MATD3 requires MATD3Config')
        self.environment_config = deepcopy(environment_config)
        self.environment_spec = self.validate_environment_spec(environment_contract)
        self.device = torch.device(device)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        h = self.config.hidden_dim
        self.normalized_actions = self.environment_spec['action_schema'] == 'normalized_speed_turn_box'
        if self.normalized_actions and not isinstance(self.config, NormalizedMATD3Config):
            raise ValueError('Normalized speed-turn actions require NormalizedMATD3Config')
        actor_type = (
            (lambda: BoxActor(self.environment_spec['observation_dim'], h))
            if self.normalized_actions
            else (lambda: DiskActor(
                self.environment_spec['observation_dim'],
                h,
                float(self.environment_spec['max_acceleration']),
            ))
        )
        self.actors = nn.ModuleList([actor_type() for _ in range(5)]).to(self.device)
        self.critics_1 = nn.ModuleList(
            [JointCritic(self.environment_spec['state_dim'], h) for _ in range(5)]
        ).to(self.device)
        self.critics_2 = nn.ModuleList(
            [JointCritic(self.environment_spec['state_dim'], h) for _ in range(5)]
        ).to(self.device)
        self.target_actors = deepcopy(self.actors).requires_grad_(False).eval()
        self.target_critics_1 = deepcopy(self.critics_1).requires_grad_(False).eval()
        self.target_critics_2 = deepcopy(self.critics_2).requires_grad_(False).eval()
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=self.config.actor_learning_rate)
            for actor in self.actors
        ]
        self.critic_1_optimizer = torch.optim.Adam(
            self.critics_1.parameters(), lr=self.config.critic_learning_rate
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critics_2.parameters(), lr=self.config.critic_learning_rate
        )
        self.actor_update_count = 0
        self.critic_update_count = 0
        self.target_update_count = 0
        self.actor_optimizer_steps = [0] * 5

    @staticmethod
    def validate_environment_spec(contract):
        return MADDPG.validate_environment_spec(contract)

    def _joint(self, observations, masks, *, target=False):
        actors = self.target_actors if target else self.actors
        return torch.stack([actor(observations[:, i]) for i, actor in enumerate(actors)], 1) * masks[..., None]

    def _smoothed_target_actions(self, next_observations, next_masks):
        next_actions = self._joint(next_observations, next_masks, target=True)
        if self.normalized_actions:
            std = np.array(
                [self.config.target_noise_std_speed, self.config.target_noise_std_turn],
                dtype=np.float32,
            )
            clip = np.array(
                [self.config.target_noise_clip_speed, self.config.target_noise_clip_turn],
                dtype=np.float32,
            )
            if np.any(std > 0) and np.any(clip > 0):
                noise = self.rng.normal(0.0, std, size=next_actions.shape).astype(np.float32)
                np.clip(noise, -clip, clip, out=noise)
                next_actions = (
                    next_actions + torch.as_tensor(noise, dtype=torch.float32, device=self.device)
                ).clamp(-1.0, 1.0)
                next_actions = next_actions * next_masks[..., None]
        elif self.config.target_noise_std_mps2 > 0 and self.config.target_noise_clip_mps2 > 0:
            clip = float(self.config.target_noise_clip_mps2)
            noise = self.rng.normal(
                0.0, self.config.target_noise_std_mps2, size=next_actions.shape
            ).astype(np.float32)
            np.clip(noise, -clip, clip, out=noise)
            next_actions = next_actions + torch.as_tensor(noise, dtype=torch.float32, device=self.device)
            radius = float(self.environment_spec['max_acceleration'])
            norms = next_actions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            next_actions = next_actions * torch.clamp(radius / norms, max=1.0)
            next_actions = next_actions * next_masks[..., None]
        return next_actions

    @torch.no_grad()
    def policy_actions(self, observations, active_masks, *, deterministic=True):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(active_masks, dtype=torch.float32, device=self.device)
        if obs.ndim != 3 or obs.shape[1:] != (5, self.environment_spec['observation_dim']) or mask.shape != obs.shape[:2]:
            raise ValueError('MATD3 observation/mask shape')
        actions = self._joint(obs, mask).cpu().numpy()
        if not deterministic:
            if self.normalized_actions:
                std = (self.config.exploration_std_speed, self.config.exploration_std_turn)
                actions += self.rng.normal(0, std, actions.shape).astype(np.float32)
                np.clip(actions, -1, 1, out=actions)
                actions *= mask.cpu().numpy()[..., None]
            elif self.config.exploration_std_mps2:
                actions += self.rng.normal(0, self.config.exploration_std_mps2, actions.shape).astype(np.float32)
                radius = self.environment_spec['max_acceleration']
                norms = np.linalg.norm(actions, axis=-1, keepdims=True)
                actions *= np.minimum(1.0, radius / np.maximum(norms, 1e-12))
                actions *= mask.cpu().numpy()[..., None]
        return actions.astype(np.float32, copy=False)

    def _batch(self, batch):
        b = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}
        size = b['rewards'].shape[0] if 'rewards' in b and b['rewards'].ndim == 2 else 0
        shapes = dict(
            observations=(size, 5, self.environment_spec['observation_dim']),
            next_observations=(size, 5, self.environment_spec['observation_dim']),
            states=(size, self.environment_spec['state_dim']),
            next_states=(size, self.environment_spec['state_dim']),
            actions=(size, 5, 2),
            rewards=(size, 5),
            dones=(size,),
            active_masks=(size, 5),
            next_active_masks=(size, 5),
        )
        if size < 1 or set(b) != set(shapes) or any(
            b[k].shape != shape or not torch.isfinite(b[k]).all() for k, shape in shapes.items()
        ):
            raise ValueError('Invalid MATD3 joint replay batch')
        if any(not ((b[k] == 0) | (b[k] == 1)).all() for k in ('dones', 'active_masks', 'next_active_masks')):
            raise ValueError('MATD3 masks/dones must be binary')
        if self.normalized_actions:
            if (b['actions'].abs() > 1 + 1e-4).any():
                raise ValueError('MATD3 normalized actions exceed [-1,1]')
        elif (b['actions'].norm(dim=-1) > self.environment_spec['max_acceleration'] + 1e-4).any():
            raise ValueError('MATD3 actions exceed physical disk')
        return b

    def _step_critic_group(self, critics, optimizer, loss, mask):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = []
        for i, critic in enumerate(critics):
            if not mask[:, i].bool().any():
                for parameter in critic.parameters():
                    parameter.grad = None
                continue
            grads.append(float(nn.utils.clip_grad_norm_(critic.parameters(), self.config.critic_max_grad_norm)))
        optimizer.step()
        return grads

    def update(self, batch, *, actor_batch=None, critic_weights=None, return_td_errors=False):
        b = self._batch(batch)
        mask = b['active_masks']
        actor_data = b if actor_batch is None else self._batch(actor_batch)
        actor_mask = actor_data['active_masks']
        if critic_weights is not None:
            weights = torch.as_tensor(critic_weights, dtype=torch.float32, device=self.device)
            if weights.shape != (len(mask),) or not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError('Invalid MATD3 importance weights')

        with torch.no_grad():
            next_actions = self._smoothed_target_actions(b['next_observations'], b['next_active_masks'])
            next_q1 = torch.stack([critic(b['next_states'], next_actions) for critic in self.target_critics_1], 1)
            next_q2 = torch.stack([critic(b['next_states'], next_actions) for critic in self.target_critics_2], 1)
            next_q = torch.minimum(next_q1, next_q2)
            targets = b['rewards'] + self.config.gamma * (1.0 - b['dones'])[:, None] * b['next_active_masks'] * next_q

        q1 = torch.stack([critic(b['states'], b['actions']) for critic in self.critics_1], 1)
        q2 = torch.stack([critic(b['states'], b['actions']) for critic in self.critics_2], 1)
        active_mask_bool = mask.bool()
        active_q1 = q1.detach()[active_mask_bool]
        active_q2 = q2.detach()[active_mask_bool]
        active_q = 0.5 * (active_q1 + active_q2)
        active_targets = targets[active_mask_bool]

        td1 = q1 - targets
        td2 = q2 - targets
        td_abs_pair = 0.5 * (td1.detach().abs() + td2.detach().abs())
        td_errors = (td_abs_pair * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        sq1 = td1.square() * mask
        sq2 = td2.square() * mask
        if critic_weights is not None:
            sq1 = sq1 * weights[:, None]
            sq2 = sq2 * weights[:, None]
        denom = mask.sum().clamp_min(1.0)
        critic_1_loss = sq1.sum() / denom
        critic_2_loss = sq2.sum() / denom
        critic_loss = critic_1_loss + critic_2_loss
        if not (torch.isfinite(critic_1_loss) and torch.isfinite(critic_2_loss)):
            raise FloatingPointError('Nonfinite MATD3 critic loss')

        grads_1 = self._step_critic_group(self.critics_1, self.critic_1_optimizer, critic_1_loss, mask)
        grads_2 = self._step_critic_group(self.critics_2, self.critic_2_optimizer, critic_2_loss, mask)
        individual_critic_grads = grads_1 + grads_2
        self.critic_update_count += 1

        actor_losses = []
        actor_grads = []
        if self.critic_update_count % self.config.policy_delay == 0:
            with torch.no_grad():
                joint = self._joint(actor_data['observations'], actor_mask).detach()
            self.critics_1.requires_grad_(False)
            try:
                for i, (actor, opt) in enumerate(zip(self.actors, self.actor_optimizers)):
                    active = actor_mask[:, i].bool()
                    if not active.any():
                        continue
                    candidate = joint[active].clone()
                    candidate[:, i] = actor(actor_data['observations'][active, i])
                    loss = -self.critics_1[i](actor_data['states'][active], candidate).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite MATD3 actor loss')
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    grad = nn.utils.clip_grad_norm_(actor.parameters(), self.config.actor_max_grad_norm)
                    opt.step()
                    self.actor_optimizer_steps[i] += 1
                    actor_losses.append(float(loss.detach()))
                    actor_grads.append(float(grad))
            finally:
                self.critics_1.requires_grad_(True)
            self.actor_update_count += 1
            with torch.no_grad():
                for online, target in (
                    (self.actors, self.target_actors),
                    (self.critics_1, self.target_critics_1),
                    (self.critics_2, self.target_critics_2),
                ):
                    for source, destination in zip(online.parameters(), target.parameters()):
                        destination.mul_(1.0 - self.config.tau).add_(source, alpha=self.config.tau)
            self.target_update_count += 1

        critic_grad = float(np.mean(individual_critic_grads)) if individual_critic_grads else 0.0
        metrics = dict(
            actor_loss=float(np.mean(actor_losses)) if actor_losses else None,
            critic_loss=float(critic_loss.detach()),
            critic_1_loss=float(critic_1_loss.detach()),
            critic_2_loss=float(critic_2_loss.detach()),
            actor_grad_norm=float(np.mean(actor_grads)) if actor_grads else None,
            actor_clip_rate=(
                float(any(value > self.config.actor_max_grad_norm for value in actor_grads))
                if actor_grads
                else None
            ),
            critic_grad_norm=critic_grad,
            critic_clip_rate=float(any(value > self.config.critic_max_grad_norm for value in individual_critic_grads)),
            active_critic_count=len(grads_1),
            critic_grad_norm_mean_agent=critic_grad,
            critic_grad_norm_max_agent=float(np.max(individual_critic_grads)) if individual_critic_grads else 0.0,
            td_abs=float((td_abs_pair * mask).sum() / denom),
            q_mean=float(active_q.mean()) if active_q.numel() else 0.0,
            q_std=float(active_q.std(unbiased=False)) if active_q.numel() else 0.0,
            q1_mean=float(active_q1.mean()) if active_q1.numel() else 0.0,
            q2_mean=float(active_q2.mean()) if active_q2.numel() else 0.0,
            q_gap_abs=float((active_q1 - active_q2).abs().mean()) if active_q1.numel() else 0.0,
            target_q_mean=float(active_targets.mean()) if active_targets.numel() else 0.0,
            actor_updates=self.actor_update_count,
            critic_updates=self.critic_update_count,
            target_updates=self.target_update_count,
            actor_optimizer_steps=sum(self.actor_optimizer_steps),
        )
        if not np.isfinite([v for v in metrics.values() if v is not None]).all():
            raise FloatingPointError('Nonfinite MATD3 metrics')
        return (metrics, td_errors.cpu().numpy()) if return_td_errors else metrics

    def save(self, path, *, global_step, environment_config):
        if environment_config != self.environment_config:
            raise ValueError('MATD3 environment mismatch at save')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                algorithm=self.algorithm,
                checkpoint_schema=self.schema,
                environment_spec=self.environment_spec,
                environment_config=self.environment_config,
                matd3_config=asdict(self.config),
                global_step=int(global_step),
                provenance=dict(
                    getattr(self, 'experiment_provenance', {}),
                    seed=self.seed,
                    output_directory=str(path.parent.resolve()),
                ),
                checkpoint_scope='inference/evaluation and learner state only; no replay, workers or exact training resume',
                actors=self.actors.state_dict(),
                critics_1=self.critics_1.state_dict(),
                critics_2=self.critics_2.state_dict(),
                target_actors=self.target_actors.state_dict(),
                target_critics_1=self.target_critics_1.state_dict(),
                target_critics_2=self.target_critics_2.state_dict(),
                actor_optimizers=[o.state_dict() for o in self.actor_optimizers],
                critic_1_optimizer=self.critic_1_optimizer.state_dict(),
                critic_2_optimizer=self.critic_2_optimizer.state_dict(),
                actor_update_count=self.actor_update_count,
                critic_update_count=self.critic_update_count,
                target_update_count=self.target_update_count,
                actor_optimizer_steps=self.actor_optimizer_steps,
                exploration_rng_state=deepcopy(self.rng.bit_generator.state),
                torch_rng_state=torch.random.get_rng_state(),
                cuda_rng_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy_rng_state=np.random.get_state(),
                python_rng_state=random.getstate(),
            ),
            path,
        )

    def load(self, path, *, load_optimizers=True):
        networks = (
            'actors',
            'critics_1',
            'critics_2',
            'target_actors',
            'target_critics_1',
            'target_critics_2',
        )
        counters = (
            'actor_update_count',
            'critic_update_count',
            'target_update_count',
            'actor_optimizer_steps',
        )
        with loading_transaction(
            tuple(getattr(self, n) for n in networks),
            (*self.actor_optimizers, self.critic_1_optimizer, self.critic_2_optimizer),
            owner=self,
            counters=(*counters, 'global_step'),
            generator_attr='rng',
            inference=not load_optimizers,
        ):
            payload = torch.load(path, map_location=self.device, weights_only=False)
            expected = dict(
                algorithm=self.algorithm,
                checkpoint_schema=self.schema,
                environment_spec=self.environment_spec,
                environment_config=self.environment_config,
                matd3_config=asdict(self.config),
            )
            required = set(expected) | set(networks) | {'global_step', 'provenance', 'checkpoint_scope'}
            if not isinstance(payload, dict) or not required <= payload.keys() or any(
                payload.get(k) != v for k, v in expected.items()
            ):
                raise ValueError('Incompatible MATD3 schema/config/environment contract')
            validate_counter(payload['global_step'], 'global_step')
            for name in networks:
                validate_model_state(getattr(self, name), payload[name])
            if load_optimizers:
                training = set(counters) | {
                    'actor_optimizers',
                    'critic_1_optimizer',
                    'critic_2_optimizer',
                    'exploration_rng_state',
                    'torch_rng_state',
                    'cuda_rng_states',
                    'numpy_rng_state',
                    'python_rng_state',
                }
                if (
                    not training <= payload.keys()
                    or len(payload['actor_optimizers']) != 5
                    or len(payload['actor_optimizer_steps']) != 5
                ):
                    raise ValueError('Incomplete MATD3 training state')
                for name in counters[:-1]:
                    validate_counter(payload[name], name)
                for value in payload['actor_optimizer_steps']:
                    validate_counter(value, 'actor_optimizer_steps')
                for optimizer, state in zip(self.actor_optimizers, payload['actor_optimizers']):
                    validate_optimizer_state(optimizer, state)
                validate_optimizer_state(self.critic_1_optimizer, payload['critic_1_optimizer'])
                validate_optimizer_state(self.critic_2_optimizer, payload['critic_2_optimizer'])
                validate_generator_rng(self.rng, payload['exploration_rng_state'])
                validate_torch_rng(payload['torch_rng_state'])
                validate_cuda_rng(payload['cuda_rng_states'], self.device)
                validate_numpy_rng(payload['numpy_rng_state'])
                validate_python_rng(payload['python_rng_state'])
            for name in networks:
                getattr(self, name).load_state_dict(payload[name], strict=True)
            if load_optimizers:
                for opt, state in zip(self.actor_optimizers, payload['actor_optimizers']):
                    opt.load_state_dict(state)
                self.critic_1_optimizer.load_state_dict(payload['critic_1_optimizer'])
                self.critic_2_optimizer.load_state_dict(payload['critic_2_optimizer'])
                self.rng.bit_generator.state = deepcopy(payload['exploration_rng_state'])
                torch.random.set_rng_state(payload['torch_rng_state'].cpu())
                if payload['cuda_rng_states'] and self.device.type == 'cuda':
                    torch.cuda.set_rng_state_all([s.cpu() for s in payload['cuda_rng_states']])
                np.random.set_state(payload['numpy_rng_state'])
                random.setstate(payload['python_rng_state'])
                for name in counters:
                    setattr(self, name, payload[name])
            return payload
