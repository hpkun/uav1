"""Five local intent-guided deterministic defenders with twin centralized MLP critics (IntentMATD3)."""
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .intent_matd3_config import IntentMATD3Config
from .maddpg import JointCritic
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


def _label_transition(item, state_seq, offset, *, n_attackers, horizons):
    """Attach time-t-anchored future enemy relative position targets using within-episode states."""
    n_horizons = len(horizons)
    max_index = len(state_seq) - 1
    att_end = 25 + 5 * n_attackers
    obs_t = item['observations']
    mask_t = item['active_masks']
    def_xy_t = state_seq[offset][:25].reshape(5, 5)[:, :2]
    enemy_vis_t = obs_t[:, 26:].reshape(5, n_attackers, 5)[:, :, 4] > 0.5
    future_pos = np.zeros((5, n_attackers, n_horizons, 2), dtype=np.float32)
    future_valid = np.zeros((5, n_attackers, n_horizons), dtype=np.float32)
    for k, h in enumerate(horizons):
        t_fut = offset + h
        if t_fut > max_index:
            continue
        att_block = state_seq[t_fut][25:att_end].reshape(n_attackers, 5)
        att_xy_fut = att_block[:, :2]
        att_alive_fut = att_block[:, 4] > 0.5
        # central_state divides positions by arena_half_extent; obs relative positions divide by 2 * arena_half_extent.
        rel_fut = (att_xy_fut[None, :, :] - def_xy_t[:, None, :]) * 0.5
        valid = (mask_t[:, None] > 0.5) & enemy_vis_t & att_alive_fut[None, :]
        future_pos[:, :, k, :] = np.where(valid[:, :, None], rel_fut, 0.0)
        future_valid[:, :, k] = valid.astype(np.float32)
    item['future_enemy_positions'] = future_pos
    item['future_enemy_valid'] = future_valid


def annotate_future_enemy_targets(episode_entries, *, n_attackers, horizons):
    """Populate per-step ground-truth future relative attacker positions from central_state."""
    if not episode_entries:
        return
    state_seq = [entry[0]['states'] for entry in episode_entries]
    state_seq.append(episode_entries[-1][0]['next_states'])
    for t, (item, _) in enumerate(episode_entries):
        _label_transition(item, state_seq, t, n_attackers=n_attackers, horizons=horizons)


class FutureTargetDelayQueue:
    """Bounded max(horizons)-step rolling delay queue for within-episode future trajectory labeling."""

    def __init__(self, *, n_attackers, horizons):
        self.n_attackers = int(n_attackers)
        self.horizons = list(horizons)
        self.max_horizon = max(self.horizons)
        self._buffer = deque()

    def __len__(self):
        return len(self._buffer)

    def push(self, item, metadata=None, *, done=False):
        self._buffer.append((item, metadata))
        ready = []
        if done:
            entries = list(self._buffer)
            self._buffer.clear()
            annotate_future_enemy_targets(entries, n_attackers=self.n_attackers, horizons=self.horizons)
            return entries
        if len(self._buffer) >= self.max_horizon:
            state_seq = [entry[0]['states'] for entry in self._buffer]
            state_seq.append(self._buffer[-1][0]['next_states'])
            oldest_item, oldest_meta = self._buffer.popleft()
            _label_transition(
                oldest_item,
                state_seq,
                0,
                n_attackers=self.n_attackers,
                horizons=self.horizons,
            )
            ready.append((oldest_item, oldest_meta))
        return ready


class IntentActor(nn.Module):
    """MLP policy augmented with an enemy trajectory predictor and single-layer prediction-guided focus."""

    def __init__(self, observation_dim, n_attackers, hidden_dim, intent_embed_dim, n_horizons):
        super().__init__()
        self.observation_dim = observation_dim
        self.n_attackers = n_attackers
        self.intent_embed_dim = intent_embed_dim
        self.n_horizons = n_horizons
        pred_dim = 2 * n_horizons
        # 1. Shared per-enemy future trajectory predictor (supervised only; detached before attention).
        self.predictor = nn.Sequential(
            nn.Linear(6 + 5, intent_embed_dim),
            nn.ReLU(),
            nn.Linear(intent_embed_dim, intent_embed_dim),
            nn.ReLU(),
            nn.Linear(intent_embed_dim, pred_dim),
        )
        # 2. Single-layer prediction-guided enemy attention with always-valid null fallback token.
        self.query_proj = nn.Linear(26, intent_embed_dim)
        self.key_proj = nn.Linear(5 + pred_dim, intent_embed_dim)
        self.val_proj = nn.Linear(5 + pred_dim, intent_embed_dim)
        self.null_key = nn.Parameter(torch.randn(1, 1, intent_embed_dim) * 0.02)
        self.null_val = nn.Parameter(torch.zeros(1, 1, intent_embed_dim))
        # 3. Standard BoxActor-style MLP head conditioned on [obs, c_intent].
        self.policy_net = nn.Sequential(
            nn.Linear(observation_dim + intent_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self._scale = 1.0 / math.sqrt(intent_embed_dim)

    def policy_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith('predictor.')]

    def forward_with_intent(self, observation):
        batch = observation.shape[0]
        own = observation[:, :6]
        self_ally = observation[:, :26]
        enemies = observation[:, 26:].reshape(batch, self.n_attackers, 5)
        enemy_valid = enemies[:, :, 4] > 0.5

        own_exp = own.unsqueeze(1).expand(-1, self.n_attackers, -1)
        pred_flat = self.predictor(torch.cat((own_exp, enemies), dim=-1))
        pred = pred_flat.reshape(batch, self.n_attackers, self.n_horizons, 2)

        # Stop-gradient prevents RL policy gradients from corrupting trajectory prediction.
        enemy_kv = torch.cat((enemies, pred_flat.detach()), dim=-1)
        q = self.query_proj(self_ally).unsqueeze(1)
        k = torch.cat((self.null_key.expand(batch, 1, -1), self.key_proj(enemy_kv)), dim=1)
        v = torch.cat((self.null_val.expand(batch, 1, -1), self.val_proj(enemy_kv)), dim=1)
        one = torch.ones((batch, 1), dtype=torch.bool, device=observation.device)
        valid = torch.cat((one, enemy_valid), dim=1)
        scores = (q * k).sum(dim=-1) * self._scale
        scores = scores.masked_fill(~valid, -1e9)
        weights = torch.softmax(scores, dim=-1)
        c_intent = (weights.unsqueeze(-1) * v).sum(dim=1)
        action = torch.tanh(self.policy_net(torch.cat((observation, c_intent), dim=-1)))
        return action, pred, weights

    def forward(self, observation):
        action, _, _ = self.forward_with_intent(observation)
        return action


class IntentMATD3:
    algorithm = 'intent_matd3'
    schema = 'interception_intent_matd3_1'

    def __init__(self, config=None, device='cpu', seed=0, *, environment_config, environment_contract):
        self.config = config or IntentMATD3Config()
        if not isinstance(self.config, IntentMATD3Config):
            raise ValueError('IntentMATD3 requires IntentMATD3Config')
        self.environment_config = deepcopy(environment_config)
        self.environment_spec = self.validate_environment_spec(environment_contract)
        self.n_attackers = int(self.environment_spec['n_attackers'])
        self.n_horizons = len(self.config.prediction_horizons)
        self.device = torch.device(device)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        h = self.config.hidden_dim
        obs_dim = int(self.environment_spec['observation_dim'])
        state_dim = int(self.environment_spec['state_dim'])
        self.actors = nn.ModuleList([
            IntentActor(obs_dim, self.n_attackers, h, self.config.intent_embed_dim, self.n_horizons)
            for _ in range(5)
        ]).to(self.device)
        self.critics_1 = nn.ModuleList([JointCritic(state_dim, h) for _ in range(5)]).to(self.device)
        self.critics_2 = nn.ModuleList([JointCritic(state_dim, h) for _ in range(5)]).to(self.device)
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
        if (
            not isinstance(contract, dict)
            or (
                contract.get('version'),
                contract.get('n_agents'),
                contract.get('n_attackers'),
                contract.get('observation_dim'),
                contract.get('state_dim'),
            )
            not in (
                ('interception_14', 5, 5, 51, 60),
                ('interception_15', 5, 3, 41, 50),
            )
            or contract.get('action_dim') != 2
            or contract.get('action_schema') != 'normalized_speed_turn_box'
        ):
            raise ValueError('IntentMATD3 requires Interception/14 (5v5) or Interception/15 (5v3)')
        return deepcopy(contract)

    def _joint(self, observations, masks, *, target=False):
        actors = self.target_actors if target else self.actors
        return torch.stack([actor(observations[:, i]) for i, actor in enumerate(actors)], 1) * masks[..., None]

    def _smoothed_target_actions(self, next_observations, next_masks):
        next_actions = self._joint(next_observations, next_masks, target=True)
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
        return next_actions

    @torch.no_grad()
    def policy_actions(self, observations, active_masks, *, deterministic=True, return_intent=False):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(active_masks, dtype=torch.float32, device=self.device)
        if (
            obs.ndim != 3
            or obs.shape[1:] != (5, self.environment_spec['observation_dim'])
            or mask.shape != obs.shape[:2]
        ):
            raise ValueError('IntentMATD3 observation/mask shape')
        if return_intent:
            acts, preds, weights = [], [], []
            for i, actor in enumerate(self.actors):
                a, p, w = actor.forward_with_intent(obs[:, i])
                acts.append(a)
                preds.append(p)
                weights.append(w)
            actions = (torch.stack(acts, 1) * mask[..., None]).cpu().numpy()
            pred_arr = torch.stack(preds, 1).cpu().numpy()
            weight_arr = torch.stack(weights, 1).cpu().numpy()
        else:
            actions = self._joint(obs, mask).cpu().numpy()
        if not deterministic:
            std = (self.config.exploration_std_speed, self.config.exploration_std_turn)
            actions += self.rng.normal(0, std, actions.shape).astype(np.float32)
            np.clip(actions, -1, 1, out=actions)
            actions *= mask.cpu().numpy()[..., None]
        actions = actions.astype(np.float32, copy=False)
        if return_intent:
            return actions, pred_arr, weight_arr
        return actions

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
            future_enemy_positions=(size, 5, self.n_attackers, self.n_horizons, 2),
            future_enemy_valid=(size, 5, self.n_attackers, self.n_horizons),
        )
        if size < 1 or set(b) != set(shapes) or any(
            b[k].shape != shape or not torch.isfinite(b[k]).all() for k, shape in shapes.items()
        ):
            raise ValueError('Invalid IntentMATD3 joint replay batch')
        if any(
            not ((b[k] == 0) | (b[k] == 1)).all()
            for k in ('dones', 'active_masks', 'next_active_masks', 'future_enemy_valid')
        ):
            raise ValueError('IntentMATD3 masks/dones must be binary')
        if (b['actions'].abs() > 1 + 1e-4).any():
            raise ValueError('IntentMATD3 normalized actions exceed [-1,1]')
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
                raise ValueError('Invalid IntentMATD3 importance weights')

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
            raise FloatingPointError('Nonfinite IntentMATD3 critic loss')

        grads_1 = self._step_critic_group(self.critics_1, self.critic_1_optimizer, critic_1_loss, mask)
        grads_2 = self._step_critic_group(self.critics_2, self.critic_2_optimizer, critic_2_loss, mask)
        individual_critic_grads = grads_1 + grads_2
        self.critic_update_count += 1

        actor_losses = []
        actor_grads = []
        pred_losses = []
        pred_valid_rates = {}
        if self.critic_update_count % self.config.policy_delay == 0:
            active_valid = actor_data['future_enemy_valid'][actor_mask.bool()]
            if active_valid.numel():
                per_h = active_valid.mean(dim=(0, 1))
                for k, h_step in enumerate(self.config.prediction_horizons):
                    pred_valid_rates[f'pred_valid_h{h_step}'] = float(per_h[k])
            with torch.no_grad():
                joint = self._joint(actor_data['observations'], actor_mask).detach()
            self.critics_1.requires_grad_(False)
            try:
                for i, (actor, opt) in enumerate(zip(self.actors, self.actor_optimizers)):
                    active = actor_mask[:, i].bool()
                    if not active.any():
                        continue
                    act_i, pred_i, _ = actor.forward_with_intent(actor_data['observations'][active, i])
                    candidate = joint[active].clone()
                    candidate[:, i] = act_i
                    policy_loss = -self.critics_1[i](actor_data['states'][active], candidate).mean()
                    total_loss = policy_loss

                    valid_i = actor_data['future_enemy_valid'][active, i]
                    valid_count = valid_i.sum()
                    if valid_count.item() > 0:
                        target_i = actor_data['future_enemy_positions'][active, i]
                        elem_loss = F.smooth_l1_loss(pred_i, target_i, reduction='none').mean(dim=-1)
                        pred_loss = (elem_loss * valid_i).sum() / valid_count
                        pred_losses.append(float(pred_loss.detach()))
                        if self.config.prediction_loss_weight > 0:
                            total_loss = total_loss + self.config.prediction_loss_weight * pred_loss

                    if not torch.isfinite(total_loss):
                        raise FloatingPointError('Nonfinite IntentMATD3 actor/prediction loss')
                    opt.zero_grad(set_to_none=True)
                    total_loss.backward()
                    # Clip disjoint policy and predictor parameter groups independently so pred_loss never steals RL clip budget.
                    policy_grad = nn.utils.clip_grad_norm_(
                        actor.policy_parameters(), self.config.actor_max_grad_norm
                    )
                    nn.utils.clip_grad_norm_(actor.predictor.parameters(), self.config.actor_max_grad_norm)
                    opt.step()
                    self.actor_optimizer_steps[i] += 1
                    actor_losses.append(float(policy_loss.detach()))
                    actor_grads.append(float(policy_grad))
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
            pred_loss=float(np.mean(pred_losses)) if pred_losses else None,
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
        for h_step in self.config.prediction_horizons:
            metrics[f'pred_valid_h{h_step}'] = pred_valid_rates.get(f'pred_valid_h{h_step}')
        if not np.isfinite([v for v in metrics.values() if v is not None]).all():
            raise FloatingPointError('Nonfinite IntentMATD3 metrics')
        return (metrics, td_errors.cpu().numpy()) if return_td_errors else metrics

    def save(self, path, *, global_step, environment_config):
        if environment_config != self.environment_config:
            raise ValueError('IntentMATD3 environment mismatch at save')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                algorithm=self.algorithm,
                checkpoint_schema=self.schema,
                environment_spec=self.environment_spec,
                environment_config=self.environment_config,
                intent_matd3_config=asdict(self.config),
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
                intent_matd3_config=asdict(self.config),
            )
            required = set(expected) | set(networks) | {'global_step', 'provenance', 'checkpoint_scope'}
            if not isinstance(payload, dict) or not required <= payload.keys() or any(
                payload.get(k) != v for k, v in expected.items()
            ):
                raise ValueError('Incompatible IntentMATD3 schema/config/environment contract')
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
                    raise ValueError('Incomplete IntentMATD3 training state')
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
