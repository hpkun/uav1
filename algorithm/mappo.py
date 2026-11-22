from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .mappo_config import MAPPOConfig
from .mappo_models import (CentralizedCritic, ContinuousGaussianActor, ACTION_PROBABILITY_VERSION,
    TANH_ACTION,DISK_ACTION,transform_raw_numpy)
from .state import (validate_model_state,validate_optimizer_state,validate_counter,
    validate_torch_rng,validate_generator_rng,loading_transaction)


class RunningValueNorm:
    """Running scalar return statistics; critic outputs normalized values."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    @property
    def std(self):
        return max((self.m2 / self.count) ** 0.5, 1e-4) if self.count else 1.0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('ValueNorm requires finite nonempty returns')
        count = len(values)
        mean = float(values.mean())
        m2 = float(np.square(values - mean).sum())
        total = self.count + count
        delta = mean - self.mean
        self.mean += delta * count / total
        self.m2 += m2 + delta * delta * self.count * count / total
        self.count = total

    def normalize(self, values):
        return (values - self.mean) / self.std

    def denormalize(self, values):
        return values * self.std + self.mean

    def state_dict(self):
        return dict(count=self.count, mean=self.mean, m2=self.m2)

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or set(state) != {'count', 'mean', 'm2'} or
                type(state['count']) is not int or state['count'] < 0 or
                not all(isinstance(state[k], (int, float)) and np.isfinite(state[k]) for k in ('mean', 'm2')) or
                state['m2'] < 0):
            raise ValueError('Invalid MAPPO ValueNorm state')
        self.count, self.mean, self.m2 = state['count'], float(state['mean']), float(state['m2'])


class RolloutBuffer:
    """Aligned [time, worker, agent] records; old logp is a Normal latent density."""

    action_probability_version = ACTION_PROBABILITY_VERSION

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        num_agents: int,
        observation_dim: int,
        state_dim: int,
        action_dim: int = 2,
        *, action_transform: str = TANH_ACTION, disk_radius: float = 8.,
    ) -> None:
        if action_transform not in (TANH_ACTION,DISK_ACTION) or action_transform==DISK_ACTION and action_dim!=2:
            raise ValueError('Unsupported rollout action transform/dimension')
        self.action_transform=action_transform
        if not np.isfinite(disk_radius) or disk_radius<=0:raise ValueError('Positive finite disk radius required')
        self.disk_radius=float(disk_radius)
        self.horizon = int(horizon)
        self.num_envs = int(num_envs)
        self.num_agents = int(num_agents)
        shape = (self.horizon, self.num_envs)
        self.observations = np.zeros(
            shape + (num_agents, observation_dim), dtype=np.float32
        )
        self.states = np.zeros(shape + (state_dim,), dtype=np.float32)
        self.actions = np.zeros(shape + (num_agents, action_dim), dtype=np.float32)
        # NaN marks records that have not supplied the mandatory latent contract.
        self.raw_actions = np.full_like(self.actions, np.nan)
        self.old_log_probabilities = np.zeros(
            shape + (num_agents,), dtype=np.float32
        )
        self.values = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.dones = np.zeros(shape, dtype=np.float32)
        self.active_masks = np.zeros(shape + (num_agents,), dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)
        self.position = 0

    def insert(
        self,
        observations: np.ndarray,
        states: np.ndarray,
        actions: np.ndarray,
        log_probabilities: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        active_masks: np.ndarray,
        *, raw_actions: np.ndarray,
    ) -> None:
        if self.position >= self.horizon:
            raise RuntimeError("rollout buffer is full")
        index = self.position
        records = (("observations", observations), ("states", states),
                   ("actions", actions), ("raw_actions", raw_actions),
                   ("old_log_probabilities", log_probabilities), ("values", values),
                   ("rewards", rewards), ("dones", dones), ("active_masks", active_masks))
        # Reject broadcasting/time-worker swaps before mutating any stored row.
        converted = {}
        for name, value in records:
            array = np.asarray(value, dtype=np.float32)
            if array.shape != getattr(self, name)[index].shape or not np.isfinite(array).all():
                raise ValueError(f"Invalid rollout {name}: expected finite {getattr(self, name)[index].shape}")
            converted[name] = array
        mask = converted['active_masks']
        if not np.isin(mask, (0., 1.)).all():
            raise ValueError("Rollout active_masks must be binary")
        if not np.isin(converted['dones'], (0., 1.)).all():
            raise ValueError("Rollout dones must be binary")
        expected = transform_raw_numpy(converted['raw_actions'],self.action_transform,self.disk_radius) * mask[..., None]
        maximum=self.disk_radius if self.action_transform==DISK_ACTION else 1.
        if np.any(np.abs(converted['actions']) > maximum) or not np.allclose(
                converted['actions'], expected, rtol=0, atol=2*maximum*np.finfo(np.float32).eps):
            raise ValueError("Executed actions must equal the configured transform of raw_actions with the active mask")
        self.observations[index] = observations
        self.states[index] = states
        self.actions[index] = actions
        self.raw_actions[index] = raw_actions
        self.old_log_probabilities[index] = log_probabilities
        self.values[index] = values
        self.rewards[index] = rewards
        self.dones[index] = dones
        self.active_masks[index] = active_masks
        self.position += 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        if self.position != self.horizon:
            raise RuntimeError("GAE requires a complete rollout")
        next_values = np.asarray(last_values, dtype=np.float32)
        if next_values.shape != (self.num_envs,) or not np.isfinite(next_values).all():
            raise ValueError(f"Invalid rollout last_values: expected finite {(self.num_envs,)}")
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for timestep in reversed(range(self.horizon)):
            continuation = 1.0 - self.dones[timestep]
            delta = (
                self.rewards[timestep]
                + gamma * next_values * continuation
                - self.values[timestep]
            )
            gae = delta + gamma * gae_lambda * continuation * gae
            self.advantages[timestep] = gae
            next_values = self.values[timestep]
        self.returns[:] = self.advantages + self.values


class MAPPO:
    action_probability_version = ACTION_PROBABILITY_VERSION

    def __init__(
        self,
        observation_dim: int,
        state_dim: int,
        n_agents: int,
        config: MAPPOConfig,
        device: str | torch.device,
        seed: int = 0,
        *, action_dim: int = 2, environment_config: dict | None = None, environment_contract: dict | None = None,
    ) -> None:
        self.action_dim = action_dim
        self.environment_snapshot = environment_config
        self.environment_contract = environment_contract or {}
        if environment_contract is not None and any(environment_contract[k]!=v for k,v in
                dict(observation_dim=observation_dim,state_dim=state_dim,n_agents=n_agents,action_dim=action_dim).items()):
            raise ValueError('MAPPO supplied environment dimensions mismatch')
        self.config = config
        version=self.environment_contract.get('version')
        dims=(observation_dim,state_dim)
        if (version not in ('interception_1','interception_2') or
                (version=='interception_2' and dims!=(54,75)) or
                self.environment_contract.get('action_schema')!='world_acceleration_disk_si'):
            raise ValueError('MAPPO requires a supported interception action contract')
        self.action_transform=DISK_ACTION
        self.disk_radius=float(self.environment_contract['max_acceleration'])
        if not np.isfinite(self.disk_radius) or self.disk_radius<=0:
            raise ValueError('Invalid acceleration disk radius')
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.value_normalizer = RunningValueNorm()
        self.use_value_norm = True
        self.actor = ContinuousGaussianActor(observation_dim, config.hidden_size, action_dim,action_transform=self.action_transform,disk_radius=self.disk_radius).to(self.device)
        self.critic = CentralizedCritic(state_dim, config.hidden_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_learning_rate, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_learning_rate, eps=1e-5
        )

    @torch.no_grad()
    def act(
        self,
        observations: np.ndarray,
        states: np.ndarray,
        active_masks: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Historical three-item inference API; PPO collection uses act_rollout."""
        observations_t = torch.as_tensor(
            observations, device=self.device, dtype=torch.float32
        )
        states_t = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        masks_t = torch.as_tensor(
            active_masks, device=self.device, dtype=torch.float32
        )
        actions, log_probabilities = self.actor.sample(
            observations_t, deterministic=deterministic
        )
        actions = actions * masks_t[..., None]
        log_probabilities = log_probabilities * masks_t
        values = self.critic(states_t)
        if self.use_value_norm:
            values = self.value_normalizer.denormalize(values)
        return (
            actions.cpu().numpy(),
            log_probabilities.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def act_rollout(
        self, observations: np.ndarray, states: np.ndarray, active_masks: np.ndarray,
        *, deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Training record: executed actions, Normal old logp, values, raw actions.

        One draw produces both representations. Even inactive slots retain their
        actual latent; their executed actions/logp are masked out of the objective.
        """
        observations_t = torch.as_tensor(observations, device=self.device, dtype=torch.float32)
        states_t = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        masks_t = torch.as_tensor(active_masks, device=self.device, dtype=torch.float32)
        actions, raw_actions, logp = self.actor.sample_rollout(observations_t, deterministic)
        actions = actions * masks_t[..., None]
        logp = logp * masks_t
        values = self.critic(states_t)
        if self.use_value_norm:
            values = self.value_normalizer.denormalize(values)
        return (actions.cpu().numpy(), logp.cpu().numpy(),
                values.cpu().numpy(), raw_actions.cpu().numpy())

    @torch.no_grad()
    def policy_actions(
        self,
        observations: np.ndarray,
        active_masks: np.ndarray,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        observations_t = torch.as_tensor(
            observations, device=self.device, dtype=torch.float32
        )
        masks_t = torch.as_tensor(
            active_masks, device=self.device, dtype=torch.float32
        )
        actions, _ = self.actor.sample(observations_t, deterministic=deterministic)
        return (actions * masks_t[..., None]).cpu().numpy()

    @torch.no_grad()
    def values(self, states: np.ndarray) -> np.ndarray:
        states_t = torch.as_tensor(states, device=self.device, dtype=torch.float32)
        values = self.critic(states_t)
        if self.use_value_norm:
            values = self.value_normalizer.denormalize(values)
        return values.cpu().numpy()

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        if (buffer.action_probability_version != self.action_probability_version or buffer.action_transform!=self.action_transform
                or buffer.position != buffer.horizon or not np.isfinite(buffer.raw_actions).all()):
            raise ValueError("PPO requires a complete raw-action/Normal-logp rollout")
        cfg = self.config
        to_tensor = lambda value: torch.as_tensor(
            value, device=self.device, dtype=torch.float32
        )
        observations = to_tensor(buffer.observations)
        states = to_tensor(buffer.states)
        raw_actions = to_tensor(buffer.raw_actions)
        old_log_probabilities = to_tensor(buffer.old_log_probabilities)
        old_values = to_tensor(buffer.values)
        returns = to_tensor(buffer.returns)
        if self.use_value_norm:
            # Rollout values were recorded with the previous running statistics.
            # Value clipping must compare against that original normalized output.
            old_values = self.value_normalizer.normalize(old_values)
            self.value_normalizer.update(buffer.returns)
            returns = self.value_normalizer.normalize(returns)
        active_masks = to_tensor(buffer.active_masks)
        team_advantages = to_tensor(buffer.advantages)

        actor_advantages = team_advantages[..., None].expand(-1, -1, self.n_agents)
        valid_advantages = actor_advantages[active_masks > 0.5]
        advantage_raw_mean = float(valid_advantages.mean().cpu()) if valid_advantages.numel() else 0.0
        advantage_raw_std = (float(valid_advantages.std(unbiased=False).cpu())
                             if valid_advantages.numel() else 0.0)
        if valid_advantages.numel():
            actor_advantages = (
                actor_advantages - valid_advantages.mean()
            ) / valid_advantages.std(unbiased=False).clamp_min(1e-8)
        else:
            actor_advantages = torch.zeros_like(actor_advantages)

        transition_count = buffer.horizon * buffer.num_envs
        observations = observations.reshape(
            transition_count, self.n_agents, observations.shape[-1]
        )
        states = states.reshape(transition_count, states.shape[-1])
        raw_actions = raw_actions.reshape(transition_count, self.n_agents, self.action_dim)
        old_log_probabilities = old_log_probabilities.reshape(
            transition_count, self.n_agents
        )
        old_values = old_values.reshape(transition_count)
        returns = returns.reshape(transition_count)
        active_masks = active_masks.reshape(transition_count, self.n_agents)
        actor_advantages = actor_advantages.reshape(
            transition_count, self.n_agents
        )

        rows: list[dict[str, float]] = []
        for _ in range(cfg.update_epochs):
            order = self.rng.permutation(transition_count)
            for start in range(0, transition_count, cfg.minibatch_size):
                indices = torch.as_tensor(
                    order[start : start + cfg.minibatch_size],
                    device=self.device,
                    dtype=torch.long,
                )
                obs_batch = observations[indices]
                raw_action_batch = raw_actions[indices]
                old_log_batch = old_log_probabilities[indices]
                mask_batch = active_masks[indices]
                advantage_batch = actor_advantages[indices]

                new_log_probability, entropy = self.actor.evaluate_raw_actions(
                    obs_batch, raw_action_batch
                )
                log_ratio = new_log_probability - old_log_batch
                ratio = log_ratio.exp()
                unclipped = ratio * advantage_batch
                clipped = ratio.clamp(
                    1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio
                ) * advantage_batch
                actor_count = mask_batch.sum().clamp_min(1.0)
                actor_loss = -(
                    torch.minimum(unclipped, clipped) * mask_batch
                ).sum() / actor_count
                entropy_mean = (entropy * mask_batch).sum() / actor_count

                self.actor_optimizer.zero_grad(set_to_none=True)
                # Scenario tails can supply critic data with zero actor samples.
                # Do not move the actor via Adam momentum on an empty batch.
                if valid_advantages.numel() and ('scenario_protocol' not in self.environment_contract
                                                 or bool(mask_batch.any())):
                    (actor_loss - cfg.entropy_coefficient * entropy_mean).backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        self.actor.parameters(), cfg.max_grad_norm
                    )
                    self.actor_optimizer.step()
                else:
                    actor_grad_norm = torch.zeros((), device=self.device)

                state_batch = states[indices]
                return_batch = returns[indices]
                old_value_batch = old_values[indices]
                values = self.critic(state_batch)
                clipped_values = old_value_batch + (values - old_value_batch).clamp(
                    -cfg.value_clip, cfg.value_clip
                )
                critic_loss = 0.5 * torch.maximum(
                    (values - return_batch).square(),
                    (clipped_values - return_batch).square(),
                ).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (cfg.value_coefficient * critic_loss).backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), cfg.max_grad_norm
                )
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = (
                        ((ratio - 1.0) - log_ratio) * mask_batch
                    ).sum() / actor_count
                    clip_fraction = (
                        ((ratio - 1.0).abs() > cfg.clip_ratio).float() * mask_batch
                    ).sum() / actor_count
                rows.append(
                    {
                        "actor_loss": float(actor_loss.detach().cpu()),
                        "critic_loss": float(critic_loss.detach().cpu()),
                        "entropy": float(entropy_mean.detach().cpu()),
                        "approx_kl": float(approx_kl.detach().cpu()),
                        "clip_fraction": float(clip_fraction.detach().cpu()),
                        "actor_grad_norm": float(actor_grad_norm.detach().cpu()),
                        "critic_grad_norm": float(critic_grad_norm.detach().cpu()),
                    }
                )

        with torch.no_grad():
            new_values = self.critic(states)
            if self.use_value_norm:
                new_values = self.value_normalizer.denormalize(new_values)
            raw_returns = to_tensor(buffer.returns).reshape(transition_count)
            return_variance = torch.var(raw_returns, unbiased=False)
            if float(return_variance) < 1e-8:
                explained_variance = torch.zeros((), device=self.device)
            else:
                explained_variance = 1.0 - torch.var(
                    raw_returns - new_values, unbiased=False
                ) / return_variance
        metrics = {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
        }
        metrics["explained_variance"] = float(explained_variance.cpu())
        if self.use_value_norm:
            metrics['value_norm_mean'] = self.value_normalizer.mean
            metrics['value_norm_std'] = self.value_normalizer.std
        metrics['policy_log_std_mean'] = float(self.actor.log_std.detach().mean().cpu())
        metrics['policy_std_mean'] = float(self.actor.log_std.detach().clamp(-5.0, 1.0).exp().mean().cpu())
        metrics['advantage_raw_mean'] = advantage_raw_mean
        metrics['advantage_raw_std'] = advantage_raw_std
        if not np.all(np.isfinite(list(metrics.values()))):
            raise FloatingPointError(f"Non-finite MAPPO metrics: {metrics}")
        return metrics

    def save(
        self,
        path: str | Path,
        *,
        global_step: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        extra = dict(extra or {}, action_probability_version=self.action_probability_version)
        if self.action_transform==DISK_ACTION:extra['action_transform']=self.action_transform
        extra['disk_radius']=self.disk_radius
        if self.environment_snapshot is not None:
            from dataclasses import asdict
            extra = dict(extra or {}, environment_contract=self.environment_contract,
                environment_snapshot=self.environment_snapshot, algorithm_config=asdict(self.config),
                learner_rng_state=self.rng.bit_generator.state, torch_rng_state=torch.random.get_rng_state(),
                cuda_rng_states=torch.cuda.get_rng_state_all() if self.device.type=='cuda' else [],
                value_norm_state=self.value_normalizer.state_dict() if self.use_value_norm else None)
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "global_step": int(global_step),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "extra": extra or {},
            },
            output,
        )

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> dict[str, Any]:
        with loading_transaction((self.actor,self.critic),(self.actor_optimizer,self.critic_optimizer),
                owner=self,counters=('global_step','value_normalizer','use_value_norm'),generator_attr='rng',inference=not load_optimizers):
            checkpoint=torch.load(Path(path),map_location=self.device,weights_only=False)
            if not isinstance(checkpoint,dict) or not {'global_step','actor','critic','extra'}<=checkpoint.keys():
                raise ValueError('Invalid MAPPO checkpoint structure')
            validate_counter(checkpoint['global_step'],'global_step')
            extra=checkpoint['extra']
            if extra.get('disk_radius')!=self.disk_radius:
                raise ValueError('Interception checkpoint disk radius mismatch')
            if self.action_transform==DISK_ACTION and (extra.get('action_transform')!=DISK_ACTION
                    or extra.get('action_probability_version')!=self.action_probability_version):
                raise ValueError('Checkpoint action/probability contract mismatch')
            if self.environment_snapshot is not None:
                from dataclasses import asdict
                if (extra.get('environment_contract') != self.environment_contract
                        or extra.get('environment_snapshot') != self.environment_snapshot
                        or extra.get('algorithm_config') != asdict(self.config)):
                    raise ValueError('MAPPO checkpoint environment/configuration mismatch')
            validate_model_state(self.actor,checkpoint['actor'])
            validate_model_state(self.critic,checkpoint['critic'])
            norm_state = extra.get('value_norm_state')
            if norm_state is not None:
                validated_norm = RunningValueNorm()
                validated_norm.load_state_dict(norm_state)
            else:
                validated_norm = RunningValueNorm()  # Historical unnormalized checkpoints.
            if load_optimizers:
                if not {'actor_optimizer','critic_optimizer'}<=checkpoint.keys():raise ValueError('Missing MAPPO optimizer states')
                validate_optimizer_state(self.actor_optimizer,checkpoint['actor_optimizer'])
                validate_optimizer_state(self.critic_optimizer,checkpoint['critic_optimizer'])
                if self.environment_snapshot is not None:
                    if not {'learner_rng_state','torch_rng_state'}<=extra.keys():raise ValueError('Missing MAPPO training RNG states')
                    validate_generator_rng(self.rng,extra['learner_rng_state'])
                    validate_torch_rng(extra['torch_rng_state'])
                    if norm_state is not None and 'cuda_rng_states' not in extra:
                        raise ValueError('Missing MAPPO CUDA RNG states')
                    if 'cuda_rng_states' in extra:
                        from .state import validate_cuda_rng
                        validate_cuda_rng(extra['cuda_rng_states'],self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.value_normalizer = validated_norm
            self.use_value_norm = norm_state is not None
            if load_optimizers:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
                if self.environment_snapshot is not None:
                    self.rng.bit_generator.state=extra['learner_rng_state']
                    torch.random.set_rng_state(extra['torch_rng_state'].cpu())
                    if extra.get('cuda_rng_states') and self.device.type=='cuda':
                        torch.cuda.set_rng_state_all([state.cpu() for state in extra['cuda_rng_states']])
            return checkpoint
