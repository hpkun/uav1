"""PS-IPPO: MAPPO PPO equations retained verbatim with clipping telemetry."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn
from algorithm.mappo.networks import SharedMAPPOActor
from algorithm.mappo.trainer import MAPPOTrainer, RolloutBatch, compute_gae, masked_mean, raw_gae_diagnostics
from algorithm.common.policy_protocol import validate_trainer_policy_protocol
from .networks import LocalValueCritic

IPPO_IMPL_VERSION = 1

class IPPOTrainer(MAPPOTrainer):
    """One shared actor and one shared local value network; no centralized network."""
    def __init__(
        self,
        observation_dim: int = 45,
        action_dim: int = 3,
        num_agents: int = 4,
        hidden_dim: int = 256,
        attention_heads: int = 2,
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coefficient: float = 0.5,
        entropy_coefficient: float = 0.001,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        minibatch_size: int = 512,
        normalize_advantages: bool = True,
        clip_value_loss: bool = True,
        device: str = "cuda",
        seed: int = 0,
        actor_activation: str = "relu",
        critic_activation: str = "relu",
        log_std_min: float = -5.0,
        log_std_max: float = 0.5,
        critic_type: str = "local",
        policy_std_mode: str = "state_independent",
        log_std_init: float = -.5,
        mean_head_init_gain: float = .01,
        target_kl: float | None = .015,
        critic_hidden_dim: int | None = None,
    ) -> None:
        from .protocol import require_cuda
        require_cuda(device)
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.num_agents = int(num_agents)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_ratio = float(clip_ratio)
        self.value_loss_coefficient = float(value_loss_coefficient)
        self.entropy_coefficient = float(entropy_coefficient)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.target_kl = None if target_kl is None else float(target_kl)
        if self.target_kl is not None and (not np.isfinite(self.target_kl) or self.target_kl <= 0):
            raise ValueError("target_kl must be positive or None")
        self.minibatch_size = int(minibatch_size)
        self.normalize_advantages = bool(normalize_advantages)
        self.clip_value_loss = bool(clip_value_loss)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.critic_type = str(critic_type)
        if self.critic_type != "local":
            raise ValueError("IPPO requires a local critic")
        if observation_dim not in (45,49) or (action_dim, num_agents) != (3,4):
            raise ValueError("IPPO requires 45 or 49 observations, 3 actions, 4 agents")
        self.attention_heads=int(attention_heads)
        self.actor_hidden_dim = int(hidden_dim)
        self.critic_hidden_dim = int(hidden_dim if critic_hidden_dim is None else critic_hidden_dim)
        if min(self.actor_hidden_dim, self.critic_hidden_dim) <= 0:
            raise ValueError("actor/critic hidden dimensions must be positive")

        self.actor = SharedMAPPOActor(
            observation_dim, action_dim, hidden_dim, log_std_min, log_std_max,
            actor_activation, policy_std_mode, log_std_init, mean_head_init_gain,
        ).to(self.device)
        self.critic = LocalValueCritic(observation_dim, self.critic_hidden_dim, critic_activation).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate
        )
        self.ppo_update_count = 0
        self.actor_update_count = 0
        self.critic_update_count = 0
        self.sampled_steps = 0
        self.vector_steps = 0

    def _ppo_update(self, rollout: RolloutBatch) -> dict[str, float]:
        to_tensor = lambda value: torch.as_tensor(
            value, dtype=torch.float32, device=self.device
        )
        observations = to_tensor(rollout.observations)
        actions = to_tensor(rollout.actions)
        raw_actions = to_tensor(rollout.raw_actions)
        old_log_probs = to_tensor(rollout.old_log_probs)
        rewards = to_tensor(rollout.rewards)
        dones = to_tensor(rollout.dones)
        alive_masks = to_tensor(rollout.alive_masks)
        next_observations = to_tensor(rollout.next_observations)
        next_alive_masks = to_tensor(rollout.next_alive_masks)
        time_steps, num_envs = observations.shape[:2]

        with torch.no_grad():
            values = self.critic(
                observations.reshape(-1, self.num_agents, observations.shape[-1]),
                alive_masks.reshape(-1, self.num_agents),
            ).view(time_steps, num_envs, self.num_agents)
            next_values = self.critic(
                next_observations.reshape(-1, self.num_agents, observations.shape[-1]),
                next_alive_masks.reshape(-1, self.num_agents),
            ).view(time_steps, num_envs, self.num_agents)
            advantages, returns = compute_gae(
                rewards, values, next_values, dones, alive_masks,
                next_alive_masks, self.gamma, self.gae_lambda,
            )
            gae_metrics = raw_gae_diagnostics(advantages, returns, values, alive_masks)
            if self.normalize_advantages:
                live_advantages = advantages[alive_masks > 0.5]
                advantages = (
                    (advantages - live_advantages.mean())
                    / live_advantages.std(unbiased=False).clamp_min(1e-8)
                ) * alive_masks

        flatten = lambda value: value.reshape(
            time_steps * num_envs, *value.shape[2:]
        )
        observations = flatten(observations)
        actions = flatten(actions)
        raw_actions = flatten(raw_actions)
        old_log_probs = flatten(old_log_probs)
        alive_masks = flatten(alive_masks)
        old_values = flatten(values)
        returns = flatten(returns)
        advantages = flatten(advantages)
        sample_count = observations.shape[0]
        metric_rows: list[dict[str, float]] = []
        epoch_rows: list[list[dict[str, float]]] = []
        first_minibatch: dict[str, float] | None = None

        for epoch in range(self.ppo_epochs):
            this_epoch: list[dict[str, float]] = []
            permutation = self.rng.permutation(sample_count)
            for start in range(0, sample_count, self.minibatch_size):
                indices = torch.as_tensor(
                    permutation[start:start + self.minibatch_size],
                    dtype=torch.long, device=self.device,
                )
                obs = observations[indices]
                act = actions[indices]
                raw_act = raw_actions[indices]
                old_log = old_log_probs[indices]
                mask = alive_masks[indices]
                if not (mask > .5).any():
                    continue
                old_value = old_values[indices]
                target_return = returns[indices]
                advantage = advantages[indices]

                new_log_prob, entropy = self.actor.evaluate_actions(
                    obs, act, raw_act
                )
                log_ratio = new_log_prob - old_log
                ratio = log_ratio.exp()
                unclipped = ratio * advantage
                clipped = ratio.clamp(
                    1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * advantage
                actor_loss = -masked_mean(torch.minimum(unclipped, clipped), mask)

                value = self.critic(obs, mask)
                if self.clip_value_loss:
                    value_clipped = old_value + (value - old_value).clamp(
                        -self.clip_ratio, self.clip_ratio
                    )
                    value_error = torch.maximum(
                        (value - target_return).square(),
                        (value_clipped - target_return).square(),
                    )
                else:
                    value_error = (value - target_return).square()
                value_loss = 0.5 * masked_mean(value_error, mask)
                entropy_mean = masked_mean(entropy, mask)

                with torch.no_grad():
                    live_ratio = ratio[mask > 0.5]
                    pre_step = {
                        "approx_kl": float(masked_mean((ratio - 1.0) - log_ratio, mask)),
                        "clip_fraction": float(masked_mean(
                            (torch.abs(ratio - 1.0) > self.clip_ratio).float(), mask
                        )),
                        "ratio_mean": float(live_ratio.mean()),
                        "ratio_std": float(live_ratio.std(unbiased=False)),
                        "ratio_p1": float(torch.quantile(live_ratio, .01)),
                        "ratio_p50": float(torch.quantile(live_ratio, .50)),
                        "ratio_p99": float(torch.quantile(live_ratio, .99)),
                        "ratio_min": float(live_ratio.min()),
                        "ratio_max": float(live_ratio.max()),
                    }
                    if first_minibatch is None:
                        first_minibatch = dict(pre_step)

                self.actor_optimizer.zero_grad()
                (actor_loss - self.entropy_coefficient * entropy_mean).backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                (self.value_loss_coefficient * value_loss).backward()
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm
                )
                self.critic_optimizer.step()
                self.actor_update_count += 1
                self.critic_update_count += 1

                with torch.no_grad():
                    approximate_kl = masked_mean(
                        (ratio - 1.0) - log_ratio, mask
                    )
                    clip_fraction = masked_mean(
                        (torch.abs(ratio - 1.0) > self.clip_ratio).float(), mask
                    )
                    row = {
                        "actor_loss": float(actor_loss),
                        "value_loss": float(value_loss),
                        "entropy": float(entropy_mean),
                        "approx_kl": float(approximate_kl),
                        "clip_fraction": float(clip_fraction),
                        "value": float(masked_mean(value, mask)),
                        "actor_grad_norm": float(actor_grad_norm),
                        "critic_grad_norm": float(critic_grad_norm),
                        "gradient_clipped_ratio": .5 * (float(actor_grad_norm > self.max_grad_norm) + float(critic_grad_norm > self.max_grad_norm)),
                        **{key: value for key, value in pre_step.items()
                           if key.startswith("ratio_")},
                    }
                    metric_rows.append(row)
                    this_epoch.append({**pre_step, "epoch": float(epoch)})
            epoch_rows.append(this_epoch)
            if self._epoch_exceeds_target(this_epoch):
                break

        self.ppo_update_count += 1
        live = alive_masks > 0.5
        return_values = returns[live]
        old_value_values = old_values[live]
        return_variance = torch.var(return_values, unbiased=False)
        explained_variance = (
            1.0 - torch.var(return_values - old_value_values, unbiased=False)
            / return_variance.clamp_min(1e-8)
        )
        metrics = {
            key: float(np.mean([row[key] for row in metric_rows]))
            for key in metric_rows[0]
        }
        metrics["explained_variance"] = float(explained_variance)
        with torch.no_grad():
            distribution = self.actor.distribution(observations)
            live_actions = actions[alive_masks > 0.5]
            live_log_std = distribution.scale.log()[alive_masks > 0.5]
            for index, name in enumerate(("psi", "theta", "v")):
                metrics[f"policy_log_std_mean_{name}"] = float(
                    live_log_std[:, index].mean()
                )
                for threshold, label in ((.9, "0_9"), (.99, "0_99"), (.999, "0_999")):
                    metrics[f"action_abs_gt_{label}_fraction_{name}"] = float(
                        (live_actions[:, index].abs() > threshold).float().mean()
                    )
        assert first_minibatch is not None
        for key, value in first_minibatch.items():
            metrics[f"first_minibatch_{key}"] = value
        for epoch, rows in enumerate(epoch_rows):
            if not rows:
                continue
            for key in rows[0]:
                if key != "epoch":
                    metrics[f"epoch_{epoch}_{key}"] = float(np.mean([
                        row[key] for row in rows
                    ]))
        metrics.update(self._stability_metrics(epoch_rows))
        metrics.update(gae_metrics)
        if not np.all(np.isfinite(list(metrics.values()))):
            raise FloatingPointError(f"non-finite IPPO update: {metrics}")
        return metrics

    def update(self, rollout: RolloutBatch) -> dict[str, float]:
        valid = int((rollout.alive_masks > .5).sum())
        before_actor, before_critic = self.actor_update_count, self.critic_update_count
        if not valid:
            self.ppo_update_count += 1
            empty = {key:0. for key in ('actor_loss','value_loss','explained_variance',
                'approximate_kl','approx_kl','clip_fraction','policy_entropy','entropy',
                'actor_gradient_norm_preclip','critic_gradient_norm_preclip',
                'gradient_clipped_ratio','actual_actor_optimizer_steps','actual_critic_optimizer_steps',
                'valid_alive_agent_samples','ratio_mean','ratio_std','ratio_min','ratio_max')}
            return dict(empty, **self._stability_metrics([]), skipped_all_dead=1.)
        metrics = self._ppo_update(rollout)
        metrics.update(approximate_kl=metrics['approx_kl'], policy_entropy=metrics['entropy'],
            actor_gradient_norm_preclip=metrics['actor_grad_norm'],
            critic_gradient_norm_preclip=metrics['critic_grad_norm'],
            actual_actor_optimizer_steps=float(self.actor_update_count-before_actor),
            actual_critic_optimizer_steps=float(self.critic_update_count-before_critic),
            valid_alive_agent_samples=float(valid), skipped_all_dead=0.)
        return metrics

    def checkpoint_state(self, extra: dict[str, Any] | None = None):
        state = super().checkpoint_state(extra)
        state.update(algorithm='ippo', ippo_impl_version=IPPO_IMPL_VERSION,
                     parameter_sharing=True, critic_scope='local_observation')
        state.pop('mappo_impl_version', None)
        state['extra'].pop('mappo_impl_version', None)
        state['extra'].update(evaluation_split='development', evaluation_seed_base=getattr(self, 'evaluation_seed_base', 10_000_000), parameter_sharing=True, critic_scope='local_observation',
            resume_capability='evaluation_reload_only', bit_exact_training_resume=False,
            ippo_impl_version=IPPO_IMPL_VERSION,
            actor_parameter_count=sum(p.numel() for p in self.actor.parameters()),
            critic_parameter_count=sum(p.numel() for p in self.critic.parameters()),
            network_architecture=dict(observation_dim=self.observation_dim, action_dim=3, num_agents=4,
                actor_hidden_layers=[self.actor_hidden_dim]*2,
                critic_hidden_layers=[self.critic_hidden_dim]*2,
                actor_activation='relu', critic_activation='relu', critic_type='local'),
            ppo_protocol=dict(gamma=self.gamma, gae_lambda=self.gae_lambda,
                clip_ratio=self.clip_ratio, value_loss_coefficient=self.value_loss_coefficient,
                entropy_coefficient=self.entropy_coefficient, max_grad_norm=self.max_grad_norm,
                ppo_epochs=self.ppo_epochs, minibatch_size=self.minibatch_size,
                normalize_advantages=self.normalize_advantages, clip_value_loss=self.clip_value_loss,
                log_std_min=self.actor.log_std_min, log_std_max=self.actor.log_std_max))
        if hasattr(self, 'algorithm_config'):
            state['extra']['algorithm_config'] = self.algorithm_config
        return state

    def load(self, path, env_config=None, algorithm_config=None):
        from .protocol import validate_checkpoint
        state = torch.load(path, map_location=self.device, weights_only=False)
        config = algorithm_config if algorithm_config is not None else getattr(self, 'algorithm_config', None)
        if config is None or env_config is None:
            raise RuntimeError('IPPO load requires explicit environment and algorithm configuration')
        validate_checkpoint(state, env_config, config)
        validate_trainer_policy_protocol(state, self.policy_protocol())
        self.actor.load_state_dict(state['actor'])
        self.critic.load_state_dict(state['critic'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        for attribute, key in [('ppo_update_count','ppo_updates'), ('actor_update_count','actor_updates'),
            ('critic_update_count','critic_updates'), ('sampled_steps','sampled_steps'), ('vector_steps','vector_steps')]:
            setattr(self, attribute, int(state[key]))
        return dict(state['extra'])
