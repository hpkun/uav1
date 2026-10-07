"""On-policy MAPPO optimization with GAE and a centralized value critic."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn

from .networks import CentralizedMLPCritic, CentralizedValueCritic, SharedMAPPOActor


MAPPO_IMPL_VERSION = 2


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    alive_masks: torch.Tensor,
    next_alive_masks: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-agent GAE, stopping recursion at death or episode end."""
    if rewards.shape != values.shape or rewards.shape != next_values.shape:
        raise ValueError("rewards and values must have matching [time, env, agent] shapes")
    if alive_masks.shape != rewards.shape or next_alive_masks.shape != rewards.shape:
        raise ValueError("alive masks must match rewards")
    if dones.shape != rewards.shape[:2]:
        raise ValueError("dones must have shape [time, env]")
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        continuation = (1.0 - dones[step].unsqueeze(-1)) * next_alive_masks[step]
        delta = rewards[step] + gamma * continuation * next_values[step] - values[step]
        gae = delta + gamma * gae_lambda * continuation * gae
        advantages[step] = gae * alive_masks[step]
    returns = (advantages + values) * alive_masks
    return advantages, returns


@dataclass
class RolloutBatch:
    observations: np.ndarray
    actions: np.ndarray
    raw_actions: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    alive_masks: np.ndarray
    next_observations: np.ndarray
    next_alive_masks: np.ndarray


class MAPPOTrainer:
    """Shared policy with centralized attention value function."""

    def __init__(
        self,
        observation_dim: int = 52,
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
        entropy_coefficient: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        minibatch_size: int = 512,
        normalize_advantages: bool = True,
        clip_value_loss: bool = True,
        device: str = "cpu",
        seed: int = 0,
        actor_activation: str = "relu",
        critic_activation: str = "relu",
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        critic_type: str = "attention",
    ) -> None:
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
        self.minibatch_size = int(minibatch_size)
        self.normalize_advantages = bool(normalize_advantages)
        self.clip_value_loss = bool(clip_value_loss)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        self.critic_type=str(critic_type)
        if self.critic_type not in {"attention","mlp"}:raise ValueError(f"unsupported critic_type: {self.critic_type}")
        self.attention_heads=int(attention_heads)

        self.actor = SharedMAPPOActor(
            observation_dim, action_dim, hidden_dim, log_std_min, log_std_max,
            actor_activation,
        ).to(self.device)
        if self.critic_type=="attention":
            self.critic=CentralizedValueCritic(observation_dim,hidden_dim,attention_heads,critic_activation).to(self.device)
        else:
            self.critic=CentralizedMLPCritic(observation_dim,num_agents,hidden_dim,critic_activation).to(self.device)
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

    @torch.no_grad()
    def act(
        self,
        observations: np.ndarray,
        alive_mask: np.ndarray | None = None,
        deterministic: bool = False,
        return_log_prob: bool = False,
        return_policy_data: bool = False,
    ):
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if deterministic:
            distribution = self.actor.distribution(tensor)
            raw_actions = distribution.mean
            actions = torch.tanh(raw_actions)
            log_prob = self.actor._squashed_log_prob(
                distribution, raw_actions, actions
            )
        else:
            actions, raw_actions, log_prob, _ = self.actor.sample(tensor)
        if alive_mask is not None:
            mask = torch.as_tensor(alive_mask, dtype=torch.float32, device=self.device)
            actions = actions * mask.unsqueeze(-1)
            if log_prob is not None:
                log_prob = log_prob * mask
            raw_actions = raw_actions * mask.unsqueeze(-1)
        action_array = actions.cpu().numpy()
        if return_policy_data:
            return action_array, raw_actions.cpu().numpy(), log_prob.cpu().numpy()
        if return_log_prob:
            return action_array, log_prob.cpu().numpy()
        return action_array

    @torch.no_grad()
    def values(self, observations: np.ndarray, alive_masks: np.ndarray) -> np.ndarray:
        observations_tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        mask_tensor = torch.as_tensor(
            alive_masks, dtype=torch.float32, device=self.device
        )
        return self.critic(observations_tensor, mask_tensor).cpu().numpy()

    def update(self, rollout: RolloutBatch) -> dict[str, float]:
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
                        **{key: value for key, value in pre_step.items()
                           if key.startswith("ratio_")},
                    }
                    metric_rows.append(row)
                    this_epoch.append({**pre_step, "epoch": float(epoch)})
            epoch_rows.append(this_epoch)

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
            for key in rows[0]:
                if key != "epoch":
                    metrics[f"epoch_{epoch}_{key}"] = float(np.mean([
                        row[key] for row in rows
                    ]))
        if not np.all(np.isfinite(list(metrics.values()))):
            raise FloatingPointError(f"non-finite MAPPO update: {metrics}")
        return metrics

    def checkpoint_state(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "algorithm": "MAPPO",
            "mappo_impl_version": MAPPO_IMPL_VERSION,
            "critic_type": self.critic_type,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "ppo_updates": self.ppo_update_count,
            "actor_updates": self.actor_update_count,
            "critic_updates": self.critic_update_count,
            "sampled_steps": self.sampled_steps,
            "vector_steps": self.vector_steps,
            "extra": extra or {},
        }

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(extra), path)

    def load(
        self, path: str | Path, allow_legacy_diagnostic: bool = False
    ) -> dict[str, Any]:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("algorithm") != "MAPPO":
            raise RuntimeError("checkpoint is not a MAPPO checkpoint")
        checkpoint_critic_type=str(state.get("critic_type",state.get("extra",{}).get("critic_type","attention")))
        if checkpoint_critic_type!=self.critic_type:
            raise RuntimeError(f"checkpoint critic_type mismatch: checkpoint={checkpoint_critic_type}, current={self.critic_type}")
        legacy = state.get("mappo_impl_version") != MAPPO_IMPL_VERSION
        if legacy and not allow_legacy_diagnostic:
            raise RuntimeError(
                "checkpoint MAPPO implementation mismatch: old checkpoints are "
                "read-only diagnostics and cannot resume formal training"
            )
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        if legacy:
            return dict(state.get("extra", {}))
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.ppo_update_count = int(state.get("ppo_updates", 0))
        self.actor_update_count = int(state.get("actor_updates", 0))
        self.critic_update_count = int(state.get("critic_updates", 0))
        self.sampled_steps = int(state.get("sampled_steps", 0))
        self.vector_steps = int(state.get("vector_steps", 0))
        return dict(state.get("extra", {}))


__all__ = [
    "MAPPO_IMPL_VERSION", "MAPPOTrainer", "RolloutBatch", "compute_gae",
    "masked_mean",
]
