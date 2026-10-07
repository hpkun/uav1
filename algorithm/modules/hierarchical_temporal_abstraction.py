"""HTA-MAPPO V1 configuration and exact semi-MDP return utilities."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch

from .base import CapabilityModule


HTA_MAPPO_VERSION = 1
HTA_MANAGER_TORCH_SEED_XOR = 0x48544131
HTA_MANAGER_NUMPY_SEED_XOR = 0x48544152


class HierarchicalTemporalAbstractionModule(CapabilityModule):
    """Frozen V1 protocol: four latent options and a 16-step manager clock."""

    name = "hierarchical_temporal_abstraction"

    def __init__(self, config=None):
        super().__init__(config)
        self.version = HTA_MAPPO_VERSION
        self.num_options = int(self.config.get("num_options", 4))
        self.decision_interval_steps = int(self.config.get("decision_interval_steps", 16))
        if self.enabled and self.num_options != 4:
            raise ValueError("HTA-MAPPO V1 requires exactly four latent options")
        if self.enabled and self.decision_interval_steps != 16:
            raise ValueError("HTA-MAPPO V1 requires decision_interval_steps=16")


def discounted_macro_reward(primitive_rewards, gamma: float):
    """Discount raw per-agent environment rewards within one macro action."""
    rewards = np.asarray(primitive_rewards)
    if rewards.ndim < 1 or rewards.shape[0] < 1:
        raise ValueError("a macro reward requires at least one primitive reward")
    powers = np.power(float(gamma), np.arange(rewards.shape[0], dtype=np.float64))
    return np.tensordot(powers, rewards, axes=(0, 0))


def smdp_boundary_masks(next_alive, *, episode_terminal: bool, rollout_truncation: bool):
    """Return distinct bootstrap/trace masks for one closed manager macro."""
    alive=np.asarray(next_alive,dtype=np.float32)
    if episode_terminal:return np.zeros_like(alive),np.zeros_like(alive)
    return alive.copy(),(np.zeros_like(alive) if rollout_truncation else alive.copy())


def compute_smdp_gae(rewards, values, next_values, durations, bootstrap_masks,
                     trace_masks, gamma: float, gae_lambda: float,
                     env_ids=None):
    """Semi-MDP GAE with separate bootstrap and trace-continuation masks.

    Inputs have leading macro-state dimension M and optional trailing agent
    dimensions.  When transitions from multiple vector environments are
    interleaved, ``env_ids`` keeps each reverse trace within its own environment.
    """
    tensor_input = torch.is_tensor(rewards)
    device = rewards.device if tensor_input else None
    r = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    v = torch.as_tensor(values, dtype=torch.float32, device=device)
    nv = torch.as_tensor(next_values, dtype=torch.float32, device=device)
    d = torch.as_tensor(durations, dtype=torch.float32, device=device)
    bootstrap = torch.as_tensor(bootstrap_masks, dtype=torch.float32, device=device)
    trace = torch.as_tensor(trace_masks, dtype=torch.float32, device=device)
    if r.shape != v.shape or r.shape != nv.shape or r.shape != bootstrap.shape or r.shape != trace.shape:
        raise ValueError("SMDP reward/value/mask shapes must match")
    if d.ndim != 1 or d.shape[0] != r.shape[0] or torch.any((d < 1) | (d > 16)):
        raise ValueError("SMDP durations must be [M] with values in [1,16]")
    discount = torch.pow(torch.as_tensor(float(gamma), device=r.device), d)
    while discount.ndim < r.ndim:
        discount = discount.unsqueeze(-1)
    delta = r + discount * nv * bootstrap - v
    advantages = torch.zeros_like(delta)
    ids = np.zeros(r.shape[0], dtype=np.int64) if env_ids is None else np.asarray(env_ids, dtype=np.int64)
    if ids.shape != (r.shape[0],):
        raise ValueError("env_ids must have shape [M]")
    running = {}
    for index in range(r.shape[0] - 1, -1, -1):
        env = int(ids[index])
        following = running.get(env, torch.zeros_like(delta[index]))
        advantages[index] = delta[index] + discount[index] * float(gae_lambda) * trace[index] * following
        running[env] = advantages[index]
    returns = advantages + v
    if tensor_input:
        return advantages, returns, delta
    return advantages.cpu().numpy(), returns.cpu().numpy(), delta.cpu().numpy()


@dataclass
class ManagerTransitionBatch:
    observations: np.ndarray
    alive_masks: np.ndarray
    options: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    next_alive_masks: np.ndarray
    durations: np.ndarray
    bootstrap_masks: np.ndarray
    trace_masks: np.ndarray
    env_ids: np.ndarray
    end_reasons: np.ndarray
    decision_reasons: np.ndarray | None = None

    def __post_init__(self):
        count = int(self.observations.shape[0])
        if count < 1:
            raise ValueError("manager batch cannot be empty")
        if any(int(getattr(self, name).shape[0]) != count for name in (
                "alive_masks", "options", "old_log_probs", "rewards",
                "next_observations", "next_alive_masks", "durations",
                "bootstrap_masks", "trace_masks", "env_ids", "end_reasons")):
            raise ValueError("manager transition fields have inconsistent lengths")
        if self.decision_reasons is not None and int(self.decision_reasons.shape[0]) != count:
            raise ValueError("manager decision reasons have inconsistent length")


__all__ = [
    "HTA_MAPPO_VERSION", "HTA_MANAGER_TORCH_SEED_XOR",
    "HTA_MANAGER_NUMPY_SEED_XOR", "HierarchicalTemporalAbstractionModule",
    "ManagerTransitionBatch", "discounted_macro_reward", "smdp_boundary_masks", "compute_smdp_gae",
]
