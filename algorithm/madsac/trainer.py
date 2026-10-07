"""Paper-faithful fixed-alpha multi-agent double-soft actor-critic."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import random
import numpy as np
import torch

from .networks import CentralizedAttentionQCritic, SharedMADSACActor
from .replay_buffer import ReplayBatch


MADSAC_IMPL_VERSION = 2


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def batch_mean_agent_sum(values: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    """Eq. (19)/(20) reduction: sum agents inside each replay item, then mean batch."""
    if values.ndim != 2 or alive.shape != values.shape:
        raise ValueError("values and alive must have identical [batch, agents] shapes")
    return (values * alive).sum(dim=1).mean()


def joint_actions_with_own_gradient(actions: torch.Tensor, agent_index: int) -> torch.Tensor:
    """Keep joint-action values while allowing gradients only through one agent action."""
    if actions.ndim != 3:
        raise ValueError("actions must have shape [batch, agents, action_dim]")
    if not 0 <= int(agent_index) < actions.shape[1]:
        raise IndexError("agent_index is outside the joint action")
    detached = actions.detach()
    own_mask = torch.zeros_like(actions)
    own_mask[:, int(agent_index), :] = 1.0
    return detached + own_mask * (actions - detached)


def own_action_twin_q(critic1, critic2, observations, actions, alive):
    """Evaluate each Q_i with only a_i connected to the shared actor graph."""
    per_agent = []
    for agent_index in range(actions.shape[1]):
        joint_actions = joint_actions_with_own_gradient(actions, agent_index)
        q1_i = critic1(observations, joint_actions, alive)[:, agent_index]
        q2_i = critic2(observations, joint_actions, alive)[:, agent_index]
        per_agent.append(torch.minimum(q1_i, q2_i))
    return torch.stack(per_agent, dim=1)


def critic_target(rewards, dones, next_alive, q1_target, q2_target,
                  next_log_prob, gamma, alpha):
    continuation = (1.0 - dones.unsqueeze(-1)) * next_alive
    return rewards + float(gamma) * continuation * (
        torch.minimum(q1_target, q2_target) - float(alpha) * next_log_prob
    )


def actor_objective(log_prob, q1, q2, alive, alpha):
    return batch_mean_agent_sum(
        float(alpha) * log_prob - torch.minimum(q1, q2), alive
    )


@torch.no_grad()
def polyak_update(target: torch.nn.Module, online: torch.nn.Module, tau: float):
    for target_parameter, online_parameter in zip(target.parameters(), online.parameters()):
        target_parameter.mul_(1.0 - float(tau)).add_(online_parameter, alpha=float(tau))


def gradient_norm(parameters) -> float:
    values = [parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
    return 0.0 if not values else float(torch.sqrt(torch.stack(values).sum()).item())


class MADSACTrainer:
    def __init__(self, observation_dim=52, action_dim=3, num_agents=4,
                 hidden_dim=256, attention_heads=2, actor_learning_rate=1e-4,
                 critic_learning_rate=1e-4, gamma=.99, alpha=.1, tau=.001,
                 policy_delay=2, device="cpu", seed=0, actor_activation="relu",
                 critic_activation="relu", log_std_min=-5., log_std_max=2.):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.num_agents = int(num_agents)
        self.gamma, self.alpha, self.tau = float(gamma), float(alpha), float(tau)
        self.policy_delay = int(policy_delay)
        if self.policy_delay <= 0:
            raise ValueError("policy_delay must be positive")
        torch.manual_seed(int(seed)); np.random.seed(int(seed)); random.seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        self.policy_generator = torch.Generator(device=self.device)
        self.policy_generator.manual_seed(int(seed) ^ 0x5A17)
        self.actor = SharedMADSACActor(
            observation_dim, action_dim, hidden_dim, log_std_min, log_std_max,
            actor_activation,
        ).to(self.device)
        self.target_actor = deepcopy(self.actor).to(self.device)
        self.critic1 = CentralizedAttentionQCritic(
            observation_dim, action_dim, hidden_dim, attention_heads, critic_activation,
        ).to(self.device)
        self.critic2 = CentralizedAttentionQCritic(
            observation_dim, action_dim, hidden_dim, attention_heads, critic_activation,
        ).to(self.device)
        self.target_critic1 = deepcopy(self.critic1).to(self.device)
        self.target_critic2 = deepcopy(self.critic2).to(self.device)
        for module in (self.target_actor, self.target_critic1, self.target_critic2):
            module.requires_grad_(False); module.eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_learning_rate)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_learning_rate)
        self.sampled_steps = 0
        self.vector_steps = 0
        self.critic_update_count = 0
        self.actor_update_count = 0

    @torch.no_grad()
    def act(self, observations, alive_mask=None, deterministic=False,
            generator: torch.Generator | None = None):
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if deterministic:
            actions = self.actor.deterministic(obs)
        else:
            actions, _, _ = self.actor.sample(obs, generator or self.policy_generator)
        if alive_mask is not None:
            mask = torch.as_tensor(alive_mask, dtype=torch.float32, device=self.device)
            actions = actions * mask.unsqueeze(-1)
        return actions.cpu().numpy()

    def _tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        obs, actions, rewards = map(self._tensor, (batch.observations, batch.actions, batch.rewards))
        next_obs = self._tensor(batch.next_observations)
        dones = self._tensor(batch.dones)
        alive, next_alive = map(self._tensor, (batch.alive_masks, batch.next_alive_masks))

        with torch.no_grad():
            next_actions, _, next_log_prob = self.target_actor.sample(next_obs, self.policy_generator)
            next_actions = next_actions * next_alive.unsqueeze(-1)
            next_log_prob = next_log_prob * next_alive
            target_q1 = self.target_critic1(next_obs, next_actions, next_alive)
            target_q2 = self.target_critic2(next_obs, next_actions, next_alive)
            target = critic_target(
                rewards, dones, next_alive, target_q1, target_q2,
                next_log_prob, self.gamma, self.alpha,
            )

        q1 = self.critic1(obs, actions, alive)
        q2 = self.critic2(obs, actions, alive)
        critic1_loss = batch_mean_agent_sum((q1 - target).square(), alive)
        critic2_loss = batch_mean_agent_sum((q2 - target).square(), alive)
        self.critic1_optimizer.zero_grad(set_to_none=True)
        critic1_loss.backward()
        critic1_grad = gradient_norm(self.critic1.parameters())
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic2_loss.backward()
        critic2_grad = gradient_norm(self.critic2.parameters())
        self.critic2_optimizer.step()
        self.critic_update_count += 1

        actor_updated = self.critic_update_count % self.policy_delay == 0
        actor_loss_value = actor_grad = 0.0
        entropy = mean_log_prob = qmin_mean = 0.0
        sampled_actions = actions.detach()
        if actor_updated:
            critics = (self.critic1, self.critic2)
            requires_grad = [
                [parameter.requires_grad for parameter in critic.parameters()]
                for critic in critics
            ]
            for critic in critics:
                critic.requires_grad_(False)
            try:
                sampled_actions, _, log_prob = self.actor.sample(obs, self.policy_generator)
                sampled_actions = sampled_actions * alive.unsqueeze(-1)
                log_prob = log_prob * alive
                actor_qmin = own_action_twin_q(
                    self.critic1, self.critic2, obs, sampled_actions, alive
                )
                actor_loss = actor_objective(
                    log_prob, actor_qmin, actor_qmin, alive, self.alpha
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = gradient_norm(self.actor.parameters())
                self.actor_optimizer.step()
            finally:
                for critic, flags in zip(critics, requires_grad):
                    for parameter, flag in zip(critic.parameters(), flags):
                        parameter.requires_grad_(flag)
            polyak_update(self.target_actor, self.actor, self.tau)
            polyak_update(self.target_critic1, self.critic1, self.tau)
            polyak_update(self.target_critic2, self.critic2, self.tau)
            self.actor_update_count += 1
            actor_loss_value = float(actor_loss.item())
            entropy = float(masked_mean(-log_prob.detach(), alive).item())
            mean_log_prob = float(masked_mean(log_prob.detach(), alive).item())
            qmin_mean = float(masked_mean(actor_qmin.detach(), alive).item())
        else:
            qmin_mean = float(masked_mean(torch.minimum(q1, q2).detach(), alive).item())

        metrics = {
            "critic1_loss": float(critic1_loss.item()),
            "critic2_loss": float(critic2_loss.item()),
            "actor_loss": actor_loss_value, "actor_updated": float(actor_updated),
            "entropy": entropy, "mean_log_prob": mean_log_prob,
            "q1_mean": float(masked_mean(q1.detach(), alive).item()),
            "q2_mean": float(masked_mean(q2.detach(), alive).item()),
            "qmin_mean": qmin_mean,
            "target_q_mean": float(masked_mean(target, alive).item()),
            "target_q_std": float(target[alive > .5].std(unbiased=False).item()),
            "actor_gradient_norm": actor_grad,
            "critic1_gradient_norm": critic1_grad,
            "critic2_gradient_norm": critic2_grad,
            "actor_update_count": float(self.actor_update_count),
            "critic_update_count": float(self.critic_update_count),
            "actor_learning_rate": float(self.actor_optimizer.param_groups[0]["lr"]),
            "critic_learning_rate": float(self.critic1_optimizer.param_groups[0]["lr"]),
            "action_abs_mean": float(masked_mean(sampled_actions.detach().abs().mean(-1), alive).item()),
            "action_saturation_fraction": float(masked_mean((sampled_actions.detach().abs() > .9).float().mean(-1), alive).item()),
        }
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite MADSAC optimization metrics: {metrics}")
        return metrics

    def checkpoint_state(self, extra=None):
        return {
            "algorithm": "madsac", "implementation_version": MADSAC_IMPL_VERSION,
            "sampled_steps": self.sampled_steps, "vector_steps": self.vector_steps,
            "critic_update_count": self.critic_update_count,
            "actor_update_count": self.actor_update_count,
            "actor": self.actor.state_dict(), "target_actor": self.target_actor.state_dict(),
            "critic1": self.critic1.state_dict(), "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "policy_generator_state": self.policy_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
            "gamma": self.gamma, "alpha": self.alpha, "tau": self.tau,
            "policy_delay": self.policy_delay,
            "formal_exact_resume_supported": False,
            "replay_buffer_included": False, "extra": extra or {},
        }

    def save(self, path, extra=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_state(extra), path)

    def load_for_evaluation(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state.get("algorithm") != "madsac" or state.get("implementation_version") != MADSAC_IMPL_VERSION:
            raise RuntimeError("incompatible MADSAC checkpoint")
        self.actor.load_state_dict(state["actor"], strict=True)
        self.target_actor.load_state_dict(state["target_actor"], strict=True)
        self.sampled_steps = int(state["sampled_steps"])
        self.vector_steps = int(state["vector_steps"])
        self.critic_update_count = int(state["critic_update_count"])
        self.actor_update_count = int(state["actor_update_count"])
        self.actor.eval(); self.target_actor.eval()
        return state

    def resume(self, *_args, **_kwargs):
        raise RuntimeError("formal exact resume unsupported: replay buffer is not checkpointed")


__all__ = [
    "MADSAC_IMPL_VERSION", "MADSACTrainer", "actor_objective",
    "batch_mean_agent_sum", "critic_target", "gradient_norm",
    "joint_actions_with_own_gradient", "masked_mean", "own_action_twin_q",
    "polyak_update",
]
