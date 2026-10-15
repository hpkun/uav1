"""MADSAC-v4 fixed-alpha objectives; the runner owns Algorithm-1 scheduling."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import random
import numpy as np
import torch

from .networks import CentralizedAttentionQCritic, SharedMADSACActor
from .replay_buffer import ReplayBatch
from .critic_contract import CRITIC_ARCHITECTURE, schedule_contract, validate_v4_contract


MADSAC_IMPL_VERSION = 4


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
                 critic_activation="relu", log_std_min=-5., log_std_max=2.,
                 steps_per_update=100, num_updates=4):
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
        self.steps_per_update = int(steps_per_update)
        self.num_updates = int(num_updates)
        if self.steps_per_update <= 0 or self.num_updates <= 0:
            raise ValueError("update-cycle parameters must be positive")
        self.update_cycle_count = 0
        self.transitions_since_update = 0
        self.replay_sample_count_critic = 0
        self.replay_sample_count_actor = 0
        self.target_update_count = 0
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

    def counters(self):
        return dict(update_cycle_count=self.update_cycle_count,
                    transitions_since_update=self.transitions_since_update,
                    critic_update_count=self.critic_update_count,
                    actor_update_count=self.actor_update_count,
                    critic_optimizer_steps=2*self.critic_update_count,
                    actor_optimizer_steps=self.actor_update_count,
                    target_update_count=self.target_update_count,
                    replay_sample_count_critic=self.replay_sample_count_critic,
                    replay_sample_count_actor=self.replay_sample_count_actor)

    @staticmethod
    def _finite(metrics):
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"non-finite MADSAC optimization metrics: {metrics}")
        return metrics

    def update_critics(self, batch: ReplayBatch) -> dict[str, float]:
        """One independent optimization step for each centralized Q; no policy/target step."""
        obs, actions, rewards = map(self._tensor, (batch.observations, batch.actions, batch.rewards))
        next_obs, dones = map(self._tensor, (batch.next_observations, batch.dones))
        alive, next_alive = map(self._tensor, (batch.alive_masks, batch.next_alive_masks))
        with torch.no_grad():
            next_actions, _, next_log_prob = self.target_actor.sample(next_obs, self.policy_generator)
            next_actions = next_actions * next_alive.unsqueeze(-1)
            next_log_prob = next_log_prob * next_alive
            target_q1 = self.target_critic1(next_obs, next_actions, next_alive)
            target_q2 = self.target_critic2(next_obs, next_actions, next_alive)
            target = critic_target(rewards, dones, next_alive, target_q1, target_q2,
                                   next_log_prob, self.gamma, self.alpha)
        # Low-frequency diagnostics reuse the Q forward; they never enter the loss.
        diagnostics = self.critic_update_count % 100 == 0
        if diagnostics:
            q1, attention1 = self.critic1(obs, actions, alive, return_attention=True)
            q2, attention2 = self.critic2(obs, actions, alive, return_attention=True)
        else:
            q1 = self.critic1(obs, actions, alive)
            q2 = self.critic2(obs, actions, alive)
        loss1 = batch_mean_agent_sum((q1 - target).square(), alive)
        loss2 = batch_mean_agent_sum((q2 - target).square(), alive)
        self.critic1_optimizer.zero_grad(set_to_none=True)
        loss1.backward()
        grad1 = gradient_norm(self.critic1.parameters())
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad(set_to_none=True)
        loss2.backward()
        grad2 = gradient_norm(self.critic2.parameters())
        self.critic2_optimizer.step()
        self.critic_update_count += 1
        metrics = dict(critic1_loss=float(loss1.item()), critic2_loss=float(loss2.item()),
            critic1_gradient_norm=grad1, critic2_gradient_norm=grad2,
            q1_mean=float(masked_mean(q1.detach(),alive)), q2_mean=float(masked_mean(q2.detach(),alive)),
            qmin_mean=float(masked_mean(torch.minimum(q1,q2).detach(),alive)),
            target_q_mean=float(masked_mean(target,alive)),
            target_q_std=float(target[alive>.5].std(unbiased=False)) if (alive>.5).any() else 0.,
            critic_learning_rate=float(self.critic1_optimizer.param_groups[0]['lr']),
            actor_updated=0., **self.counters())
        if diagnostics:
            with torch.no_grad():
                for index, weights in enumerate((attention1.detach(),attention2.detach()),1):
                    live_queries = (alive>.5)[:,None,:].expand(weights.shape[:-1])
                    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
                    for name, values in (('entropy',entropy),('max_weight',weights.max(-1).values)):
                        metrics[f'critic{index}_attention_{name}'] = float(values[live_queries].mean()) if live_queries.any() else 0.
        return self._finite(metrics)

    def update_actor(self, batch: ReplayBatch) -> dict[str, float]:
        """One own-action reparameterized policy step; no critic/target optimization."""
        obs, alive = map(self._tensor,(batch.observations,batch.alive_masks))
        critics=(self.critic1,self.critic2)
        flags=[[p.requires_grad for p in c.parameters()] for c in critics]
        for critic in critics:
            critic.requires_grad_(False)
        try:
            sampled_actions, _, log_prob = self.actor.sample(obs,self.policy_generator)
            sampled_actions = sampled_actions * alive.unsqueeze(-1)
            log_prob = log_prob * alive
            qmin = own_action_twin_q(self.critic1,self.critic2,obs,sampled_actions,alive)
            loss = actor_objective(log_prob,qmin,qmin,alive,self.alpha)
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = gradient_norm(self.actor.parameters())
            self.actor_optimizer.step()
        finally:
            for critic, values in zip(critics,flags):
                for parameter, flag in zip(critic.parameters(),values):
                    parameter.requires_grad_(flag)
        self.actor_update_count += 1
        return self._finite(dict(actor_loss=float(loss.item()),actor_updated=1.,
            actor_gradient_norm=grad,entropy=float(masked_mean(-log_prob.detach(),alive)),
            mean_log_prob=float(masked_mean(log_prob.detach(),alive)),
            qmin_mean=float(masked_mean(qmin.detach(),alive)),
            action_abs_mean=float(masked_mean(sampled_actions.detach().abs().mean(-1),alive)),
            action_saturation_fraction=float(masked_mean((sampled_actions.detach().abs()>.9).float().mean(-1),alive)),
            actor_learning_rate=float(self.actor_optimizer.param_groups[0]['lr']),**self.counters()))

    def update_targets(self):
        """One Polyak update of the three targets after a complete delayed actor cycle."""
        polyak_update(self.target_actor,self.actor,self.tau)
        polyak_update(self.target_critic1,self.critic1,self.tau)
        polyak_update(self.target_critic2,self.critic2,self.tau)
        self.target_update_count += 1

    def checkpoint_state(self, extra=None):
        return {
            "algorithm": "madsac", "implementation_version": MADSAC_IMPL_VERSION,
            "sampled_steps": self.sampled_steps, "vector_steps": self.vector_steps,
            **self.counters(),
            "network_architecture": dict(CRITIC_ARCHITECTURE,hidden_dim=self.critic1.hidden_dim,
                attention_heads=self.critic1.attention_heads,attention_head_dim=self.critic1.head_dim),
            "training_schedule": schedule_contract(self.steps_per_update,self.num_updates,self.policy_delay),
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
        validate_v4_contract(state,self.critic1.hidden_dim,self.critic1.attention_heads)
        if state['training_schedule'] != schedule_contract(self.steps_per_update,self.num_updates,self.policy_delay):
            raise RuntimeError('MADSAC v4 checkpoint configured schedule mismatch')
        for name in ('actor','target_actor','critic1','critic2','target_critic1','target_critic2'):
            getattr(self,name).load_state_dict(state[name], strict=True)
            getattr(self,name).eval()
        self.sampled_steps = int(state["sampled_steps"])
        self.vector_steps = int(state["vector_steps"])
        self.critic_update_count = int(state["critic_update_count"])
        self.actor_update_count = int(state["actor_update_count"])
        for key in ('update_cycle_count','transitions_since_update','replay_sample_count_critic',
                    'replay_sample_count_actor','target_update_count'):
            setattr(self,key,int(state[key]))
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
