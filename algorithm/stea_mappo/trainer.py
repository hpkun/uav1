"""Sequence-preserving PPO for the independently identified STEA-MAPPO actor."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn

from algorithm.mappo.trainer import MAPPOTrainer, RolloutBatch, compute_gae, masked_mean
from .networks import SpatioTemporalEntityAttentionActor, attention_diagnostics
from algorithm.common.policy_protocol import validate_trainer_policy_protocol

STEA_MAPPO_IMPL_VERSION = 1


@dataclass
class RecurrentRolloutBatch(RolloutBatch):
    actor_hidden_states: np.ndarray  # before acting: [T,E,A,H]
    episode_starts: np.ndarray       # before acting: [T,E]


def sequence_chunks(time_steps, num_envs, length):
    """Environment-major contiguous chunks, including the final partial chunk."""
    return [(env, start, min(start + length, time_steps))
            for env in range(num_envs) for start in range(0, time_steps, length)]


def pack_sequences(tensor, chunks, length, pad=0):
    """Pack [T,E,...] into [B,L,...]; callers explicitly mask padding."""
    output = tensor.new_full((len(chunks), length, *tensor.shape[2:]), pad)
    for index, (env, start, end) in enumerate(chunks):
        output[index, :end-start] = tensor[start:end, env]
    return output


class STEAMAPPOTrainer(MAPPOTrainer):
    def __init__(self, *, entity_dim=64, entity_attention_heads=2,
                 spatial_hidden_dim=128, gru_hidden_dim=128, gru_layers=1,
                 recurrent_sequence_length=32, self_feature_dim=7, **kwargs):
        if kwargs.get("critic_type", "attention") not in {"attention", "mlp"}:
            raise ValueError("STEA-MAPPO critic_type must be attention or mlp")
        if kwargs.get("actor_activation", "relu") != "relu":
            raise ValueError("STEA-MAPPO actor activation must be relu")
        if kwargs.get("observation_dim", 52) != 13*kwargs.get("num_agents", 4)+self_feature_dim-7 or kwargs.get("action_dim", 3) != 3:
            raise ValueError("STEA-MAPPO requires observation/action/agents=13*N/3/N")
        self.sequence_length = int(recurrent_sequence_length)
        if self.sequence_length <= 0 or int(kwargs.get("minibatch_size", 512)) % self.sequence_length:
            raise ValueError("minibatch_size must be divisible by recurrent_sequence_length")
        # Construct the exact baseline critic with the same seeded initialization.
        # Replace only the temporary baseline actor and its optimizer.
        super().__init__(**kwargs)
        self.actor = SpatioTemporalEntityAttentionActor(
            entity_dim, entity_attention_heads, spatial_hidden_dim,
            gru_hidden_dim, gru_layers, self.actor.mean.out_features,
            kwargs.get("log_std_min", -5.), kwargs.get("log_std_max", 2.),
            kwargs.get("policy_std_mode", "state_dependent"), kwargs.get("log_std_init", -.5),
            kwargs.get("mean_head_init_gain", .01),
            self_feature_dim,
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=kwargs.get("actor_learning_rate", 3e-4))
        self.network_architecture = {
            "actor_type": "stea", "entity_dim": int(entity_dim),
            "entity_attention_heads": int(entity_attention_heads),
            "spatial_hidden_dim": int(spatial_hidden_dim),
            "gru_hidden_dim": int(gru_hidden_dim), "gru_layers": int(gru_layers),
            "recurrent_sequence_length": self.sequence_length,
            "critic_type": self.critic_type, "critic_hidden_dim": self.critic_hidden_dim,
        }
        self.actor_hidden_dim = int(gru_hidden_dim)
        if self_feature_dim == 8: self.network_architecture['self_feature_dim'] = 8
        if self.critic_type == "attention":
            self.network_architecture["critic_attention_heads"] = self.attention_heads
        if self.actor.policy_std_mode == "state_independent":
            self.network_architecture.update({key: value for key, value in self.policy_protocol().items() if key != "target_kl"})

    def tensor(self, value):
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, observations, alive_masks, actor_hidden_states, episode_starts,
            deterministic=False, return_diagnostics=False):
        alive = self.tensor(alive_masks)
        distribution, hidden, attention = self.actor.distribution_step(
            self.tensor(observations), self.tensor(actor_hidden_states), alive,
            self.tensor(episode_starts))
        raw = distribution.mean if deterministic else distribution.rsample()
        actions = torch.tanh(raw)
        log_prob = self.actor._squashed_log_prob(distribution, raw, actions) * alive
        result = (actions * alive.unsqueeze(-1), raw * alive.unsqueeze(-1), log_prob, hidden)
        arrays = tuple(value.cpu().numpy() for value in result)
        if return_diagnostics:
            diagnostics = {key:float(value) for key,value in attention_diagnostics(attention, hidden, alive).items()}
            diagnostics.update({f"policy_log_std_mean_{name}": float(masked_mean(
                distribution.scale.log()[..., index], alive))
                for index, name in enumerate(("psi", "theta", "v"))})
            return (*arrays, diagnostics)
        return arrays

    def _packed(self, tensors, chunks):
        packed = {key: pack_sequences(value, chunks, self.sequence_length,
                    pad=1 if key == "episode_starts" else 0)
                  for key, value in tensors.items() if key != "actor_hidden_states"}
        packed["initial_hidden"] = torch.stack([
            tensors["actor_hidden_states"][start, env] for env, start, _ in chunks]).detach()
        packed["valid_steps"] = torch.stack([
            torch.arange(self.sequence_length, device=self.device) < end-start
            for _, start, end in chunks]).float()
        packed["alive_masks"] *= packed["valid_steps"].unsqueeze(-1)
        return packed

    def _evaluate(self, batch, entropy=True):
        return self.actor.evaluate_sequence(batch["observations"], batch["actions"],
            batch["raw_actions"], batch["initial_hidden"], batch["alive_masks"],
            batch["episode_starts"], compute_entropy=entropy)

    @torch.no_grad()
    def audit_rollout_ratio(self, rollout):
        tensors = {key: self.tensor(value) for key, value in vars(rollout).items()}
        chunks = sequence_chunks(*tensors["observations"].shape[:2], self.sequence_length)
        ratios = []
        for start in range(0, len(chunks), self.minibatch_size // self.sequence_length):
            batch = self._packed(tensors, chunks[start:start+self.minibatch_size//self.sequence_length])
            log_prob = self._evaluate(batch, entropy=False)[0]
            ratios.append((log_prob - batch["old_log_probs"]).exp()[batch["alive_masks"] > .5])
        live = torch.cat(ratios)
        if not live.numel():
            raise ValueError("rollout has no live agents")
        error = float((live-1).abs().max())
        if not torch.isfinite(live).all() or error > 1e-4:
            raise RuntimeError(f"recurrent PPO pre-update ratio mismatch: max_abs_error={error}")
        return {"pre_update_ratio_mean": float(live.mean()),
                "pre_update_ratio_min": float(live.min()), "pre_update_ratio_max": float(live.max()),
                "pre_update_ratio_max_abs_error": error}

    def update(self, rollout):
        audit = self.audit_rollout_ratio(rollout)
        tensors = {key: self.tensor(value) for key, value in vars(rollout).items()}
        obs, mask = tensors["observations"], tensors["alive_masks"]
        t, e = obs.shape[:2]
        if tensors["actor_hidden_states"].shape != (t,e,self.num_agents,self.actor.gru_hidden_dim):
            raise ValueError("actor_hidden_states must have shape [T,E,N,H]")
        if tensors["episode_starts"].shape != (t,e):
            raise ValueError("episode_starts must have shape [T,E]")
        with torch.no_grad():
            values = self.critic(obs.reshape(-1,self.num_agents,self.observation_dim),mask.reshape(-1,self.num_agents)).reshape(t,e,self.num_agents)
            next_values = self.critic(tensors["next_observations"].reshape(-1,self.num_agents,self.observation_dim),
                tensors["next_alive_masks"].reshape(-1,self.num_agents)).reshape(t,e,self.num_agents)
            advantages, returns = compute_gae(tensors["rewards"], values, next_values,
                tensors["dones"], mask, tensors["next_alive_masks"], self.gamma, self.gae_lambda)
            if self.normalize_advantages:
                live = advantages[mask > .5]
                advantages = ((advantages-live.mean())/live.std(unbiased=False).clamp_min(1e-8))*mask
        tensors.update(old_values=values, advantages=advantages, returns=returns)
        chunks = sequence_chunks(t,e,self.sequence_length)
        chunk_batch = self.minibatch_size // self.sequence_length
        rows, epoch_rows = [], []
        for epoch in range(self.ppo_epochs):
            this_epoch = []
            order = self.rng.permutation(len(chunks))
            for start in range(0,len(chunks),chunk_batch):
                batch = self._packed(tensors,[chunks[i] for i in order[start:start+chunk_batch]])
                live_mask = batch["alive_masks"]
                if not (live_mask > .5).any():
                    continue
                log_prob, entropy, hidden, attention, log_std = self._evaluate(batch)
                log_ratio = log_prob - batch["old_log_probs"]
                ratio = log_ratio.exp()
                advantage = batch["advantages"]
                actor_loss = -masked_mean(torch.minimum(ratio*advantage,
                    ratio.clamp(1-self.clip_ratio,1+self.clip_ratio)*advantage),live_mask)
                value = self.critic(batch["observations"].reshape(-1,self.num_agents,self.observation_dim),
                    live_mask.reshape(-1,self.num_agents)).reshape_as(live_mask)
                value_error = (value-batch["returns"]).square()
                if self.clip_value_loss:
                    clipped = batch["old_values"] + (value-batch["old_values"]).clamp(-self.clip_ratio,self.clip_ratio)
                    value_error = torch.maximum(value_error,(clipped-batch["returns"]).square())
                value_loss = .5*masked_mean(value_error,live_mask)
                entropy_mean = masked_mean(entropy,live_mask)
                with torch.no_grad():
                    live_ratio = ratio[live_mask > .5]
                    row = {"actor_loss":float(actor_loss),"value_loss":float(value_loss),
                        "entropy":float(entropy_mean),"value":float(masked_mean(value,live_mask)),
                        "approx_kl":float(masked_mean((ratio-1)-log_ratio,live_mask)),
                        "clip_fraction":float(masked_mean((abs(ratio-1)>self.clip_ratio).float(),live_mask)),
                        "ratio_mean":float(live_ratio.mean()),"ratio_std":float(live_ratio.std(unbiased=False)),
                        "ratio_min":float(live_ratio.min()),"ratio_max":float(live_ratio.max()),
                        **{key:float(value) for key,value in attention_diagnostics(attention,hidden,live_mask).items()}}
                    for quantile,label in ((.01,"p1"),(.5,"p50"),(.99,"p99")):
                        row[f"ratio_{label}"] = float(torch.quantile(live_ratio,quantile))
                    for index,name in enumerate(("psi","theta","v")):
                        row[f"policy_log_std_mean_{name}"] = float(masked_mean(log_std[...,index],live_mask))
                        for threshold,label in ((.9,"0_9"),(.99,"0_99"),(.999,"0_999")):
                            row[f"action_abs_gt_{label}_fraction_{name}"] = float(masked_mean(
                                (batch["actions"][...,index].abs()>threshold).float(),live_mask))
                self.actor_optimizer.zero_grad()
                (actor_loss-self.entropy_coefficient*entropy_mean).backward()
                row["actor_grad_norm"] = float(nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm))
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                (self.value_loss_coefficient*value_loss).backward()
                row["critic_grad_norm"] = float(nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm))
                self.critic_optimizer.step()
                self.actor_update_count += 1
                self.critic_update_count += 1
                rows.append(row); this_epoch.append(row)
            epoch_rows.append(this_epoch)
            if self._epoch_exceeds_target(this_epoch):
                break
        if not rows:
            raise ValueError("rollout has no trainable live sequence")
        self.ppo_update_count += 1
        metrics = {key:float(np.mean([row[key] for row in rows])) for key in rows[0]}
        for key in ("approx_kl","clip_fraction","ratio_mean","ratio_std","ratio_min","ratio_max","ratio_p1","ratio_p50","ratio_p99"):
            metrics[f"first_minibatch_{key}"] = rows[0][key]
            for epoch,items in enumerate(epoch_rows):
                if items:
                    metrics[f"epoch_{epoch}_{key}"] = float(np.mean([row[key] for row in items]))
        with torch.no_grad():
            live = mask > .5
            metrics["explained_variance"] = float(1-torch.var((returns-values)[live],unbiased=False)/torch.var(returns[live],unbiased=False).clamp_min(1e-8))
        metrics.update(audit, sequence_length=float(self.sequence_length),
            sequence_chunks=float(len(chunks)), valid_environment_transitions=float(t*e),
            padded_environment_transitions=float(len(chunks)*self.sequence_length-t*e))
        metrics.update(self._stability_metrics(epoch_rows))
        if not np.isfinite(list(metrics.values())).all():
            raise FloatingPointError(f"non-finite STEA-MAPPO update: {metrics}")
        return metrics

    def checkpoint_state(self, extra=None):
        state = super().checkpoint_state(extra)
        state.pop("mappo_impl_version")
        state.update(algorithm="STEA-MAPPO", stea_mappo_impl_version=STEA_MAPPO_IMPL_VERSION,
                     network_architecture=dict(self.network_architecture))
        return state

    def load(self, path: str | Path):
        state = torch.load(path,map_location=self.device,weights_only=False)
        if state.get("algorithm") != "STEA-MAPPO":
            raise RuntimeError("checkpoint is not a STEA-MAPPO checkpoint")
        if state.get("stea_mappo_impl_version") != STEA_MAPPO_IMPL_VERSION:
            raise RuntimeError("STEA-MAPPO implementation version mismatch")
        validate_trainer_policy_protocol(state, self.policy_protocol())
        if state.get("network_architecture") != self.network_architecture:
            raise RuntimeError("STEA-MAPPO network_architecture mismatch")
        if state.get("critic_type") != self.critic_type:
            raise RuntimeError("STEA-MAPPO critic_type mismatch")
        extra = state.get("extra",{})
        for key in ("observation_dim","action_dim","num_agents"):
            if key in extra and extra[key] != getattr(self,key):
                raise RuntimeError(f"STEA-MAPPO checkpoint {key} mismatch")
        if "network_architecture" in extra and extra["network_architecture"] != self.network_architecture:
            raise RuntimeError("STEA-MAPPO extra network_architecture mismatch")
        if "stea_mappo_impl_version" in extra and extra["stea_mappo_impl_version"] != STEA_MAPPO_IMPL_VERSION:
            raise RuntimeError("STEA-MAPPO extra implementation version mismatch")
        self.actor.load_state_dict(state["actor"],strict=True)
        self.critic.load_state_dict(state["critic"],strict=True)
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        for name,key in (("ppo_update_count","ppo_updates"),("actor_update_count","actor_updates"),
                         ("critic_update_count","critic_updates"),("sampled_steps","sampled_steps"),("vector_steps","vector_steps")):
            setattr(self,name,int(state[key]))
        return dict(state.get("extra",{}))
