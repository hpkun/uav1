"""Persistent-wave trajectory memory, stratification and V-trace utilities."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
from typing import Any, Callable

import numpy as np
import torch

from .base import CapabilityModule


PWTR_MAPPO_VERSION = 1
PWTR_SEQUENCE_LENGTH = 128
PWTR_BRIDGE_HALF_LENGTH = 64
PWTR_MIN_SEGMENT_LENGTH = 32
PWTR_PARTITION_CAPACITY = 32
PWTR_ACTOR_MAX_AGE_UPDATES = 2
PWTR_REPLAY_BATCH_STATES = 512
PWTR_SEGMENTS_PER_BATCH = 4

W2_INTERNAL = "W2_INTERNAL"
W3_INTERNAL = "W3_INTERNAL"
BRIDGE_12 = "BRIDGE_12"
BRIDGE_23 = "BRIDGE_23"
PWTR_PARTITIONS = (W2_INTERNAL, W3_INTERNAL, BRIDGE_12, BRIDGE_23)
_PWTR_RNG_XOR = 0x50575452


def wave_stratified_permutation(
    wave_indices: np.ndarray, minibatch_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Return one no-replacement epoch ordered into proportionate wave batches."""
    waves = np.asarray(wave_indices, dtype=np.int64).reshape(-1)
    if waves.size == 0:
        return np.empty(0, dtype=np.int64)
    if minibatch_size <= 0:
        raise ValueError("minibatch_size must be positive")
    buckets = {wave: np.flatnonzero(waves == wave) for wave in np.unique(waves)}
    for values in buckets.values():
        rng.shuffle(values)
    offsets = {wave: 0 for wave in buckets}
    remaining = {wave: len(values) for wave, values in buckets.items()}
    ordered: list[int] = []
    while sum(remaining.values()):
        batch_size = min(minibatch_size, sum(remaining.values()))
        total = sum(remaining.values())
        expected = {wave: batch_size * count / total for wave, count in remaining.items()}
        quota = {wave: min(remaining[wave], int(math.floor(expected[wave]))) for wave in remaining}
        spare = batch_size - sum(quota.values())
        ranking = sorted(
            remaining,
            key=lambda wave: (expected[wave] - quota[wave], remaining[wave], -int(wave)),
            reverse=True,
        )
        while spare:
            progressed = False
            for wave in ranking:
                if quota[wave] < remaining[wave]:
                    quota[wave] += 1
                    spare -= 1
                    progressed = True
                    if not spare:
                        break
            if not progressed:
                raise RuntimeError("PWTR stratification could not allocate remainder")
        batch: list[int] = []
        for wave in sorted(buckets):
            start = offsets[wave]
            stop = start + quota[wave]
            batch.extend(buckets[wave][start:stop].tolist())
            offsets[wave] = stop
            remaining[wave] -= quota[wave]
        batch_array = np.asarray(batch, dtype=np.int64)
        rng.shuffle(batch_array)
        ordered.extend(batch_array.tolist())
    result = np.asarray(ordered, dtype=np.int64)
    if result.size != waves.size or not np.array_equal(np.sort(result), np.arange(waves.size)):
        raise RuntimeError("PWTR stratification violated exact sample conservation")
    return result


def replay_batch_budget(later_wave_environment_states: int) -> int:
    count = int(later_wave_environment_states)
    if count < 0:
        raise ValueError("later-wave state count cannot be negative")
    return int(math.ceil(count / PWTR_REPLAY_BATCH_STATES)) if count else 0


def replay_budget_fill_fraction(actual_batches: int, budget: int) -> float:
    if int(actual_batches) < 0 or int(budget) < 0 or int(actual_batches) > int(budget):
        raise ValueError("invalid PWTR replay budget accounting")
    return 1.0 if int(budget) == 0 else float(actual_batches) / float(budget)


def valid_collection_generations(segment: dict[str, Any]) -> np.ndarray:
    """Return the true collection generation of every valid transition."""
    valid = np.asarray(segment["valid_time_mask"]) > 0.5
    generations = np.asarray(segment["collection_ppo_update_ids"], dtype=np.int64)
    if generations.shape != valid.shape:
        raise ValueError("PWTR collection generations/valid mask shape mismatch")
    return generations[valid]


def transition_actor_age_mask(
    collection_ppo_update_ids: torch.Tensor,
    current_generation: int,
    valid_time_mask: torch.Tensor,
    alive_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return transition ages and the exact valid/alive Actor replay mask."""
    ages = int(current_generation) - collection_ppo_update_ids
    mask = (ages <= PWTR_ACTOR_MAX_AGE_UPDATES).to(alive_masks.dtype)
    mask = mask.unsqueeze(-1) * valid_time_mask.unsqueeze(-1) * alive_masks
    return ages, mask


def clipped_importance_weights(
    new_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    alive_masks: torch.Tensor,
    valid_time_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Individual actor and joint critic ratios, both clipped above at one."""
    log_ratio = new_log_probs - behavior_log_probs
    alive = alive_masks * valid_time_mask.unsqueeze(-1)
    individual_rho = torch.exp(torch.clamp(log_ratio, max=0.0)).detach()
    joint_log_ratio = (log_ratio * alive).sum(dim=-1)
    joint_rho = torch.exp(torch.clamp(joint_log_ratio, max=0.0)).detach()
    return log_ratio, individual_rho, joint_log_ratio, joint_rho


def normalized_ess_and_freshness(
    joint_log_ratio: torch.Tensor, valid_state_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = joint_log_ratio[valid_state_mask > 0.5]
    if values.numel() == 0:
        zero = joint_log_ratio.new_zeros(())
        return zero, zero, zero
    weights = torch.exp(values - values.max())
    ess = weights.sum().square() / (values.numel() * weights.square().sum().clamp_min(1e-30))
    divergence = values.abs().mean()
    # Preserve the mathematical (0, 1] range even for extreme but finite
    # policy drift; this is a numerical floor, not a tunable coefficient.
    freshness = (ess * torch.exp(-divergence)).clamp_min(torch.finfo(values.dtype).tiny)
    return ess, divergence, freshness


def vtrace_targets_and_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    alive_masks: torch.Tensor,
    next_alive_masks: torch.Tensor,
    valid_time_mask: torch.Tensor,
    joint_rho: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rho_bar=c_bar=1 V-trace targets on padded [B,T,N] data."""
    if rewards.ndim != 3:
        raise ValueError("PWTR V-trace tensors must be [B,T,N]")
    valid = valid_time_mask.unsqueeze(-1).to(rewards.dtype)
    continuation = (1.0 - dones).unsqueeze(-1) * next_alive_masks * valid
    rho = joint_rho.unsqueeze(-1)
    delta = rho * (rewards + float(gamma) * continuation * next_values - values) * valid
    accumulator = torch.zeros_like(values[:, 0])
    corrections = torch.zeros_like(values)
    for time_index in range(values.shape[1] - 1, -1, -1):
        accumulator = delta[:, time_index] + (
            float(gamma)
            * continuation[:, time_index]
            * rho[:, time_index]
            * accumulator
        )
        accumulator = accumulator * valid[:, time_index]
        corrections[:, time_index] = accumulator
    targets = (values + corrections).detach()
    next_targets = next_values.detach().clone()
    if values.shape[1] > 1:
        has_next = valid_time_mask[:, 1:].unsqueeze(-1).to(torch.bool)
        next_targets[:, :-1] = torch.where(has_next, targets[:, 1:], next_targets[:, :-1])
    advantages = (
        rewards + float(gamma) * continuation * next_targets - values
    ).detach() * valid
    return targets, advantages


class PersistentWaveTrajectoryReplayModule(CapabilityModule):
    name = "persistent_wave_trajectory_replay"
    version = PWTR_MAPPO_VERSION

    def __init__(self, config: dict[str, Any] | None = None, seed: int = 0) -> None:
        super().__init__(config)
        allowed = {
            "enabled", "fresh_wave_stratification", "replay_enabled", "replay_source",
            "priority_enabled", "bridge_enabled", "actor_replay", "critic_replay",
            "sequence_length", "bridge_half_length", "min_segment_length",
            "partition_capacity", "actor_max_age_updates",
        }
        unknown = set(self.config) - allowed
        if unknown:
            raise ValueError(f"PWTR has unsupported configuration keys: {sorted(unknown)}")
        self.fresh_wave_stratification = bool(self.config.get("fresh_wave_stratification", False))
        self.replay_enabled = bool(self.config.get("replay_enabled", False))
        self.replay_source = str(self.config.get("replay_source", "recent_uniform"))
        self.priority_enabled = bool(self.config.get("priority_enabled", False))
        self.bridge_enabled = bool(self.config.get("bridge_enabled", False))
        self.actor_replay = bool(self.config.get("actor_replay", self.replay_enabled))
        self.critic_replay = bool(self.config.get("critic_replay", self.replay_enabled))
        fixed = {
            "sequence_length": PWTR_SEQUENCE_LENGTH,
            "bridge_half_length": PWTR_BRIDGE_HALF_LENGTH,
            "min_segment_length": PWTR_MIN_SEGMENT_LENGTH,
            "partition_capacity": PWTR_PARTITION_CAPACITY,
            "actor_max_age_updates": PWTR_ACTOR_MAX_AGE_UPDATES,
        }
        for key, expected in fixed.items():
            if int(self.config.get(key, expected)) != expected:
                raise ValueError(f"PWTR V1 fixes {key}={expected}")
        if self.replay_source not in {"current", "recent_uniform", "recent_priority"}:
            raise ValueError("PWTR replay_source must be current, recent_uniform, or recent_priority")
        if self.priority_enabled != (self.replay_source == "recent_priority"):
            raise ValueError("PWTR priority_enabled must match recent_priority source")
        if not self.replay_enabled and any((self.actor_replay, self.critic_replay, self.priority_enabled, self.bridge_enabled)):
            raise ValueError("disabled PWTR replay cannot enable replay-only capabilities")
        if self.replay_enabled and not (self.actor_replay or self.critic_replay):
            raise ValueError("PWTR replay requires actor_replay and/or critic_replay")
        self.rng = np.random.default_rng(int(seed) ^ _PWTR_RNG_XOR)
        self.partitions = {name: deque(maxlen=PWTR_PARTITION_CAPACITY) for name in PWTR_PARTITIONS}
        self.pending_internal: dict[int, dict[int, list[dict[str, Any]]]] = {}
        self.source_tails: dict[int, dict[int, deque]] = {}
        self.pending_bridges: dict[int, dict[int, dict[str, Any]]] = {}
        self.entry_survivors: dict[int, dict[int, int]] = {}
        self.segment_id_counter = 0
        self.fresh_rollout_count = 0
        self.replay_phase_count = 0
        self.replay_actor_optimizer_steps = 0
        self.replay_critic_optimizer_steps = 0
        self.pending_dropped_on_resume = 0
        self.current_rollout_generation = -1
        self.current_later_states = 0

    def _ensure_env(self, env_id: int) -> None:
        if env_id not in self.pending_internal:
            self.pending_internal[env_id] = {2: [], 3: []}
            self.source_tails[env_id] = {
                1: deque(maxlen=PWTR_BRIDGE_HALF_LENGTH),
                2: deque(maxlen=PWTR_BRIDGE_HALF_LENGTH),
            }
            self.pending_bridges[env_id] = {}
            self.entry_survivors[env_id] = {2: 0, 3: 0}

    @staticmethod
    def _transition(batch: Any, time_index: int, env_id: int, generation: int) -> dict[str, Any]:
        return {
            "observations": np.asarray(batch.observations[time_index, env_id], dtype=np.float32).copy(),
            "next_observations": np.asarray(batch.next_observations[time_index, env_id], dtype=np.float32).copy(),
            "raw_actions": np.asarray(batch.raw_actions[time_index, env_id], dtype=np.float32).copy(),
            "behavior_log_probs": np.asarray(batch.old_log_probs[time_index, env_id], dtype=np.float32).copy(),
            "rewards": np.asarray(batch.rewards[time_index, env_id], dtype=np.float32).copy(),
            "dones": float(batch.dones[time_index, env_id]),
            "alive_masks": np.asarray(batch.alive_masks[time_index, env_id], dtype=np.float32).copy(),
            "next_alive_masks": np.asarray(batch.next_alive_masks[time_index, env_id], dtype=np.float32).copy(),
            "wave_indices": int(batch.wave_indices[time_index, env_id]),
            "collection_ppo_update_ids": int(generation),
        }

    def _finalize(self, rows: list[dict[str, Any]], segment_type: str, env_id: int,
                  entry_survivor_count: int, *, ordinary: bool) -> dict[str, Any] | None:
        actual = len(rows)
        if actual < PWTR_MIN_SEGMENT_LENGTH:
            return None
        if actual > PWTR_SEQUENCE_LENGTH:
            raise RuntimeError("PWTR segment exceeds fixed sequence length")
        fields = (
            "observations", "next_observations", "raw_actions", "behavior_log_probs",
            "rewards", "dones", "alive_masks", "next_alive_masks", "wave_indices",
            "collection_ppo_update_ids",
        )
        segment: dict[str, Any] = {}
        for field in fields:
            values = np.asarray([row[field] for row in rows])
            padded = np.zeros((PWTR_SEQUENCE_LENGTH, *values.shape[1:]), dtype=values.dtype)
            padded[:actual] = values
            segment[field] = padded
        segment["valid_time_mask"] = np.concatenate((
            np.ones(actual, dtype=np.float32),
            np.zeros(PWTR_SEQUENCE_LENGTH - actual, dtype=np.float32),
        ))
        segment.update({
            "entry_survivor_count": int(entry_survivor_count),
            "segment_type": segment_type,
            "source_env_id": int(env_id),
            "segment_id": int(self.segment_id_counter),
            "segment_generation": int(max(row["collection_ppo_update_ids"] for row in rows)),
            "actual_length": int(actual),
            "ordinary_internal": bool(ordinary),
            "priority_cache": None,
            "priority_cache_generation": None,
        })
        self.segment_id_counter += 1
        self.partitions[segment_type].append(segment)
        return segment

    def _finish_internal(self, env_id: int, wave: int) -> None:
        rows = self.pending_internal[env_id][wave]
        if rows:
            self._finalize(rows, W2_INTERNAL if wave == 2 else W3_INTERNAL, env_id,
                           self.entry_survivors[env_id][wave], ordinary=True)
        self.pending_internal[env_id][wave] = []

    def _append_internal(self, env_id: int, wave: int, row: dict[str, Any]) -> None:
        pending = self.pending_internal[env_id][wave]
        pending.append(row)
        if len(pending) == PWTR_SEQUENCE_LENGTH:
            self._finalize(pending, W2_INTERNAL if wave == 2 else W3_INTERNAL, env_id,
                           self.entry_survivors[env_id][wave], ordinary=True)
            self.pending_internal[env_id][wave] = []

    def _finish_bridge(self, env_id: int, source_wave: int) -> None:
        pending = self.pending_bridges[env_id].pop(source_wave, None)
        if pending is None:
            return
        rows = pending["tail"] + pending["head"]
        self._finalize(rows, BRIDGE_12 if source_wave == 1 else BRIDGE_23, env_id,
                       pending["entry_survivor_count"], ordinary=False)

    def ingest_rollout(self, batch: Any, generation: int) -> None:
        """Extract natural segments without changing rollout data or environment state."""
        if not self.enabled:
            return
        self.current_rollout_generation = int(generation)
        waves = np.asarray(batch.wave_indices)
        self.current_later_states = int(np.isin(waves, (2, 3)).sum())
        for time_index in range(waves.shape[0]):
            for env_id in range(waves.shape[1]):
                self._ensure_env(env_id)
                row = self._transition(batch, time_index, env_id, generation)
                wave = int(row["wave_indices"])
                done = bool(row["dones"])
                spawned = bool(batch.wave_transition_flags[time_index, env_id])

                for source_wave, bridge in list(self.pending_bridges[env_id].items()):
                    if wave == source_wave + 1:
                        bridge["head"].append(row)
                        if len(bridge["head"]) == PWTR_BRIDGE_HALF_LENGTH:
                            self._finish_bridge(env_id, source_wave)

                if wave in (2, 3):
                    self._append_internal(env_id, wave, row)
                if wave in (1, 2):
                    self.source_tails[env_id][wave].append(row)

                if spawned and wave in (1, 2):
                    target_wave = wave + 1
                    survivors = int(np.asarray(row["next_alive_masks"]).sum())
                    self.entry_survivors[env_id][target_wave] = survivors
                    if wave == 2:
                        # W2 is the immutable target head of an older Bridge12.
                        # Once W2 ends, that bridge can never receive more head
                        # transitions and must be finalized immediately.
                        self._finish_bridge(env_id, 1)
                        self._finish_internal(env_id, 2)
                    if self.bridge_enabled:
                        self.pending_bridges[env_id][wave] = {
                            "tail": list(self.source_tails[env_id][wave]),
                            "head": [],
                            "entry_survivor_count": survivors,
                        }
                    self.source_tails[env_id][wave].clear()

                if done:
                    if wave in (2, 3):
                        self._finish_internal(env_id, wave)
                    for source_wave in list(self.pending_bridges[env_id]):
                        self._finish_bridge(env_id, source_wave)
                    for tail in self.source_tails[env_id].values():
                        tail.clear()
                    self.entry_survivors[env_id] = {2: 0, 3: 0}
        self.fresh_rollout_count += 1

    def fresh_epoch_permutation(self, waves: np.ndarray, minibatch_size: int,
                                rng: np.random.Generator) -> np.ndarray:
        if not self.enabled or not self.fresh_wave_stratification:
            return rng.permutation(len(np.asarray(waves).reshape(-1)))
        return wave_stratified_permutation(waves, minibatch_size, rng)

    def eligible(self, generation: int) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for partition, rows in self.partitions.items():
            if partition.startswith("BRIDGE") and not self.bridge_enabled:
                continue
            if self.replay_source == "current":
                candidates = [row for row in rows if (
                    valid_collection_generations(row).size > 0
                    and np.all(valid_collection_generations(row) == int(generation))
                )]
            else:
                candidates = [row for row in rows if (
                    valid_collection_generations(row).size > 0
                    and np.all(valid_collection_generations(row) < int(generation))
                )]
            if candidates:
                result[partition] = candidates
        return result

    def sample_batch(
        self, generation: int,
        priority_function: Callable[[dict[str, Any]], dict[str, float]] | None = None,
    ) -> list[dict[str, Any]]:
        eligible = self.eligible(generation)
        if not eligible:
            return []
        volumes = np.asarray([
            sum(int(row["valid_time_mask"].sum()) for row in eligible[name])
            for name in eligible
        ], dtype=np.float64)
        names = list(eligible)
        partition_probabilities = volumes / volumes.sum()
        selected: list[dict[str, Any]] = []
        for _ in range(PWTR_SEGMENTS_PER_BATCH):
            partition = names[int(self.rng.choice(len(names), p=partition_probabilities))]
            candidates = eligible[partition]
            if self.priority_enabled:
                if priority_function is None:
                    raise RuntimeError("PWTR priority sampling requires a score function")
                scores = []
                for candidate in candidates:
                    diagnostics = candidate.get("priority_cache")
                    if diagnostics is None or candidate.get("priority_cache_generation") != int(generation):
                        diagnostics = priority_function(candidate)
                        candidate["priority_cache"] = diagnostics
                        candidate["priority_cache_generation"] = int(generation)
                    scores.append(max(float(diagnostics["priority"]), 0.0))
                probabilities = np.asarray(scores, dtype=np.float64)
                probabilities = probabilities / probabilities.sum() if probabilities.sum() > 0 else np.full(len(candidates), 1 / len(candidates))
                index = int(self.rng.choice(len(candidates), p=probabilities))
            else:
                index = int(self.rng.integers(len(candidates)))
            selected.append(candidates[index])
        return selected

    @staticmethod
    def stack_segments(segments: list[dict[str, Any]]) -> dict[str, Any]:
        if not segments:
            raise ValueError("cannot stack an empty PWTR replay batch")
        array_fields = (
            "observations", "next_observations", "raw_actions", "behavior_log_probs",
            "rewards", "dones", "alive_masks", "next_alive_masks", "wave_indices",
            "valid_time_mask", "collection_ppo_update_ids",
        )
        result = {field: np.stack([segment[field] for segment in segments]) for field in array_fields}
        result.update({
            "segment_types": [segment["segment_type"] for segment in segments],
            "segment_generations": np.asarray([segment["segment_generation"] for segment in segments], dtype=np.int64),
            "segment_ids": np.asarray([segment["segment_id"] for segment in segments], dtype=np.int64),
        })
        return result

    def memory_counts(self) -> dict[str, float]:
        return {
            "pwtr_memory_w2_segments": float(len(self.partitions[W2_INTERNAL])),
            "pwtr_memory_w3_segments": float(len(self.partitions[W3_INTERNAL])),
            "pwtr_memory_bridge12_segments": float(len(self.partitions[BRIDGE_12])),
            "pwtr_memory_bridge23_segments": float(len(self.partitions[BRIDGE_23])),
        }

    def default_metrics(self, waves: np.ndarray | None = None) -> dict[str, float | str]:
        counts = {wave: 0 for wave in (1, 2, 3)}
        if waves is not None:
            values = np.asarray(waves)
            counts = {wave: int((values == wave).sum()) for wave in counts}
        return {
            "pwtr_enabled": float(self.enabled),
            "pwtr_version": float(self.version),
            "pwtr_fresh_stratified": float(self.enabled and self.fresh_wave_stratification),
            "pwtr_replay_enabled": float(self.enabled and self.replay_enabled),
            # Optimization metrics are intentionally numeric (the established
            # trainer contract); run/checkpoint identity stores the readable
            # replay_source string. 0=disabled, 1=current, 2=recent_uniform,
            # 3=recent_priority.
            "pwtr_replay_source": float(0 if not self.enabled else {
                "current": 1, "recent_uniform": 2, "recent_priority": 3,
            }[self.replay_source]),
            "pwtr_priority_enabled": float(self.enabled and self.priority_enabled),
            "pwtr_bridge_enabled": float(self.enabled and self.bridge_enabled),
            "pwtr_pending_dropped_on_resume": float(self.pending_dropped_on_resume),
            **self.memory_counts(),
            **{f"pwtr_fresh_w{wave}_states": float(counts[wave]) for wave in (1, 2, 3)},
            "pwtr_replay_batches": 0.0, "pwtr_replay_valid_states": 0.0,
            "pwtr_replay_alive_samples": 0.0, "pwtr_replay_w2_fraction": 0.0,
            "pwtr_replay_w3_fraction": 0.0, "pwtr_replay_bridge12_fraction": 0.0,
            "pwtr_replay_bridge23_fraction": 0.0, "pwtr_mean_segment_age": 0.0,
            "pwtr_max_segment_age": 0.0, "pwtr_mean_learning_potential": 0.0,
            "pwtr_mean_transition_age": 0.0, "pwtr_max_transition_age": 0.0,
            "pwtr_mean_freshness": 0.0, "pwtr_mean_priority": 0.0,
            "pwtr_mean_joint_abs_log_ratio": 0.0, "pwtr_mean_normalized_ess": 0.0,
            "pwtr_actor_eligible_fraction": 0.0, "pwtr_vtrace_rho_mean": 0.0,
            "pwtr_vtrace_rho_min": 0.0, "pwtr_vtrace_rho_max": 0.0,
            "pwtr_actor_replay_loss": 0.0, "pwtr_critic_replay_loss": 0.0,
            "pwtr_actor_replay_grad_norm_preclip": 0.0,
            "pwtr_critic_replay_grad_norm_preclip": 0.0,
            "pwtr_replay_actor_optimizer_steps": float(self.replay_actor_optimizer_steps),
            "pwtr_replay_critic_optimizer_steps": float(self.replay_critic_optimizer_steps),
            "pwtr_replay_phase_count": float(self.replay_phase_count),
            "pwtr_replay_budget_fill_fraction": 1.0,
        }

    def discard_pending_after_environment_restart(self) -> int:
        """Drop only trajectory fragments that belong to the old physical env."""
        dropped = 0
        for rows in self.pending_internal.values():
            dropped += sum(bool(values) for values in rows.values())
        for tails in self.source_tails.values():
            dropped += sum(bool(values) for values in tails.values())
        dropped += sum(len(values) for values in self.pending_bridges.values())
        self.pending_internal = {}
        self.source_tails = {}
        self.pending_bridges = {}
        self.entry_survivors = {}
        self.pending_dropped_on_resume += int(dropped)
        return int(dropped)

    def state_dict(self) -> dict[str, Any]:
        return deepcopy({
            "version": self.version,
            "config": self.config,
            "partitions": {name: list(rows) for name, rows in self.partitions.items()},
            "pending_internal": self.pending_internal,
            "source_tails": {env: {wave: list(rows) for wave, rows in tails.items()} for env, tails in self.source_tails.items()},
            "pending_bridges": self.pending_bridges,
            "entry_survivors": self.entry_survivors,
            "segment_id_counter": self.segment_id_counter,
            "fresh_rollout_count": self.fresh_rollout_count,
            "replay_phase_count": self.replay_phase_count,
            "replay_actor_optimizer_steps": self.replay_actor_optimizer_steps,
            "replay_critic_optimizer_steps": self.replay_critic_optimizer_steps,
            "pending_dropped_on_resume": self.pending_dropped_on_resume,
            "current_rollout_generation": self.current_rollout_generation,
            "current_later_states": self.current_later_states,
            "rng_state": self.rng.bit_generator.state,
        })

    def load_state_dict(self, state: dict[str, Any] | None, *, branch_from_plain: bool = False) -> None:
        if state is None:
            if branch_from_plain:
                return
            raise RuntimeError("PWTR checkpoint is missing trajectory replay state")
        if int(state.get("version", -1)) != self.version or state.get("config") != self.config:
            raise RuntimeError("PWTR checkpoint version/config mismatch")
        self.partitions = {
            name: deque(deepcopy(state["partitions"].get(name, [])), maxlen=PWTR_PARTITION_CAPACITY)
            for name in PWTR_PARTITIONS
        }
        self.pending_internal = deepcopy(state.get("pending_internal", {}))
        self.source_tails = {
            int(env): {int(wave): deque(deepcopy(rows), maxlen=PWTR_BRIDGE_HALF_LENGTH) for wave, rows in tails.items()}
            for env, tails in state.get("source_tails", {}).items()
        }
        self.pending_bridges = deepcopy(state.get("pending_bridges", {}))
        self.entry_survivors = deepcopy(state.get("entry_survivors", {}))
        for name in (
            "segment_id_counter", "fresh_rollout_count", "replay_phase_count",
            "replay_actor_optimizer_steps", "replay_critic_optimizer_steps",
            "pending_dropped_on_resume",
            "current_rollout_generation", "current_later_states",
        ):
            setattr(self, name, int(state.get(name, getattr(self, name))))
        self.rng.bit_generator.state = deepcopy(state["rng_state"])


__all__ = [
    "PWTR_MAPPO_VERSION", "PWTR_SEQUENCE_LENGTH", "PWTR_BRIDGE_HALF_LENGTH",
    "PWTR_MIN_SEGMENT_LENGTH", "PWTR_PARTITION_CAPACITY", "PWTR_ACTOR_MAX_AGE_UPDATES",
    "PWTR_REPLAY_BATCH_STATES", "PWTR_SEGMENTS_PER_BATCH", "W2_INTERNAL", "W3_INTERNAL",
    "BRIDGE_12", "BRIDGE_23", "PWTR_PARTITIONS", "PersistentWaveTrajectoryReplayModule",
    "wave_stratified_permutation", "replay_batch_budget", "replay_budget_fill_fraction",
    "clipped_importance_weights",
    "valid_collection_generations", "transition_actor_age_mask",
    "normalized_ess_and_freshness", "vtrace_targets_and_advantages",
]
