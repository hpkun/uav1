"""Read-only wave-wise gradient-conflict audit for the frozen Plain MAPPO runs.

This module deliberately contains no optimizer or trainer update path.  It uses
``torch.autograd.grad`` on natural stochastic rollouts and verifies that model,
optimizer, and update-counter fingerprints are unchanged after every checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.mappo.trainer import compute_gae, masked_mean
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_modular_checkpoint
from algorithm.modular_mappo.trainer import stable_ratio_terms


RUNS = {
    5301: ROOT / "outputs/diag_mappo_learnability/l3_seed5301",
    5302: ROOT / "outputs/diag_mappo_learnability/l3_seed5302",
    5303: ROOT / "outputs/diag_mappo_learnability/l3_seed5303",
}
TARGET_STAGES = (("mid", 1_500_000, 120_000, False),
                 ("late", 2_000_000, 120_000, False),
                 ("exact3m", 3_000_000, 0, True))
DIAGNOSTIC_ENV_SEED_BASE = 88_110_000
DIAGNOSTIC_TORCH_SEED = 88_120_000
FORBIDDEN_RANGES = ((44_000_000, 44_000_049), (45_000_000, 45_000_199))
WAVES = (1, 2, 3)
PAIRS = ((1, 2), (1, 3), (2, 3))
MIN_ALIVE_AGENT_SAMPLES = 64
CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)\.pt$")
RAW_REQUIRED_FIELDS = (
    "training_seed", "checkpoint_path", "checkpoint_step", "rollout_index",
    "diagnostic_env_seed_base", "diagnostic_torch_seed",
    "transition_samples_w1", "transition_samples_w2", "transition_samples_w3",
    "alive_agent_samples_w1", "alive_agent_samples_w2", "alive_agent_samples_w3",
    "actor_loss_w1", "actor_loss_w2", "actor_loss_w3",
    "actor_grad_norm_w1", "actor_grad_norm_w2", "actor_grad_norm_w3",
    "critic_loss_w1", "critic_loss_w2", "critic_loss_w3",
    "critic_grad_norm_w1", "critic_grad_norm_w2", "critic_grad_norm_w3",
    "actor_normadv_w1_w2_cosine", "actor_normadv_w1_w3_cosine", "actor_normadv_w2_w3_cosine",
    "actor_rawadv_w1_w2_cosine", "actor_rawadv_w1_w3_cosine", "actor_rawadv_w2_w3_cosine",
    "critic_w1_w2_cosine", "critic_w1_w3_cosine", "critic_w2_w3_cosine",
    "actor_within_w1_cos", "actor_within_w2_cos", "actor_within_w3_cos",
    "critic_within_w1_cos", "critic_within_w2_cos", "critic_within_w3_cos",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_update(digest: Any, value: Any) -> None:
    """Hash nested tensor/optimizer state deterministically, without serialization."""
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0" + str(array.dtype).encode() + b"\0")
        digest.update(json.dumps(list(array.shape)).encode() + b"\0" + array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: str(item)):
            _hash_update(digest, str(key)); _hash_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(("tuple" if isinstance(value, tuple) else "list").encode() + b"\0")
        for item in value:
            _hash_update(digest, item)
    else:
        digest.update(type(value).__name__.encode() + b"\0" + repr(value).encode() + b"\0")


def state_sha256(value: Any) -> str:
    digest = hashlib.sha256(); _hash_update(digest, value); return digest.hexdigest()


def mutation_fingerprint(trainer: Any) -> dict[str, Any]:
    return {
        "actor_parameters_sha256": state_sha256(trainer.actor.state_dict()),
        "critic_parameters_sha256": state_sha256(trainer.critic.state_dict()),
        "actor_optimizer_sha256": state_sha256(trainer.actor_optimizer.state_dict()),
        "critic_optimizer_sha256": state_sha256(trainer.critic_optimizer.state_dict()),
        "actor_update_count": int(trainer.actor_update_count),
        "critic_update_count": int(trainer.critic_update_count),
        "ppo_update_count": int(trainer.ppo_update_count),
    }


def flatten_gradients(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter],
                      *, retain_graph: bool = True) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True,
                                    retain_graph=retain_graph, create_graph=False)
    pieces = [torch.zeros_like(parameter).reshape(-1) if gradient is None
              else gradient.reshape(-1) for parameter, gradient in zip(parameters, gradients)]
    if not pieces:
        return torch.empty(0, device=loss.device)
    return torch.cat(pieces)


def cosine_metrics(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> dict[str, float]:
    dot = float(torch.dot(left, right).detach().cpu())
    left_norm = float(torch.linalg.vector_norm(left).detach().cpu())
    right_norm = float(torch.linalg.vector_norm(right).detach().cpu())
    cosine = dot / (left_norm * right_norm + eps)
    norm_ratio = min(left_norm, right_norm) / max(left_norm, right_norm, eps)
    return {"dot": dot, "cosine": cosine, "left_norm": left_norm,
            "right_norm": right_norm, "norm_ratio": norm_ratio}


def wave_agent_mask(alive_masks: torch.Tensor, wave_indices: torch.Tensor, wave: int) -> torch.Tensor:
    if alive_masks.ndim != 3 or wave_indices.shape != alive_masks.shape[:2]:
        raise ValueError("alive_masks must be [T,E,N] and wave_indices [T,E]")
    return alive_masks * (wave_indices == int(wave)).to(alive_masks.dtype).unsqueeze(-1)


def global_live_advantage_normalization(advantages: torch.Tensor,
                                        alive_masks: torch.Tensor) -> torch.Tensor:
    live = advantages[alive_masks > .5]
    if live.numel() == 0:
        raise ValueError("cannot normalize an empty live-agent set")
    return ((advantages - live.mean()) / live.std(unbiased=False).clamp_min(1e-8)) * alive_masks


def ppo_clipped_surrogate(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor,
                          advantages: torch.Tensor, clip_ratio: float) -> torch.Tensor:
    _, ratio = stable_ratio_terms(new_log_prob, old_log_prob)
    return torch.minimum(ratio * advantages,
                         ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages)


def clipped_value_error(current: torch.Tensor, old: torch.Tensor, target: torch.Tensor,
                        clip_ratio: float, clip_value_loss: bool) -> torch.Tensor:
    plain = (current - target).square()
    if not clip_value_loss:
        return plain
    clipped = old + (current - old).clamp(-clip_ratio, clip_ratio)
    return torch.maximum(plain, (clipped - target).square())


def sufficient_wave_samples(count: int, minimum: int = MIN_ALIVE_AGENT_SAMPLES) -> bool:
    return int(count) >= int(minimum)


def split_half_masks(mask: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically split scalar live-agent positions as evenly as possible."""
    indices = torch.nonzero(mask.reshape(-1) > .5, as_tuple=False).squeeze(-1).cpu().numpy()
    rng = np.random.default_rng(int(seed)); indices = rng.permutation(indices)
    midpoint = len(indices) // 2
    first = torch.zeros(mask.numel(), dtype=mask.dtype, device=mask.device)
    second = torch.zeros_like(first)
    if midpoint:
        first[torch.as_tensor(indices[:midpoint], device=mask.device)] = 1
    if len(indices) - midpoint:
        second[torch.as_tensor(indices[midpoint:], device=mask.device)] = 1
    return first.view_as(mask), second.view_as(mask)


def seed_is_forbidden(seed: int) -> bool:
    return any(start <= int(seed) <= end for start, end in FORBIDDEN_RANGES)


def scan_checkpoints(run_dir: Path) -> dict[int, Path]:
    result = {}
    for path in run_dir.glob("checkpoint_*.pt"):
        match = CHECKPOINT_PATTERN.match(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def select_checkpoint(checkpoints: dict[int, Path], target: int, max_deviation: int,
                      exact: bool = False) -> tuple[int, Path]:
    if exact:
        if target not in checkpoints:
            raise RuntimeError(f"exact checkpoint_{target}.pt is required")
        return target, checkpoints[target]
    if not checkpoints:
        raise RuntimeError("no checkpoint_*.pt files found")
    step = min(checkpoints, key=lambda value: (abs(value - target), value))
    if abs(step - target) > max_deviation:
        raise RuntimeError(f"nearest checkpoint {step} exceeds {max_deviation} from {target}")
    return step, checkpoints[step]


def enabled_modules(config: dict[str, Any]) -> list[str]:
    return sorted(name for name, value in config.get("modules", {}).items()
                  if isinstance(value, dict) and bool(value.get("enabled", False)))


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid mapping in {path}")
    return value


def validate_run_and_checkpoint(run_dir: Path, training_seed: int, checkpoint: Path,
                                checkpoint_step: int, device: str) -> tuple[Any, dict, dict, dict]:
    env_config = load_yaml(run_dir / "env_config.yaml")
    algorithm_config = load_yaml(run_dir / "algorithm_config.yaml")
    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    required = {
        "algorithm": run_config.get("algorithm") == "modular_mappo",
        "environment_variant": env_config.get("environment_variant") == "persistent_wave_v2",
        "total_waves": int(env_config.get("persistent_waves", {}).get("total_waves", -1)) == 3,
        "max_steps": int(env_config.get("simulation", {}).get("max_steps", -1)) == 3000,
        "observation_dim": int(algorithm_config["network"]["observation_dim"]) == 52,
        "action_dim": int(algorithm_config["network"]["action_dim"]) == 3,
        "num_agents": int(algorithm_config["network"]["num_agents"]) == 4,
        "training_seed": int(run_config.get("seed", -1)) == int(training_seed),
        "plain_modules": enabled_modules(algorithm_config) == ["actor_lr_decay"],
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise RuntimeError(f"run protocol mismatch for {run_dir}: {failed}")
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(state.get("sampled_steps", -1)) != checkpoint_step:
        raise RuntimeError("checkpoint filename/state sampled_steps mismatch")
    validate_modular_checkpoint(state, env_config, algorithm_config,
                                expected_runtime={"training_seed": int(training_seed)})
    if sorted(state.get("enabled_modules", [])) != ["actor_lr_decay"]:
        raise RuntimeError("checkpoint is not exact Plain+actor_lr_decay protocol")
    trainer = build_modular_mappo_trainer(
        algorithm_config, device=device,
        hidden_dim=int(algorithm_config["network"]["actor_hidden_layers"][0]),
        total_sampled_steps=int(algorithm_config["training"]["total_sampled_steps"]),
    )
    trainer.load(checkpoint, strict_protocol=True, restore_rng=False)
    trainer.actor.eval(); trainer.critic.eval()
    return trainer, env_config, algorithm_config, state


def _actor_group_indices(trainer: Any) -> dict[str, list[int]]:
    mapping = {"actor_backbone": [], "actor_mean_head": [], "actor_log_std": []}
    parameters = list(trainer.actor.trainable_policy_parameters())
    positions = {id(parameter): index for index, parameter in enumerate(parameters)}
    for name, parameter in trainer.actor.named_parameters():
        if id(parameter) not in positions:
            continue
        group = ("actor_backbone" if name.startswith("backbone.") else
                 "actor_mean_head" if name.startswith("mean.") else
                 "actor_log_std" if name.startswith("log_std.") else None)
        if group is not None:
            mapping[group].append(positions[id(parameter)])
    return {key: value for key, value in mapping.items() if value}


def _group_vector(full_gradients: list[torch.Tensor], indices: Sequence[int]) -> torch.Tensor:
    return torch.cat([full_gradients[index].reshape(-1) for index in indices])


def _gradient_list(loss: torch.Tensor, parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    values = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
    return [torch.zeros_like(parameter) if value is None else value
            for parameter, value in zip(parameters, values)]


def _flat(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in values])


def _loss_gradient(loss_values: torch.Tensor, mask: torch.Tensor,
                   parameters: Sequence[torch.nn.Parameter]) -> tuple[float | None, list[torch.Tensor] | None]:
    if int(mask.sum().item()) == 0:
        return None, None
    loss = masked_mean(loss_values, mask)
    return float(loss.detach().cpu()), _gradient_list(loss, parameters)


def _within_cos(loss_values: torch.Tensor, mask: torch.Tensor,
                parameters: Sequence[torch.nn.Parameter], seed: int) -> float | None:
    first, second = split_half_masks(mask, seed)
    if int(first.sum().item()) == 0 or int(second.sum().item()) == 0:
        return None
    _, grad_first = _loss_gradient(loss_values, first, parameters)
    _, grad_second = _loss_gradient(loss_values, second, parameters)
    return cosine_metrics(_flat(grad_first), _flat(grad_second))["cosine"]


def collect_rollout(trainer: Any, vector: ParallelVectorEnv, observations: np.ndarray,
                    waves: np.ndarray, rollout_steps: int) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    storage: dict[str, list[np.ndarray]] = defaultdict(list)
    for _ in range(int(rollout_steps)):
        alive = vector.current_alive_masks.copy(); pre_wave = waves.copy()
        actions, raw, log_prob, _ = trainer.act(
            observations, alive, deterministic=False, return_policy_data=True)
        values, _ = trainer.values_step(observations, alive)
        result = vector.step_batch(actions)
        done = np.asarray(result.terminated | result.truncated, dtype=np.float32)
        next_values, _ = trainer.values_step(result.transition_next_observations,
                                             result.next_alive_masks)
        for key, value in {
            "observations": observations, "actions": actions, "raw_actions": raw,
            "old_log_probs": log_prob, "rewards": result.rewards,
            "dones": done, "alive_masks": alive,
            "next_alive_masks": result.next_alive_masks,
            "values": values, "next_values": next_values,
            "wave_indices": pre_wave,
        }.items():
            storage[key].append(np.asarray(value).copy())
        next_wave = np.asarray([int(info.get("wave_index", 1)) for info in result.infos])
        waves = np.where(done.astype(bool), 1, next_wave)
        observations = result.observations
    return {key: np.asarray(value) for key, value in storage.items()}, observations, waves


def audit_rollout(trainer: Any, rollout: dict[str, np.ndarray], rollout_index: int,
                  split_seed: int, minimum_samples: int) -> dict[str, Any]:
    device = trainer.device
    tensor = lambda value, dtype=torch.float32: torch.as_tensor(value, dtype=dtype, device=device)
    obs = tensor(rollout["observations"]); actions = tensor(rollout["actions"])
    raw = tensor(rollout["raw_actions"]); old_log = tensor(rollout["old_log_probs"])
    rewards = tensor(rollout["rewards"]); dones = tensor(rollout["dones"])
    alive = tensor(rollout["alive_masks"]); next_alive = tensor(rollout["next_alive_masks"])
    old_values = tensor(rollout["values"]); next_values = tensor(rollout["next_values"])
    waves = tensor(rollout["wave_indices"], torch.long)
    with torch.no_grad():
        raw_advantages, returns = compute_gae(
            rewards, old_values, next_values, dones, alive, next_alive,
            trainer.gamma, trainer.gae_lambda)
        normalized_advantages = global_live_advantage_normalization(raw_advantages, alive)
    shape = obs.shape; flat_obs = obs.reshape(-1, shape[-2], shape[-1])
    flat_alive = alive.reshape(-1, shape[-2])
    current_values = trainer.critic.forward_step(flat_obs, flat_alive)[0].reshape_as(old_values)
    distribution, _ = trainer.actor.distribution_step(obs, None, None, None, alive)
    new_log = trainer.actor._squashed_log_prob(distribution, raw, actions)
    normalized_surrogate = ppo_clipped_surrogate(
        new_log, old_log, normalized_advantages, trainer.clip_ratio)
    raw_surrogate = ppo_clipped_surrogate(new_log, old_log, raw_advantages, trainer.clip_ratio)
    value_error = .5 * clipped_value_error(
        current_values, old_values, returns, trainer.clip_ratio, trainer.clip_value_loss)

    actor_parameters = list(trainer.actor.trainable_policy_parameters())
    critic_parameters = list(trainer.critic.parameters())
    masks = {wave: wave_agent_mask(alive, waves, wave) for wave in WAVES}
    transition_counts = {wave: int((waves == wave).sum().item()) for wave in WAVES}
    alive_counts = {wave: int(masks[wave].sum().item()) for wave in WAVES}
    actor_losses: dict[int, float | None] = {}; critic_losses: dict[int, float | None] = {}
    actor_gradients: dict[int, list[torch.Tensor] | None] = {}
    actor_raw_gradients: dict[int, list[torch.Tensor] | None] = {}
    critic_gradients: dict[int, list[torch.Tensor] | None] = {}
    for wave in WAVES:
        actor_losses[wave], actor_gradients[wave] = _loss_gradient(
            -normalized_surrogate, masks[wave], actor_parameters)
        _, actor_raw_gradients[wave] = _loss_gradient(-raw_surrogate, masks[wave], actor_parameters)
        critic_losses[wave], critic_gradients[wave] = _loss_gradient(
            value_error, masks[wave], critic_parameters)

    row: dict[str, Any] = {"rollout_index": rollout_index}
    for wave in WAVES:
        row[f"transition_samples_w{wave}"] = transition_counts[wave]
        row[f"alive_agent_samples_w{wave}"] = alive_counts[wave]
        row[f"actor_loss_w{wave}"] = actor_losses[wave]
        row[f"critic_loss_w{wave}"] = critic_losses[wave]
        row[f"actor_grad_norm_w{wave}"] = (None if actor_gradients[wave] is None else
                                              float(torch.linalg.vector_norm(_flat(actor_gradients[wave])).cpu()))
        row[f"critic_grad_norm_w{wave}"] = (None if critic_gradients[wave] is None else
                                               float(torch.linalg.vector_norm(_flat(critic_gradients[wave])).cpu()))
        row[f"actor_within_w{wave}_cos"] = _within_cos(
            -normalized_surrogate, masks[wave], actor_parameters,
            split_seed + rollout_index * 100 + wave)
        row[f"critic_within_w{wave}_cos"] = _within_cos(
            value_error, masks[wave], critic_parameters,
            split_seed + rollout_index * 100 + 10 + wave)

    group_indices = _actor_group_indices(trainer)
    valid_reasons = []
    for first, second in PAIRS:
        label = f"w{first}_w{second}"
        valid = sufficient_wave_samples(alive_counts[first], minimum_samples) and sufficient_wave_samples(alive_counts[second], minimum_samples)
        row[f"valid_{label}"] = bool(valid)
        reason = "VALID" if valid else ";".join(
            f"W{wave}_ALIVE_SAMPLES_{alive_counts[wave]}_LT_{minimum_samples}"
            for wave in (first, second) if alive_counts[wave] < minimum_samples)
        row[f"invalid_reason_{label}"] = "" if valid else reason; valid_reasons.append(reason)
        for network, gradients in (("actor_normadv", actor_gradients),
                                   ("actor_rawadv", actor_raw_gradients),
                                   ("critic", critic_gradients)):
            metrics = None if gradients[first] is None or gradients[second] is None else cosine_metrics(
                _flat(gradients[first]), _flat(gradients[second]))
            prefix = f"{network}_{label}"
            for metric in ("cosine", "dot", "norm_ratio"):
                row[f"{prefix}_{metric}"] = None if metrics is None else metrics[metric]
            row[f"{prefix}_conflict"] = None if metrics is None else metrics["cosine"] < 0
            row[f"{prefix}_strong_conflict"] = None if metrics is None else metrics["cosine"] < -.1
        for group, indices in group_indices.items():
            metrics = None if actor_gradients[first] is None or actor_gradients[second] is None else cosine_metrics(
                _group_vector(actor_gradients[first], indices),
                _group_vector(actor_gradients[second], indices))
            row[f"{group}_{label}_cosine"] = None if metrics is None else metrics["cosine"]
    sensitive = []
    for first, second in PAIRS:
        label = f"w{first}_w{second}"
        norm = row[f"actor_normadv_{label}_cosine"]; raw_value = row[f"actor_rawadv_{label}_cosine"]
        if norm is not None and raw_value is not None and (norm < 0) != (raw_value < 0):
            sensitive.append(label)
    row["advantage_normalization_sensitive_pairs"] = ";".join(sensitive)
    row["validity_summary"] = ";".join(valid_reasons)
    return row


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(values) if values else None


def _quantile(values: Sequence[float], q: float) -> float | None:
    values = np.asarray([value for value in values if value is not None and math.isfinite(float(value))], dtype=float)
    return None if not len(values) else float(np.quantile(values, q))


def summarize_checkpoints(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in raw_rows:
        for network in ("actor_normadv", "critic"):
            for first, second in PAIRS:
                groups[(row["training_seed"], row["stage"], row["checkpoint_step"], network, first, second)].append(row)
    result = []
    for (seed, stage, step, network, first, second), rows in sorted(groups.items()):
        label = f"w{first}_w{second}"; prefix = f"{network}_{label}"
        valid = [row for row in rows if row[f"valid_{label}"] and row[f"{prefix}_cosine"] is not None]
        cosines = [float(row[f"{prefix}_cosine"]) for row in valid]
        dots = [float(row[f"{prefix}_dot"]) for row in valid]
        result.append({
            "training_seed": seed, "stage": stage, "checkpoint_step": step,
            "network": "actor" if network == "actor_normadv" else "critic",
            "wave_pair": f"W{first}-W{second}", "total_rollouts": len(rows),
            "valid_pair_rollouts": len(valid), "mean_cosine": _mean(cosines),
            "median_cosine": statistics.median(cosines) if cosines else None,
            "std_cosine": statistics.stdev(cosines) if len(cosines) > 1 else None,
            "q25_cosine": _quantile(cosines, .25), "q75_cosine": _quantile(cosines, .75),
            "negative_fraction": _mean([value < 0 for value in cosines]),
            "strong_negative_fraction": _mean([value < -.1 for value in cosines]),
            "mean_dot_product": _mean(dots), "median_dot_product": statistics.median(dots) if dots else None,
            "mean_norm_ratio": _mean([row[f"{prefix}_norm_ratio"] for row in valid]),
            "wave_i_mean_grad_norm": _mean([row[f"{network.split('_')[0]}_grad_norm_w{first}"] for row in valid]),
            "wave_j_mean_grad_norm": _mean([row[f"{network.split('_')[0]}_grad_norm_w{second}"] for row in valid]),
            "wave_i_mean_sample_count": _mean([row[f"alive_agent_samples_w{first}"] for row in rows]),
            "wave_j_mean_sample_count": _mean([row[f"alive_agent_samples_w{second}"] for row in rows]),
        })
    return result


def summarize_seeds(checkpoint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per training seed plus explicit cross-seed directional summaries."""
    rows = [dict(row, summary_scope="within_checkpoint_rollout_descriptive")
            for row in checkpoint_rows]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in checkpoint_rows:
        groups[(row["stage"], row["network"], row["wave_pair"])].append(row)
    for (stage, network, pair), values in sorted(groups.items()):
        medians = [row["median_cosine"] for row in values if row["median_cosine"] is not None]
        negative = [row["negative_fraction"] for row in values if row["negative_fraction"] is not None]
        rows.append({
            "summary_scope": "cross_training_seed_directional_consistency",
            "training_seed": "ALL_n=3", "stage": stage, "checkpoint_step": "per-seed-actual",
            "network": network, "wave_pair": pair, "total_rollouts": sum(row["total_rollouts"] for row in values),
            "valid_pair_rollouts": sum(row["valid_pair_rollouts"] for row in values),
            "mean_cosine": _mean(medians), "median_cosine": statistics.median(medians) if medians else None,
            "std_cosine": statistics.stdev(medians) if len(medians) > 1 else None,
            "q25_cosine": _quantile(medians, .25), "q75_cosine": _quantile(medians, .75),
            "negative_fraction": _mean(negative),
            "strong_negative_fraction": _mean([row["strong_negative_fraction"] for row in values]),
            "training_seed_replications": len(values),
        })
    return rows


def candidate_labels(checkpoint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels = []
    for stage in ("mid", "late", "exact3m"):
        for network in ("actor", "critic"):
            for pair in ("W1-W2", "W1-W3", "W2-W3"):
                rows = [row for row in checkpoint_rows if row["stage"] == stage and row["network"] == network and row["wave_pair"] == pair]
                if len(rows) != 3 or any(row["valid_pair_rollouts"] == 0 for row in rows):
                    label = "INCONCLUSIVE_DUE_TO_LOW_WAVE_SAMPLES"
                elif all(row["negative_fraction"] > .5 and row["median_cosine"] < 0 for row in rows):
                    label = "ACTOR_CONFLICT_CANDIDATE" if network == "actor" else "CRITIC_CONFLICT_CANDIDATE"
                else:
                    label = "NO_STABLE_CONFLICT_SIGNAL"
                labels.append({"stage": stage, "network": network, "wave_pair": pair,
                               "candidate_label": label})
    return labels


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def git_version() -> dict[str, Any]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                             capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True,
                                    capture_output=True, text=True).stdout.strip())
        return {"commit_sha": sha, "working_tree_dirty": dirty}
    except Exception as error:
        return {"commit_sha": None, "working_tree_dirty": None, "error": str(error)}


def make_report(manifest: dict[str, Any], checkpoint_rows: list[dict[str, Any]],
                labels: list[dict[str, Any]], mutation: list[dict[str, Any]]) -> str:
    selected = "\n".join(
        f"- seed {row['training_seed']} / {row['stage']}: {row['checkpoint_step']} (`{row['checkpoint_path']}`)"
        for row in manifest["selected_checkpoints"])
    label_lines = "\n".join(
        f"- {row['stage']} {row['network']} {row['wave_pair']}: **{row['candidate_label']}**"
        for row in labels)
    return f"""# Wave-wise Gradient Conflict Audit

## 1. Protocol integrity

Natural stochastic Plain MAPPO rollouts only; no update, optimizer step, curriculum restore, artificial kill, or wave skip. Mutation guards passed for {len(mutation)} checkpoint(s). Diagnostic seeds are separate from development evaluation and final holdout. 44M and 45M were unused.

## 2. Selected checkpoints

{selected}

## 3. Sample availability by wave

See `wave_gradient_raw.csv`; a cross-wave pair enters primary aggregation only when both waves have at least {manifest['min_alive_agent_samples_per_wave']} live-agent samples in that rollout.

## 4. Actor cross-wave gradient cosine

Primary actor gradients use globally normalized advantages and the clipped PPO surrogate without entropy. See `wave_gradient_checkpoint_summary.csv`.

## 5. Critic cross-wave gradient cosine

Critic gradients reproduce the configured clipped value objective. See `wave_gradient_checkpoint_summary.csv`.

## 6. Negative-gradient fractions

{label_lines}

## 7. Gradient norm imbalance

Per-wave norms and pairwise norm ratios are recorded in raw and checkpoint summaries.

## 8. Within-wave split-half consistency

Each wave's live-agent samples are deterministically split into balanced halves. These cosines contextualize cross-wave negative cosines as signal versus sampling noise.

## 9. Normalized-vs-raw advantage sensitivity

Primary normalized-advantage and secondary raw-GAE actor cosines are both recorded. Sign reversals are marked `ADVANTAGE_NORMALIZATION_SENSITIVE` in raw rows.

## 10. Cross-seed directional consistency

The scientific replication unit is the training seed (n=3), not an evaluation episode or rollout. Rollout statistics are within-checkpoint descriptive statistics only.

## 11. What the audit supports

The predefined labels mechanically identify stable, cross-training-seed directional conflict candidates when all three seeds have median cosine below zero and negative fraction above 0.5.

## 12. What the audit does NOT prove

Gradient conflict is not causal proof that PCGrad, CAGrad, or any other intervention will improve performance. This audit makes no method recommendation.
"""


def run_audit(output_dir: Path, device: str, num_envs: int, rollout_steps: int,
              rollouts_per_checkpoint: int, minimum_samples: int,
              smoke: bool = False) -> dict[str, Any]:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for this checkpoint audit")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing audit output: {output_dir}")
    output_dir.mkdir(parents=True)
    selections = []
    run_items = [(5301, RUNS[5301])] if smoke else list(RUNS.items())
    stages = (TARGET_STAGES[-1],) if smoke else TARGET_STAGES
    for seed, run_dir in run_items:
        checkpoints = scan_checkpoints(run_dir)
        for stage, target, deviation, exact in stages:
            step, path = select_checkpoint(checkpoints, target, deviation, exact)
            selections.append({"training_seed": seed, "run_dir": str(run_dir.relative_to(ROOT)),
                               "stage": stage, "target_step": target, "checkpoint_step": step,
                               "checkpoint_path": str(path.relative_to(ROOT)),
                               "checkpoint_sha256": file_sha256(path)})
    if seed_is_forbidden(DIAGNOSTIC_ENV_SEED_BASE) or seed_is_forbidden(DIAGNOSTIC_TORCH_SEED):
        raise RuntimeError("diagnostic seed overlaps a protected range")

    raw_rows = []; mutation_rows = []
    env_hashes = set(); algorithm_hashes = set()
    for selected in selections:
        seed = selected["training_seed"]; checkpoint = ROOT / selected["checkpoint_path"]
        trainer, env_config, algorithm_config, _ = validate_run_and_checkpoint(
            ROOT / selected["run_dir"], seed, checkpoint, selected["checkpoint_step"], device)
        env_hashes.add(config_sha256(env_config)); algorithm_hashes.add(config_sha256(algorithm_config))
        before = mutation_fingerprint(trainer)
        checkpoint_env_seed = DIAGNOSTIC_ENV_SEED_BASE
        checkpoint_torch_seed = DIAGNOSTIC_TORCH_SEED
        torch.manual_seed(checkpoint_torch_seed); torch.cuda.manual_seed_all(checkpoint_torch_seed)
        vector = ParallelVectorEnv(num_envs, env_config, base_seed=checkpoint_env_seed,
                                   forbidden_seeds=tuple(range(44_000_000, 44_000_050)) + tuple(range(45_000_000, 45_000_200)))
        try:
            observations = vector.reset(); waves = np.ones(num_envs, dtype=np.int64)
            for rollout_index in range(rollouts_per_checkpoint):
                rollout, observations, waves = collect_rollout(
                    trainer, vector, observations, waves, rollout_steps)
                row = audit_rollout(trainer, rollout, rollout_index,
                                    DIAGNOSTIC_TORCH_SEED + seed, minimum_samples)
                row.update({
                    "training_seed": seed, "stage": selected["stage"],
                    "checkpoint_path": selected["checkpoint_path"],
                    "checkpoint_step": selected["checkpoint_step"],
                    "rollout_index": rollout_index,
                    "diagnostic_env_seed_base": checkpoint_env_seed,
                    "diagnostic_torch_seed": checkpoint_torch_seed,
                })
                raw_rows.append(row)
        finally:
            vector.close()
        after = mutation_fingerprint(trainer)
        if before != after:
            raise RuntimeError(f"READ-ONLY MUTATION GUARD FAILED: {checkpoint}")
        mutation_rows.append({"checkpoint_path": selected["checkpoint_path"],
                              "before": before, "after": after, "exact_unchanged": True})

    checkpoint_rows = summarize_checkpoints(raw_rows); seed_rows = summarize_seeds(checkpoint_rows)
    missing_fields = sorted(set(RAW_REQUIRED_FIELDS) - set(raw_rows[0])) if raw_rows else list(RAW_REQUIRED_FIELDS)
    if missing_fields:
        raise RuntimeError(f"raw audit schema is incomplete: {missing_fields}")
    labels = candidate_labels(checkpoint_rows)
    manifest = {
        "audit": "wave_wise_gradient_conflict", "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "tiny_smoke" if smoke else "full", "code_version": git_version(),
        "environment_config_sha256": sorted(env_hashes), "algorithm_config_sha256": sorted(algorithm_hashes),
        "runs": [str(path.relative_to(ROOT)) for _, path in run_items],
        "training_seeds": [seed for seed, _ in run_items], "selected_checkpoints": selections,
        "diagnostic_seed_protocol": {
            "environment_seed_base": DIAGNOSTIC_ENV_SEED_BASE,
            "torch_policy_seed": DIAGNOSTIC_TORCH_SEED,
            "purpose": "diagnostic seeds only; not development evaluation; not final holdout",
        },
        "num_envs": num_envs, "rollout_steps": rollout_steps,
        "rollouts_per_checkpoint": rollouts_per_checkpoint,
        "min_alive_agent_samples_per_wave": minimum_samples,
        "gamma": .999, "gae_lambda": .95,
        "advantage_normalization": "global live-agent mean/std per rollout; never per-wave",
        "actor_gradient_definition": "negative clipped PPO surrogate mean over live agents of each wave; entropy excluded",
        "critic_gradient_definition": "0.5 * clipped value-error mean over live agents of each wave",
        "44M_unused": True, "45M_unused": True,
        "mutation_guards": mutation_rows,
    }
    summary = {
        "protocol_integrity": "PASS", "candidate_labels": labels,
        "replication_unit": "training_seed", "training_seed_n": len(run_items),
        "rollouts_are_descriptive_not_independent_replications": True,
        "advantage_normalization_sensitive_rows": sum(bool(row["advantage_normalization_sensitive_pairs"]) for row in raw_rows),
        "mutation_guards_passed": all(row["exact_unchanged"] for row in mutation_rows),
        "limitations": ["gradient conflict is not causal proof that PCGrad will improve performance",
                        "low-wave-sample pairs are excluded rather than artificially replenished"],
    }
    (output_dir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_csv(output_dir / "wave_gradient_raw.csv", raw_rows)
    write_csv(output_dir / "wave_gradient_checkpoint_summary.csv", checkpoint_rows)
    write_csv(output_dir / "wave_gradient_seed_summary.csv", seed_rows)
    (output_dir / "wave_gradient_conflict_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "wave_gradient_conflict_report.md").write_text(
        make_report(manifest, checkpoint_rows, labels, mutation_rows), encoding="utf-8")
    return {"output_dir": str(output_dir), "raw_rows": len(raw_rows),
            "checkpoints": len(selections), "mutation_guards_passed": True,
            "mode": manifest["mode"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--output-dir", default="outputs/wave_gradient_conflict_audit")
    parser.add_argument("--num-envs", type=int, default=24)
    parser.add_argument("--rollout-steps", type=int, default=256)
    parser.add_argument("--rollouts-per-checkpoint", type=int, default=16)
    parser.add_argument("--min-alive-agent-samples", type=int, default=MIN_ALIVE_AGENT_SAMPLES)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = ROOT / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if args.smoke:
        if args.output_dir == "outputs/wave_gradient_conflict_audit":
            output = ROOT / "outputs/wave_gradient_conflict_audit_smoke"
        args.num_envs = min(args.num_envs, 2); args.rollout_steps = min(args.rollout_steps, 8)
        args.rollouts_per_checkpoint = 1; args.min_alive_agent_samples = min(args.min_alive_agent_samples, 1)
    result = run_audit(output, args.device, args.num_envs, args.rollout_steps,
                       args.rollouts_per_checkpoint, args.min_alive_agent_samples, args.smoke)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
