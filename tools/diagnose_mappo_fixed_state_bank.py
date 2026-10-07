#!/usr/bin/env python3
"""Read-only fixed-state-bank diagnostics for the MAPPO attention baseline.

The state bank is generated exactly once by FreshStrong with deterministic
actions. Every checkpoint is evaluated on the same records; fixed-bank
evaluation never calls ``env.step``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.mappo.factory import build_mappo_trainer
from env.factory import make_combat_environment
from env.geometry import engagement_geometry

ENV_CONFIG = ROOT / "configs/persistent_wave_v2_environment.yaml"
BANK_GENERATOR = "FreshStrong"
BANK_SEED_BASE = 88_700_000
DEPLOYMENT_SEED_BASE = 88_710_000
FORBIDDEN_SEED_INTERVALS = tuple(
    (million * 1_000_000, million * 1_000_000 + 999_999)
    for million in (44, 45, 46, 47)
)
AXES = ("psi", "theta", "v")
WAVES = (1, 2, 3)
CHECKPOINTS = {
    "FreshStrong": ROOT / "outputs/mappo_attention_seed5303_1p5m/best_eval.pt",
    # The historical fresh run did not emit final.pt. This exact-budget
    # checkpoint is the explicit, audited final fallback.
    "FreshFinal": ROOT / "outputs/mappo_attention_seed5303_1p5m/checkpoint_1500000.pt",
    "ControlFinal": ROOT / "outputs/mappo_attn_lr3e4_cont_seed5303_1p5m/final.pt",
    "TreatmentFinal": ROOT / "outputs/mappo_attn_lr1e4_cont_seed5303_1p5m/final.pt",
    "CommonSource": ROOT / "outputs/mappo_attention_seed5303_1p5m/checkpoint_1001472.pt",
}
PAIR_SPECS = (
    ("FreshStrong", "FreshFinal", "FRESH_STRONG_VS_FINAL"),
    ("ControlFinal", "TreatmentFinal", "CONTROL_VS_TREATMENT"),
)
EXPECTED_STEPS = {
    "FreshStrong": 1_400_832,
    "FreshFinal": 1_500_000,
    "ControlFinal": 1_500_000,
    "TreatmentFinal": 1_500_000,
    "CommonSource": 1_001_472,
}
EXPECTED_ENVIRONMENT_SHA256 = "ca2108c449065f17a3ad8ea287c94e8aa94dadac8b1e20a7b063afbfd22333ee"
EXPECTED_FRESH_STRONG_SHA256 = "65c265f865735483688ee2c93d2db8889a2c0ec4710cd0831ace152cd234690d"
EXPECTED_COMMON_SOURCE_SHA256 = "9e6ce2a0c03de98ff70a8162f069594d28fabe4e1496e9e8523b85f3edac1ef8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def algorithm_config(checkpoint: Path) -> dict[str, Any]:
    return load_yaml(checkpoint.parent / "algorithm_config.yaml")


def validate_checkpoint_provenance(
    name: str, path: Path, state: dict[str, Any], environment_sha: str,
    checkpoint_sha: str,
) -> None:
    extra = state.get("extra", {})
    if environment_sha != EXPECTED_ENVIRONMENT_SHA256:
        raise RuntimeError("current persistent-wave environment SHA is not frozen")
    if extra.get("environment_config_sha256") != environment_sha:
        raise RuntimeError(f"{name} environment config fingerprint mismatch")
    if name == "FreshStrong" and checkpoint_sha != EXPECTED_FRESH_STRONG_SHA256:
        raise RuntimeError("FreshStrong frozen checkpoint SHA mismatch")
    if name == "CommonSource":
        if path.name != "checkpoint_1001472.pt":
            raise RuntimeError("CommonSource must be checkpoint_1001472.pt")
        if checkpoint_sha != EXPECTED_COMMON_SOURCE_SHA256:
            raise RuntimeError("CommonSource frozen checkpoint SHA mismatch")
    if name in {"ControlFinal", "TreatmentFinal"}:
        run = json.loads((path.parent / "run_config.json").read_text(encoding="utf-8"))
        branch = json.loads((path.parent / "branch_from.json").read_text(encoding="utf-8"))
        for label, document in (("run_config", run), ("branch_from", branch)):
            if document.get("parent_checkpoint_sha256") != EXPECTED_COMMON_SOURCE_SHA256:
                raise RuntimeError(f"{name} {label} parent checkpoint SHA mismatch")
            if int(document.get("source_sampled_steps", -1)) != 1_001_472:
                raise RuntimeError(f"{name} {label} source step mismatch")
            seed = document.get("seed", document.get("source_training_seed"))
            if int(seed if seed is not None else -1) != 5303:
                raise RuntimeError(f"{name} {label} seed mismatch")
            if document.get("environment_config_sha256") != environment_sha:
                raise RuntimeError(f"{name} {label} environment SHA mismatch")


def checkpoint_identity(
    name: str, path: Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=map_location, weights_only=False)
    extra = state.get("extra", {})
    config = algorithm_config(path)
    network = config["network"]
    expected = {
        "algorithm": "MAPPO", "critic_type": "attention",
        "environment_variant": "persistent_wave_v2", "observation_dim": 52,
        "action_dim": 3, "num_agents": 4, "training_seed": 5303,
    }
    actual = {
        "algorithm": state.get("algorithm"),
        "critic_type": state.get("critic_type", extra.get("critic_type")),
        "environment_variant": extra.get("environment_variant"),
        "observation_dim": extra.get("observation_dim", network.get("observation_dim")),
        "action_dim": extra.get("action_dim", network.get("action_dim")),
        "num_agents": extra.get("num_agents", network.get("num_agents")),
        "training_seed": extra.get("training_seed"),
    }
    if actual != expected:
        raise RuntimeError(f"{name} checkpoint identity mismatch: {actual!r}")
    step = int(state.get("sampled_steps", -1))
    if step != EXPECTED_STEPS[name]:
        raise RuntimeError(f"{name} sampled_steps mismatch: {step}")
    if name == "FreshStrong":
        best = extra.get("best_evaluation")
        if not isinstance(best, dict) or int(best.get("sampled_steps", -1)) != step:
            raise RuntimeError("FreshStrong is not the recorded best evaluation checkpoint")
    if extra.get("algorithm_config_sha256") != config_sha256(config):
        raise RuntimeError(f"{name} algorithm config fingerprint mismatch")
    checkpoint_sha = sha256(path)
    environment_sha = config_sha256(load_yaml(ENV_CONFIG))
    validate_checkpoint_provenance(
        name, path, state, environment_sha, checkpoint_sha
    )
    result = {
        **actual, "sampled_steps": step, "sha256": checkpoint_sha,
        "environment_config_sha256": environment_sha,
        "path": str(path.resolve()),
    }
    if name == "FreshStrong":
        result["best_evaluation_sampled_steps"] = int(
            extra["best_evaluation"]["sampled_steps"]
        )
        result["frozen_generator_sha_verified"] = True
    if name == "CommonSource":
        result["source_identity"] = "checkpoint_1001472.pt"
        result["frozen_source_sha_verified"] = True
    if name in {"ControlFinal", "TreatmentFinal"}:
        run = json.loads((path.parent / "run_config.json").read_text(encoding="utf-8"))
        result["parent_checkpoint_sha256"] = run["parent_checkpoint_sha256"]
        result["source_sampled_steps"] = int(run["source_sampled_steps"])
        result["continuation_provenance_verified"] = True
    del state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def load_policy(path: Path, device: str):
    trainer = build_mappo_trainer(algorithm_config(path), device)
    trainer.load(path)
    trainer.actor.eval()
    trainer.critic.eval()
    return trainer


def quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"n": 0, **{key: None for key in (
            "min", "p1", "p10", "median", "mean", "p90", "p99", "max"
        )}}
    points = np.quantile(array, (0.01, 0.10, 0.50, 0.90, 0.99))
    return {
        "n": int(array.size), "min": float(array.min()), "p1": float(points[0]),
        "p10": float(points[1]), "median": float(points[2]),
        "mean": float(array.mean()), "p90": float(points[3]),
        "p99": float(points[4]), "max": float(array.max()),
    }


def policy_episode_seed(base_policy_seed: int, environment_seed: int) -> int:
    payload = f"mappo-fixed-bank:{base_policy_seed}:{environment_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def ranges_overlap(start: int, end: int, interval: tuple[int, int]) -> bool:
    return not (end < interval[0] or start > interval[1])


def validate_seed_protocol(
    seed_base: int, bank_episodes: int, episodes: int
) -> dict[str, list[int]]:
    bank = (int(seed_base), int(seed_base) + int(bank_episodes) - 1)
    deployment = (
        int(seed_base) + 10_000,
        int(seed_base) + 10_000 + int(episodes) - 1,
    )
    if seed_base != BANK_SEED_BASE:
        raise RuntimeError(f"seed-base must be the frozen {BANK_SEED_BASE}")
    if bank_episodes < 1 or episodes < 1 or bank[1] >= DEPLOYMENT_SEED_BASE:
        raise RuntimeError("invalid or overlapping bank/deployment seed ranges")
    if any(
        ranges_overlap(*bank, forbidden)
        or ranges_overlap(*deployment, forbidden)
        for forbidden in FORBIDDEN_SEED_INTERVALS
    ):
        raise RuntimeError("diagnostic attempted to access protected 44M-47M ranges")
    return {"bank": list(bank), "deployment": list(deployment)}


def formation_dispersion(env) -> float:
    positions = [
        np.asarray((state.x, state.y, state.z), dtype=float)
        for state in env.red if state.alive
    ]
    if len(positions) < 2:
        return 0.0
    return float(np.mean([
        np.linalg.norm(positions[i] - positions[j])
        for i in range(len(positions)) for j in range(i + 1, len(positions))
    ]))


def focal_geometry(env, focal: int) -> dict[str, Any]:
    own = env.red[focal]
    living_blue = [blue for blue in env.blue if blue.alive]
    nearest = min(
        living_blue, key=lambda blue: engagement_geometry(own, blue).distance
    )
    geometry = engagement_geometry(own, nearest)
    relative_position = np.asarray(
        (nearest.x - own.x, nearest.y - own.y, nearest.z - own.z), dtype=float
    )
    relative_velocity = nearest.velocity_vector() - own.velocity_vector()
    closing_velocity = (
        0.0 if geometry.distance <= 0 else
        -float(np.dot(relative_position, relative_velocity) / geometry.distance)
    )
    return {
        "distance_to_boundary": float(env.arena_radius - math.hypot(own.x, own.y)),
        "altitude": float(own.altitude), "speed": float(own.v),
        "heading": float(own.psi), "pitch": float(own.theta),
        "formation_dispersion": formation_dispersion(env),
        "nearest_blue_distance": float(geometry.distance),
        "nearest_blue_ata": float(geometry.ata),
        "nearest_blue_aa": float(geometry.aa),
        "nearest_blue_closing_velocity": closing_velocity,
    }


def make_bank_records(
    env, observation: np.ndarray, episode_seed: int, episode_index: int,
    entry_wave: int | None = None,
) -> list[dict[str, Any]]:
    current = env._observations()
    if observation.shape != (4, 52) or not np.array_equal(observation, current):
        raise RuntimeError("bank observation/context time alignment violation")
    alive = env.red_alive_mask.astype(np.float32).copy()
    team_observation = np.asarray(observation, dtype=np.float32).copy()
    team_fire = np.asarray(
        [state.armed for state in env.red_fire_states], dtype=bool
    )
    records = []
    for focal in np.flatnonzero(alive > 0.5).tolist():
        local = team_observation[focal].copy()
        if not np.isfinite(local).all() or np.count_nonzero(local) == 0:
            continue
        records.append({
            "episode_seed": int(episode_seed),
            "episode_index": int(episode_index), "global_step": int(env.steps),
            "wave_index": int(env.wave_index),
            "waves_remaining": int(env.total_waves - env.wave_index),
            "focal_agent_index": int(focal), "local_observation": local,
            "team_observation": team_observation.copy(),
            "alive_mask": alive.copy(), "red_survivors": int(alive.sum()),
            "blue_survivors": int(env.blue_alive_mask.sum()),
            "remaining_horizon_steps": int(env.max_steps - env.steps),
            "own_fire_ready": bool(team_fire[focal]),
            "team_fire_readiness": team_fire.copy(),
            "entry_wave": None if entry_wave is None else int(entry_wave),
            **focal_geometry(env, focal),
        })
    return records


def generate_bank(
    trainer, env_config: dict[str, Any], seeds: Iterable[int], stride: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for episode_index, seed in enumerate(seeds):
        env = make_combat_environment(env_config)
        observation, _ = env.reset(int(seed))
        while True:
            if env.steps % stride == 0:
                for row in make_bank_records(
                    env, observation, seed, episode_index
                ):
                    key = (
                        row["episode_index"], row["global_step"],
                        row["focal_agent_index"],
                    )
                    if key not in seen:
                        records.append(row)
                        seen.add(key)
            actions = trainer.act(
                observation, env.red_alive_mask, deterministic=True
            )
            observation, _, terminated, truncated, info = env.step(actions)
            if info.get("spawned_next_wave", False):
                entry_rows = make_bank_records(
                    env, observation, seed, episode_index,
                    entry_wave=int(info["wave_index"]),
                )
                entries.extend(entry_rows)
                for row in entry_rows:
                    key = (
                        row["episode_index"], row["global_step"],
                        row["focal_agent_index"],
                    )
                    if key not in seen:
                        records.append(row)
                        seen.add(key)
            if terminated or truncated:
                break
    return records, entries


@torch.no_grad()
def evaluate_fixed_bank(
    trainer, bank: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not bank:
        return []
    team = torch.as_tensor(
        np.stack([row["team_observation"] for row in bank]),
        dtype=torch.float32, device=trainer.device,
    )
    alive = torch.as_tensor(
        np.stack([row["alive_mask"] for row in bank]),
        dtype=torch.float32, device=trainer.device,
    )
    focal = torch.as_tensor(
        [row["focal_agent_index"] for row in bank],
        dtype=torch.long, device=trainer.device,
    )
    batch = torch.arange(len(bank), device=trainer.device)
    local = team[batch, focal]
    distribution = trainer.actor.distribution(local)
    mean_raw = distribution.mean
    log_std = distribution.scale.log()
    action = torch.tanh(mean_raw)
    values = trainer.critic(team, alive)[batch, focal]
    return [{
        "mean_raw_action": mean_raw[index].cpu().numpy(),
        "deterministic_action": action[index].cpu().numpy(),
        "log_std": log_std[index].cpu().numpy(),
        "critic_value": float(values[index].cpu()),
    } for index in range(len(bank))]


def log_std_table(
    outputs: dict[str, list[dict[str, Any]]], bank: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for checkpoint, values in outputs.items():
        for wave in WAVES:
            indices = [
                index for index, record in enumerate(bank)
                if record["wave_index"] == wave
            ]
            for axis_index, axis in enumerate(AXES):
                data = [values[index]["log_std"][axis_index] for index in indices]
                rows.append({
                    "checkpoint": checkpoint, "wave": wave, "axis": axis,
                    **quantiles(data),
                    "fraction_le_neg4_99": (
                        None if not data else float(np.mean(np.asarray(data) <= -4.99))
                    ),
                    "fraction_ge_1_99": (
                        None if not data else float(np.mean(np.asarray(data) >= 1.99))
                    ),
                })
    return rows


def drift_table(
    outputs: dict[str, list[dict[str, Any]]], bank: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for left, right, label in PAIR_SPECS:
        for wave in WAVES:
            indices = [
                index for index, record in enumerate(bank)
                if record["wave_index"] == wave
            ]
            for field in (
                "mean_raw_action", "deterministic_action", "log_std"
            ):
                values = [float(np.linalg.norm(
                    outputs[right][index][field] - outputs[left][index][field]
                )) for index in indices]
                rows.append({
                    "comparison": label, "left": left, "right": right,
                    "wave": wave, "quantity": f"{field}_l2",
                    **quantiles(values),
                })
    return rows


def critic_table(
    outputs: dict[str, list[dict[str, Any]]], bank: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for checkpoint, values in outputs.items():
        for wave in WAVES:
            indices = [
                index for index, record in enumerate(bank)
                if record["wave_index"] == wave
            ]
            stats = quantiles(values[index]["critic_value"] for index in indices)
            rows.append({
                "row_type": "surface", "checkpoint": checkpoint,
                "comparison": "", "wave": wave,
                **{key: stats[key] for key in (
                    "n", "mean", "p10", "median", "p90"
                )},
            })
    for left, right, label in PAIR_SPECS:
        for wave in WAVES:
            indices = [
                index for index, record in enumerate(bank)
                if record["wave_index"] == wave
            ]
            stats = quantiles(
                abs(outputs[right][index]["critic_value"]
                    - outputs[left][index]["critic_value"])
                for index in indices
            )
            rows.append({
                "row_type": "absolute_surface_difference", "checkpoint": "",
                "comparison": label, "wave": wave,
                **{key: stats[key] for key in (
                    "n", "mean", "p10", "median", "p90"
                )},
            })
    return rows


def nearest_index(
    source: np.ndarray, candidates: np.ndarray,
    scale: np.ndarray | None = None,
) -> tuple[int, float]:
    delta = candidates - source
    if scale is not None:
        delta = delta / scale
    distances = np.linalg.norm(delta, axis=1)
    index = int(np.argmin(distances))
    return index, float(distances[index])


def cross_wave_alias_rows(
    bank: list[dict[str, Any]],
    outputs: dict[str, list[dict[str, Any]]],
    std_floor: float = 1e-6,
) -> list[dict[str, Any]]:
    observations = np.stack([
        row["local_observation"] for row in bank
    ]).astype(np.float64)
    scale = np.maximum(observations.std(axis=0), std_floor)
    rows: list[dict[str, Any]] = []
    for wave_a, wave_b in ((1, 2), (1, 3), (2, 3)):
        source_indices = [
            index for index, row in enumerate(bank)
            if row["wave_index"] == wave_a
        ]
        base_targets = [
            index for index, row in enumerate(bank)
            if row["wave_index"] == wave_b
        ]
        for source_index in source_indices:
            for restriction in ("all_cross_wave", "cross_episode_only"):
                target_indices = [
                    index for index in base_targets
                    if restriction == "all_cross_wave"
                    or bank[index]["episode_index"]
                    != bank[source_index]["episode_index"]
                ]
                if not target_indices:
                    continue
                candidates = observations[target_indices]
                for distance_kind, divisor in (
                    ("raw_normalized_observation_l2", None),
                    ("bank_standardized_l2", scale),
                ):
                    local_index, distance = nearest_index(
                        observations[source_index], candidates, divisor
                    )
                    target_index = target_indices[local_index]
                    source, target = bank[source_index], bank[target_index]
                    exact = bool(np.array_equal(
                        source["local_observation"], target["local_observation"]
                    ))
                    row: dict[str, Any] = {
                        "pair": f"W{wave_a}-W{wave_b}",
                        "direction": f"W{wave_a}_TO_W{wave_b}",
                        "restriction": restriction,
                        "distance_kind": distance_kind, "distance": distance,
                        "exact_alias": exact,
                        "pair_class": (
                            "exact_cross_wave_alias" if exact
                            else "cross_wave_nearest_neighbor"
                        ),
                        "source_episode_seed": source["episode_seed"],
                        "source_episode_index": source["episode_index"],
                        "source_step": source["global_step"],
                        "source_agent": source["focal_agent_index"],
                        "source_wave": source["wave_index"],
                        "target_episode_seed": target["episode_seed"],
                        "target_episode_index": target["episode_index"],
                        "target_step": target["global_step"],
                        "target_agent": target["focal_agent_index"],
                        "target_wave": target["wave_index"],
                        "same_episode": (
                            source["episode_index"] == target["episode_index"]
                        ),
                    }
                    for context in (
                        "waves_remaining", "remaining_horizon_steps",
                        "own_fire_ready", "red_survivors",
                    ):
                        row[f"source_{context}"] = source[context]
                        row[f"target_{context}"] = target[context]
                    for name, values in outputs.items():
                        prefix = name.lower()
                        row[f"{prefix}_deterministic_action_l2"] = float(
                            np.linalg.norm(
                                values[source_index]["deterministic_action"]
                                - values[target_index]["deterministic_action"]
                            )
                        )
                        row[f"{prefix}_log_std_l2"] = float(np.linalg.norm(
                            values[source_index]["log_std"]
                            - values[target_index]["log_std"]
                        ))
                        row[f"{prefix}_critic_value_abs_diff"] = abs(
                            values[source_index]["critic_value"]
                            - values[target_index]["critic_value"]
                        )
                    rows.append(row)
    return rows


def alias_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    keys = sorted({
        (row["pair"], row["direction"], row["restriction"], row["distance_kind"])
        for row in rows
    })
    for pair, direction, restriction, kind in keys:
        selected = [
            row for row in rows
            if (row["pair"], row["direction"], row["restriction"], row["distance_kind"])
            == (pair, direction, restriction, kind)
        ]
        result.append({
            "pair": pair, "direction": direction, "restriction": restriction,
            "distance_kind": kind,
            "exact_alias_count": sum(row["exact_alias"] for row in selected),
            **quantiles(row["distance"] for row in selected),
        })
    return result


def bank_coverage(
    bank: list[dict[str, Any]], entries: list[dict[str, Any]],
    bank_episodes: int,
) -> dict[str, Any]:
    by_wave = {
        wave: [row for row in bank if row["wave_index"] == wave]
        for wave in WAVES
    }
    entry_events = {
        (row["episode_seed"], row["entry_wave"], row["global_step"])
        for row in entries if row["entry_wave"] in (2, 3)
    }
    result: dict[str, Any] = {
        "total_decision_states": len({
            (row["episode_index"], row["global_step"]) for row in bank
        }),
        "live_focal_records": len(bank), "bank_episodes": bank_episodes,
        "w2_entry_events": sum(wave == 2 for _, wave, _ in entry_events),
        "w3_entry_events": sum(wave == 3 for _, wave, _ in entry_events),
        "w2_entry_live_focal_records": sum(row["entry_wave"] == 2 for row in entries),
        "w3_entry_live_focal_records": sum(row["entry_wave"] == 3 for row in entries),
        "episodes_reaching_w2_entry": len({
            seed for seed, wave, _ in entry_events if wave == 2
        }),
        "episodes_reaching_w3_entry": len({
            seed for seed, wave, _ in entry_events if wave == 3
        }),
    }
    for wave in WAVES:
        result[f"wave_{wave}_live_focal_records"] = len(by_wave[wave])
        result[f"episodes_reaching_wave_{wave}"] = len({
            row["episode_index"] for row in by_wave[wave]
        })
        for agent in range(4):
            result[f"wave_{wave}_agent_{agent}_records"] = sum(
                row["focal_agent_index"] == agent for row in by_wave[wave]
            )
    threshold = max(20, bank_episodes)
    result["minimum_w3_live_focal_records"] = threshold
    minimum_w3_episodes = 3
    result["minimum_recommended_w3_episodes"] = minimum_w3_episodes
    low_diversity = result["episodes_reaching_wave_3"] < minimum_w3_episodes
    insufficient = len(by_wave[3]) < threshold
    result["coverage_status"] = (
        "INSUFFICIENT_W3_BANK_COVERAGE" if insufficient or low_diversity
        else "COVERAGE_RECORDED"
    )
    warnings = []
    if insufficient:
        warnings.append(
            "INSUFFICIENT_W3_BANK_COVERAGE: increase --bank-episodes; generator was not changed"
        )
    if low_diversity:
        warnings.append("LOW_W3_EPISODE_DIVERSITY")
    result["warnings"] = warnings
    return result


def entry_summary(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    team_metrics = (
        "red_survivors", "remaining_horizon_steps", "formation_dispersion",
    )
    focal_metrics = (
        "distance_to_boundary", "altitude", "speed", "heading", "pitch",
        "nearest_blue_distance", "nearest_blue_ata", "nearest_blue_aa",
        "nearest_blue_closing_velocity", "own_fire_ready",
    )
    rows = []
    checkpoints = sorted({str(row["checkpoint"]) for row in entries})
    for checkpoint in checkpoints:
      for wave in (2, 3):
        selected = [
            row for row in entries
            if row["checkpoint"] == checkpoint and row["entry_wave"] == wave
        ]
        unique_events = {}
        for row in selected:
            key = (
                row["checkpoint"], row["episode_seed"], row["entry_wave"],
                row["entry_global_step"],
            )
            unique_events.setdefault(key, row)
        for metric in team_metrics:
            rows.append({
                "checkpoint": checkpoint, "entry_wave": wave,
                "metric": metric, "aggregation_level": "entry_event",
                **quantiles(float(row[metric]) for row in unique_events.values()),
            })
        for metric in focal_metrics:
            rows.append({
                "checkpoint": checkpoint, "entry_wave": wave,
                "metric": metric, "aggregation_level": "live_focal_agent",
                **quantiles(float(row[metric]) for row in selected),
            })
    return rows


def entry_detail_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in entries:
        rows.append({
            "checkpoint": row["checkpoint"],
            "episode_seed": row["episode_seed"],
            "entry_wave": row["entry_wave"],
            "entry_global_step": row["entry_global_step"],
            "red_survivors": row["red_survivors"],
            "remaining_horizon_steps": row["remaining_horizon_steps"],
            "formation_dispersion": row["formation_dispersion"],
            "focal_agent_index": row["focal_agent_index"],
            "distance_to_boundary": row["distance_to_boundary"],
            "altitude": row["altitude"], "speed": row["speed"],
            "heading": row["heading"], "pitch": row["pitch"],
            "nearest_blue_distance": row["nearest_blue_distance"],
            "nearest_blue_ATA": row["nearest_blue_ata"],
            "nearest_blue_AA": row["nearest_blue_aa"],
            "nearest_blue_closing_velocity": row["nearest_blue_closing_velocity"],
            "own_fire_ready": row["own_fire_ready"],
        })
    return rows


def run_deployment_episode(
    trainer, env_config: dict[str, Any], environment_seed: int,
    deterministic: bool, base_policy_seed: int = 89_710_000,
    checkpoint: str | None = None,
    entry_sink: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    env = make_combat_environment(env_config)
    observation, _ = env.reset(int(environment_seed))
    returns = np.zeros(4, dtype=np.float64)
    devices = [trainer.device.index or 0] if trainer.device.type == "cuda" else []
    # fork_rng always preserves the CPU generator and, when applicable, the
    # selected CUDA generator. Diagnostic evaluation cannot perturb training
    # or a caller's random stream.
    with torch.random.fork_rng(devices=devices):
        seed = policy_episode_seed(base_policy_seed, environment_seed)
        torch.manual_seed(seed)
        if trainer.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        while True:
            action = trainer.act(
                observation, env.red_alive_mask, deterministic=deterministic
            )
            observation, reward, terminated, truncated, info = env.step(action)
            returns += reward
            if (
                deterministic and entry_sink is not None and checkpoint is not None
                and info.get("spawned_next_wave", False)
            ):
                entry_rows = make_bank_records(
                    env, observation, environment_seed, environment_seed,
                    entry_wave=int(info["wave_index"]),
                )
                for row in entry_rows:
                    row["checkpoint"] = checkpoint
                    row["entry_global_step"] = row["global_step"]
                entry_sink.extend(entry_rows)
            if terminated or truncated:
                return {
                    "environment_seed": int(environment_seed),
                    "deterministic": bool(deterministic),
                    "waves_cleared": int(info["waves_cleared"]),
                    "return": float(returns.sum()),
                    "red_loss": int(info["red_losses"]),
                    "ground": int(info["red_ground_losses"]),
                    "boundary": int(info["red_boundary_exits"]),
                }


def deployment_summary(
    policies: dict[str, Any], env_config: dict[str, Any], seeds: Iterable[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    entries: list[dict[str, Any]] = []
    seed_list = list(seeds)
    for checkpoint in (
        "FreshStrong", "FreshFinal", "ControlFinal", "TreatmentFinal"
    ):
        for deterministic in (True, False):
            episodes = [
                run_deployment_episode(
                    policies[checkpoint], env_config, seed, deterministic,
                    checkpoint=checkpoint,
                    entry_sink=entries if deterministic else None,
                ) for seed in seed_list
            ]
            rows.append({
                "checkpoint": checkpoint,
                "mode": "deterministic" if deterministic else "stochastic",
                "episodes": len(episodes),
                "W1": float(np.mean([
                    row["waves_cleared"] >= 1 for row in episodes
                ])),
                "W2": float(np.mean([
                    row["waves_cleared"] >= 2 for row in episodes
                ])),
                "W3": float(np.mean([
                    row["waves_cleared"] >= 3 for row in episodes
                ])),
                "average_waves": float(np.mean([
                    row["waves_cleared"] for row in episodes
                ])),
                "average_return": float(np.mean([
                    row["return"] for row in episodes
                ])),
                "average_red_loss": float(np.mean([
                    row["red_loss"] for row in episodes
                ])),
                "average_ground": float(np.mean([
                    row["ground"] for row in episodes
                ])),
                "average_boundary": float(np.mean([
                    row["boundary"] for row in episodes
                ])),
                "environment_seed_start": seed_list[0],
                "environment_seed_end": seed_list[-1],
                "action_noise_matched_claimed": False,
            })
    return rows, entries


def ensure_output_available(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty output directory: {path}"
        )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_bank(path: Path, bank: list[dict[str, Any]]) -> None:
    np.savez_compressed(
        path,
        record_index=np.arange(len(bank), dtype=np.int64),
        local_observation=np.stack([row["local_observation"] for row in bank]),
        team_observation=np.stack([row["team_observation"] for row in bank]),
        alive_mask=np.stack([row["alive_mask"] for row in bank]),
        episode_seed=np.asarray([
            row["episode_seed"] for row in bank
        ], dtype=np.int64),
        episode_index=np.asarray([
            row["episode_index"] for row in bank
        ], dtype=np.int32),
        global_step=np.asarray([
            row["global_step"] for row in bank
        ], dtype=np.int32),
        wave_index=np.asarray([
            row["wave_index"] for row in bank
        ], dtype=np.int8),
        focal_agent_index=np.asarray([
            row["focal_agent_index"] for row in bank
        ], dtype=np.int8),
    )


def save_bank_context(path: Path, bank: list[dict[str, Any]]) -> None:
    fields = (
        "episode_seed", "episode_index", "global_step", "wave_index",
        "waves_remaining", "focal_agent_index", "red_survivors",
        "blue_survivors", "remaining_horizon_steps", "own_fire_ready",
        "team_fire_readiness", "entry_wave", "distance_to_boundary",
        "altitude", "speed", "heading", "pitch", "formation_dispersion",
        "nearest_blue_distance", "nearest_blue_ata", "nearest_blue_aa",
        "nearest_blue_closing_velocity",
    )
    with path.open("w", encoding="utf-8") as stream:
        for record_index, record in enumerate(bank):
            row = {"record_index": record_index}
            for field in fields:
                value = record[field]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                elif isinstance(value, np.generic):
                    value = value.item()
                row[field] = value
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def report_text(summary: dict[str, Any]) -> str:
    return "# MAPPO fixed-state-bank diagnostic\n\n" + "\n".join((
        f"- Status: **{summary['status']}**",
        f"- Bank generator: `{summary['bank_generator']['checkpoint']}` at "
        f"{summary['bank_generator']['sampled_steps']} steps (deterministic tanh(mean)).",
        f"- Fixed bank records: {summary['bank_coverage']['live_focal_records']}",
        f"- Coverage: **{summary['bank_coverage']['coverage_status']}**",
        "- Every checkpoint was evaluated on one immutable bank without environment advancement.",
        "- Cross-wave rows are directional nearest neighbours; only elementwise equality is an exact cross-wave alias.",
        "- Critic values are surface comparisons, not value errors (no return-to-go labels).",
        "- Deployment modes share environment seeds but make no action-noise-matching claim.",
        "- No automatic root-cause winner is emitted; causal interpretation remains manual.",
    )) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--bank-episodes", type=int, default=12)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=BANK_SEED_BASE)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/mappo_fixed_state_bank_diagnostic",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for checkpoint/fixed-bank diagnostics")
    if args.stride < 1:
        raise ValueError("stride must be positive")
    ranges = validate_seed_protocol(
        args.seed_base, args.bank_episodes, args.episodes
    )
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    ensure_output_available(output)
    identities = {
        name: checkpoint_identity(name, path, torch.device("cuda:0"))
        for name, path in CHECKPOINTS.items()
    }
    env_config = load_yaml(ENV_CONFIG)
    policies = {
        name: load_policy(path, args.device)
        for name, path in CHECKPOINTS.items()
    }
    bank, entries = generate_bank(
        policies[BANK_GENERATOR], env_config,
        range(ranges["bank"][0], ranges["bank"][1] + 1), args.stride,
    )
    if not bank:
        raise RuntimeError("fixed state bank is empty")
    outputs = {
        name: evaluate_fixed_bank(policy, bank)
        for name, policy in policies.items()
    }
    if any(len(values) != len(bank) for values in outputs.values()):
        raise RuntimeError("not all checkpoints were evaluated on one fixed bank")
    coverage = bank_coverage(bank, entries, args.bank_episodes)
    alias_rows = cross_wave_alias_rows(bank, outputs)
    deployment, checkpoint_entries = deployment_summary(
        policies, env_config,
        range(ranges["deployment"][0], ranges["deployment"][1] + 1),
    )
    output.mkdir(parents=True, exist_ok=True)
    requested_final = CHECKPOINTS["FreshFinal"].parent / "final.pt"
    metadata = {
        "bank_generator_checkpoint": str(CHECKPOINTS[BANK_GENERATOR].resolve()),
        "bank_generator_sha256": identities[BANK_GENERATOR]["sha256"],
        "bank_generator_sampled_steps": identities[BANK_GENERATOR]["sampled_steps"],
        "bank_generation_mode": "deterministic_tanh_mean",
        "bank_seed_range": ranges["bank"], "stride": args.stride,
        "single_generator": True, "checkpoints": identities,
        "fresh_final_requested_path": str(requested_final.resolve()),
        "fresh_final_requested_path_exists": requested_final.exists(),
        "fresh_final_exact_budget_fallback": str(CHECKPOINTS["FreshFinal"].resolve()),
    }
    summary = {
        "status": "DIAGNOSTIC_COMPLETE_DESCRIPTIVE_NO_CAUSAL_WINNER",
        "environment": "persistent_wave_v2",
        "bank_generator": {
            "checkpoint": BANK_GENERATOR,
            "sampled_steps": identities[BANK_GENERATOR]["sampled_steps"],
            "sha256": identities[BANK_GENERATOR]["sha256"],
            "mode": "deterministic_tanh_mean",
        },
        "seed_ranges": ranges, "protected_44m_47m_accessed": False,
        "bank_coverage": coverage,
        "fixed_bank_shared_by_all_checkpoints": True,
        "fixed_bank_environment_steps_during_evaluation": 0,
        "classifications": {
            "observation_aliasing": "FORMAL_ALIASING_EXISTS",
            "empirical_aliasing": "EMPIRICAL_CROSS_WAVE_NEAREST_NEIGHBOURS_QUANTIFIED",
            "performance_causal_role": "PERFORMANCE_CAUSAL_ROLE_NOT_ESTABLISHED",
        },
        "comparisons": [label for _, _, label in PAIR_SPECS],
        "root_cause_winner": None,
    }
    (output / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "bank_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (output / "bank_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    write_csv(output / "policy_std_by_wave.csv", log_std_table(outputs, bank))
    write_csv(output / "policy_mean_drift_by_wave.csv", drift_table(outputs, bank))
    write_csv(output / "critic_value_by_wave.csv", critic_table(outputs, bank))
    write_csv(output / "cross_wave_alias_pairs.csv", alias_rows)
    write_csv(output / "cross_wave_alias_summary.csv", alias_summary(alias_rows))
    write_csv(output / "entry_state_by_checkpoint.csv", entry_detail_rows(checkpoint_entries))
    write_csv(output / "entry_state_summary.csv", entry_summary(checkpoint_entries))
    write_csv(output / "deployment_mode_comparison.csv", deployment)
    save_bank(output / "bank_states.npz", bank)
    save_bank_context(output / "bank_context.jsonl", bank)
    (output / "diagnostic_report.md").write_text(
        report_text(summary), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
