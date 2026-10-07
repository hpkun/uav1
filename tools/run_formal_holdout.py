"""One-shot CUDA evaluator for the frozen M5 formal holdout.

No training functionality is present. The formal seed range, methods,
checkpoints, metrics and checkpoint roles are loaded from an immutable protocol
manifest and revalidated before evaluation. Per-episode caches support only an
exact-protocol recovery after interruption.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_formal_holdout import (
    EPISODES_PER_POLICY, FORMAL_SEED_END, FORMAL_SEED_START, MANIFEST_HASH_PATH,
    MANIFEST_PATH, METHODS, PRIMARY_METRICS, TRAINING_SEEDS, canonical_json,
    file_sha256, manifest_sha256, read_json, relative_path, resolve_manifest_path,
    validate_manifest_files, validate_manifest_schema, value_sha256,
)

OUT = ROOT / "outputs" / "formal_holdout"
SMOKE_OUT = ROOT / "outputs" / "formal_holdout_smoke"
STATE_NAME = "formal_holdout_state.json"
COMPLETED_NAME = "COMPLETED.json"
SMOKE_SEED_START = 99_000_000
PER_EPISODE_FIELDS = (
    "method", "training_seed", "checkpoint_role", "checkpoint_step", "episode_seed",
    "W1_clear", "W2_clear", "W3_clear", "waves_cleared", "episode_return",
    "red_losses", "blue_losses", "kill_loss_ratio", "red_boundary_exits",
    "red_ground_losses", "timeout", "episode_length", "reached_W2", "reached_W3",
    "time_to_clear_W1", "time_to_clear_W2", "time_spent_in_W3",
)
SUMMARY_FIELDS = (
    "W1", "W2", "W3", "average_waves", "return", "red_loss", "blue_loss", "K_L",
    "boundary", "ground", "timeout", "episode_length", "probability_reaching_W2",
    "probability_reaching_W3", "timeout_conditioned_reached_W2",
    "timeout_conditioned_reached_W3", "mean_time_to_clear_W1",
    "mean_time_to_clear_W2", "mean_time_spent_in_W3",
    "episode_length_conditioned_waves_cleared_0",
    "episode_length_conditioned_waves_cleared_1",
    "episode_length_conditioned_waves_cleared_2",
    "episode_length_conditioned_waves_cleared_3",
)
DELTA_METRICS = (
    "average_waves", "W3", "return", "red_loss", "K_L", "W1", "W2",
    "boundary", "ground", "timeout",
)
FAVORABLE_LOWER = {"red_loss", "boundary", "ground", "timeout"}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_locked_manifest() -> tuple[dict[str, Any], str]:
    if not MANIFEST_PATH.is_file() or not MANIFEST_HASH_PATH.is_file():
        raise FileNotFoundError("formal protocol manifest/hash is missing; run prepare_formal_holdout.py first")
    manifest = read_json(MANIFEST_PATH)
    digest = manifest_sha256(manifest)
    if digest != MANIFEST_HASH_PATH.read_text(encoding="utf-8").strip():
        raise RuntimeError("formal protocol manifest SHA mismatch")
    validate_manifest_schema(manifest)
    validate_manifest_files(manifest)
    return manifest, digest


def require_cuda() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal holdout; CPU fallback is forbidden")
    return torch.cuda.get_device_name(0)


def build_tasks(manifest: dict[str, Any], smoke: bool = False) -> list[dict[str, Any]]:
    validate_manifest_schema(manifest)
    seeds = [SMOKE_SEED_START] if smoke else list(range(FORMAL_SEED_START, FORMAL_SEED_END + 1))
    if not smoke and (len(seeds) != EPISODES_PER_POLICY or seeds[-1] != FORMAL_SEED_END):
        raise RuntimeError("formal seed range/episode count changed")
    tasks: list[dict[str, Any]] = []
    for run in manifest["runs"]:
        for role, path_key, step_key, hash_key in (
            ("best", "best_checkpoint_path", "best_checkpoint_sampled_steps", "best_checkpoint_sha256"),
            ("latest", "latest_checkpoint_path", "latest_checkpoint_sampled_steps", "latest_checkpoint_sha256"),
        ):
            tasks.append({
                "method": run["method"],
                "training_seed": int(run["training_seed"]),
                "checkpoint_role": role,
                "checkpoint": run[path_key],
                "checkpoint_sha256": run[hash_key],
                "checkpoint_step": int(run[step_key]),
                "environment": manifest["environment_config_path"],
                "episode_seeds": seeds,
            })
    expected = {(method, seed, role) for method in METHODS for seed in TRAINING_SEEDS for role in ("best", "latest")}
    actual = {(task["method"], task["training_seed"], task["checkpoint_role"]) for task in tasks}
    if actual != expected or len(tasks) != 12:
        raise RuntimeError("formal task method/seed/checkpoint-role list changed")
    return tasks


def task_id(task: dict[str, Any]) -> str:
    method = "alloff" if task["method"] == "All-Off" else "m5"
    return f"{method}_seed{task['training_seed']}_{task['checkpoint_role']}"


def task_fingerprint(task: dict[str, Any], manifest_digest: str) -> str:
    return value_sha256({"manifest_sha256": manifest_digest, "task": task})


def initial_state(manifest: dict[str, Any], digest: str, tasks: list[dict[str, Any]], smoke: bool) -> dict[str, Any]:
    return {
        "protocol_name": manifest["protocol_name"],
        "manifest_sha256": digest,
        "smoke": smoke,
        "formal_seed_start": None if smoke else FORMAL_SEED_START,
        "formal_seed_end": None if smoke else FORMAL_SEED_END,
        "episodes_per_policy": 1 if smoke else EPISODES_PER_POLICY,
        "task_fingerprints": {task_id(task): task_fingerprint(task, digest) for task in tasks},
        "created_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def prepare_output_directory(
    output: Path, manifest: dict[str, Any], digest: str, tasks: list[dict[str, Any]],
    resume_exact: bool, smoke: bool,
) -> dict[str, Any]:
    state_path = output / STATE_NAME
    completed = output / COMPLETED_NAME
    expected = initial_state(manifest, digest, tasks, smoke)
    if completed.exists():
        raise RuntimeError(f"formal output is already complete; rerun refused: {output}")
    existing_entries = list(output.iterdir()) if output.exists() else []
    if existing_entries and not resume_exact:
        raise RuntimeError(f"output directory already contains data; use --resume-exact only for interruption recovery: {output}")
    if resume_exact:
        if not state_path.is_file():
            raise RuntimeError("--resume-exact requires an existing formal_holdout_state.json")
        current = read_json(state_path)
        for key in ("protocol_name", "manifest_sha256", "smoke", "formal_seed_start", "formal_seed_end", "episodes_per_policy", "task_fingerprints"):
            if current.get(key) != expected.get(key):
                raise RuntimeError(f"exact-resume protocol mismatch: {key}")
        return current
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(state_path, expected)
    return expected


def _load_policy(task: dict[str, Any]):
    import torch
    import yaml
    from algorithm.modular_mappo.factory import build_modular_mappo_trainer
    from algorithm.modular_mappo.protocol import is_formal_v2_checkpoint, validate_modular_checkpoint
    checkpoint = resolve_manifest_path(task["checkpoint"])
    if file_sha256(checkpoint) != task["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint SHA changed: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("algorithm") != "modular_mappo" or not is_formal_v2_checkpoint(state):
        raise RuntimeError(f"non-formal modular checkpoint refused: {checkpoint}")
    extra = state.get("extra", {})
    if int(extra.get("training_seed", -1)) != int(task["training_seed"]):
        raise RuntimeError("checkpoint training seed mismatch")
    if int(state.get("sampled_steps", -1)) != int(task["checkpoint_step"]):
        raise RuntimeError("checkpoint sampled step mismatch")
    config = extra.get("algorithm_config")
    if config is None:
        raise RuntimeError("checkpoint lacks self-describing algorithm config")
    environment = yaml.safe_load(resolve_manifest_path(task["environment"]).read_text(encoding="utf-8"))
    validate_modular_checkpoint(state, environment, config)
    trainer = build_modular_mappo_trainer(
        config, "cuda", hidden_dim=int(extra["network_architecture"]["hidden_dim"])
    )
    trainer.load(checkpoint)
    return trainer, environment


def run_episode(trainer, env_config: dict[str, Any], episode_seed: int, task: dict[str, Any]) -> dict[str, Any]:
    from env.factory import make_combat_environment
    from algorithm.modules.wave_survival_pbrs import mission_context_numpy
    env = make_combat_environment(env_config)
    observation, _ = env.reset(int(episode_seed))
    alive = env.red_alive_mask.copy()
    returns = np.zeros(4, dtype=np.float64)
    wave = 1
    total_waves = int(env_config.get("persistent_waves", {}).get("total_waves", 1))
    actor_hidden, critic_hidden = trainer.initial_hidden(1)
    episode_mask = np.zeros(1, dtype=np.float32)
    while True:
        context = mission_context_numpy(
            trainer, np.asarray([wave]), np.asarray([total_waves]),
            env.blue_alive_mask[None], np.asarray([env.steps]), env.max_steps,
        )
        actions, actor_hidden = trainer.act(
            observation[None], alive[None], True, False, context, actor_hidden, episode_mask
        )
        _, critic_hidden = trainer.values_step(
            observation[None], alive[None], context, critic_hidden, episode_mask
        )
        observation, reward, terminated, truncated, info = env.step(actions[0])
        returns += reward
        alive = np.asarray(info["red_alive_mask"], dtype=np.float32)
        episode_mask[:] = 1
        wave = int(info.get("wave_index", 1))
        total_waves = int(info.get("total_waves", total_waves))
        if terminated or truncated:
            break
    waves_cleared = int(info.get("waves_cleared", 0))
    per_wave = [dict(row) for row in info.get("per_wave_metrics", [])]
    timing: dict[str, Any] = {}
    for index in (1, 2, 3):
        cleared = next(
            (row for row in per_wave if int(row.get("wave_index", 0)) == index and row.get("wave_cleared")), None
        )
        reached = next((row for row in per_wave if int(row.get("wave_index", 0)) == index), None)
        timing[f"time_to_clear_W{index}"] = int(cleared["end_step"]) if cleared else None
        timing[f"time_spent_in_W{index}"] = int(reached["duration_steps"]) if reached else None
    red_losses = float(info.get("red_losses", 0))
    blue_losses = float(info.get("blue_losses", 0))
    return {
        "method": task["method"], "training_seed": int(task["training_seed"]),
        "checkpoint_role": task["checkpoint_role"], "checkpoint_step": int(task["checkpoint_step"]),
        "episode_seed": int(episode_seed), "W1_clear": int(waves_cleared >= 1),
        "W2_clear": int(waves_cleared >= 2), "W3_clear": int(waves_cleared >= 3),
        "waves_cleared": waves_cleared, "episode_return": float(returns.sum()),
        "red_losses": red_losses, "blue_losses": blue_losses,
        "kill_loss_ratio": blue_losses / max(red_losses, 1.0),
        "red_boundary_exits": float(info.get("red_boundary_exits", 0)),
        "red_ground_losses": float(info.get("red_ground_losses", 0)),
        "timeout": int(info.get("termination_reason") == "red_failure_timeout"),
        "episode_length": int(info.get("episode_length", 0)),
        "reached_W2": int(waves_cleared >= 1), "reached_W3": int(waves_cleared >= 2),
        "time_to_clear_W1": timing["time_to_clear_W1"],
        "time_to_clear_W2": timing["time_to_clear_W2"],
        "time_spent_in_W3": timing["time_spent_in_W3"],
    }


def episode_cache_path(output: Path, task: dict[str, Any], seed: int) -> Path:
    return output / "cache" / task_id(task) / f"episode_{seed}.json"


def progress_path(output: Path, task: dict[str, Any]) -> Path:
    return output / "cache" / task_id(task) / "progress.jsonl"


def load_progress(output: Path, task: dict[str, Any]) -> dict[int, str]:
    path = progress_path(output, task)
    progress: dict[int, str] = {}
    if not path.is_file():
        return progress
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        seed = int(row["episode_seed"]); digest = str(row["cache_sha256"])
        if seed in progress and progress[seed] != digest:
            raise RuntimeError(f"conflicting append-only progress entries for {task_id(task)} seed {seed}")
        progress[seed] = digest
    return progress


def append_progress(output: Path, task: dict[str, Any], seed: int, cache_sha256: str) -> None:
    path = progress_path(output, task)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"episode_seed": int(seed), "cache_sha256": cache_sha256}) + "\n")
        stream.flush()


def validate_task_progress(output: Path, task: dict[str, Any]) -> dict[int, str]:
    """Reject deletion or mutation of any episode already committed to the ledger."""
    progress = load_progress(output, task)
    allowed = set(int(seed) for seed in task["episode_seeds"])
    if not set(progress).issubset(allowed):
        raise RuntimeError(f"progress ledger contains out-of-protocol seed for {task_id(task)}")
    for seed, expected_hash in progress.items():
        cache = episode_cache_path(output, task, seed)
        if not cache.is_file():
            raise RuntimeError(f"committed episode cache was deleted; selective rerun refused: {cache}")
        if file_sha256(cache) != expected_hash:
            raise RuntimeError(f"committed episode cache changed; exact resume refused: {cache}")
    return progress


def validate_cached_episode(payload: dict[str, Any], task: dict[str, Any], digest: str, seed: int) -> dict[str, Any]:
    if payload.get("task_fingerprint") != task_fingerprint(task, digest):
        raise RuntimeError(f"cached episode task/protocol mismatch: {task_id(task)} seed {seed}")
    record = payload.get("record", {})
    if int(record.get("episode_seed", -1)) != seed:
        raise RuntimeError("cached episode seed mismatch")
    return record


def evaluate_task(task: dict[str, Any], output_value: str, digest: str) -> dict[str, Any]:
    require_cuda()
    output = Path(output_value)
    progress = validate_task_progress(output, task)
    missing: list[int] = []
    for seed in task["episode_seeds"]:
        path = episode_cache_path(output, task, seed)
        if path.is_file():
            validate_cached_episode(read_json(path), task, digest, seed)
            if seed not in progress:
                # Recovery for a crash after atomic cache replacement but before
                # the append-only progress commit.
                append_progress(output, task, seed, file_sha256(path))
                progress[seed] = file_sha256(path)
        else:
            missing.append(seed)
    if not missing:
        return {"task": task_id(task), "evaluated": 0, "cached": len(task["episode_seeds"])}
    trainer, env_config = _load_policy(task)
    for seed in missing:
        record = run_episode(trainer, env_config, seed, task)
        cache = episode_cache_path(output, task, seed)
        atomic_write_json(
            cache,
            {"task_fingerprint": task_fingerprint(task, digest), "record": record},
        )
        append_progress(output, task, seed, file_sha256(cache))
    return {"task": task_id(task), "evaluated": len(missing), "cached": len(task["episode_seeds"]) - len(missing)}


def evaluate_all(tasks: list[dict[str, Any]], output: Path, digest: str, workers: int) -> None:
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between 1 and 4")
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(evaluate_task, task, str(output), digest) for task in tasks]
        for future in as_completed(futures):
            print(json.dumps(future.result()), flush=True)


def collect_records(tasks: list[dict[str, Any]], output: Path, digest: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for task in tasks:
        for seed in task["episode_seeds"]:
            path = episode_cache_path(output, task, seed)
            if not path.is_file():
                raise RuntimeError(f"missing exact-protocol episode cache: {path}")
            records.append(validate_cached_episode(read_json(path), task, digest, seed))
    expected = sum(len(task["episode_seeds"]) for task in tasks)
    if len(records) != expected:
        raise RuntimeError("formal episode record count mismatch")
    return records


def mean_present(records: list[dict[str, Any]], key: str, predicate=None) -> float | None:
    rows = records if predicate is None else [row for row in records if predicate(row)]
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def summarize_policy(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty policy")
    total_blue = sum(float(row["blue_losses"]) for row in records)
    total_red = sum(float(row["red_losses"]) for row in records)
    summary = {
        "method": records[0]["method"], "training_seed": records[0]["training_seed"],
        "checkpoint_role": records[0]["checkpoint_role"], "checkpoint_step": records[0]["checkpoint_step"],
        "episodes": len(records), "W1": mean_present(records, "W1_clear"),
        "W2": mean_present(records, "W2_clear"), "W3": mean_present(records, "W3_clear"),
        "average_waves": mean_present(records, "waves_cleared"), "return": mean_present(records, "episode_return"),
        "red_loss": mean_present(records, "red_losses"), "blue_loss": mean_present(records, "blue_losses"),
        "K_L": total_blue / max(total_red, 1.0), "boundary": mean_present(records, "red_boundary_exits"),
        "ground": mean_present(records, "red_ground_losses"), "timeout": mean_present(records, "timeout"),
        "episode_length": mean_present(records, "episode_length"),
        "probability_reaching_W2": mean_present(records, "reached_W2"),
        "probability_reaching_W3": mean_present(records, "reached_W3"),
        "timeout_conditioned_reached_W2": mean_present(records, "timeout", lambda row: bool(row["reached_W2"])),
        "timeout_conditioned_reached_W3": mean_present(records, "timeout", lambda row: bool(row["reached_W3"])),
        "mean_time_to_clear_W1": mean_present(records, "time_to_clear_W1"),
        "mean_time_to_clear_W2": mean_present(records, "time_to_clear_W2"),
        "mean_time_spent_in_W3": mean_present(records, "time_spent_in_W3"),
    }
    for waves in range(4):
        summary[f"episode_length_conditioned_waves_cleared_{waves}"] = mean_present(
            records, "episode_length", lambda row, count=waves: int(row["waves_cleared"]) == count
        )
    return summary


def policy_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault((row["method"], int(row["training_seed"]), row["checkpoint_role"]), []).append(row)
    if len(groups) != 12:
        raise RuntimeError("policy summary must contain exactly 12 policies")
    return [summarize_policy(groups[key]) for key in sorted(groups, key=lambda value: (value[1], value[0], value[2]))]


def delta_value(metric: str, m5: float, alloff: float) -> float:
    return float(m5 - alloff)


def seed_level_deltas(summaries: list[dict[str, Any]], role: str = "best") -> list[dict[str, Any]]:
    lookup = {(row["method"], row["training_seed"], row["checkpoint_role"]): row for row in summaries}
    rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        alloff = lookup[("All-Off", seed, role)]
        m5 = lookup[("M5 Wave Balance", seed, role)]
        for metric in DELTA_METRICS:
            rows.append({"row_type": "training_seed", "checkpoint_role": role, "training_seed": seed,
                         "metric": metric, "delta": delta_value(metric, m5[metric], alloff[metric])})
    for metric in DELTA_METRICS:
        values = [row["delta"] for row in rows if row["metric"] == metric]
        mean = statistics.mean(values); std = statistics.stdev(values); half = 4.302652729 * std / math.sqrt(3)
        rows.append({"row_type": "aggregate", "checkpoint_role": role, "training_seed": None,
                     "metric": metric, "delta": None, "n_training_seeds": 3, "mean": mean,
                     "std": std, "median": statistics.median(values), "ci95_low": mean - half,
                     "ci95_high": mean + half, "ci_note": "descriptive t-CI; n=3; unstable"})
    return rows


def paired_bootstrap(values: list[float], seed: int, iterations: int = 10_000) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(iterations, len(array)))
    means = array[indices].mean(axis=1)
    return float(array.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def episode_paired_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(row["method"], int(row["training_seed"]), row["checkpoint_role"], int(row["episode_seed"])): row for row in records}
    mapping = {"average_waves": "waves_cleared", "W3": "W3_clear", "return": "episode_return",
               "red_loss": "red_losses", "K_L": "kill_loss_ratio", "W1": "W1_clear", "W2": "W2_clear",
               "boundary": "red_boundary_exits", "ground": "red_ground_losses", "timeout": "timeout"}
    output: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        episode_seeds = sorted({key[3] for key in lookup if key[1] == seed and key[2] == "best"})
        for index, (metric, field) in enumerate(mapping.items()):
            deltas = [
                float(lookup[("M5 Wave Balance", seed, "best", episode_seed)][field])
                - float(lookup[("All-Off", seed, "best", episode_seed)][field])
                for episode_seed in episode_seeds
            ]
            mean, low, high = paired_bootstrap(deltas, 81_000 + seed * 100 + index)
            output.append({"training_seed": seed, "metric": metric, "paired_episodes": len(deltas),
                           "mean_paired_delta": mean, "bootstrap_ci95_low": low,
                           "bootstrap_ci95_high": high,
                           "interpretation": "fixed-training-seed scenario robustness; not training-seed significance"})
    return output


def direction_counts(delta_rows: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    seed_rows = [row for row in delta_rows if row["row_type"] == "training_seed"]
    for metric in PRIMARY_METRICS:
        local = {"average_waves_cleared": "average_waves", "clear_wave_3_probability": "W3",
                 "average_return": "return", "average_red_loss": "red_loss", "kill_loss_ratio": "K_L"}[metric]
        values = [row["delta"] for row in seed_rows if row["metric"] == local]
        favorable = sum(value < 0 if local == "red_loss" else value > 0 for value in values)
        result[local] = f"{favorable}/3 favorable"
    return result


def formal_conclusion(delta_rows: list[dict[str, Any]]) -> str:
    seed_rows = [row for row in delta_rows if row["row_type"] == "training_seed"]
    values = {metric: [row["delta"] for row in seed_rows if row["metric"] == metric]
              for metric in ("average_waves", "return", "red_loss", "K_L", "W3")}
    core = (
        sum(value > 0 for value in values["average_waves"]) >= 2,
        sum(value > 0 for value in values["return"]) >= 2,
        sum(value < 0 for value in values["red_loss"]) >= 2,
        sum(value > 0 for value in values["K_L"]) >= 2,
    )
    return "M5_FORMAL_HOLDOUT_SUPPORTED" if all(core) and statistics.mean(values["W3"]) >= 0 else "M5_FORMAL_HOLDOUT_FAILED_MIXED"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    names = list(fieldnames) if fieldnames is not None else list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def stability_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row, row_type="policy") for row in summaries]
    rows.extend(seed_level_deltas(summaries, "best"))
    rows.extend(seed_level_deltas(summaries, "latest"))
    return rows


def mean_std_table(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        policies = [row for row in summaries if row["method"] == method and row["checkpoint_role"] == "best"]
        if len(policies) != 3:
            raise RuntimeError("mean/std aggregation must use exactly three training-seed policy summaries")
        result: dict[str, Any] = {"Method": method, "n_training_seeds": 3}
        for metric in ("W1", "W2", "W3", "average_waves", "return", "red_loss", "K_L", "boundary", "ground", "timeout"):
            values = [float(row[metric]) for row in policies]
            result[metric] = f"{statistics.mean(values):.6g} ± {statistics.stdev(values):.6g}"
        rows.append(result)
    return rows


def create_plots(summaries: list[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    primary = [row for row in summaries if row["checkpoint_role"] == "best"]
    for metric, filename, ylabel in (
        ("average_waves", "formal_waves_by_seed.png", "Average waves cleared"),
        ("W3", "formal_W3_by_seed.png", "W3 clear rate"),
        ("return", "formal_return_by_seed.png", "Average return"),
        ("red_loss", "formal_red_loss_by_seed.png", "Average Red loss"),
        ("K_L", "formal_KL_by_seed.png", "Kill/loss ratio"),
    ):
        fig, axis = plt.subplots(figsize=(6, 4))
        for method in METHODS:
            rows = sorted((row for row in primary if row["method"] == method), key=lambda row: row["training_seed"])
            axis.plot([row["training_seed"] for row in rows], [row[metric] for row in rows], marker="o", label=method)
        axis.set(xlabel="Training seed", ylabel=ylabel, xticks=list(TRAINING_SEEDS)); axis.legend(); fig.tight_layout()
        fig.savefig(output / filename, dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(TRAINING_SEEDS)); width = .18
    for index, method in enumerate(METHODS):
        rows = sorted((row for row in primary if row["method"] == method), key=lambda row: row["training_seed"])
        axis.bar(x + (index - .5) * width, [row["timeout_conditioned_reached_W3"] or 0 for row in rows], width, label=method)
    axis.set(xticks=x, xticklabels=TRAINING_SEEDS, xlabel="Training seed", ylabel="Timeout | reached W3")
    axis.legend(); fig.tight_layout(); fig.savefig(output / "formal_timeout_conditioned.png", dpi=180); plt.close(fig)


def build_outputs(manifest: dict[str, Any], digest: str, records: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    summaries = policy_summaries(records)
    primary_delta = seed_level_deltas(summaries, "best")
    paired = episode_paired_summary(records)
    conclusion = formal_conclusion(primary_delta)
    write_csv(output / "formal_holdout_per_episode.csv", records, PER_EPISODE_FIELDS)
    write_csv(output / "formal_holdout_policy_summary.csv", summaries)
    write_csv(output / "formal_holdout_seed_level_delta.csv", primary_delta)
    write_csv(output / "formal_holdout_episode_paired.csv", paired)
    write_csv(output / "formal_holdout_best_latest_stability.csv", stability_rows(summaries))
    primary_table = [
        {"Method": row["method"], "Training Seed": row["training_seed"], "W1": row["W1"], "W2": row["W2"],
         "W3": row["W3"], "Avg Waves": row["average_waves"], "Return": row["return"],
         "Red Loss": row["red_loss"], "K/L": row["K_L"], "Boundary": row["boundary"],
         "Ground": row["ground"], "Timeout": row["timeout"]}
        for row in summaries if row["checkpoint_role"] == "best"
    ]
    write_csv(output / "formal_holdout_primary_table.csv", primary_table)
    write_csv(output / "formal_holdout_mean_std_table.csv", mean_std_table(summaries))
    create_plots(summaries, output)
    result = {
        "protocol_name": manifest["protocol_name"], "protocol_manifest_sha256": digest,
        "formal_seed_start": FORMAL_SEED_START, "formal_seed_end": FORMAL_SEED_END,
        "episodes_per_policy": EPISODES_PER_POLICY, "total_episodes": len(records),
        "statistical_unit": "training_seed", "n_training_seeds": 3,
        "training_seed_ci_warning": "n=3; t-based intervals are unstable and descriptive only",
        "paired_episode_warning": "paired bootstrap describes fixed-training-seed scenario robustness, not training-seed significance",
        "direction_consistency": direction_counts(primary_delta), "conclusion": conclusion,
    }
    atomic_write_json(output / "formal_holdout_summary.json", result)
    report = [
        "# Frozen M5 Formal Holdout", "", f"Conclusion: **{conclusion}**", "",
        "Statistical unit: training seed (n=3). Student-t intervals are descriptive and unstable.",
        "Episode-paired bootstrap intervals describe scenario robustness within a fixed trained policy; they are not algorithm-level significance tests.",
        "", "## Direction consistency", "",
    ] + [f"- {metric}: {value}" for metric, value in result["direction_consistency"].items()]
    report += ["", "## Protocol", "", f"- Manifest SHA256: `{digest}`",
               f"- Formal seeds: {FORMAL_SEED_START}-{FORMAL_SEED_END}",
               "- Primary: best_eval.pt", "- Secondary robustness: latest.pt",
               "- No post-holdout method, checkpoint, metric or seed selection is permitted."]
    (output / "formal_holdout_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume-exact", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="run one episode/policy using only seed 99,000,000")
    args = parser.parse_args()
    manifest, digest = load_locked_manifest()
    require_cuda()
    tasks = build_tasks(manifest, smoke=args.smoke)
    output = SMOKE_OUT if args.smoke else OUT
    prepare_output_directory(output, manifest, digest, tasks, args.resume_exact, args.smoke)
    evaluate_all(tasks, output, digest, args.workers)
    records = collect_records(tasks, output, digest)
    if args.smoke:
        atomic_write_json(output / COMPLETED_NAME, {
            "smoke": True, "episode_seed": SMOKE_SEED_START, "policies": len(tasks),
            "cuda_device": require_cuda(), "manifest_sha256": digest,
        })
        print(json.dumps({"smoke": "passed", "policies": len(tasks), "episode_seed": SMOKE_SEED_START}, indent=2))
        return
    result = build_outputs(manifest, digest, records, output)
    atomic_write_json(output / COMPLETED_NAME, {
        "completed_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": digest, "formal_seed_start": FORMAL_SEED_START,
        "formal_seed_end": FORMAL_SEED_END, "total_episodes": len(records),
        "summary_sha256": file_sha256(output / "formal_holdout_summary.json"),
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
