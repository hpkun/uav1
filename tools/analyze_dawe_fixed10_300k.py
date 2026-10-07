#!/usr/bin/env python3
"""Fail-closed, read-only exact-endpoint analyzer for the six-run DAWE screen."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256, runtime_source_manifest
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
from algorithm.modular_mappo.trainer import MODULAR_MAPPO_IMPL_VERSION
from algorithm.train_modular_mappo import load_config
from tools.preflight_dawe_fixed10_300k import validate_plain_feedforward_architecture

SEEDS = (5301, 5302, 5303)
SOURCE_STEP = 1_505_280
TARGET = 1_805_280
OUT = ROOT / "outputs/dawe_fixed10_300k_analysis"
ENVIRONMENT = ROOT / "configs/persistent_wave_v2_environment.yaml"
CONFIGS = {
    "Control": ROOT / "configs/dev_dawe_fixed10_control_300k.yaml",
    "DAWE": ROOT / "configs/dev_dawe_fixed10_v1_300k.yaml",
}
RUNS = {
    "Control": lambda seed: ROOT / f"outputs/dev_dawe_control_seed{seed}_300k",
    "DAWE": lambda seed: ROOT / f"outputs/dev_dawe_v1_seed{seed}_300k",
}
METHODS = {"Control": "dawe_fixed10_control", "DAWE": "dawe_fixed10_v1"}
ENABLED = {
    "Control": ["actor_gradient_clipping", "actor_lr_decay"],
    "DAWE": ["actor_gradient_clipping", "actor_lr_decay", "deployment_aligned_wave_exploration"],
}
KEY_RUNTIME_FILES = {
    "algorithm/modular_mappo/trainer.py",
    "algorithm/modules/deployment_aligned_wave_exploration.py",
    "algorithm/modular_mappo/runner.py",
    "algorithm/train_modular_mappo.py",
}
FIELDS = {
    "AW": "average_waves_cleared", "W1": "clear_wave_1_probability",
    "W2": "clear_wave_2_probability", "W3": "clear_wave_3_probability",
    "Return": "average_return", "RedLoss": "average_red_loss",
    "Boundary": "average_red_boundary_exits", "Ground": "average_red_ground_losses",
    "EpisodeLength": "average_episode_length",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in data:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def metrics(row: dict[str, Any]) -> dict[str, float | None]:
    result = {key: float(row[field]) for key, field in FIELDS.items()}
    result["Q2"] = None if result["W1"] == 0 else result["W2"] / result["W1"]
    result["Q3"] = None if result["W2"] == 0 else result["W3"] / result["W2"]
    return result


def delta(high: dict[str, Any], low: dict[str, Any], key: str) -> float | None:
    return None if high[key] is None or low[key] is None else high[key] - low[key]


def evaluation_interval(row: dict[str, Any]) -> tuple[int, int, int]:
    start_value = row.get("seed_start", row.get("evaluation_seed_base"))
    if start_value in (None, ""):
        raise RuntimeError("evaluation row lacks seed_start/evaluation_seed_base")
    start = int(float(start_value))
    episodes_value = row.get("evaluation_episodes", row.get("episodes"))
    if episodes_value in (None, ""):
        raise RuntimeError("evaluation row lacks episode count")
    episodes = int(float(episodes_value))
    end_value = row.get("seed_end", row.get("evaluation_seed_end"))
    end = start + episodes - 1 if end_value in (None, "") else int(float(end_value))
    if episodes <= 0 or end != start + episodes - 1:
        raise RuntimeError(f"inconsistent evaluation interval: {start}..{end}, episodes={episodes}")
    return start, end, episodes


def intervals_intersect(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])


def validate_evaluation_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = []
    for row in history:
        start, end, episodes = evaluation_interval(row)
        if intervals_intersect((start, end), (45_000_000, 45_000_199)):
            raise RuntimeError(f"evaluation interval intersects reserved 45M: {start}..{end}")
        if int(float(row["sampled_steps"])) == TARGET:
            endpoints.append(row)
    if len(endpoints) != 1:
        raise RuntimeError(f"expected exactly one exact endpoint row, got {len(endpoints)}")
    if evaluation_interval(endpoints[0]) != (44_000_000, 44_000_049, 50):
        raise RuntimeError("exact endpoint is not the registered 44M 50-episode interval")
    return endpoints[0]


def validate_run_config(run: dict[str, Any], method: str, seed: int, environment_hash: str,
                        algorithm_hash: str, module_hash: str) -> None:
    expected = {
        "algorithm": "modular_mappo", "seed": seed, "total_sampled_steps": TARGET,
        "num_envs": 24, "smoke": False, "device": "cuda",
        "environment_variant": "persistent_wave_v2",
        "environment_config_sha256": environment_hash,
        "algorithm_config_sha256": algorithm_hash,
        "development_method": METHODS[method],
        "module_config_sha256": module_hash,
        "modular_mappo_impl_version": MODULAR_MAPPO_IMPL_VERSION,
        "baseline_mappo_impl_version": MAPPO_IMPL_VERSION,
    }
    mismatches = {key: (run.get(key), value) for key, value in expected.items() if run.get(key) != value}
    if mismatches:
        raise RuntimeError(f"run_config mismatch: {mismatches}")
    if run.get("enabled_modules") != ENABLED[method]:
        raise RuntimeError(f"run_config enabled_modules mismatch: {run.get('enabled_modules')}")
    validate_plain_feedforward_architecture(run.get("network_architecture", {}))


def validate_checkpoint(state: dict[str, Any], run: dict[str, Any], seed: int,
                        source_state: dict[str, Any], label: str) -> None:
    extra = state.get("extra", {})
    expected = {
        "algorithm": "modular_mappo",
        "modular_mappo_impl_version": MODULAR_MAPPO_IMPL_VERSION,
        "baseline_mappo_impl_version": MAPPO_IMPL_VERSION,
        "sampled_steps": TARGET,
        "module_config_sha256": run["module_config_sha256"],
    }
    mismatches = {key: (state.get(key), value) for key, value in expected.items() if state.get(key) != value}
    if mismatches:
        raise RuntimeError(f"{label} checkpoint mismatch: {mismatches}")
    extra_expected = {
        "training_seed": seed,
        "training_total_sampled_steps": TARGET,
        "training_num_envs": 24,
        "training_smoke": False,
        "environment_variant": "persistent_wave_v2",
        "environment_config_sha256": run["environment_config_sha256"],
        "algorithm_config_sha256": run["algorithm_config_sha256"],
        "runtime_source_manifest_sha256": run["runtime_source_manifest_sha256"],
    }
    extra_mismatches = {key: (extra.get(key), value) for key, value in extra_expected.items() if extra.get(key) != value}
    if extra_mismatches:
        raise RuntimeError(f"{label} checkpoint extra mismatch: {extra_mismatches}")
    if state.get("enabled_modules") != run["enabled_modules"]:
        raise RuntimeError(f"{label} checkpoint module identity mismatch")
    validate_plain_feedforward_architecture(extra.get("network_architecture", {}))
    deltas = {
        "ppo_updates": int(state.get("ppo_updates", -1)) - int(source_state.get("ppo_updates", -1)),
        "actor_updates": int(state.get("actor_updates", -1)) - int(source_state.get("actor_updates", -1)),
        "critic_updates": int(state.get("critic_updates", -1)) - int(source_state.get("critic_updates", -1)),
    }
    if deltas != {"ppo_updates": 49, "actor_updates": 5860, "critic_updates": 5860}:
        raise RuntimeError(f"{label} checkpoint update deltas mismatch: {deltas}")


def validate_optimization_metrics(rows: list[dict[str, Any]]) -> tuple[int, int]:
    if len(rows) != 49:
        raise RuntimeError(f"expected 49 optimization rows, got {len(rows)}")
    actor_steps = sum(int(row.get("actor_optimizer_steps_this_update", -1)) for row in rows)
    critic_steps = sum(int(row.get("critic_optimizer_steps_this_update", -1)) for row in rows)
    if (actor_steps, critic_steps) != (5860, 5860):
        raise RuntimeError(f"optimization step sums mismatch: {(actor_steps, critic_steps)}")
    if any(int(row.get("ppo_epochs_executed", -1)) != 10 for row in rows):
        raise RuntimeError("not every optimization row executed 10 PPO epochs")
    return actor_steps, critic_steps


def validate_run_summary(summary: dict[str, Any], run: dict[str, Any], method: str) -> None:
    expected = {
        "algorithm": "modular_mappo", "sampled_steps": TARGET,
        "latest_step": TARGET, "development_method": METHODS[method],
    }
    mismatches = {key: (summary.get(key), value) for key, value in expected.items() if summary.get(key) != value}
    if mismatches:
        raise RuntimeError(f"run_summary mismatch: {mismatches}")
    protocol = summary.get("protocol", {})
    protocol_expected = {
        "environment_config_sha256": run["environment_config_sha256"],
        "algorithm_config_sha256": run["algorithm_config_sha256"],
        "module_config_sha256": run["module_config_sha256"],
    }
    protocol_mismatches = {key: (protocol.get(key), value) for key, value in protocol_expected.items() if protocol.get(key) != value}
    if protocol_mismatches:
        raise RuntimeError(f"run_summary protocol mismatch: {protocol_mismatches}")
    if protocol.get("enabled_modules") != ENABLED[method]:
        raise RuntimeError("run_summary enabled_modules mismatch")
    latest = summary.get("latest_evaluation")
    if latest is not None:
        if int(latest.get("sampled_steps", -1)) != TARGET:
            raise RuntimeError("run_summary latest_evaluation is not exact endpoint")
        if evaluation_interval(latest) != (44_000_000, 44_000_049, 50):
            raise RuntimeError("run_summary latest_evaluation protocol mismatch")


def runtime_file_map(run: dict[str, Any]) -> dict[str, str]:
    rows = run.get("runtime_source_manifest_files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("run lacks complete runtime_source_manifest_files")
    result = {str(row["path"]).replace("\\", "/"): row["sha256"] for row in rows}
    missing = sorted(KEY_RUNTIME_FILES - result.keys())
    if missing:
        raise RuntimeError(f"runtime manifest lacks key files: {missing}")
    if not any(path.startswith("env/") and path.endswith(".py") for path in result):
        raise RuntimeError("runtime manifest lacks env Python sources")
    return result


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {OUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for checkpoint audit")

    environment = load_config(ENVIRONMENT)
    environment_hash = config_sha256(environment)
    configurations = {method: load_config(path) for method, path in CONFIGS.items()}
    algorithm_hashes = {method: config_sha256(config) for method, config in configurations.items()}
    module_hashes = {method: config_sha256(config["modules"]) for method, config in configurations.items()}
    endpoint: dict[int, dict[str, dict[str, Any]]] = {}
    paired: list[dict[str, Any]] = []
    mechanism: list[dict[str, Any]] = []
    protocol: dict[str, Any] = {}
    training_runtime_sha: str | None = None
    training_manifest: dict[str, str] | None = None

    for seed in SEEDS:
        endpoint[seed] = {}
        source_path = ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt"
        source_sha = file_sha256(source_path)
        source_state = torch.load(source_path, map_location="cuda", weights_only=False)
        if int(source_state.get("sampled_steps", -1)) != SOURCE_STEP:
            raise RuntimeError(f"source checkpoint step mismatch: seed{seed}")
        for method, path_fn in RUNS.items():
            path = path_fn(seed)
            required = ("run_config.json", "run_summary.json", "branch_from.json", "evaluation_history.csv", "optimization_metrics.jsonl", "latest.pt", "final.pt")
            if not path.is_dir() or any(not (path / name).is_file() for name in required):
                raise RuntimeError(f"incomplete run: {path}")
            run = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
            summary = json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
            branch = json.loads((path / "branch_from.json").read_text(encoding="utf-8"))
            history = csv_rows(path / "evaluation_history.csv")
            optimization = jsonl_rows(path / "optimization_metrics.jsonl")

            validate_run_config(run, method, seed, environment_hash, algorithm_hashes[method], module_hashes[method])
            if branch.get("parent_checkpoint_sha256") != source_sha or int(branch.get("source_training_seed", -1)) != seed:
                raise RuntimeError(f"parent checkpoint provenance mismatch: {path}")
            final_row = validate_evaluation_history(history)
            actor_steps, critic_steps = validate_optimization_metrics(optimization)
            validate_run_summary(summary, run, method)
            final_state = torch.load(path / "final.pt", map_location="cuda", weights_only=False)
            latest_state = torch.load(path / "latest.pt", map_location="cuda", weights_only=False)
            validate_checkpoint(final_state, run, seed, source_state, "final.pt")
            validate_checkpoint(latest_state, run, seed, source_state, "latest.pt")
            if any(final_state.get(key) != latest_state.get(key) for key in ("sampled_steps", "ppo_updates", "actor_updates", "critic_updates")):
                raise RuntimeError(f"final.pt/latest.pt endpoint counters differ: {path}")

            manifest = runtime_file_map(run)
            run_runtime_sha = run.get("runtime_source_manifest_sha256")
            if not isinstance(run_runtime_sha, str):
                raise RuntimeError(f"missing runtime source SHA: {path}")
            if training_runtime_sha is None:
                training_runtime_sha, training_manifest = run_runtime_sha, manifest
            elif run_runtime_sha != training_runtime_sha or manifest != training_manifest:
                raise RuntimeError("six runs do not share one complete runtime source manifest")

            endpoint[seed][method] = metrics(final_row)
            protocol[f"{method}_seed{seed}"] = {
                "status": "PASS", "runtime_source_sha": run_runtime_sha,
                "parent_sha": source_sha, "optimization_rows": 49,
                "actor_steps": actor_steps, "critic_steps": critic_steps,
                "final_latest_same_endpoint": True, "uses_45m": False,
            }
            windows = {"early": optimization[:10], "late": optimization[-16:], "final5": optimization[-5:]}
            for window, selected in windows.items():
                for wave in (1, 2, 3):
                    def mean(key: str) -> float | None:
                        values = [float(row[key]) for row in selected if row.get(key) not in (None, "")]
                        return statistics.mean(values) if values else None
                    mechanism.append({
                        "method": method, "seed": seed, "window": window, "wave": wave,
                        "base_std_mean": mean(f"dawe_wave{wave}_base_std_mean"),
                        "effective_std_mean": mean(f"dawe_wave{wave}_effective_std_mean"),
                        "effective_to_base_std_ratio": mean(f"dawe_wave{wave}_effective_to_base_std_ratio"),
                        "base_log_std_mean": mean(f"dawe_wave{wave}_base_log_std_mean"),
                        "latent_deviation_abs_mean": mean(f"dawe_wave{wave}_latent_deviation_abs_mean"),
                    })
        del source_state
        for key in (*FIELDS, "Q2", "Q3"):
            paired.append({
                "seed": seed, "metric": key,
                "Control": endpoint[seed]["Control"][key], "DAWE": endpoint[seed]["DAWE"][key],
                "delta": delta(endpoint[seed]["DAWE"], endpoint[seed]["Control"], key),
            })

    aggregate: dict[str, Any] = {}
    for key in (*FIELDS, "Q2", "Q3"):
        rows = [row for row in paired if row["metric"] == key]
        values = [row["delta"] for row in rows if row["delta"] is not None]
        aggregate[key] = {
            "mean_delta": statistics.mean(values) if values else None,
            "wins": sum(value > 0 for value in values), "defined_n": len(values),
            "per_seed": {str(row["seed"]): row["delta"] for row in rows},
        }
    safety = all(aggregate["AW"]["per_seed"][str(seed)] > -0.50 and aggregate["W1"]["per_seed"][str(seed)] > -0.25 for seed in SEEDS) and aggregate["W1"]["mean_delta"] >= -0.05
    efficacy = aggregate["AW"]["wins"] >= 2 and aggregate["AW"]["mean_delta"] > 0 and aggregate["W3"]["wins"] >= 2 and aggregate["W3"]["mean_delta"] > 0
    label = "SAFETY_FAIL" if not safety else "PROMISING" if efficacy else "NOT_SUPPORTED"

    current_manifest = runtime_source_manifest(ROOT)
    current_sha = current_manifest["runtime_source_manifest_sha256"]
    runtime_matches = current_sha == training_runtime_sha
    report = {
        "status": "DAWE_FIXED10_300K_ANALYSIS_COMPLETE",
        "replication_unit": "training_seed", "n": 3, "primary_endpoint": TARGET,
        "protocol": protocol, "endpoint": endpoint,
        "paired_DAWE_minus_Control": aggregate, "mechanism_diagnostics": mechanism,
        "gates": {"safety": safety, "efficacy": efficacy}, "DAWE_SCREEN": label,
        "evaluation_rerun": False, "uses_45m": False,
        "runtime_source_provenance": {
            "training_runtime_source_sha256": training_runtime_sha,
            "current_runtime_source_sha256": current_sha,
            "runtime_source_matches_current": runtime_matches,
            "warning": None if runtime_matches else "CURRENT_SOURCE_DIFFERS_FROM_TRAINING_SOURCE",
            "six_run_manifests_identical": True,
            "key_training_source_files": {key: training_manifest[key] for key in sorted(KEY_RUNTIME_FILES)},
        },
    }
    OUT.mkdir(parents=True)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "protocol_validation.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    write_csv(OUT / "paired_endpoint.csv", paired)
    write_csv(OUT / "mechanism_diagnostics.csv", mechanism)
    (OUT / "decision_support.txt").write_text(f"DAWE_SCREEN={label}\nSafety={safety}\nEfficacy={efficacy}\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "label": label, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
