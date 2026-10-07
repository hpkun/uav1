"""Create and verify the immutable M5 formal-holdout protocol manifest.

This tool never evaluates a policy and never imports the training entry point.
It discovers runs from recorded metadata, validates the frozen comparison, and
locks all paths and hashes before any reserved 20M episode can be evaluated.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUTPUTS = ROOT / "outputs"
MANIFEST_PATH = OUTPUTS / "formal_holdout_protocol.json"
MANIFEST_HASH_PATH = OUTPUTS / "formal_holdout_protocol.sha256"
TRAINING_SEEDS = (2023, 2024, 2025)
FORMAL_SEED_START = 20_000_000
FORMAL_SEED_END = 20_000_199
EPISODES_PER_POLICY = 200
METHODS = ("All-Off", "M5 Wave Balance")
PRIMARY_CHECKPOINT = "best_eval.pt"
SECONDARY_CHECKPOINT = "latest.pt"
PRIMARY_METRICS = (
    "average_waves_cleared",
    "clear_wave_3_probability",
    "average_return",
    "average_red_loss",
    "kill_loss_ratio",
)
SECONDARY_METRICS = (
    "clear_wave_1_probability", "clear_wave_2_probability",
    "average_blue_loss", "average_red_boundary_exits",
    "average_red_ground_losses", "timeout_rate", "average_episode_length",
    "conditional_timeout_and_timing",
)
EXPECTED_ENVIRONMENT_VARIANT = "persistent_wave_v2"
EXPECTED_GAMMA = 0.999
EXPECTED_NUM_ENVS = 24
EXPECTED_SAMPLED_STEPS = 1_500_000
PROTOCOL_NAME = "M5_WAVE_BALANCE_FORMAL_HOLDOUT_V1"
REPOSITORY_IMPLEMENTATION_VERSION = "formal_holdout_protocol_v1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def formal_episode_seeds() -> list[int]:
    seeds = list(range(FORMAL_SEED_START, FORMAL_SEED_END + 1))
    if len(seeds) != EPISODES_PER_POLICY:
        raise RuntimeError("formal seed range no longer contains exactly 200 seeds")
    return seeds


def method_from_modules(modules: set[str]) -> str | None:
    if not modules:
        return "All-Off"
    if modules == {"wave_balancing"}:
        return "M5 Wave Balance"
    return None


def effective_training_config(config: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    """Merge recorded CLI/runtime overrides into the persisted base YAML."""
    effective = deepcopy(config)
    training = effective["training"]
    training["seed"] = int(run["seed"])
    training["device"] = str(run["device"])
    training["num_train_envs"] = int(run["num_envs"])
    training["total_sampled_steps"] = int(run["total_sampled_steps"])
    return effective


def validate_run(directory: Path) -> dict[str, Any] | None:
    required = {
        "run_config": directory / "run_config.json",
        "run_summary": directory / "run_summary.json",
        "algorithm_config": directory / "algorithm_config.yaml",
        "best": directory / PRIMARY_CHECKPOINT,
        "latest": directory / SECONDARY_CHECKPOINT,
    }
    if not all(path.is_file() for path in required.values()):
        return None
    run = read_json(required["run_config"])
    summary = read_json(required["run_summary"])
    base_config = read_yaml(required["algorithm_config"])
    if run.get("algorithm") != "modular_mappo":
        return None
    if run.get("environment_variant") != EXPECTED_ENVIRONMENT_VARIANT:
        return None
    seed = int(run.get("seed", -1))
    if seed not in TRAINING_SEEDS:
        return None
    method = method_from_modules(set(run.get("enabled_modules", [])))
    if method is None:
        return None
    config = effective_training_config(base_config, run)
    checks = {
        "gamma": abs(float(config["training"]["gamma"]) - EXPECTED_GAMMA) < 1e-12,
        "num_envs": int(run.get("num_envs", -1)) == EXPECTED_NUM_ENVS,
        "run_budget": int(run.get("total_sampled_steps", -1)) == EXPECTED_SAMPLED_STEPS,
        "summary_steps": int(summary.get("sampled_steps", -1)) == EXPECTED_SAMPLED_STEPS,
        "latest_step": int(summary.get("latest_step", -1)) == EXPECTED_SAMPLED_STEPS,
        "resume_count": int(summary.get("resume_count", -1)) == 0,
        "training_seed": int(config["training"]["seed"]) == seed,
        "config_num_envs": int(config["training"]["num_train_envs"]) == EXPECTED_NUM_ENVS,
        "config_budget": int(config["training"]["total_sampled_steps"]) == EXPECTED_SAMPLED_STEPS,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        # Historical smoke/diagnostic runs can share method modules and a
        # training seed. They are not formal candidates unless every frozen
        # budget/protocol field also matches. A missing valid candidate is
        # rejected by discover_formal_runs after this filtering step.
        return None
    best_step = int(summary.get("best_checkpoint_step", -1))
    latest_step = int(summary.get("latest_step", -1))
    if best_step <= 0 or latest_step != EXPECTED_SAMPLED_STEPS:
        raise RuntimeError(f"invalid checkpoint steps in {directory}")
    return {
        "method": method,
        "training_seed": seed,
        "directory": directory.resolve(),
        "run": run,
        "summary": summary,
        "config": config,
        "base_config": base_config,
        "paths": required,
        "best_step": best_step,
        "latest_step": latest_step,
    }


def discover_formal_runs(outputs: Path = OUTPUTS) -> dict[tuple[str, int], dict[str, Any]]:
    discovered: dict[tuple[str, int], dict[str, Any]] = {}
    for directory in outputs.iterdir():
        if not directory.is_dir():
            continue
        record = validate_run(directory)
        if record is None:
            continue
        key = (record["method"], record["training_seed"])
        if key in discovered:
            raise RuntimeError(f"ambiguous formal run for {key}: {discovered[key]['directory']} and {directory}")
        discovered[key] = record
    expected = {(method, seed) for method in METHODS for seed in TRAINING_SEEDS}
    missing = sorted(expected - set(discovered))
    extra = sorted(set(discovered) - expected)
    if missing or extra:
        raise RuntimeError(f"formal run set mismatch: missing={missing}, extra={extra}")
    return discovered


def normalized_matched_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    modules = normalized.get("modules", {})
    if "wave_balancing" not in modules:
        raise RuntimeError("wave_balancing block missing from resolved config")
    modules["wave_balancing"]["enabled"] = False
    return normalized


def validate_matched_pair(alloff: dict[str, Any], m5: dict[str, Any]) -> dict[str, Any]:
    left = normalized_matched_config(alloff["config"])
    right = normalized_matched_config(m5["config"])
    if left != right:
        raise RuntimeError(
            f"All-Off/M5 training configs are not strictly matched for seed {alloff['training_seed']}"
        )
    if alloff["training_seed"] != m5["training_seed"]:
        raise RuntimeError("matched pair training-seed mismatch")
    return {
        "matched": True,
        "allowed_difference": "modules.wave_balancing.enabled",
        "normalized_training_config_sha256": value_sha256(left),
    }


def run_manifest_record(record: dict[str, Any], match: dict[str, Any]) -> dict[str, Any]:
    paths = record["paths"]
    return {
        "method": record["method"],
        "training_seed": record["training_seed"],
        "run_path": relative_path(record["directory"]),
        "run_sampled_steps": int(record["summary"]["sampled_steps"]),
        "enabled_modules": list(record["run"].get("enabled_modules", [])),
        "gamma": float(record["config"]["training"]["gamma"]),
        "num_envs": int(record["run"]["num_envs"]),
        "environment_variant": record["run"]["environment_variant"],
        "resume_count": int(record["summary"]["resume_count"]),
        "best_checkpoint_path": relative_path(paths["best"]),
        "best_checkpoint_sha256": file_sha256(paths["best"]),
        "best_checkpoint_sampled_steps": record["best_step"],
        "latest_checkpoint_path": relative_path(paths["latest"]),
        "latest_checkpoint_sha256": file_sha256(paths["latest"]),
        "latest_checkpoint_sampled_steps": record["latest_step"],
        "resolved_training_config_sha256": value_sha256(record["config"]),
        "normalized_matched_config_sha256": match["normalized_training_config_sha256"],
        "run_config_path": relative_path(paths["run_config"]),
        "run_config_sha256": file_sha256(paths["run_config"]),
        "run_summary_path": relative_path(paths["run_summary"]),
        "run_summary_sha256": file_sha256(paths["run_summary"]),
        "algorithm_config_path": relative_path(paths["algorithm_config"]),
        "algorithm_config_sha256": file_sha256(paths["algorithm_config"]),
        "matched_validation": match,
    }


def build_manifest(outputs: Path = OUTPUTS) -> dict[str, Any]:
    runs = discover_formal_runs(outputs)
    matched = {
        seed: validate_matched_pair(
            runs[("All-Off", seed)], runs[("M5 Wave Balance", seed)]
        ) for seed in TRAINING_SEEDS
    }
    run_records = [
        run_manifest_record(runs[(method, seed)], matched[seed])
        for seed in TRAINING_SEEDS for method in METHODS
    ]
    implementation_pairs = {
        (
            int(record["run"].get("baseline_mappo_impl_version", -1)),
            int(record["run"].get("modular_mappo_impl_version", -1)),
        ) for record in runs.values()
    }
    if len(implementation_pairs) != 1:
        raise RuntimeError(f"implementation version mismatch: {implementation_pairs}")
    mappo_version, modular_version = implementation_pairs.pop()
    return {
        "protocol_name": PROTOCOL_NAME,
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repository_implementation_version": REPOSITORY_IMPLEMENTATION_VERSION,
        "mappo_implementation_version": mappo_version,
        "modular_implementation_version": modular_version,
        "environment_variant": EXPECTED_ENVIRONMENT_VARIANT,
        "environment_config_path": "configs/persistent_wave_v2_environment.yaml",
        "environment_config_sha256": file_sha256(ROOT / "configs/persistent_wave_v2_environment.yaml"),
        "formal_seed_start": FORMAL_SEED_START,
        "formal_seed_end": FORMAL_SEED_END,
        "episodes_per_policy": EPISODES_PER_POLICY,
        "training_seeds": list(TRAINING_SEEDS),
        "methods": list(METHODS),
        "primary_checkpoint_protocol": "best_eval.pt",
        "secondary_checkpoint_protocol": "latest.pt",
        "primary_metric_order": list(PRIMARY_METRICS),
        "secondary_metrics": list(SECONDARY_METRICS),
        "statistical_unit": "training_seed",
        "n_training_seeds": 3,
        "primary_episode_count": 1200,
        "secondary_episode_count": 1200,
        "total_episode_count": 2400,
        "formal_holdout_used_before": False,
        "matched_validation": {str(seed): matched[seed] for seed in TRAINING_SEEDS},
        "runs": run_records,
        "conclusion_rule": {
            "core_metrics": ["average_waves_cleared", "average_return", "average_red_loss", "kill_loss_ratio"],
            "support_requirement": "M5 favorable in at least 2/3 training seeds for every core metric and mean W3 delta >= 0",
            "supported_label": "M5_FORMAL_HOLDOUT_SUPPORTED",
            "failed_label": "M5_FORMAL_HOLDOUT_FAILED_MIXED",
            "no_post_holdout_changes": True,
        },
    }


def manifest_sha256(manifest: dict[str, Any]) -> str:
    return value_sha256(manifest)


def validate_manifest_schema(manifest: dict[str, Any]) -> None:
    exact = {
        "protocol_name": PROTOCOL_NAME,
        "environment_variant": EXPECTED_ENVIRONMENT_VARIANT,
        "formal_seed_start": FORMAL_SEED_START,
        "formal_seed_end": FORMAL_SEED_END,
        "episodes_per_policy": EPISODES_PER_POLICY,
        "training_seeds": list(TRAINING_SEEDS),
        "methods": list(METHODS),
        "primary_checkpoint_protocol": PRIMARY_CHECKPOINT,
        "secondary_checkpoint_protocol": SECONDARY_CHECKPOINT,
        "primary_metric_order": list(PRIMARY_METRICS),
        "formal_holdout_used_before": False,
    }
    mismatches = [key for key, value in exact.items() if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError(f"frozen formal protocol mismatch: {mismatches}")
    if len(manifest.get("runs", [])) != 6:
        raise RuntimeError("formal manifest must contain exactly six runs")
    if formal_episode_seeds() != list(range(manifest["formal_seed_start"], manifest["formal_seed_end"] + 1)):
        raise RuntimeError("formal seed list mismatch")


def validate_manifest_files(manifest: dict[str, Any]) -> None:
    validate_manifest_schema(manifest)
    for run in manifest["runs"]:
        for path_key, hash_key in (
            ("best_checkpoint_path", "best_checkpoint_sha256"),
            ("latest_checkpoint_path", "latest_checkpoint_sha256"),
            ("run_config_path", "run_config_sha256"),
            ("run_summary_path", "run_summary_sha256"),
            ("algorithm_config_path", "algorithm_config_sha256"),
        ):
            path = resolve_manifest_path(run[path_key])
            if not path.is_file() or file_sha256(path) != run[hash_key]:
                raise RuntimeError(f"locked file changed or missing: {path}")
        base_config = read_yaml(resolve_manifest_path(run["algorithm_config_path"]))
        run_config = read_json(resolve_manifest_path(run["run_config_path"]))
        config = effective_training_config(base_config, run_config)
        if value_sha256(config) != run["resolved_training_config_sha256"]:
            raise RuntimeError(f"resolved training config changed for {run['method']} seed {run['training_seed']}")
    environment = resolve_manifest_path(manifest["environment_config_path"])
    if file_sha256(environment) != manifest["environment_config_sha256"]:
        raise RuntimeError("locked environment config changed")
    # Rediscovery catches method-list, metadata, budget, module and matched-pair changes.
    rediscovered = discover_formal_runs(OUTPUTS)
    for seed in TRAINING_SEEDS:
        validate_matched_pair(rediscovered[("All-Off", seed)], rediscovered[("M5 Wave Balance", seed)])


def write_locked_manifest(manifest: dict[str, Any]) -> str:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    digest = manifest_sha256(manifest)
    if MANIFEST_PATH.exists() or MANIFEST_HASH_PATH.exists():
        if not (MANIFEST_PATH.exists() and MANIFEST_HASH_PATH.exists()):
            raise RuntimeError("partial protocol lock exists")
        existing = read_json(MANIFEST_PATH)
        locked_digest = MANIFEST_HASH_PATH.read_text(encoding="utf-8").strip()
        if manifest_sha256(existing) != locked_digest:
            raise RuntimeError("existing protocol manifest/hash mismatch")
        validate_manifest_files(existing)
        return locked_digest
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MANIFEST_HASH_PATH.write_text(digest + "\n", encoding="utf-8")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="verify the existing immutable manifest")
    args = parser.parse_args()
    if args.verify:
        if not MANIFEST_PATH.is_file() or not MANIFEST_HASH_PATH.is_file():
            raise FileNotFoundError("formal protocol manifest has not been prepared")
        manifest = read_json(MANIFEST_PATH)
        digest = manifest_sha256(manifest)
        if digest != MANIFEST_HASH_PATH.read_text(encoding="utf-8").strip():
            raise RuntimeError("protocol manifest hash mismatch")
        validate_manifest_files(manifest)
    else:
        manifest = build_manifest()
        digest = write_locked_manifest(manifest)
    print(json.dumps({"manifest": relative_path(MANIFEST_PATH), "sha256": digest, "verified": True}, indent=2))


if __name__ == "__main__":
    main()
