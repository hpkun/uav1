"""Evaluate Direct- and Persistent-trained MAPPO policies in both environments."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch

from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
from algorithm.common.protocol import config_sha256
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint


def resolved(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def validate_matrix_source(
    role: str,
    checkpoint: Path,
    algorithm_config: dict[str, Any],
    environment_config: dict[str, Any],
) -> dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_for_evaluation(
        state, environment_config, algorithm_config, allow_cross_variant=False
    )
    extra = state.get("extra", {})
    expected_variant = str(
        environment_config.get("environment_variant", "direct_v2_3")
    )
    if str(extra.get("environment_variant", "direct_v2_3")) != expected_variant:
        raise RuntimeError(f"{role} checkpoint/source environment role mismatch")
    required = (
        "training_seed", "training_gamma", "training_num_envs",
        "training_total_sampled_steps", "training_smoke",
        "environment_config_sha256", "algorithm_config_sha256",
    )
    missing = [field for field in required if field not in extra]
    if missing:
        raise RuntimeError(
            f"{role} checkpoint has incomplete protocol metadata: "
            + ", ".join(missing)
        )
    expected_algorithm_hash = config_sha256(algorithm_config)
    if extra["algorithm_config_sha256"] != expected_algorithm_hash:
        raise RuntimeError(f"{role} checkpoint algorithm config fingerprint mismatch")
    expected_environment_hash = config_sha256(environment_config)
    if extra["environment_config_sha256"] != expected_environment_hash:
        raise RuntimeError(f"{role} checkpoint environment config fingerprint mismatch")
    return {
        "training_gamma": extra["training_gamma"],
        "algorithm_config_sha256": extra["algorithm_config_sha256"],
        "environment_config_sha256": extra["environment_config_sha256"],
        "sampled_steps": int(state.get("sampled_steps", 0)),
        "protocol_complete": True,
    }


def evaluate_policy_matrix(
    direct_checkpoint: Path,
    persistent_checkpoint: Path,
    direct_algorithm_config: dict[str, Any],
    persistent_algorithm_config: dict[str, Any],
    direct_environment_config: dict[str, Any],
    persistent_environment_config: dict[str, Any],
    seeds: list[int],
    device: str,
    output_dir: Path,
    manifest_paths: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    if not seeds:
        raise ValueError("seeds must not be empty")
    if any(right != left + 1 for left, right in zip(seeds, seeds[1:])):
        raise ValueError("seeds must be strictly increasing and contiguous by 1")
    direct_protocol = validate_matrix_source(
        "direct", direct_checkpoint, direct_algorithm_config,
        direct_environment_config,
    )
    persistent_protocol = validate_matrix_source(
        "persistent", persistent_checkpoint, persistent_algorithm_config,
        persistent_environment_config,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = {
        "direct_to_direct": (
            direct_checkpoint, direct_algorithm_config, direct_environment_config
        ),
        "direct_to_persistent": (
            direct_checkpoint, direct_algorithm_config, persistent_environment_config
        ),
        "persistent_to_direct": (
            persistent_checkpoint, persistent_algorithm_config, direct_environment_config
        ),
        "persistent_to_persistent": (
            persistent_checkpoint,
            persistent_algorithm_config,
            persistent_environment_config,
        ),
    }
    results: dict[str, dict[str, Any]] = {}
    cross_cells = {"direct_to_persistent", "persistent_to_direct"}
    for name, (checkpoint, algorithm_config, environment_config) in cells.items():
        result = evaluate_mappo_checkpoint(
            checkpoint,
            algorithm_config,
            environment_config,
            device,
            seeds,
            allow_cross_variant=name in cross_cells,
        )
        results[name] = result
        write_json(output_dir / f"{name}.json", result)

    write_json(output_dir / "matrix_summary.json", results)
    rows = []
    for cell, result in results.items():
        rows.append({
            "cell": cell,
            "source_training_variant": result["checkpoint_environment_variant"],
            "target_evaluation_variant": result["evaluation_environment_variant"],
            "cross_variant": result["cross_variant_evaluation"],
            "checkpoint": result["checkpoint"],
            "episodes": result["evaluation_episodes"],
            "seed_base": result["holdout_seed_base"],
            **result,
        })
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "matrix_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        **(manifest_paths or {}),
        "direct_checkpoint": str(direct_checkpoint.resolve()),
        "persistent_checkpoint": str(persistent_checkpoint.resolve()),
        "holdout_seed_base": seeds[0],
        "holdout_seed_end": seeds[-1],
        "evaluation_episodes": len(seeds),
        "device": device,
        "direct_training_gamma": direct_protocol["training_gamma"],
        "persistent_training_gamma": persistent_protocol["training_gamma"],
        "direct_algorithm_config_sha256": direct_protocol[
            "algorithm_config_sha256"
        ],
        "persistent_algorithm_config_sha256": persistent_protocol[
            "algorithm_config_sha256"
        ],
        "direct_environment_config_sha256": direct_protocol[
            "environment_config_sha256"
        ],
        "persistent_environment_config_sha256": persistent_protocol[
            "environment_config_sha256"
        ],
        "direct_checkpoint_sampled_steps": direct_protocol["sampled_steps"],
        "persistent_checkpoint_sampled_steps": persistent_protocol[
            "sampled_steps"
        ],
        "direct_protocol_complete": direct_protocol["protocol_complete"],
        "persistent_protocol_complete": persistent_protocol[
            "protocol_complete"
        ],
    }
    write_json(output_dir / "evaluation_manifest.json", manifest)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-checkpoint", required=True)
    parser.add_argument("--persistent-checkpoint", required=True)
    parser.add_argument(
        "--direct-algorithm-config", default="configs/mappo.yaml"
    )
    parser.add_argument(
        "--persistent-algorithm-config",
        default="configs/mappo_persistent_wave.yaml",
    )
    parser.add_argument("--direct-env-config", default="configs/combat_environment.yaml")
    parser.add_argument(
        "--persistent-env-config",
        default="configs/persistent_wave_v2_environment.yaml",
    )
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    paths = {
        "direct_algorithm_config": resolved(args.direct_algorithm_config),
        "persistent_algorithm_config": resolved(args.persistent_algorithm_config),
        "direct_environment_config": resolved(args.direct_env_config),
        "persistent_environment_config": resolved(args.persistent_env_config),
    }
    configs = {
        name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    results = evaluate_policy_matrix(
        resolved(args.direct_checkpoint),
        resolved(args.persistent_checkpoint),
        configs["direct_algorithm_config"],
        configs["persistent_algorithm_config"],
        configs["direct_environment_config"],
        configs["persistent_environment_config"],
        list(range(args.seed_base, args.seed_base + args.episodes)),
        args.device,
        resolved(args.output_dir),
        {name: str(path) for name, path in paths.items()},
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
