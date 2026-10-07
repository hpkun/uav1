"""Aggregate independent trained-policy holdout evaluations with protocol checks."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.aggregate_training_runs import CI_METHOD, summarize_values


PROTOCOL_FIELDS = (
    "algorithm",
    "checkpoint_environment_version",
    "checkpoint_environment_variant",
    "evaluation_environment_version",
    "evaluation_environment_variant",
    "cross_variant_evaluation",
    "mappo_impl_version",
    "observation_dim",
    "action_dim",
    "num_agents",
    "checkpoint_training_gamma",
    "checkpoint_training_num_envs",
    "checkpoint_training_total_sampled_steps",
    "checkpoint_training_smoke",
    "checkpoint_effective_hidden_dim",
    "checkpoint_environment_config_sha256",
    "checkpoint_algorithm_config_sha256",
    "evaluation_environment_config_sha256",
    "provided_algorithm_config_sha256",
    "evaluation_episodes",
    "holdout_seed_base",
    "holdout_seed_end",
    "protocol_complete",
)
NUMERIC_METADATA_FIELDS = set(PROTOCOL_FIELDS) | {
    "checkpoint_sampled_steps",
    "checkpoint_training_seed",
    "device",
    "checkpoint",
}


def _performance_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError
    number = float(value)
    if not math.isfinite(number):
        raise ValueError
    return number


def aggregate_holdout_results(
    input_paths: list[Path], output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(input_paths) < 2:
        raise ValueError("at least two holdout result files are required")
    results = [
        json.loads(path.read_text(encoding="utf-8")) for path in input_paths
    ]
    for index, result in enumerate(results, start=1):
        if result.get("protocol_complete") is not True:
            raise RuntimeError(
                f"holdout result {index} has incomplete experiment protocol"
            )
    training_seeds = [result.get("checkpoint_training_seed") for result in results]
    if any(seed is None for seed in training_seeds):
        raise RuntimeError("holdout results are missing checkpoint_training_seed")
    if len(set(training_seeds)) != len(training_seeds):
        raise RuntimeError(
            "holdout aggregation requires unique checkpoint_training_seed values"
        )
    protocol = {field: results[0].get(field) for field in PROTOCOL_FIELDS}
    for field, expected in protocol.items():
        for index, result in enumerate(results[1:], start=2):
            if result.get(field) != expected:
                raise RuntimeError(
                    f"holdout protocol mismatch for {field}: result 1 has "
                    f"{expected!r}, result {index} has {result.get(field)!r}"
                )
    common_fields = set.intersection(*(set(result) for result in results))
    metrics = []
    for field in sorted(common_fields - NUMERIC_METADATA_FIELDS):
        try:
            [_performance_number(result[field]) for result in results]
        except (TypeError, ValueError):
            continue
        metrics.append(field)
    if not metrics:
        raise RuntimeError("holdout results have no common numeric performance metrics")
    statistics = {
        metric: summarize_values([
            _performance_number(result[metric]) for result in results
        ])
        for metric in metrics
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"metric": metric, **statistics[metric]} for metric in metrics]
    with (output_dir / "holdout_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"protocol": protocol, "metrics": statistics}
    (output_dir / "holdout_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    manifest = {
        "input_results": [str(path.resolve()) for path in input_paths],
        "number_of_results": len(input_paths),
        "aggregated_metrics": metrics,
        "ci_method": CI_METHOD,
        "protocol": protocol,
        "checkpoint_training_seeds": training_seeds,
        "checkpoint_sampled_steps": [
            result.get("checkpoint_sampled_steps") for result in results
        ],
    }
    (output_dir / "aggregation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return summary, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = [
        path if path.is_absolute() else PROJECT_ROOT / path for path in args.results
    ]
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else PROJECT_ROOT / args.output_dir
    )
    _, manifest = aggregate_holdout_results(inputs, output_dir)
    print(
        f"aggregated {manifest['number_of_results']} holdout results and "
        f"{len(manifest['aggregated_metrics'])} metrics into {output_dir}"
    )


if __name__ == "__main__":
    main()
