"""Offline-only HTA-MAPPO V2 primary and retention analysis."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5301, 5302, 5303)
METRIC_KEYS = (
    "W1", "W2", "W3", "Q2", "Q3", "average_waves", "return",
    "red_loss", "blue_loss", "ground", "boundary", "episode_length",
)


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def jsonl_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def number(row, key):
    return None if row.get(key) in (None, "") else float(row[key])


def completed_run(run_dir):
    run_dir = Path(run_dir)
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"run_summary.json is absent: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("sampled_steps", -1)) != 3_000_000:
        raise RuntimeError(f"run is not complete at exact 3M: {run_dir}")
    rows = csv_rows(run_dir / "evaluation_history.csv")
    exact = next((row for row in rows if int(row["sampled_steps"]) == 3_000_000), None)
    if exact is None:
        raise RuntimeError(f"exact-3M evaluation is absent: {run_dir}")
    return rows, exact


def metrics(row):
    w1, w2, w3 = (number(row, f"clear_wave_{index}_probability") for index in (1, 2, 3))
    return {
        "W1": w1, "W2": w2, "W3": w3,
        "Q2": None if not w1 else w2 / w1,
        "Q3": None if not w2 else w3 / w2,
        "average_waves": number(row, "average_waves_cleared"),
        "return": number(row, "average_return"),
        "red_loss": number(row, "average_red_loss"),
        "blue_loss": number(row, "average_blue_loss"),
        "ground": number(row, "average_red_ground_losses"),
        "boundary": number(row, "average_red_boundary_exits"),
        "episode_length": number(row, "average_episode_length"),
    }


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}


def endpoint_summary(per_seed, method):
    return {
        key: summary([per_seed[str(seed)][method][key] for seed in SEEDS])
        for key in METRIC_KEYS
    }


def nearest_lr(rows, target):
    row = min(rows, key=lambda item: abs(int(item["sampled_steps"]) - target))
    return {
        "requested_step": target,
        "actual_step": int(row["sampled_steps"]),
        "worker_lr": number(row, "hta_worker_actor_lr") or number(row, "actor_learning_rate"),
        "manager_lr": number(row, "hta_manager_actor_lr"),
        "multiplier": number(row, "hta_worker_consolidation_multiplier"),
    }


def displacement_phases(rows):
    phases = {
        "through_1p5m": [row for row in rows if int(row["sampled_steps"]) <= 1_500_000],
        "1p5m_to_2p5m": [row for row in rows if 1_500_000 < int(row["sampled_steps"]) < 2_500_000],
        "from_2p5m": [row for row in rows if int(row["sampled_steps"]) >= 2_500_000],
    }
    result = {}
    for name, phase_rows in phases.items():
        values = [number(row, "hta_worker_relative_parameter_update") for row in phase_rows]
        values = np.asarray([value for value in values if value is not None], dtype=np.float64)
        result[name] = ({"count": 0, "mean": None, "p90": None, "max": None} if not values.size else {
            "count": int(values.size), "mean": float(values.mean()),
            "p90": float(np.quantile(values, 0.9)), "max": float(values.max()),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain-root", default="outputs/diag_mappo_learnability")
    parser.add_argument("--v1-root", default="outputs/dev_hta_mappo_v1_3m")
    parser.add_argument("--v2-root", default="outputs/dev_hta_mappo_v2_3m")
    args = parser.parse_args()
    plain_root, v1_root, v2_root = (ROOT / args.plain_root, ROOT / args.v1_root, ROOT / args.v2_root)
    endpoints, paired_primary, paired_secondary = {}, [], []
    retention, learning_rates, displacement = {}, {}, {}
    try:
        for seed in SEEDS:
            plain_rows, plain_exact = completed_run(plain_root / f"l3_seed{seed}")
            v1_rows, v1_exact = completed_run(v1_root / f"seed{seed}")
            v2_rows, v2_exact = completed_run(v2_root / f"seed{seed}")
            p, v1, v2 = metrics(plain_exact), metrics(v1_exact), metrics(v2_exact)
            endpoints[str(seed)] = {"plain": p, "hta_v1": v1, "hta_v2": v2}
            paired_primary.append({"seed": seed, **{
                f"delta_{key}": v2[key] - p[key] for key in METRIC_KEYS
            }})
            paired_secondary.append({"seed": seed, **{
                f"delta_{key}": v2[key] - v1[key] for key in METRIC_KEYS
            }})
            v1_max = max(number(row, "average_waves_cleared") for row in v1_rows)
            v2_max = max(number(row, "average_waves_cleared") for row in v2_rows)
            retention[str(seed)] = {
                "hta_v1": {"max_average_waves": v1_max, "exact3m_average_waves": v1["average_waves"], "retention_gap": v1_max - v1["average_waves"]},
                "hta_v2": {"max_average_waves": v2_max, "exact3m_average_waves": v2["average_waves"], "retention_gap": v2_max - v2["average_waves"]},
            }
            optimization = jsonl_rows(v2_root / f"seed{seed}" / "optimization_metrics.jsonl")
            learning_rates[str(seed)] = [nearest_lr(optimization, step) for step in (1_500_000, 2_000_000, 2_500_000, 3_000_000)]
            displacement[str(seed)] = displacement_phases(optimization)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(json.dumps({"offline_only": True, "result": "HTA_V2_DEVELOPMENT_INCOMPLETE", "reason": str(error)}, indent=2))
        raise SystemExit(2)

    primary_means = {
        key: float(np.mean([row[key] for row in paired_primary]))
        for key in paired_primary[0] if key.startswith("delta_")
    }
    secondary_means = {
        key: float(np.mean([row[key] for row in paired_secondary]))
        for key in paired_secondary[0] if key.startswith("delta_")
    }
    primary_supported = (
        primary_means["delta_average_waves"] >= 0.15
        and sum(row["delta_average_waves"] > 0 for row in paired_primary) >= 2
        and primary_means["delta_W1"] >= -0.05
        and primary_means["delta_Q2"] > 0
        and primary_means["delta_Q3"] > 0
        and max(primary_means["delta_Q2"], primary_means["delta_Q3"]) >= 0.05
    )
    v1_gaps = [retention[str(seed)]["hta_v1"]["retention_gap"] for seed in SEEDS]
    v2_gaps = [retention[str(seed)]["hta_v2"]["retention_gap"] for seed in SEEDS]
    result = {
        "offline_only": True,
        "primary_comparison": "Plain exact-3M vs HTA V2 exact-3M",
        "secondary_comparison": "HTA V1 exact-3M vs HTA V2 exact-3M",
        "endpoints": endpoints,
        "cross_seed_endpoint_summary": {
            method: endpoint_summary(endpoints, method) for method in ("plain", "hta_v1", "hta_v2")
        },
        "paired_primary_deltas": paired_primary,
        "primary_mean_deltas": primary_means,
        "paired_secondary_deltas": paired_secondary,
        "secondary_mean_deltas": secondary_means,
        "retention": retention,
        "retention_gap_means": {"hta_v1": float(np.mean(v1_gaps)), "hta_v2": float(np.mean(v2_gaps))},
        "v2_learning_rates": learning_rates,
        "v2_worker_relative_parameter_update": displacement,
        "primary_gate": "HTA_V2_SUPPORTED" if primary_supported else "HTA_V2_NOT_SUPPORTED",
        "route_if_failed": None if primary_supported else "HTA_ROUTE_CLOSED",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
