"""Offline paired analysis of Plain versus FireReady-Plain development runs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5301, 5302, 5303)
MILESTONES = (900_000, 1_500_000, 2_000_000, 2_500_000, 3_000_000)
FIELDS = {
    "W1": "clear_wave_1_probability", "W2": "clear_wave_2_probability",
    "W3": "clear_wave_3_probability", "AverageWaves": "average_waves_cleared",
    "Return": "average_return", "red_loss": "average_red_loss",
    "blue_loss": "average_blue_loss", "boundary": "average_red_boundary_exits",
    "ground": "average_red_ground_losses", "episode_length": "average_episode_length",
}


def read_history(run: Path) -> list[dict]:
    path = run / "evaluation_history.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"empty evaluation history: {path}")
    summary_path = run / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("sampled_steps", -1)) != 3_000_000:
        raise RuntimeError(f"run is not exact-3M complete: {run}")
    if not any(int(float(row["sampled_steps"])) == 3_000_000 for row in rows):
        raise RuntimeError(f"run lacks exact-3M evaluation: {run}")
    return rows


def select(rows: list[dict], target: int) -> dict:
    row = min(rows, key=lambda item: (abs(int(float(item["sampled_steps"])) - target),
                                      -int(float(item["sampled_steps"]))))
    result = {"sampled_steps": int(float(row["sampled_steps"]))}
    for short, source in FIELDS.items():
        value = row.get(source, "")
        result[short] = None if value in (None, "") else float(value)
    result["Q2"] = None if result["W1"] in (None, 0) else result["W2"] / result["W1"]
    result["Q3"] = None if result["W2"] in (None, 0) else result["W3"] / result["W2"]
    return result


def summarize(values: list[float]) -> dict:
    return {"mean": mean(values), "std": stdev(values), "min": min(values), "max": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain-root", default="outputs/diag_mappo_learnability")
    parser.add_argument("--fire-ready-root", default="outputs/dev_fire_ready_plain_3m")
    parser.add_argument("--output", default="outputs/dev_fire_ready_plain_3m/fire_ready_plain_analysis.json")
    args = parser.parse_args()
    roots = {"Plain": ROOT / args.plain_root, "FireReady": ROOT / args.fire_ready_root}
    data: dict[str, dict] = {name: {} for name in roots}
    for method, root in roots.items():
        for seed in SEEDS:
            run = root / (f"l3_seed{seed}" if method == "Plain" else f"seed{seed}")
            rows = read_history(run)
            data[method][str(seed)] = {
                "final": select(rows, 3_000_000),
                "learning_curve": {str(step): select(rows, step) for step in MILESTONES},
            }
    metrics = (*FIELDS.keys(), "Q2", "Q3")
    paired = {
        str(seed): {
            metric: data["FireReady"][str(seed)]["final"][metric]
                    - data["Plain"][str(seed)]["final"][metric]
            for metric in metrics
            if data["FireReady"][str(seed)]["final"][metric] is not None
            and data["Plain"][str(seed)]["final"][metric] is not None
        } for seed in SEEDS
    }
    summary = {}
    for method in roots:
        summary[method] = {
            metric: summarize([data[method][str(seed)]["final"][metric] for seed in SEEDS])
            for metric in metrics
            if all(data[method][str(seed)]["final"][metric] is not None for seed in SEEDS)
        }
    delta = {metric: mean([paired[str(seed)][metric] for seed in SEEDS]) for metric in metrics}
    aw_wins = sum(paired[str(seed)]["AverageWaves"] > 0 for seed in SEEDS)
    checks = {
        "mean_delta_average_waves_gte_0.15": delta["AverageWaves"] >= 0.15,
        "paired_aw_wins_gte_2_of_3": aw_wins >= 2,
        "mean_delta_w1_gte_minus_0.05": delta["W1"] >= -0.05,
        "mean_delta_q2_gt_0": delta["Q2"] > 0,
        "mean_delta_q3_gt_0": delta["Q3"] > 0,
        "max_delta_q2_q3_gte_0.05": max(delta["Q2"], delta["Q3"]) >= 0.05,
    }
    verdict = ("FIRE_READY_STATE_SUPPORTED" if all(checks.values())
               else "FIRE_READY_STATE_NOT_SUPPORTED_AS_PRIMARY_LIMITATION")
    result = {"analysis": "offline_existing_evaluations_only", "seeds": list(SEEDS),
              "methods": data, "summary": summary, "paired_deltas": paired,
              "mean_paired_deltas": delta, "paired_aw_wins": aw_wins,
              "primary_gate": checks, "verdict": verdict}
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "verdict": verdict, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
