"""Same-gamma Direct/Persistent 2x2 diagnostic with paired episode records."""
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

import numpy as np
import torch
import yaml

from algorithm.common.protocol import config_sha256
from tools.audit_mappo_run import aggregate_diagnostic, checkpoint_audit, plain, spatial_features
from tools.evaluate_policy_matrix import validate_matrix_source
from tools.plot_best_model_trajectories import build_mappo, rollout, write_trajectory_artifacts


METRICS = (
    "win_rate", "clear_wave_1_probability", "clear_wave_2_probability",
    "clear_wave_3_probability", "average_waves_cleared", "average_return",
    "average_red_loss", "average_blue_loss", "kill_loss_ratio",
    "average_red_survivors_after_wave_1", "average_red_survivors_after_wave_2",
    "average_red_survivors_after_wave_3", "average_red_boundary_exits",
    "evaluation_boundary_exit_rate", "average_red_ground_losses",
    "average_blue_ground_losses", "timeout_rate", "average_episode_length",
)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(plain(payload), indent=2), encoding="utf-8")


def difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {key: float(left[key]) - float(right[key]) for key in METRICS
            if key in left and key in right}


def median_seed(rows: list[dict[str, Any]], score) -> int | None:
    if not rows:
        return None
    values = [float(score(row)) for row in rows]
    median = float(np.median(values))
    return int(min(rows, key=lambda row: abs(float(score(row)) - median))["seed"])


def select_pairs(direct: list[dict[str, Any]], persistent: list[dict[str, Any]]) -> dict[str, Any]:
    p_by_seed = {int(row["seed"]): row for row in persistent}
    pairs = [{"seed": int(d["seed"]), "direct": d, "persistent": p_by_seed[int(d["seed"])]}
             for d in direct]
    priority = [p for p in pairs if p["direct"]["waves_cleared"] <= 1
                and p["persistent"]["waves_cleared"] == 3]
    level = 1
    if not priority:
        priority = [p for p in pairs if p["direct"]["waves_cleared"] == 2
                    and p["persistent"]["waves_cleared"] == 3]
        level = 2
    key_seed = median_seed(priority, lambda p: p["persistent"]["team_return"] - p["direct"]["team_return"])
    counter = [p for p in pairs if p["direct"]["waves_cleared"] >= p["persistent"]["waves_cleared"]]
    counter_seed = median_seed(counter, lambda p: p["direct"]["team_return"] - p["persistent"]["team_return"])
    return {"key_seed": key_seed, "key_priority": level if key_seed is not None else None,
            "key_candidate_count": len(priority), "counterexample_seed": counter_seed,
            "counterexample_candidate_count": len(counter)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-run", type=Path, required=True)
    parser.add_argument("--persistent-run", type=Path, required=True)
    parser.add_argument("--algorithm-config", type=Path, required=True)
    parser.add_argument("--direct-env-config", type=Path, required=True)
    parser.add_argument("--persistent-env-config", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-trajectories", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 50:
        raise ValueError("episodes must be 1..50")
    seeds = list(range(args.seed_base, args.seed_base + args.episodes))
    if any(20_000_000 <= seed <= 20_000_199 for seed in seeds):
        raise ValueError("formal holdout seed range is forbidden")
    torch.set_num_threads(1)
    resolve = lambda path: path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    direct_run, persistent_run, output = map(resolve, (args.direct_run, args.persistent_run, args.output_dir))
    alg_path, direct_env_path, persistent_env_path = map(
        resolve, (args.algorithm_config, args.direct_env_config, args.persistent_env_config))
    alg, direct_env, persistent_env = map(load_yaml, (alg_path, direct_env_path, persistent_env_path))
    d_checkpoint, p_checkpoint = direct_run / "best_eval.pt", persistent_run / "best_eval.pt"
    direct_protocol = validate_matrix_source("direct", d_checkpoint, alg, direct_env)
    persistent_protocol = validate_matrix_source("persistent", p_checkpoint, alg, persistent_env)
    if float(direct_protocol["training_gamma"]) != float(persistent_protocol["training_gamma"]):
        raise RuntimeError("training gamma mismatch")
    output.mkdir(parents=True, exist_ok=True)
    d_actor = build_mappo(d_checkpoint, alg_path, args.device)
    p_actor = build_mappo(p_checkpoint, alg_path, args.device)
    source_meta = {
        "direct": checkpoint_audit(d_checkpoint, config_sha256(direct_env), config_sha256(alg)),
        "persistent": checkpoint_audit(p_checkpoint, config_sha256(persistent_env), config_sha256(alg)),
    }
    specs = {
        "direct_to_direct": (d_actor, direct_env, source_meta["direct"]),
        "direct_to_persistent": (d_actor, persistent_env, source_meta["direct"]),
        "persistent_to_direct": (p_actor, direct_env, source_meta["persistent"]),
        "persistent_to_persistent": (p_actor, persistent_env, source_meta["persistent"]),
    }
    results: dict[str, dict[str, Any]] = {}
    episode_rows: dict[str, list[dict[str, Any]]] = {}
    for cell, (actor, environment, metadata) in specs.items():
        rows = []
        for index, seed in enumerate(seeds, 1):
            rows.append(rollout(actor, environment, seed, capture=False)[0])
            if index % 5 == 0:
                print(f"[MATRIX] {cell}: {index}/{len(seeds)}", flush=True)
        aggregate = aggregate_diagnostic(rows, metadata, environment, alg)
        aggregate.update({"cell": cell, "episodes_detail": rows})
        results[cell] = aggregate
        episode_rows[cell] = rows
        write_json(output / f"{cell}.json", aggregate)
    write_json(output / "matrix_summary.json", results)
    matrix_fields = ["cell"] + list(dict.fromkeys(key for result in results.values() for key in result
                                                   if key != "episodes_detail"))
    with (output / "matrix_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=matrix_fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(results.values())

    d_pw, p_pw = episode_rows["direct_to_persistent"], episode_rows["persistent_to_persistent"]
    p_by_seed = {int(row["seed"]): row for row in p_pw}
    paired = []
    for d in d_pw:
        p = p_by_seed[int(d["seed"])]
        paired.append({
            "seed": int(d["seed"]), "d_to_pw_waves_cleared": d["waves_cleared"],
            "pw_to_pw_waves_cleared": p["waves_cleared"],
            "waves_difference_pw_minus_d": p["waves_cleared"] - d["waves_cleared"],
            "d_to_pw_success": d["waves_cleared"] == 3, "pw_to_pw_success": p["waves_cleared"] == 3,
            "d_to_pw_red_loss": d["red_losses"], "pw_to_pw_red_loss": p["red_losses"],
            "red_loss_difference_pw_minus_d": p["red_losses"] - d["red_losses"],
            "d_to_pw_boundary_exits": d["red_boundary_exits"],
            "pw_to_pw_boundary_exits": p["red_boundary_exits"],
            "boundary_difference_pw_minus_d": p["red_boundary_exits"] - d["red_boundary_exits"],
            "d_to_pw_ground_losses": d["red_ground_losses"], "pw_to_pw_ground_losses": p["red_ground_losses"],
            "d_to_pw_return": d["team_return"], "pw_to_pw_return": p["team_return"],
        })
    with (output / "persistent_transfer_paired.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    pair_selection = select_pairs(d_pw, p_pw)
    trajectory_records = []
    trajectory_dir = output / "trajectories"; trajectory_dir.mkdir(exist_ok=True)
    trajectory_jobs = () if args.skip_trajectories else (
        ("key", pair_selection["key_seed"]),
        ("counterexample", pair_selection["counterexample_seed"]),
    )
    for category, seed in trajectory_jobs:
        if seed is None:
            continue
        for label, actor in (("d999_to_pw", d_actor), ("pw999_to_pw", p_actor)):
            summary, tracks = rollout(actor, persistent_env, int(seed), capture=True)
            stem = trajectory_dir / f"seed_{seed}_{label}_3d"
            artifacts = write_trajectory_artifacts(label, summary, tracks, stem, True, render=False)
            trajectory_records.append({"category": category, "policy": label, "seed": seed,
                                       "summary": summary,
                                       "spatial_features": spatial_features(summary, tracks, float(persistent_env["arena"]["radius"])),
                                       "artifacts": artifacts})
    paired_stats = {
        "d_failed_pw_succeeded": sum(not r["d_to_pw_success"] and r["pw_to_pw_success"] for r in paired),
        "d_one_wave_pw_two_or_three": sum(r["d_to_pw_waves_cleared"] == 1 and r["pw_to_pw_waves_cleared"] >= 2 for r in paired),
        "both_succeeded": sum(r["d_to_pw_success"] and r["pw_to_pw_success"] for r in paired),
        "d_succeeded_pw_failed": sum(r["d_to_pw_success"] and not r["pw_to_pw_success"] for r in paired),
        "waves_difference_distribution": {
            str(value): sum(r["waves_difference_pw_minus_d"] == value for r in paired)
            for value in sorted({r["waves_difference_pw_minus_d"] for r in paired})},
        "waves_difference_mean": float(np.mean([r["waves_difference_pw_minus_d"] for r in paired])),
        "waves_difference_std": float(np.std([r["waves_difference_pw_minus_d"] for r in paired])),
        "red_loss_difference_mean": float(np.mean([r["red_loss_difference_pw_minus_d"] for r in paired])),
        "red_loss_difference_std": float(np.std([r["red_loss_difference_pw_minus_d"] for r in paired])),
        "boundary_difference_mean": float(np.mean([r["boundary_difference_pw_minus_d"] for r in paired])),
        "boundary_difference_std": float(np.std([r["boundary_difference_pw_minus_d"] for r in paired])),
    }
    manifest = {
        "purpose": "diagnostic paired episodes; not independent training seeds or formal holdout",
        "seed_base": seeds[0], "seed_end": seeds[-1], "episodes_per_cell": len(seeds),
        "device": args.device, "training_gamma": direct_protocol["training_gamma"],
        "algorithm_config": str(alg_path), "algorithm_config_sha256": config_sha256(alg),
        "direct_environment_config": str(direct_env_path),
        "direct_environment_config_sha256": config_sha256(direct_env),
        "persistent_environment_config": str(persistent_env_path),
        "persistent_environment_config_sha256": config_sha256(persistent_env),
        "direct_checkpoint": str(d_checkpoint), "persistent_checkpoint": str(p_checkpoint),
        "direct_protocol": direct_protocol, "persistent_protocol": persistent_protocol,
    }
    summary = {
        "manifest": manifest, "persistent_training_advantage": difference(
            results["persistent_to_persistent"], results["direct_to_persistent"]),
        "direct_ability_difference_pw_minus_d": difference(
            results["persistent_to_direct"], results["direct_to_direct"]),
        "paired_statistics": paired_stats, "representative_selection": pair_selection,
        "trajectory_records": trajectory_records,
    }
    write_json(output / "evaluation_manifest.json", manifest)
    write_json(output / "same_gamma_comparison_summary.json", summary)
    print(json.dumps(plain({"output": str(output), "paired": paired_stats,
                            "selection": pair_selection}), indent=2), flush=True)


if __name__ == "__main__":
    main()
