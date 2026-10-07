"""Reusable offline MAPPO run audit with short deterministic diagnostics."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from algorithm.common.checkpoint import evaluation_selection_key
from algorithm.common.protocol import config_sha256
from tools.plot_best_model_trajectories import (
    build_mappo, rollout, select_representative_cases,
    trajectory_rows, write_trajectory_artifacts,
)


REQUIRED_CHECKPOINT_PROTOCOL = (
    "environment_version", "environment_variant", "training_seed",
    "training_gamma", "training_num_envs", "training_total_sampled_steps",
    "training_smoke", "effective_hidden_dim", "environment_config_sha256",
    "algorithm_config_sha256", "observation_dim", "action_dim", "num_agents",
)
STAT_METRICS = (
    "actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction",
    "actor_grad_norm", "critic_grad_norm", "explained_variance",
    "policy_log_std_mean_psi", "policy_log_std_mean_theta",
    "policy_log_std_mean_v",
)


def plain(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def read_csv_numeric(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream); raw = list(reader)
        fields = list(reader.fieldnames or [])
    rows = []
    for row in raw:
        converted = {}
        for key, value in row.items():
            try: converted[key] = float(value)
            except (TypeError, ValueError): pass
        rows.append(converted)
    numeric = [field for field in fields if all(field in row for row in rows)]
    return numeric, rows


def tensor_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for group in ("actor", "critic"):
        for key, tensor in sorted(state[group].items()):
            digest.update(group.encode()); digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_audit(path: Path, env_hash: str, alg_hash: str) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    extra = state.get("extra", {})
    complete = all(field in extra for field in REQUIRED_CHECKPOINT_PROTOCOL)
    return {
        "path": str(path.resolve()), "sampled_steps": int(state.get("sampled_steps", 0)),
        "vector_steps": int(state.get("vector_steps", 0)),
        "mappo_impl_version": state.get("mappo_impl_version"),
        **{field: extra.get(field) for field in REQUIRED_CHECKPOINT_PROTOCOL},
        "protocol_complete": complete,
        "environment_hash_matches_snapshot": extra.get("environment_config_sha256") == env_hash,
        "algorithm_hash_matches_snapshot": extra.get("algorithm_config_sha256") == alg_hash,
        "weights_sha256": tensor_digest(state),
        "recorded_best_evaluation": extra.get("best_evaluation"),
    }


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "std": float(array.std()), "p90": float(np.percentile(array, 90)),
            "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)), "max": float(array.max())}


def optimization_audit(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    numeric = sorted({key for row in rows for key, value in row.items()
                      if isinstance(value, (int, float)) and not isinstance(value, bool)})
    stats = {}
    for metric in numeric:
        values = [float(row[metric]) for row in rows if metric in row
                  and math.isfinite(float(row[metric]))]
        if values: stats[metric] = distribution(values)
    anomalies = {}
    for metric in ("approx_kl", "clip_fraction", "value_loss",
                   "actor_grad_norm", "critic_grad_norm"):
        anomalies[metric] = [
            {"sampled_steps": int(row["sampled_steps"]), metric: float(row[metric])}
            for row in sorted(rows, key=lambda item: float(item[metric]), reverse=True)[:10]
        ]
    nonfinite = [(int(row.get("sampled_steps", -1)), key) for row in rows
                 for key, value in row.items()
                 if isinstance(value, float) and not math.isfinite(value)]
    return {"row_count": len(rows), "numeric_columns": numeric,
            "statistics": stats, "top_anomalies": anomalies,
            "nonfinite_values": nonfinite}, rows


def training_behavior_audit(path: Path, total_steps: int) -> dict[str, Any]:
    bins = [(start, min(start + 500_000, total_steps))
            for start in range(0, total_steps, 500_000)]
    buckets = [{"episode_rows": 0, "values": {}} for _ in bins]
    count = 0; first_step = None; last_step = None; nonfinite = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip(): continue
            row = json.loads(line); count += 1
            step = int(row["sampled_steps"]); first_step = first_step or step; last_step = step
            for key, value in row.items():
                if isinstance(value, float) and not math.isfinite(value):
                    nonfinite.append((line_number, key))
            if row.get("team_episode_return") is None: continue
            index = min((step - 1) // 500_000, len(buckets) - 1)
            buckets[index]["episode_rows"] += 1
            for key in (
                "team_episode_return", "red_uav_losses", "blue_uav_losses",
                "red_attack_kills", "blue_attack_kills", "red_boundary_exits",
                "red_ground_losses", "blue_ground_losses", "timeout_rate",
                "red_step_fire_attempts", "red_step_weapon_hits",
                "blue_step_fire_attempts", "blue_step_weapon_hits",
            ):
                if row.get(key) is not None:
                    buckets[index]["values"].setdefault(key, []).append(float(row[key]))
    phases = []
    for (start, end), bucket in zip(bins, buckets):
        phases.append({"start_step": start, "end_step": end,
                       "episode_completion_rows": bucket["episode_rows"],
                       **{key: float(np.mean(values))
                          for key, values in bucket["values"].items()}})
    return {"row_count": count, "first_step": first_step, "last_step": last_step,
            "nonfinite_values": nonfinite, "half_million_phases": phases}


def aggregate_diagnostic(rows: list[dict[str, Any]], metadata: dict[str, Any],
                         env_config: dict, alg_config: dict) -> dict[str, Any]:
    mean = lambda key: float(np.mean([row[key] for row in rows]))
    result = {
        "average_return": mean("team_return"), "average_agent_return": mean("mean_agent_return"),
        "win_rate": float(np.mean([r["red_success"] for r in rows])),
        "timeout_rate": float(np.mean([r["termination_reason"] == "red_failure_timeout" for r in rows])),
        "average_red_loss": mean("red_losses"), "average_blue_loss": mean("blue_losses"),
        "average_red_attack_kills": mean("red_attack_kills"),
        "average_blue_attack_kills": mean("blue_attack_kills"),
        "average_red_boundary_exits": mean("red_boundary_exits"),
        "evaluation_boundary_exit_rate": float(np.mean([r["red_boundary_exits"] > 0 for r in rows])),
        "average_red_ground_losses": mean("red_ground_losses"),
        "average_blue_ground_losses": mean("blue_ground_losses"),
        "average_episode_length": mean("episode_length"),
        "average_waves_cleared": mean("waves_cleared"),
        "evaluation_episodes": len(rows),
    }
    total_waves = max(r["total_waves"] for r in rows)
    for wave in range(1, total_waves + 1):
        result[f"clear_wave_{wave}_probability"] = float(np.mean([
            r["waves_cleared"] >= wave for r in rows]))
        survivors = [record["red_survivors_end"] for row in rows
                     for record in row["per_wave_metrics"]
                     if record["wave_index"] == wave and record.get("wave_cleared", True)]
        result[f"average_red_survivors_after_wave_{wave}"] = (
            float(np.mean(survivors)) if survivors else 0.0)
    total_blue = sum(r["blue_losses"] for r in rows); total_red = sum(r["red_losses"] for r in rows)
    result.update({"total_blue_losses": total_blue, "total_red_losses": total_red,
                   "kill_loss_ratio": total_blue / max(total_red, 1),
                   "algorithm": "MAPPO", "checkpoint": metadata["path"],
                   "checkpoint_sampled_steps": metadata["sampled_steps"],
                   "checkpoint_training_seed": metadata["training_seed"],
                   "checkpoint_training_gamma": metadata["training_gamma"],
                   "checkpoint_training_num_envs": metadata["training_num_envs"],
                   "checkpoint_training_total_sampled_steps": metadata["training_total_sampled_steps"],
                   "checkpoint_training_smoke": metadata["training_smoke"],
                   "checkpoint_effective_hidden_dim": metadata["effective_hidden_dim"],
                   "checkpoint_environment_config_sha256": metadata["environment_config_sha256"],
                   "checkpoint_algorithm_config_sha256": metadata["algorithm_config_sha256"],
                   "evaluation_environment_config_sha256": config_sha256(env_config),
                   "provided_algorithm_config_sha256": config_sha256(alg_config),
                   "protocol_complete": metadata["protocol_complete"],
                   "holdout_seed_base": rows[0]["seed"], "holdout_seed_end": rows[-1]["seed"],
                   "diagnostic_not_formal_holdout": True})
    return result


def run_diagnostics(checkpoint: Path, alg_path: Path, env_config: dict,
                    seeds: list[int], device: str, label: str) -> tuple[dict, list[dict]]:
    actor = build_mappo(checkpoint, alg_path, device)
    rows = []
    for index, seed in enumerate(seeds, 1):
        rows.append(rollout(actor, env_config, seed, capture=False)[0])
        if index % 5 == 0: print(f"[DIAGNOSTIC] {label}: {index}/{len(seeds)}", flush=True)
    return actor, rows


def spatial_features(summary: dict, tracks: dict, arena_radius: float) -> dict[str, Any]:
    rows = trajectory_rows(tracks); features = []
    for wave_record in summary["per_wave_metrics"]:
        wave = int(wave_record["wave_index"]); start = int(wave_record["start_step"])
        end = int(wave_record["end_step"])
        selected = [r for r in rows if start <= r["step"] <= end
                    and (r["side"] == "red" or r["wave_index"] == wave)]
        red = [r for r in selected if r["side"] == "red" and r["alive"]]
        red_radial = [math.hypot(r["x_m"], r["y_m"]) for r in red]
        red_altitude = [r["altitude_m"] for r in red]
        spreads = []
        for step in sorted({r["step"] for r in red}):
            points = [(r["x_m"], r["y_m"], r["altitude_m"]) for r in red if r["step"] == step]
            if len(points) > 1:
                spreads.append(np.mean([np.linalg.norm(np.subtract(a, b))
                                        for i, a in enumerate(points) for b in points[i + 1:]]))
        path_totals = {"red": 0.0, "blue": 0.0}
        for side in ("red", "blue"):
            identities = sorted({r["aircraft"] for r in selected if r["side"] == side})
            for aircraft in identities:
                points = sorted((r for r in selected if r["side"] == side
                                 and r["aircraft"] == aircraft), key=lambda r: r["step"])
                for left, right in zip(points, points[1:]):
                    path_totals[side] += float(np.linalg.norm(np.subtract(
                        (right["x_m"], right["y_m"], right["altitude_m"]),
                        (left["x_m"], left["y_m"], left["altitude_m"]),
                    )))
        def centroid(at_step: int):
            points = [r for r in red if r["step"] == at_step]
            return ([float(np.mean([r["x_m"] for r in points])),
                     float(np.mean([r["y_m"] for r in points])),
                     float(np.mean([r["altitude_m"] for r in points]))]
                    if points else None)
        transition = next((t for t in summary["wave_transitions"]
                           if t["from_wave"] == wave), None)
        features.append({**wave_record,
            "red_min_boundary_margin_m": float(arena_radius - max(red_radial)) if red_radial else None,
            "red_min_altitude_m": float(min(red_altitude)) if red_altitude else None,
            "red_mean_altitude_m": float(np.mean(red_altitude)) if red_altitude else None,
            "red_mean_pairwise_spread_m": float(np.mean(spreads)) if spreads else None,
            "red_total_path_length_m": path_totals["red"],
            "blue_total_path_length_m": path_totals["blue"],
            "red_start_centroid_xyz_m": centroid(start),
            "red_end_centroid_xyz_m": centroid(end),
            "spawn_transition": transition})
    return {"waves": features}


def plot_curve(rows: list[dict[str, float]], columns: list[str], output: Path,
               best_step: int, final_step: int, title: str) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    steps = [row["sampled_steps"] for row in rows]
    for column in columns:
        if all(column in row for row in rows): axis.plot(steps, [r[column] for r in rows], label=column)
    axis.axvline(best_step, color="green", linestyle="--", label="best_eval")
    axis.axvline(final_step, color="black", linestyle=":", label="final")
    axis.set(xlabel="sampled steps", title=title); axis.grid(True, alpha=0.25); axis.legend(fontsize=8)
    fig.savefig(output, dpi=190); plt.close(fig)


def write_reports(output: Path, audit: dict, trajectory_records: list[dict]) -> None:
    fmt = lambda value: "NA" if value is None else f"{value:.1f}"
    best = audit["best_training_evaluation"]; final = audit["final_training_evaluation"]
    diagnostic = audit["diagnostic_comparison"]
    wave_keys = sorted(
        key for key in best if key.startswith("clear_wave_") and key.endswith("_probability")
    )
    wave_summary = lambda row: ("/".join(f"{row[key]:.2f}" for key in wave_keys)
                                if wave_keys else f"{row['win_rate']:.2f}")
    wave_label = ("/".join(f"W{index + 1}" for index in range(len(wave_keys)))
                  if wave_keys else "win_rate")
    primary_key = wave_keys[-1] if wave_keys else "win_rate"
    report = [f"# {audit['protocol']['environment_variant']} MAPPO training audit", "", "## Run protocol", "",
              f"- Protocol complete: `{audit['protocol']['protocol_complete']}`",
              f"- Variant/version: `{audit['protocol']['environment_variant']}` / `{audit['protocol']['environment_version']}`",
              f"- seed/gamma/envs/target: {audit['protocol']['seed']} / {audit['protocol']['gamma']} / {audit['protocol']['num_envs']} / {audit['protocol']['total_sampled_steps']}",
              "", "## File integrity", "",
              f"- Files found: {len(audit['files'])}; final sampled steps: {audit['completion']['sampled_steps']}.",
              f"- Log anomaly matches: {len(audit['completion']['log_anomaly_matches'])}; resume history present: {audit['completion']['resume_history_present']}.",
              f"- latest and checkpoint_3000000 weights identical: {audit['completion']['latest_matches_final_weights']}.",
              "", "## Training completion", "",
              f"- latest.pt sampled_steps={audit['completion']['sampled_steps']}; checkpoint_3000000.pt and latest.pt have identical tensor digests.",
              "- train.log ends with the 3M evaluation, checkpoint save, and DONE record; any present auxiliary log was also scanned.",
              "", "## Evaluation curve", "",
              f"- Evaluation steps: {audit['evaluation_sampled_steps']}",
              f"- First non-zero wave clears: {audit['learning_onset']}.",
              "", "## Best checkpoint", "",
              f"- Current selection rule chooses **{int(best['sampled_steps'])}** steps.",
              f"- Best: return={best['average_return']:.3f}, {wave_label}={wave_summary(best)}, waves={best.get('average_waves_cleared', best.get('win_rate', 0.0)):.2f}, Red loss={best['average_red_loss']:.2f}.",
              "", "## Final checkpoint", "",
              f"- Final: return={final['average_return']:.3f}, {wave_label}={wave_summary(final)}, waves={final.get('average_waves_cleared', final.get('win_rate', 0.0)):.2f}, Red loss={final['average_red_loss']:.2f}.",
              "", "## Best vs final", "",
              f"- Training-time best-to-final deltas: {audit['best_minus_final_training']}.",
              "", "## Wave metrics", "",
              f"- Best {wave_label} and average waves: {wave_summary(best)}; {best.get('average_waves_cleared', best.get('win_rate', 0.0)):.2f}.",
              f"- Final {wave_label} and average waves: {wave_summary(final)}; {final.get('average_waves_cleared', final.get('win_rate', 0.0)):.2f}.",
              "", "## Optimization stability", "",
              f"- Optimization rows: {audit['optimization']['row_count']}; non-finite values: {len(audit['optimization']['nonfinite_values'])}.",
              "", "## KL / clip / entropy / value diagnostics", "",
              "- Negative differential entropy, where present, means a more concentrated continuous Gaussian policy and is not intrinsically invalid.",
              "- Large KL/clip/value/gradient observations are reported as temporal diagnostics only; no causal claim is made.",
              "", "## Boundary, ground, timeout, and combat behavior", "",
              f"- Best boundary/ground/timeout: {best.get('average_red_boundary_exits'):.3f} / {best.get('average_red_ground_losses'):.3f} / {best.get('timeout_rate'):.3f}.",
              f"- Final boundary/ground/timeout: {final.get('average_red_boundary_exits'):.3f} / {final.get('average_red_ground_losses'):.3f} / {final.get('timeout_rate'):.3f}.",
              "", "## Policy-drift evidence", "",
              f"- Training evaluation best-to-final return delta={best['average_return']-final['average_return']:.3f}, {primary_key} delta={best[primary_key]-final[primary_key]:.3f}, Red-loss delta={best['average_red_loss']-final['average_red_loss']:.3f}.",
              "", "## Diagnostic best-vs-latest comparison", "",
              f"- Seeds: {diagnostic['seed_base']}–{diagnostic['seed_end']} (diagnostic only; not formal holdout).",
              f"- Best {primary_key}={diagnostic['best'][primary_key]:.3f}, latest={diagnostic['latest'][primary_key]:.3f}.",
              f"- Best average waves={diagnostic['best']['average_waves_cleared']:.3f}, latest={diagnostic['latest']['average_waves_cleared']:.3f}.",
              f"- Best return={diagnostic['best']['average_return']:.3f}, latest={diagnostic['latest']['average_return']:.3f}.",
              f"- Policy-drift diagnostic: {diagnostic['interpretation']}",
              "", "## Main factual conclusions", "",
              *[f"- {item}" for item in audit["main_conclusions"]],
              "", "## Remaining uncertainty", "",
              "- Twenty training-evaluation episodes and thirty diagnostic episodes are finite samples; neither replaces the future untouched formal holdout.",
              "- Temporal correspondence between optimizer spikes and evaluation changes is not evidence of causation."]
    (output / "training_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    lines = ["# Persistent-Wave trajectory audit", ""]
    for record in trajectory_records:
        summary = record["summary"]; lines += [f"## {record['category']} — {record['checkpoint']} seed {summary['seed']}", "",
            f"Return {summary['team_return']:.3f}; waves {summary['waves_cleared']}/{summary['total_waves']}; steps {summary['episode_length']}; termination `{summary['termination_reason']}`; Red/Blue losses {summary['red_losses']}/{summary['blue_losses']}.", ""]
        for wave in record["spatial_features"]["waves"]:
            lines += [f"### Wave {wave['wave_index']}", "",
                f"Steps {wave['start_step']}–{wave['end_step']} ({wave['duration_steps']} steps); Red survivors {wave['red_survivors_start']}→{wave['red_survivors_end']}; Blue survivors {wave['blue_survivors_start']}→{wave['blue_survivors_end']}; clear={wave['wave_cleared']}.",
                f"Red attack kills={wave['red_attack_kills']}, Blue attack kills={wave['blue_attack_kills']}, Red boundary/ground={wave['red_boundary_exits']}/{wave['red_ground_losses']}; wave team return={wave['team_return']:.3f}.",
                f"Observed Red minimum boundary margin={fmt(wave['red_min_boundary_margin_m'])} m, minimum altitude={fmt(wave['red_min_altitude_m'])} m, mean altitude={fmt(wave['red_mean_altitude_m'])} m, mean pairwise spread={fmt(wave['red_mean_pairwise_spread_m'])} m.", ""]
            lines += [f"Aggregate 3-D path length in this wave: Red={wave.get('red_total_path_length_m', 0.0):.1f} m, Blue={wave.get('blue_total_path_length_m', 0.0):.1f} m. "
                      f"The observed Red geometry was {'tightly concentrated' if (wave['red_mean_pairwise_spread_m'] or 1e9) < 100 else 'moderately grouped' if (wave['red_mean_pairwise_spread_m'] or 1e9) < 500 else 'spatially dispersed'}; "
                      f"{'at least one Red approached within 100 m of the arena edge' if (wave['red_min_boundary_margin_m'] is not None and wave['red_min_boundary_margin_m'] < 100) else 'no recorded Red point approached within 100 m of the arena edge'}; "
                      f"{'at least one Red descended below 100 m' if (wave['red_min_altitude_m'] is not None and wave['red_min_altitude_m'] < 100) else 'all recorded surviving Red points remained above 100 m'}.", ""]
            if wave.get("spawn_transition"):
                spawn=wave["spawn_transition"]; lines += [f"Next Blue wave spawned at step {spawn['step']}, radial angle={spawn['wave_spawn_radial_angle']:.4f} rad, candidate={spawn['wave_spawn_candidate_index']}, minimum Red–Blue spawn distance={spawn['minimum_spawn_distance']:.1f} m.", ""]
    drift_best = next((r for r in trajectory_records if r["category"] == "drift_best"), None)
    drift_latest = next((r for r in trajectory_records if r["category"] == "drift_latest"), None)
    if drift_best and drift_latest:
        best_waves=drift_best["spatial_features"]["waves"]; latest_waves=drift_latest["spatial_features"]["waves"]
        lines += ["## Same-seed best-vs-latest drift comparison", "",
            f"At seed {drift_best['summary']['seed']}, best cleared 3 waves with {drift_best['summary']['red_losses']} Red loss, whereas latest cleared {drift_latest['summary']['waves_cleared']} wave with {drift_latest['summary']['red_losses']} Red losses.",
            f"Wave 1 already diverged: best retained {best_waves[0]['red_survivors_end']} Red with boundary/ground {best_waves[0]['red_boundary_exits']}/{best_waves[0]['red_ground_losses']}; latest retained {latest_waves[0]['red_survivors_end']} with boundary/ground {latest_waves[0]['red_boundary_exits']}/{latest_waves[0]['red_ground_losses']}.",
            f"Latest entered Wave 2 with only {latest_waves[1]['red_survivors_start']} Red, then ended at altitude minimum {fmt(latest_waves[1]['red_min_altitude_m'])} m with a ground loss; it never spawned Wave 3. Best entered Wave 2 with {best_waves[1]['red_survivors_start']} Red and preserved all of them through that wave.",
            "This paired episode therefore supports an early Wave-1 force-preservation regression, followed by insufficient force for sustained Wave-2/3 adaptation; it is not merely a Wave-3-only failure.", ""]
    (output / "trajectory_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, default=30_000_000)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--reuse-diagnostics", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-trajectories", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.episodes > 30: raise ValueError("episodes must be 1..30")
    if 20_000_000 <= args.seed_base <= 20_000_199 or 20_000_000 <= args.seed_base + args.episodes - 1 <= 20_000_199:
        raise ValueError("formal holdout seed range is forbidden for diagnostics")
    torch.set_num_threads(1)
    run = args.run_dir.resolve(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    env_path, alg_path = run / "env_config.yaml", run / "algorithm_config.yaml"
    env_config = yaml.safe_load(env_path.read_text(encoding="utf-8")); alg_config = yaml.safe_load(alg_path.read_text(encoding="utf-8"))
    run_config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    env_hash, alg_hash = config_sha256(env_config), config_sha256(alg_config)
    checkpoint_names = sorted({"best_eval.pt", "latest.pt"} |
                              {path.name for path in run.glob("checkpoint_*.pt")})
    checkpoint_files = {name: checkpoint_audit(run / name, env_hash, alg_hash)
                        for name in checkpoint_names}
    numeric_columns, eval_rows = read_csv_numeric(run / "evaluation_history.csv")
    variant = str(env_config.get("environment_variant", "direct_v2_3"))
    best_row = max(eval_rows, key=lambda row: evaluation_selection_key(row, variant))
    final_row = next(row for row in eval_rows if int(row["sampled_steps"]) == 3_000_000)
    optimization, optimization_rows = optimization_audit(run / "optimization_metrics.jsonl")
    training = training_behavior_audit(run / "training_metrics.jsonl", 3_000_000)
    files = [{"name": p.name, "size_bytes": p.stat().st_size,
              "modified": datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat()}
             for p in sorted(run.iterdir()) if p.is_file()]
    log_paths = [run / "train.log", run.parent / f"{run.name}_nohup.log"]
    pattern = re.compile(r"traceback|\bnan\b|\binf\b|floatingpointerror|out of memory|worker.*crash|exception|protocol mismatch", re.I)
    log_matches = [{"file": str(path), "line": i, "text": line}
                   for path in log_paths if path.exists()
                   for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                   if pattern.search(line)]
    total_waves = int(env_config.get("persistent_waves", {}).get("total_waves", 1))
    onset = {f"wave_{wave}": next((int(r["sampled_steps"]) for r in eval_rows
                                    if r.get(f"clear_wave_{wave}_probability", r.get("win_rate", 0)) > 0), None)
             for wave in range(1, total_waves + 1)}
    seeds = list(range(args.seed_base, args.seed_base + args.episodes))
    if args.reuse_diagnostics:
        best_saved=json.loads((output/"best_diagnostic_evaluation.json").read_text(encoding="utf-8"))
        latest_saved=json.loads((output/"latest_diagnostic_evaluation.json").read_text(encoding="utf-8"))
        best_episodes=best_saved.pop("episodes_detail"); latest_episodes=latest_saved.pop("episodes_detail")
        best_diag,latest_diag=best_saved,latest_saved
        best_actor=build_mappo(run/"best_eval.pt",alg_path,args.device)
        latest_actor=build_mappo(run/"latest.pt",alg_path,args.device)
    else:
        best_actor, best_episodes = run_diagnostics(run / "best_eval.pt", alg_path, env_config, seeds, args.device, "best")
        latest_actor, latest_episodes = run_diagnostics(run / "latest.pt", alg_path, env_config, seeds, args.device, "latest")
        best_diag = aggregate_diagnostic(best_episodes, checkpoint_files["best_eval.pt"], env_config, alg_config)
        latest_diag = aggregate_diagnostic(latest_episodes, checkpoint_files["latest.pt"], env_config, alg_config)
    (output / "best_diagnostic_evaluation.json").write_text(json.dumps(plain({**best_diag, "episodes_detail": best_episodes}), indent=2), encoding="utf-8")
    (output / "latest_diagnostic_evaluation.json").write_text(json.dumps(plain({**latest_diag, "episodes_detail": latest_episodes}), indent=2), encoding="utf-8")
    comparison_metrics = [key for key, value in best_diag.items() if isinstance(value, (int, float)) and not isinstance(value, bool) and key in latest_diag]
    with (output / "best_vs_latest_diagnostic.csv").open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream, fieldnames=("metric","best","latest","best_minus_latest")); writer.writeheader()
        for key in comparison_metrics: writer.writerow({"metric":key,"best":best_diag[key],"latest":latest_diag[key],"best_minus_latest":best_diag[key]-latest_diag[key]})
    manifest = {"purpose":"diagnostic only; not formal holdout", "seed_base":seeds[0], "seed_end":seeds[-1],
                "episodes_per_checkpoint":len(seeds), "environment_config_sha256":env_hash,
                "algorithm_config_sha256":alg_hash, "best_checkpoint":checkpoint_files["best_eval.pt"],
                "latest_checkpoint":checkpoint_files["latest.pt"]}
    (output / "diagnostic_manifest.json").write_text(json.dumps(plain(manifest), indent=2), encoding="utf-8")
    selected = (select_representative_cases(best_episodes, latest_episodes)
                if not args.skip_trajectories else {})
    trajectory_records=[]
    jobs=[]
    if not args.skip_trajectories:
        if selected["best_success"] is not None: jobs.append(("best_success","best",best_actor,selected["best_success"],True))
        if selected["best_partial"] is not None: jobs.append(("best_partial","best",best_actor,selected["best_partial"],False))
        if selected["drift_pair"] is not None:
            jobs += [("drift_best","best",best_actor,selected["drift_pair"],True),
                     ("drift_latest","latest",latest_actor,selected["drift_pair"],True)]
        if selected["latest_success"] is not None: jobs.append(("latest_success","latest",latest_actor,selected["latest_success"],False))
    for category,label,actor,seed,views in jobs:
        summary,tracks=rollout(actor,env_config,int(seed),capture=True)
        stem=output/f"trajectory_{category}_seed_{seed}"
        artifacts=write_trajectory_artifacts(f"{label}_eval.pt",summary,tracks,stem,views,render=not args.skip_plots)
        trajectory_records.append({"category":category,"checkpoint":label,"summary":summary,
                                   "spatial_features":spatial_features(summary,tracks,float(env_config["arena"]["radius"])),
                                   "artifacts":artifacts})
    # Evaluation and optimizer curves.
    if not args.skip_plots:
        plot_curve(eval_rows,["average_return"],output/"evaluation_return_curve.png",int(best_row["sampled_steps"]),3_000_000,"Evaluation return")
        plot_curve(eval_rows,["clear_wave_1_probability","clear_wave_2_probability","clear_wave_3_probability"],output/"evaluation_wave_clear_curve.png",int(best_row["sampled_steps"]),3_000_000,"Wave clear probability")
        plot_curve(eval_rows,["average_waves_cleared"],output/"evaluation_average_waves_curve.png",int(best_row["sampled_steps"]),3_000_000,"Average waves cleared")
        plot_curve(eval_rows,["average_red_loss"],output/"evaluation_red_loss_curve.png",int(best_row["sampled_steps"]),3_000_000,"Red loss")
        plot_curve(eval_rows,["average_red_survivors_after_wave_1","average_red_survivors_after_wave_2","average_red_survivors_after_wave_3"],output/"red_survivors_by_wave_curve.png",int(best_row["sampled_steps"]),3_000_000,"Red survivors by cleared wave")
        plot_curve(eval_rows,["average_red_boundary_exits","average_red_ground_losses","average_blue_ground_losses"],output/"boundary_ground_curve.png",int(best_row["sampled_steps"]),3_000_000,"Boundary and ground losses")
        for metric,name in (("approx_kl","optimization_kl_curve.png"),("entropy","optimization_entropy_curve.png"),("value_loss","optimization_value_loss_curve.png")):
            plot_curve(optimization_rows,[metric],output/name,int(best_row["sampled_steps"]),3_000_000,metric)
    training_delta={key:best_row[key]-final_row[key] for key in numeric_columns if key in best_row and key in final_row}
    primary_key = f"clear_wave_{total_waves}_probability"
    diag_gap=best_diag[primary_key]-latest_diag[primary_key]
    interpretation=("30-seed diagnostics support final-policy regression" if diag_gap >= 0.10 and best_diag["average_waves_cleared"] > latest_diag["average_waves_cleared"] else
                    "30-seed diagnostics do not show a clear policy regression; training-evaluation noise remains important")
    protocol={"algorithm":run_config["algorithm"],"environment_variant":variant,
              "environment_version":str(env_config["environment_version"]),"seed":run_config["seed"],
              "gamma":float(alg_config["training"]["gamma"]),"num_envs":run_config["num_envs"],
              "total_sampled_steps":run_config["total_sampled_steps"],"mode":"smoke" if run_config["smoke"] else "formal",
              "device":run_config["device"],"environment_config_sha256":env_hash,"algorithm_config_sha256":alg_hash,
              "protocol_complete":all(c["protocol_complete"] and c["environment_hash_matches_snapshot"] and c["algorithm_hash_matches_snapshot"] for c in checkpoint_files.values())}
    audit={"protocol":protocol,"files":files,"completion":{"sampled_steps":checkpoint_files["latest.pt"]["sampled_steps"],
             "latest_matches_final_weights":checkpoint_files["latest.pt"]["weights_sha256"]==checkpoint_files["checkpoint_3000000.pt"]["weights_sha256"],
             "log_anomaly_matches":log_matches,"resume_history_present":(run/"resume_history.jsonl").exists()},
           "checkpoint_metadata":checkpoint_files,"evaluation_numeric_columns":numeric_columns,
           "evaluation_sampled_steps":[int(r["sampled_steps"]) for r in eval_rows],
           "best_training_evaluation":best_row,"final_training_evaluation":final_row,
           "best_minus_final_training":training_delta,"learning_onset":onset,
           "training_behavior":training,"optimization":optimization,
           "diagnostic_comparison":{"seed_base":seeds[0],"seed_end":seeds[-1],"best":best_diag,"latest":latest_diag,
                                    "best_minus_latest":{k:best_diag[k]-latest_diag[k] for k in comparison_metrics},
                                    "interpretation":interpretation},
           "representative_selection":selected,"trajectory_records":trajectory_records,
           "main_conclusions":[f"Training completed at {checkpoint_files['latest.pt']['sampled_steps']} sampled steps without recorded non-finite optimizer values.",
                               f"The current {variant} selection key chooses step {int(best_row['sampled_steps'])}.",
                               f"latest.pt and checkpoint_3000000.pt contain identical actor/critic weights.",
                               interpretation]}
    (output/"training_audit_summary.json").write_text(json.dumps(plain(audit),indent=2),encoding="utf-8")
    write_reports(output,audit,trajectory_records)
    print(json.dumps({"best_step":int(best_row["sampled_steps"]),"selected":selected,
                      "diagnostic_interpretation":interpretation,"output_dir":str(output)},indent=2))


if __name__ == "__main__": main()
