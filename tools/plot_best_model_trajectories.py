"""Capture and plot deterministic Direct or Persistent-Wave MAPPO trajectories."""
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from algorithm.mappo.factory import build_mappo_trainer
from algorithm.mappo.trainer import MAPPOTrainer
from env.factory import make_combat_environment


ROOT = PROJECT_ROOT
POINT_FIELDS = (
    "step", "wave_index", "side", "aircraft", "x_m", "y_m",
    "altitude_m", "alive", "speed_mps", "theta_rad", "psi_rad", "event",
)


def resolved(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def build_mappo(checkpoint: Path, algorithm_config: Path, device: str) -> MAPPOTrainer:
    cfg = yaml.safe_load(algorithm_config.read_text(encoding="utf-8"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    hidden = state.get("extra", {}).get("effective_hidden_dim")
    trainer = build_mappo_trainer(cfg, device, hidden_dim=hidden)
    trainer.load(checkpoint, allow_legacy_diagnostic=True)
    trainer.actor.eval(); trainer.critic.eval()
    return trainer


def _point(step: int, wave: int, side: str, aircraft: int, state,
           event: str = "") -> dict[str, Any]:
    return {
        "step": int(step), "wave_index": int(wave), "side": side,
        "aircraft": int(aircraft + 1), "x_m": float(state.x),
        "y_m": float(state.y), "altitude_m": float(state.altitude),
        "alive": bool(state.alive), "speed_mps": float(state.v),
        "theta_rad": float(state.theta), "psi_rad": float(state.psi),
        "event": event,
    }


def _track_key(side: str, wave: int, aircraft: int) -> tuple[str, int, int]:
    return side, 0 if side == "red" else int(wave), int(aircraft + 1)


def rollout(actor: MAPPOTrainer, environment_config: dict, seed: int,
            capture: bool = False) -> tuple[dict[str, Any], dict]:
    """Run one deterministic episode and optionally retain wave-safe tracks."""
    env = make_combat_environment(environment_config)
    observation, reset_info = env.reset(seed)
    returns = np.zeros(4, dtype=np.float64)
    tracks: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    transitions: list[dict[str, Any]] = []

    def append_state(step: int, wave: int, side: str, aircraft: int, state,
                     event: str = "") -> None:
        if capture:
            tracks.setdefault(_track_key(side, wave, aircraft), []).append(
                _point(step, wave, side, aircraft, state, event)
            )

    initial_wave = int(getattr(env, "wave_index", 1))
    for side, states in (("red", env.red), ("blue", env.blue)):
        for index, state in enumerate(states):
            append_state(0, initial_wave, side, index, state, "start")

    while True:
        wave_before = int(getattr(env, "wave_index", 1))
        red_before_alive = [state.alive for state in env.red]
        blue_before_alive = [state.alive for state in env.blue]
        old_blue = list(env.blue)
        actions = actor.act(observation, env.red_alive_mask, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(actions)
        returns += reward
        wave_after = int(info.get("wave_index", wave_before))
        spawned = bool(info.get("spawned_next_wave", False))
        for index, state in enumerate(env.red):
            if red_before_alive[index] or state.alive:
                event = "lost" if red_before_alive[index] and not state.alive else ""
                append_state(env.steps, wave_after, "red", index, state, event)
        blue_after_step = old_blue if spawned else env.blue
        for index, state in enumerate(blue_after_step):
            if blue_before_alive[index] or state.alive:
                event = "lost" if blue_before_alive[index] and not state.alive else ""
                append_state(env.steps, wave_before, "blue", index, state, event)
        if spawned:
            transitions.append({
                "from_wave": wave_before, "to_wave": wave_after,
                "step": int(env.steps), "spawned_next_wave": True,
                "wave_spawn_radial_angle": info.get("wave_spawn_radial_angle"),
                "wave_spawn_candidate_index": info.get("wave_spawn_candidate_index"),
                "minimum_spawn_distance": info.get("minimum_spawn_distance"),
            })
            for index, state in enumerate(env.blue):
                append_state(env.steps, wave_after, "blue", index, state, "spawn")
        if terminated or truncated:
            break

    summary = {
        "seed": int(seed), "team_return": float(returns.sum()),
        "mean_agent_return": float(returns.mean()),
        "termination_reason": info["termination_reason"],
        "episode_length": int(info["episode_length"]),
        "red_success": bool(info.get("red_success", False)),
        "waves_cleared": int(info.get("waves_cleared", int(info.get("red_success", 0)))),
        "total_waves": int(info.get("total_waves", 1)),
        **{key: int(info[key]) for key in (
            "red_losses", "blue_losses", "red_attack_kills", "blue_attack_kills",
            "red_boundary_exits", "blue_boundary_exits", "red_ground_losses",
            "blue_ground_losses",
        )},
        **{f"episode_{name}_total": float(info[f"episode_{name}_total"])
           for name in ("r1", "r2", "r3", "r4")},
        "per_wave_metrics": info.get("per_wave_metrics", []),
        "wave_transitions": transitions,
        "environment_variant": info.get(
            "environment_variant", reset_info.get("environment_variant", "direct_v2_3")
        ),
    }
    return summary, tracks


def representative_seed(actor: MAPPOTrainer, environment: dict, seed_base: int,
                        episodes: int) -> int:
    rows = [rollout(actor, environment, seed)[0]
            for seed in range(seed_base, seed_base + episodes)]
    total_waves = max(row["total_waves"] for row in rows)
    successful = [row for row in rows if row["waves_cleared"] >= total_waves]
    candidates = successful or rows
    median_return = float(np.median([row["team_return"] for row in candidates]))
    return min(candidates, key=lambda row: abs(row["team_return"] - median_return))["seed"]


def select_representative_cases(
    best_rows: list[dict[str, Any]], latest_rows: list[dict[str, Any]]
) -> dict[str, int | None]:
    """Select mission-aware representative seeds without single-round win logic."""
    if not best_rows or not latest_rows:
        raise ValueError("best_rows and latest_rows must not be empty")
    total_waves = max(row["total_waves"] for row in best_rows + latest_rows)

    def median_case(rows: list[dict[str, Any]]) -> int | None:
        if not rows:
            return None
        median = float(np.median([row["team_return"] for row in rows]))
        return int(min(rows, key=lambda row: abs(row["team_return"] - median))["seed"])

    best_successes = [r for r in best_rows if r["waves_cleared"] >= total_waves]
    best_partial = [r for r in best_rows if r["waves_cleared"] == total_waves - 1]
    if not best_partial:
        maximum = max((r["waves_cleared"] for r in best_rows
                       if r["waves_cleared"] < total_waves), default=-1)
        best_partial = [r for r in best_rows if r["waves_cleared"] == maximum]
    latest_by_seed = {int(r["seed"]): r for r in latest_rows}
    drift = [(row, latest_by_seed[int(row["seed"])]) for row in best_successes
             if int(row["seed"]) in latest_by_seed
             and latest_by_seed[int(row["seed"])]["waves_cleared"] < total_waves]
    drift_seed = None
    if drift:
        gaps = [left["team_return"] - right["team_return"] for left, right in drift]
        median_gap = float(np.median(gaps))
        drift_seed = int(min(drift, key=lambda pair: abs(
            (pair[0]["team_return"] - pair[1]["team_return"]) - median_gap
        ))[0]["seed"])
    latest_successes = [r for r in latest_rows if r["waves_cleared"] >= total_waves]
    return {
        "best_success": median_case(best_successes),
        "best_partial": median_case(best_partial),
        "drift_pair": drift_seed,
        "latest_success": median_case(latest_successes),
    }


def trajectory_rows(tracks: dict) -> list[dict[str, Any]]:
    rows = [point for points in tracks.values() for point in points]
    return sorted(rows, key=lambda row: (
        row["step"], row["side"], row["wave_index"], row["aircraft"]
    ))


def plot_tracks(seed: int, summary: dict, tracks: dict, output: Path,
                checkpoint_label: str = "MAPPO checkpoint") -> None:
    figure = plt.figure(figsize=(12, 9), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    red_colors = ["#7f0000", "#b2182b", "#d6604d", "#f4a582"]
    blue_colors = ["#053061", "#2166ac", "#4393c3", "#92c5de"]
    all_points = []
    for (side, wave, aircraft), points in sorted(tracks.items()):
        if not points: continue
        xyz = np.asarray([[p["x_m"], p["y_m"], p["altitude_m"]]
                          for p in points]) / 1000.0
        all_points.append(xyz)
        color = (red_colors if side == "red" else blue_colors)[aircraft - 1]
        label = f"Red {aircraft}" if side == "red" else f"Blue W{wave}-{aircraft}"
        axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                  "-" if side == "red" else "--", color=color,
                  linewidth=1.8 if side == "red" else 1.2,
                  alpha=0.9 if side == "red" else 0.65, label=label)
        axis.scatter(*xyz[0], color=color,
                     marker="D" if points[0]["event"] == "spawn" else "o", s=35)
        axis.scatter(*xyz[-1], color=color,
                     marker="X" if not points[-1]["alive"] else "^", s=55)
    angle = np.linspace(0.0, 2.0 * np.pi, 240)
    axis.plot(5.0 * np.cos(angle), 5.0 * np.sin(angle), np.zeros_like(angle),
              color="#555555", linewidth=1.0, alpha=0.7, label="5 km arena")
    if all_points:
        points = np.concatenate(all_points)
        axis.set_box_aspect((max(np.ptp(points[:, 0]), 1.0),
                             max(np.ptp(points[:, 1]), 1.0),
                             max(np.ptp(points[:, 2]), 1.0) * 1.4))
    axis.set(xlabel="x (km)", ylabel="y (km)", zlabel="altitude (km)")
    axis.view_init(elev=25, azim=-56)
    axis.set_title(
        f"{checkpoint_label} | seed={seed} | waves={summary['waves_cleared']}/"
        f"{summary['total_waves']} | return={summary['team_return']:.2f}\n"
        f"Red loss={summary['red_losses']} | Blue loss={summary['blue_losses']} | "
        f"{summary['termination_reason']} | steps={summary['episode_length']}"
    )
    axis.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), ncol=2, fontsize=7)
    axis.grid(True, alpha=0.25); figure.savefig(output, dpi=220); plt.close(figure)


def plot_diagnostic_views(summary: dict, tracks: dict, output_prefix: Path) -> list[Path]:
    outputs = []
    transitions = [int(item["step"]) for item in summary["wave_transitions"]]
    for kind in ("topdown", "altitude", "alive"):
        fig, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
        if kind in {"topdown", "altitude"}:
            for (side, wave, aircraft), points in sorted(tracks.items()):
                if not points: continue
                color = "#b2182b" if side == "red" else "#2166ac"
                label = f"R{aircraft}" if side == "red" else f"B W{wave}-{aircraft}"
                if kind == "topdown":
                    axis.plot([p["x_m"] / 1000 for p in points],
                              [p["y_m"] / 1000 for p in points],
                              color=color, alpha=0.65, linewidth=1.0, label=label)
                else:
                    axis.plot([p["step"] for p in points],
                              [p["altitude_m"] / 1000 for p in points],
                              color=color, alpha=0.65, linewidth=1.0, label=label)
            if kind == "topdown":
                angle = np.linspace(0, 2 * np.pi, 240)
                axis.plot(5 * np.cos(angle), 5 * np.sin(angle), color="black")
                axis.set_aspect("equal"); axis.set(xlabel="x (km)", ylabel="y (km)")
            else:
                axis.set(xlabel="step", ylabel="altitude (km)")
        else:
            steps = np.arange(int(summary["episode_length"]) + 1)
            red_alive, blue_alive = [], []
            rows = trajectory_rows(tracks)
            latest = {}
            cursor = 0
            for step in steps:
                while cursor < len(rows) and rows[cursor]["step"] <= step:
                    row = rows[cursor]
                    key = (row["side"], row["wave_index"] if row["side"] == "blue" else 0,
                           row["aircraft"])
                    latest[key] = row; cursor += 1
                active_wave = 1 + sum(step >= transition for transition in transitions)
                red_alive.append(sum(r["alive"] for k, r in latest.items() if k[0] == "red"))
                blue_alive.append(sum(r["alive"] for k, r in latest.items()
                                      if k[0] == "blue" and k[1] == active_wave))
            axis.step(steps, red_alive, where="post", color="#b2182b", label="Red alive")
            axis.step(steps, blue_alive, where="post", color="#2166ac", label="Active Blue alive")
            axis.set(xlabel="step", ylabel="alive aircraft", ylim=(-0.1, 4.3))
        for step in transitions:
            if kind != "topdown": axis.axvline(step, color="grey", linestyle=":")
        axis.set_title(f"seed={summary['seed']} | {kind}")
        axis.legend(fontsize=7, ncol=2); axis.grid(True, alpha=0.25)
        path = output_prefix.with_name(f"{output_prefix.name}_{kind}.png")
        fig.savefig(path, dpi=190); plt.close(fig); outputs.append(path)
    return outputs


def write_trajectory_artifacts(checkpoint_label: str, summary: dict, tracks: dict,
                               stem: Path, extra_views: bool = False,
                               render: bool = True) -> dict[str, Any]:
    png, csv_path, json_path = stem.with_suffix(".png"), stem.with_suffix(".csv"), stem.with_suffix(".json")
    if render:
        plot_tracks(summary["seed"], summary, tracks, png, checkpoint_label)
    rows = trajectory_rows(tracks)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=POINT_FIELDS)
        writer.writeheader(); writer.writerows(rows)
    json_path.write_text(json.dumps({"checkpoint": checkpoint_label, "summary": summary,
                                     "trajectory_point_count": len(rows)}, indent=2), encoding="utf-8")
    views = plot_diagnostic_views(summary, tracks, stem) if extra_views and render else []
    return {"png": str(png) if render else None, "csv": str(csv_path), "json": str(json_path),
            "diagnostic_views": [str(path) for path in views]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--algorithm-config", default="configs/mappo.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed-base", type=int, default=10_000_000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--selected-seed", type=int)
    parser.add_argument("--env-config", default="configs/combat_environment.yaml")
    args = parser.parse_args()
    environment = yaml.safe_load(resolved(args.env_config).read_text(encoding="utf-8"))
    checkpoint = resolved(args.checkpoint)
    actor = build_mappo(checkpoint, resolved(args.algorithm_config), args.device)
    seed = int(args.selected_seed) if args.selected_seed is not None else representative_seed(
        actor, environment, args.seed_base, args.episodes)
    summary, tracks = rollout(actor, environment, seed, capture=True)
    output_dir = resolved(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"mappo_checkpoint_3d_trajectory_seed_{seed}"
    artifacts = write_trajectory_artifacts(checkpoint.name, summary, tracks, stem, True)
    report = {"checkpoint": str(checkpoint), "summary": summary, **artifacts}
    (output_dir / "best_model_trajectory_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
