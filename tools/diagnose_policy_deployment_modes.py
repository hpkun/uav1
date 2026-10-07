"""Development-only diagnostic for squashed-Gaussian deployment modes.

The tool reads self-describing modular MAPPO checkpoints and runs environment
episodes without modifying training or evaluator semantics. Its outputs are
diagnostic development evidence and must never be treated as formal holdout
results.
"""
from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import numpy as np
import torch
from torch.distributions import Normal
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.evaluator import episode_return_metrics
from algorithm.modular_mappo.evaluation import per_wave_episode_diagnostics
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import (
    is_formal_v2_checkpoint,
    validate_modular_checkpoint,
)
from env.factory import make_combat_environment


ROLE = "development_mechanism_diagnostic_only"
NOT_FORMAL_EVIDENCE = True
DEFAULT_SEED_BASE = 31_000_000
POLICY_NOISE_BASE = 720_000_000
FORBIDDEN_SEED_RANGES = (
    (29_000_000, 29_000_019, "monitoring"),
    (30_000_000, 30_000_199, "formal_holdout"),
)
DEPLOYMENT_MODES = (
    "mean", "squashed_expectation", "noise_025", "noise_050", "noise_100",
)
NOISE_SCALES = {
    "mean": 0.0,
    "squashed_expectation": 0.0,
    "noise_025": 0.25,
    "noise_050": 0.50,
    "noise_100": 1.00,
}
NOISE_MODE_INDICES = {"noise_025": 1, "noise_050": 2, "noise_100": 3}
SUMMARY_METRICS = (
    "average_waves", "W1", "W2", "W3", "return", "red_loss", "blue_loss",
    "K_L", "ground", "boundary", "timeout", "episode_length",
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=3)
def gauss_hermite_nodes(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute physicists' Gauss-Hermite nodes and normalized weights."""
    if nodes not in (16, 32, 64):
        raise ValueError("quadrature nodes must be one of 16, 32, 64")
    points, weights = np.polynomial.hermite.hermgauss(nodes)
    return points.astype(np.float64), (weights / np.sqrt(np.pi)).astype(np.float64)


def squashed_normal_expectation(
    mean: torch.Tensor, std: torch.Tensor, nodes: int = 32,
) -> torch.Tensor:
    """Compute E[tanh(Z)] for independent Z~Normal(mean,std) by quadrature.

    The identity used is E[f(Z)] = 1/sqrt(pi) * sum_i w_i
    f(mean + sqrt(2)*std*x_i). Nodes are cached and reused across calls.
    """
    if mean.shape != std.shape:
        raise ValueError("mean and std shapes must match")
    if torch.any(std < 0) or not torch.all(torch.isfinite(mean)) or not torch.all(torch.isfinite(std)):
        raise ValueError("mean/std must be finite and std non-negative")
    points_np, weights_np = gauss_hermite_nodes(nodes)
    points = torch.as_tensor(points_np, dtype=mean.dtype, device=mean.device)
    weights = torch.as_tensor(weights_np, dtype=mean.dtype, device=mean.device)
    expanded = mean.unsqueeze(-1) + math.sqrt(2.0) * std.unsqueeze(-1) * points
    return (torch.tanh(expanded) * weights).sum(dim=-1)


def squashed_noise_action(
    mean: torch.Tensor, std: torch.Tensor, scale: float,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    """Return tanh(mean + scale*std*epsilon) for a fixed standard-normal draw."""
    if mean.shape != std.shape or mean.shape != epsilon.shape:
        raise ValueError("mean, std and epsilon shapes must match")
    if scale < 0:
        raise ValueError("noise scale must be non-negative")
    return torch.tanh(mean + float(scale) * std * epsilon)


def select_deployment_action(
    distribution: Normal,
    mode: str,
    quadrature_nodes: int = 32,
    epsilon: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select one bounded action without changing the trainer's act interface."""
    if mode not in DEPLOYMENT_MODES:
        raise ValueError(f"unknown deployment mode: {mode}")
    mean, std = distribution.mean, distribution.stddev
    if mode == "mean":
        return torch.tanh(mean)
    if mode == "squashed_expectation":
        return squashed_normal_expectation(mean, std, quadrature_nodes)
    if epsilon is None:
        epsilon = torch.randn_like(mean)
    return squashed_noise_action(mean, std, NOISE_SCALES[mode], epsilon)


def validate_development_seed_range(seed_base: int, episodes: int) -> list[int]:
    """Reject monitoring/formal ranges and return an explicit development range."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    seeds = list(range(int(seed_base), int(seed_base) + int(episodes)))
    for start, end, role in FORBIDDEN_SEED_RANGES:
        overlap = [seed for seed in seeds if start <= seed <= end]
        if overlap:
            raise ValueError(
                f"development diagnostic refuses {role} seed range "
                f"{start}..{end}; first conflict={overlap[0]}"
            )
    return seeds


def validate_checkpoint_filename(path: Path) -> None:
    """Refuse validation-selected best_eval checkpoints in this diagnostic."""
    if path.name.lower() == "best_eval.pt":
        raise ValueError("best_eval.pt is forbidden; use an explicit development/latest checkpoint")


def prepare_output_directory(path: Path) -> None:
    """Create an output directory while refusing to overwrite any existing data."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is non-empty; overwrite refused: {path}")
    path.mkdir(parents=True, exist_ok=True)


def policy_noise_seed(mode: str, environment_seed: int, replicate: int) -> int:
    """Map a paired mode/scenario/repeat tuple to a method-independent RNG seed."""
    if mode not in NOISE_MODE_INDICES:
        raise ValueError("policy noise seeds exist only for noise deployment modes")
    if replicate < 0:
        raise ValueError("replicate must be non-negative")
    return (
        POLICY_NOISE_BASE
        + NOISE_MODE_INDICES[mode] * 100_000
        + 2 * (int(environment_seed) - DEFAULT_SEED_BASE)
        + int(replicate)
    )


def seed_policy_rng(seed: int) -> None:
    """Seed Python, NumPy, Torch CPU and every CUDA generator."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enabled_module_names(config: dict[str, Any]) -> list[str]:
    """Return enabled module names from a self-describing algorithm config."""
    return sorted(
        name for name, block in config.get("modules", {}).items()
        if isinstance(block, dict) and bool(block.get("enabled", False))
    )


def infer_method(config: dict[str, Any]) -> str:
    """Infer the EA/WB diagnostic label without relying on a directory name."""
    modules = config.get("modules", {})
    ea = bool(modules.get("entity_attention", {}).get("enabled", False))
    wb = bool(modules.get("wave_balancing", {}).get("enabled", False))
    if ea and wb:
        return "EA-WB-MAPPO"
    if ea:
        return "EA-MAPPO"
    if wb:
        return "WB-MAPPO"
    return "MAPPO"


def load_policy(checkpoint: Path, env_config: dict[str, Any], device: str):
    """Strictly validate and load a modular-v2 self-describing checkpoint."""
    validate_checkpoint_filename(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not is_formal_v2_checkpoint(state):
        raise RuntimeError("checkpoint is not a complete modular-v2 self-describing checkpoint")
    config = state.get("extra", {}).get("algorithm_config")
    if not isinstance(config, dict):
        raise RuntimeError("checkpoint lacks embedded algorithm_config")
    validate_modular_checkpoint(state, env_config, config)
    extra = state["extra"]
    architecture = extra["network_architecture"]
    trainer = build_modular_mappo_trainer(
        config, device=device, hidden_dim=int(architecture["hidden_dim"])
    )
    trainer.load(checkpoint, restore_rng=False)
    trainer.actor.eval()
    trainer.critic.eval()
    source = {
        "method": infer_method(config),
        "training_seed": int(extra["training_seed"]),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_checkpoint_sampled_steps": int(state["sampled_steps"]),
        "source_training_seed": int(extra["training_seed"]),
        "source_enabled_modules": enabled_module_names(config),
        "source_algorithm_hash": str(extra["algorithm_config_sha256"]),
        "source_environment_hash": str(extra["environment_config_sha256"]),
        "modular_mappo_impl_version": int(state["modular_mappo_impl_version"]),
        "baseline_mappo_impl_version": int(state["baseline_mappo_impl_version"]),
        "observation_dim": int(extra["observation_dim"]),
        "action_dim": int(extra["action_dim"]),
        "num_agents": int(extra["num_agents"]),
        "environment_variant": str(extra["environment_variant"]),
    }
    return trainer, source


@torch.no_grad()
def custom_action_step(
    trainer,
    observations: np.ndarray,
    alive_mask: np.ndarray,
    context: np.ndarray,
    hidden: np.ndarray | None,
    episode_mask: np.ndarray,
    mode: str,
    quadrature_nodes: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run actor.distribution_step and apply only the requested deployment rule."""
    convert = lambda value: None if value is None else torch.as_tensor(
        value, dtype=torch.float32, device=trainer.device
    )
    obs = convert(observations)
    alive = convert(alive_mask)
    ctx = convert(context)
    hid = convert(hidden)
    ep = convert(episode_mask)
    distribution, new_hidden = trainer.actor.distribution_step(
        obs, trainer._ctx(ctx, True), hid, ep, alive
    )
    epsilon = torch.randn_like(distribution.mean) if mode in NOISE_MODE_INDICES else None
    actions = select_deployment_action(
        distribution, mode, quadrature_nodes=quadrature_nodes, epsilon=epsilon
    )
    actions = actions * alive[..., None]
    return actions.cpu().numpy(), None if new_hidden is None else new_hidden.cpu().numpy()


def run_episode(
    trainer,
    env_config: dict[str, Any],
    source: dict[str, Any],
    environment_seed: int,
    mode: str,
    replicate: int | None,
    quadrature_nodes: int,
) -> dict[str, Any]:
    """Run one episode with the canonical modular evaluator state lifecycle."""
    environment = make_combat_environment(env_config)
    observation, _ = environment.reset(int(environment_seed))
    noise_seed = None
    if mode in NOISE_MODE_INDICES:
        if replicate is None:
            raise ValueError("noise mode requires a replicate")
        noise_seed = policy_noise_seed(mode, environment_seed, replicate)
        seed_policy_rng(noise_seed)
    alive = environment.red_alive_mask.copy()
    actor_hidden, critic_hidden = trainer.initial_hidden(1)
    wave = 1
    total_waves = int(env_config.get("persistent_waves", {}).get("total_waves", 1))
    episode_mask = np.zeros(1, dtype=np.float32)
    returns = np.zeros(int(source["num_agents"]), dtype=np.float64)
    while True:
        from algorithm.modules.wave_survival_pbrs import mission_context_numpy
        context = mission_context_numpy(
            trainer, np.asarray([wave]), np.asarray([total_waves]),
            environment.blue_alive_mask[None], np.asarray([environment.steps]), environment.max_steps,
        )
        actions, actor_hidden = custom_action_step(
            trainer, observation[None], alive[None], context, actor_hidden,
            episode_mask, mode, quadrature_nodes,
        )
        _, critic_hidden = trainer.values_step(
            observation[None], alive[None], context, critic_hidden, episode_mask
        )
        observation, reward, terminated, truncated, info = environment.step(actions[0])
        returns += reward
        alive = np.asarray(info["red_alive_mask"], dtype=np.float32)
        actor_hidden = trainer.recurrent.apply_alive(actor_hidden, alive[None])
        critic_hidden = trainer.recurrent.apply_alive(critic_hidden, alive[None])
        episode_mask[:] = 1
        wave = int(info.get("wave_index", 1))
        total_waves = int(info.get("total_waves", total_waves))
        if terminated or truncated:
            break
    team_return, _ = episode_return_metrics(returns)
    waves = int(info.get("waves_cleared", 0))
    red_loss = float(info.get("red_losses", 0))
    blue_loss = float(info.get("blue_losses", 0))
    result = {
        "method": source["method"],
        "training_seed": source["training_seed"],
        "checkpoint": source["checkpoint"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "environment_seed": int(environment_seed),
        "deployment_mode": mode,
        "noise_scale": NOISE_SCALES[mode],
        "noise_replicate": "" if replicate is None else int(replicate),
        "policy_noise_seed": "" if noise_seed is None else int(noise_seed),
        "waves_cleared": waves,
        "clear_wave_1": int(waves >= 1),
        "clear_wave_2": int(waves >= 2),
        "clear_wave_3": int(waves >= 3),
        "episode_return": team_return,
        "red_loss": red_loss,
        "blue_loss": blue_loss,
        "kill_loss_ratio": blue_loss / max(red_loss, 1.0),
        "ground_loss": float(info.get("red_ground_losses", 0)),
        "boundary_exit": float(info.get("red_boundary_exits", 0)),
        "timeout": int(info.get("termination_reason") == "red_failure_timeout"),
        "episode_length": int(info.get("episode_length", 0)),
    }
    result.update(per_wave_episode_diagnostics(info, total_waves))
    if not all(
        math.isfinite(float(result[key]))
        for key in (
            "waves_cleared", "episode_return", "red_loss", "blue_loss",
            "kill_loss_ratio", "ground_loss", "boundary_exit", "episode_length",
        )
    ):
        raise FloatingPointError(f"non-finite episode result: {result}")
    return result


def scenario_level_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, float]]:
    """Average policy-noise repeats inside each environment scenario first."""
    selected = [row for row in rows if row["deployment_mode"] == mode]
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in selected:
        grouped.setdefault(int(row["environment_seed"]), []).append(row)
    source_keys = {
        "average_waves": "waves_cleared", "W1": "clear_wave_1",
        "W2": "clear_wave_2", "W3": "clear_wave_3", "return": "episode_return",
        "red_loss": "red_loss", "blue_loss": "blue_loss", "ground": "ground_loss",
        "boundary": "boundary_exit", "timeout": "timeout",
        "episode_length": "episode_length",
    }
    output = []
    for environment_seed, records in sorted(grouped.items()):
        item = {"environment_seed": environment_seed}
        for destination, source in source_keys.items():
            item[destination] = float(np.mean([float(row[source]) for row in records]))
        item["K_L"] = item["blue_loss"] / max(item["red_loss"], 1.0)
        output.append(item)
    return output


def summarize_checkpoint(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build mode summaries and paired deltas using environment seed as the unit."""
    summaries = []
    scenario_by_mode = {mode: scenario_level_rows(rows, mode) for mode in DEPLOYMENT_MODES}
    mean_reference = {int(row["environment_seed"]): row for row in scenario_by_mode["mean"]}
    for mode in DEPLOYMENT_MODES:
        scenarios = scenario_by_mode[mode]
        summary: dict[str, Any] = {
            "method": rows[0]["method"], "training_seed": rows[0]["training_seed"],
            "checkpoint": rows[0]["checkpoint"], "deployment_mode": mode,
            "environment_scenario_units": len(scenarios),
            "noise_repeats_per_scenario": (
                len([row for row in rows if row["deployment_mode"] == mode]) // max(len(scenarios), 1)
            ),
        }
        for metric in SUMMARY_METRICS:
            summary[metric] = float(np.mean([row[metric] for row in scenarios]))
        # Protocol-style K/L uses aggregate scenario-mean losses.
        summary["K_L"] = sum(row["blue_loss"] for row in scenarios) / max(
            sum(row["red_loss"] for row in scenarios), 1.0
        )
        summaries.append(summary)
    deltas = []
    for mode in DEPLOYMENT_MODES[1:]:
        scenarios = scenario_by_mode[mode]
        row: dict[str, Any] = {
            "method": rows[0]["method"], "training_seed": rows[0]["training_seed"],
            "checkpoint": rows[0]["checkpoint"], "deployment_mode": mode,
            "reference_mode": "mean", "paired_environment_scenarios": len(scenarios),
        }
        for metric in SUMMARY_METRICS:
            differences = [
                scenario[metric] - mean_reference[int(scenario["environment_seed"])][metric]
                for scenario in scenarios
            ]
            row[f"delta_{metric}"] = float(np.mean(differences))
            row[f"scenarios_improved_{metric}"] = int(sum(value > 0 for value in differences))
            row[f"scenarios_degraded_{metric}"] = int(sum(value < 0 for value in differences))
        deltas.append(row)
    return summaries, deltas


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a homogeneous list of diagnostic dictionaries."""
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_metadata(
    sources: list[dict[str, Any]], env_path: Path, env_config: dict[str, Any],
    seeds: list[int], noise_repeats: int, nodes: int, device_name: str,
) -> dict[str, Any]:
    """Create explicit provenance that permanently labels output non-formal."""
    return {
        "role": ROLE,
        "not_formal_evidence": NOT_FORMAL_EVIDENCE,
        "diagnostic_name": "policy_deployment_mode_diagnostic",
        "environment_config": str(env_path.resolve()),
        "environment_config_sha256": file_sha256(env_path),
        "environment_variant": env_config.get("environment_variant"),
        "environment_seeds": seeds,
        "forbidden_seed_ranges": [
            {"start": start, "end": end, "role": role}
            for start, end, role in FORBIDDEN_SEED_RANGES
        ],
        "deployment_modes": list(DEPLOYMENT_MODES),
        "quadrature_nodes": nodes,
        "noise_repeats": noise_repeats,
        "policy_noise_seed_formula": (
            f"{POLICY_NOISE_BASE} + mode_index*100000 + "
            f"2*(environment_seed-{DEFAULT_SEED_BASE}) + replicate"
        ),
        "noise_mode_indices": NOISE_MODE_INDICES,
        "paired_unit": "environment_seed",
        "stochastic_repeats_are_not_independent_scenarios": True,
        "device": "cuda",
        "cuda_device_name": device_name,
        "sources": sources,
    }


def build_report(
    metadata: dict[str, Any], summaries: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
) -> str:
    """Build a concise report with the required mode and seed comparisons."""
    lines = [
        "# Policy deployment mode diagnostic", "",
        "**Development mechanism diagnostic only — not formal evidence.**", "",
        "## Protocol", "",
        f"- Environment seeds: `{metadata['environment_seeds'][0]}..{metadata['environment_seeds'][-1]}`",
        f"- Gauss-Hermite nodes: {metadata['quadrature_nodes']}",
        f"- Noise repeats per mode/scenario: {metadata['noise_repeats']}",
        "- Noise repeats are averaged inside each environment seed before cross-scenario aggregation.", "",
        "## Deployment-mode results", "",
        "| Method | Seed | Mode | Waves | W1 | W2 | W3 | Return | Red loss | K/L | Ground | Boundary | Timeout |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['training_seed']} | {row['deployment_mode']} | "
            f"{row['average_waves']:.3f} | {row['W1']:.3f} | {row['W2']:.3f} | "
            f"{row['W3']:.3f} | {row['return']:.3f} | {row['red_loss']:.3f} | "
            f"{row['K_L']:.3f} | {row['ground']:.3f} | {row['boundary']:.3f} | {row['timeout']:.3f} |"
        )
    lines += ["", "## Paired deltas versus mean", "",
              "| Method | Seed | Mode | Δ Waves | Δ W1 | Δ W3 | Δ Return | Δ Ground | Δ Boundary | Δ Timeout |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in deltas:
        lines.append(
            f"| {row['method']} | {row['training_seed']} | {row['deployment_mode']} | "
            f"{row['delta_average_waves']:+.3f} | {row['delta_W1']:+.3f} | "
            f"{row['delta_W3']:+.3f} | {row['delta_return']:+.3f} | "
            f"{row['delta_ground']:+.3f} | {row['delta_boundary']:+.3f} | {row['delta_timeout']:+.3f} |"
        )
    lines += [
        "", "## Interpretation guide", "",
        "- **Case A:** squashed_expectation materially improves over mean and approaches noise modes: deployment representative problem.",
        "- **Case B:** squashed_expectation stays near mean while noise modes improve: online stochastic perturbation dependence / control architecture instability.",
        "- **Case C:** noise_025 or noise_050 is best and noise_100 declines: exploration-noise sweet spot.",
        "", "The complete run should compare EA-WB3102 against EA-WB3101/3103, identify which noise dose rescues weak seeds, and check whether noise degrades the strong seed.", "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, help="Repeat for each modular checkpoint.")
    parser.add_argument("--env-config", default="configs/persistent_wave_v2_environment.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--noise-repeats", type=int, default=2)
    parser.add_argument("--quadrature-nodes", type=int, choices=(16, 32, 64), default=32)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--methods", nargs="*", help="Optional inferred-method filter.")
    parser.add_argument("--training-seeds", nargs="*", type=int, help="Optional source training-seed filter.")
    return parser.parse_args()


def resolved(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for checkpoint diagnostics; CPU fallback is forbidden")
    if args.noise_repeats not in (1, 2):
        raise ValueError("noise repeats must be 1 (smoke) or 2 (complete diagnostic)")
    seeds = validate_development_seed_range(args.seed_base, args.episodes)
    output = resolved(args.output_dir)
    prepare_output_directory(output)
    env_path = resolved(args.env_config)
    env_config = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    policies = []
    sources = []
    for value in args.checkpoint:
        checkpoint = resolved(value)
        trainer, source = load_policy(checkpoint, env_config, args.device)
        if args.methods and source["method"] not in args.methods:
            continue
        if args.training_seeds and source["training_seed"] not in args.training_seeds:
            continue
        if any(
            existing["method"] == source["method"]
            and existing["training_seed"] == source["training_seed"]
            for existing in sources
        ):
            raise ValueError(f"duplicate method/training-seed source: {source['method']} {source['training_seed']}")
        policies.append(trainer)
        sources.append(source)
    if not policies:
        raise RuntimeError("no checkpoint remains after filters")
    metadata = diagnostic_metadata(
        sources, env_path, env_config, seeds, args.noise_repeats,
        args.quadrature_nodes, torch.cuda.get_device_name(0),
    )
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    all_episodes: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    all_deltas: list[dict[str, Any]] = []
    for trainer, source in zip(policies, sources):
        policy_rows = []
        print(f"[POLICY] {source['method']} seed{source['training_seed']}", flush=True)
        for mode in DEPLOYMENT_MODES:
            repeats = args.noise_repeats if mode in NOISE_MODE_INDICES else 1
            for environment_seed in seeds:
                for replicate in range(repeats):
                    row = run_episode(
                        trainer, env_config, source, environment_seed, mode,
                        replicate if mode in NOISE_MODE_INDICES else None,
                        args.quadrature_nodes,
                    )
                    policy_rows.append(row)
                    all_episodes.append(row)
            print(f"[MODE] {mode}: episodes={len(seeds) * repeats}", flush=True)
        summaries, deltas = summarize_checkpoint(policy_rows)
        all_summaries.extend(summaries)
        all_deltas.extend(deltas)
    write_csv(output / "episode_results.csv", all_episodes)
    write_csv(output / "deployment_mode_summary.csv", all_summaries)
    write_csv(output / "paired_mode_deltas.csv", all_deltas)
    (output / "report.md").write_text(
        build_report(metadata, all_summaries, all_deltas), encoding="utf-8"
    )
    print(
        f"[COMPLETE] policies={len(policies)} episodes={len(all_episodes)} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
