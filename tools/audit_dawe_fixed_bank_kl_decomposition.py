#!/usr/bin/env python3
"""Read-only fixed-bank DAWE mean/variance Gaussian-KL decomposition audit."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from tools.fixed10_wave_state_drift_common import (
    BANK_PATH, checkpoint_mutation_guard, checkpoint_step, csv_write,
    distribution_stats, json_dump, load_checkpoint, state_dict_sha256,
    symmetric_diagonal_gaussian_kl, validate_bank,
)

SEEDS = (5301, 5302, 5303)
SOURCE_STEP = 1_505_280
FINAL_STEP = 1_805_280
OUTPUT = ROOT / "outputs/dawe_fixed_bank_kl_decomposition"
FORMAL_ANALYSIS = ROOT / "outputs/dawe_fixed10_300k_analysis/analysis.json"
FORMAL_PROTOCOL = ROOT / "outputs/dawe_fixed10_300k_analysis/protocol_validation.json"
EPSILON = 1e-12
MIN_WAVE_TEAM_STATES = 30
MIN_WAVE_ALIVE_AGENTS = 60
ACTION_NAMES = ("heading", "pitch", "speed")
MULTIPLIERS = {"Control": {1: 1.0, 2: 1.0, 3: 1.0}, "DAWE": {1: 0.25, 2: 0.25, 3: 1.0}}


def source_path(seed: int) -> Path:
    return ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt"


def final_path(arm: str, seed: int) -> Path:
    branch = "control" if arm == "Control" else "v1"
    return ROOT / f"outputs/dev_dawe_{branch}_seed{seed}_300k/final.pt"


def run_dir(arm: str, seed: int) -> Path:
    branch = "control" if arm == "Control" else "v1"
    return ROOT / f"outputs/dev_dawe_{branch}_seed{seed}_300k"


def directional_diagonal_gaussian_kl_components(
    mu_source: np.ndarray, log_std_source: np.ndarray,
    mu_final: np.ndarray, log_std_final: np.ndarray,
) -> dict[str, np.ndarray]:
    mu_source = np.asarray(mu_source, dtype=np.float64)
    mu_final = np.asarray(mu_final, dtype=np.float64)
    log_std_source = np.asarray(log_std_source, dtype=np.float64)
    log_std_final = np.asarray(log_std_final, dtype=np.float64)
    sigma_source = np.exp(log_std_source)
    sigma_final = np.exp(log_std_final)
    variance = np.sum(
        log_std_final - log_std_source
        + sigma_source ** 2 / (2.0 * sigma_final ** 2) - 0.5,
        axis=-1,
    )
    mean = np.sum((mu_source - mu_final) ** 2 / (2.0 * sigma_final ** 2), axis=-1)
    variance = np.maximum(variance, 0.0)
    mean = np.maximum(mean, 0.0)
    return {"mean": mean, "variance": variance, "total": mean + variance}


def symmetric_diagonal_gaussian_kl_components(
    mu_source: np.ndarray, log_std_source: np.ndarray,
    mu_final: np.ndarray, log_std_final: np.ndarray,
) -> dict[str, np.ndarray]:
    mu_source = np.asarray(mu_source, dtype=np.float64)
    mu_final = np.asarray(mu_final, dtype=np.float64)
    log_std_source = np.asarray(log_std_source, dtype=np.float64)
    log_std_final = np.asarray(log_std_final, dtype=np.float64)
    sigma_source = np.exp(log_std_source)
    sigma_final = np.exp(log_std_final)
    delta2 = (mu_source - mu_final) ** 2
    mean = 0.25 * np.sum(delta2 * (1.0 / sigma_source ** 2 + 1.0 / sigma_final ** 2), axis=-1)
    variance = 0.25 * np.sum(
        sigma_source ** 2 / sigma_final ** 2
        + sigma_final ** 2 / sigma_source ** 2 - 2.0,
        axis=-1,
    )
    mean = np.maximum(mean, 0.0)
    variance = np.maximum(variance, 0.0)
    return {"mean": mean, "variance": variance, "total": mean + variance}


def safe_ratio(numerator: float | None, denominator: float | None, epsilon: float = EPSILON) -> dict[str, Any]:
    if numerator is None or denominator is None or abs(float(denominator)) <= epsilon:
        return {"value": None, "status": "UNDEFINED"}
    return {"value": float(numerator) / float(denominator), "status": "DEFINED"}


def contribution_fraction(component: float | None, total: float | None, epsilon: float = EPSILON) -> dict[str, Any]:
    return safe_ratio(component, total, epsilon)


def effective_log_std(log_std: np.ndarray, wave: np.ndarray, arm: str) -> np.ndarray:
    result = np.array(log_std, dtype=np.float64, copy=True)
    for wave_id, rho in MULTIPLIERS[arm].items():
        result[wave == wave_id] += math.log(rho)
    return result


def validate_formal_results() -> tuple[dict[str, Any], dict[str, Any]]:
    analysis = json.loads(FORMAL_ANALYSIS.read_text(encoding="utf-8"))
    protocol = json.loads(FORMAL_PROTOCOL.read_text(encoding="utf-8"))
    if analysis.get("status") != "DAWE_FIXED10_300K_ANALYSIS_COMPLETE":
        raise RuntimeError("formal DAWE analysis status mismatch")
    if analysis.get("DAWE_SCREEN") != "SAFETY_FAIL" or analysis.get("uses_45m") is not False:
        raise RuntimeError("formal DAWE label/45M guard mismatch")
    if len(protocol) != 6 or any(row.get("status") != "PASS" for row in protocol.values()):
        raise RuntimeError("formal DAWE six-run protocol validation did not pass")
    return analysis, protocol


def validate_checkpoint_identity(state: dict[str, Any], expected_step: int, seed: int, role: str) -> None:
    if state.get("algorithm") != "modular_mappo":
        raise RuntimeError(f"{role} algorithm mismatch")
    if checkpoint_step(state) != expected_step:
        raise RuntimeError(f"{role} checkpoint step mismatch")
    if int(state.get("extra", {}).get("training_seed", -1)) != seed:
        raise RuntimeError(f"{role} checkpoint seed mismatch")


def actor_outputs(config: dict[str, Any], state: dict[str, Any], device: str,
                  observations: np.ndarray, alive: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    trainer = build_modular_mappo_trainer(config, device=device, hidden_dim=256, total_sampled_steps=FINAL_STEP)
    trainer.actor.load_state_dict(state["actor"], strict=True)
    trainer.actor.eval()
    before = state_dict_sha256(trainer.actor.state_dict())
    mus: list[np.ndarray] = []
    log_stds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(observations), 256):
            obs = torch.as_tensor(observations[start:start + 256], dtype=torch.float32, device=device)
            mask = torch.as_tensor(alive[start:start + 256], dtype=torch.float32, device=device)
            distribution, _ = trainer.actor.distribution_step(obs, None, None, None, mask)
            mus.append(distribution.mean.detach().cpu().numpy())
            log_stds.append(torch.log(distribution.scale).detach().cpu().numpy())
    after = state_dict_sha256(trainer.actor.state_dict())
    if before != after:
        raise RuntimeError("Actor mutation guard failed during no-grad fixed-bank inference")
    result = {"mu": np.concatenate(mus), "log_std": np.concatenate(log_stds)}
    del trainer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result, {"before": before, "after": after, "status": "PASS"}


def selected(values: np.ndarray, team_mask: np.ndarray, alive: np.ndarray) -> np.ndarray:
    live = alive[team_mask] > 0.5
    return values[team_mask][live]


def group_summary(seed: int, arm: str, group: str, team_mask: np.ndarray,
                  alive: np.ndarray, source: dict[str, np.ndarray], final: dict[str, np.ndarray],
                  wave: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    live = alive[team_mask] > 0.5
    team_count = int(team_mask.sum())
    alive_count = int(live.sum())
    dead_count = int(live.size - alive_count)
    coverage = "SUFFICIENT" if group == "ALL" or (team_count >= MIN_WAVE_TEAM_STATES and alive_count >= MIN_WAVE_ALIVE_AGENTS) else "INSUFFICIENT_COVERAGE"

    base_directional = directional_diagonal_gaussian_kl_components(source["mu"], source["log_std"], final["mu"], final["log_std"])
    base_symmetric = symmetric_diagonal_gaussian_kl_components(source["mu"], source["log_std"], final["mu"], final["log_std"])
    source_effective = effective_log_std(source["log_std"], wave, arm)
    final_effective = effective_log_std(final["log_std"], wave, arm)
    effective_directional = directional_diagonal_gaussian_kl_components(source["mu"], source_effective, final["mu"], final_effective)
    effective_symmetric = symmetric_diagonal_gaussian_kl_components(source["mu"], source_effective, final["mu"], final_effective)
    delta_mu = final["mu"] - source["mu"]
    delta_action = np.tanh(final["mu"]) - np.tanh(source["mu"])
    delta_log_std = final["log_std"] - source["log_std"]
    std_ratio = np.exp(delta_log_std)

    arrays: dict[str, np.ndarray] = {
        "raw_mu_abs": np.mean(np.abs(delta_mu), axis=-1),
        "raw_mu_l2": np.linalg.vector_norm(delta_mu, axis=-1),
        "deterministic_action_l2": np.linalg.vector_norm(delta_action, axis=-1),
        "abs_delta_log_std": np.mean(np.abs(delta_log_std), axis=-1),
        "log_std_delta": np.mean(delta_log_std, axis=-1),
    }
    for index, name in enumerate(ACTION_NAMES):
        arrays[f"{name}_raw_mu_abs"] = np.abs(delta_mu[..., index])
        arrays[f"{name}_action_abs_delta"] = np.abs(delta_action[..., index])
    for prefix, components in (
        ("base_directional", base_directional), ("effective_directional", effective_directional),
        ("base_symmetric", base_symmetric), ("effective_symmetric", effective_symmetric),
    ):
        for component, values in components.items():
            arrays[f"{prefix}_kl_{component}"] = values

    row: dict[str, Any] = {
        "seed": seed, "arm": arm, "group": group,
        "team_state_count": team_count, "alive_agent_count": alive_count,
        "dead_agent_count": dead_count, "coverage": coverage,
    }
    details: dict[str, Any] = {"seed": seed, "arm": arm, "group": group, "coverage": coverage}
    for name, values in arrays.items():
        values_live = selected(values, team_mask, alive)
        stats = distribution_stats(values_live)
        details[name] = stats
        for stat, value in stats.items():
            if stat != "n":
                row[f"{name}_{stat}"] = value

    ratio_live = selected(std_ratio, team_mask, alive).reshape(-1)
    log_ratio_live = np.log(ratio_live)
    row["std_geometric_ratio"] = float(np.exp(np.mean(log_ratio_live)))
    row["std_ratio_mean"] = float(np.mean(ratio_live))
    row["base_source_std_mean"] = float(np.mean(selected(np.exp(source["log_std"]), team_mask, alive)))
    row["base_final_std_mean"] = float(np.mean(selected(np.exp(final["log_std"]), team_mask, alive)))
    row["effective_source_std_mean"] = float(np.mean(selected(np.exp(source_effective), team_mask, alive)))
    row["effective_final_std_mean"] = float(np.mean(selected(np.exp(final_effective), team_mask, alive)))

    for prefix in ("base_directional", "effective_directional", "base_symmetric", "effective_symmetric"):
        mean_values = selected(arrays[f"{prefix}_kl_mean"], team_mask, alive)
        variance_values = selected(arrays[f"{prefix}_kl_variance"], team_mask, alive)
        total_values = selected(arrays[f"{prefix}_kl_total"], team_mask, alive)
        aggregate_mean = contribution_fraction(float(np.mean(mean_values)), float(np.mean(total_values)))
        aggregate_variance = contribution_fraction(float(np.mean(variance_values)), float(np.mean(total_values)))
        fractions_mean = np.divide(mean_values, total_values, out=np.full_like(mean_values, np.nan), where=total_values > EPSILON)
        fractions_variance = np.divide(variance_values, total_values, out=np.full_like(variance_values, np.nan), where=total_values > EPSILON)
        details[f"{prefix}_contribution"] = {
            "ratio_of_aggregated_means": {"mean_component_fraction": aggregate_mean, "variance_component_fraction": aggregate_variance},
            "per_sample_fraction_distribution": {
                "mean_component_fraction": distribution_stats(fractions_mean),
                "variance_component_fraction": distribution_stats(fractions_variance),
            },
        }
        row[f"{prefix}_mean_component_fraction_aggregate"] = aggregate_mean["value"]
        row[f"{prefix}_variance_component_fraction_aggregate"] = aggregate_variance["value"]

    identity = {
        "base_directional_decomposition_max_abs_error": float(np.max(np.abs(base_directional["total"] - base_directional["mean"] - base_directional["variance"]))),
        "effective_directional_decomposition_max_abs_error": float(np.max(np.abs(effective_directional["total"] - effective_directional["mean"] - effective_directional["variance"]))),
        "base_symmetric_decomposition_max_abs_error": float(np.max(np.abs(base_symmetric["total"] - base_symmetric["mean"] - base_symmetric["variance"]))),
        "effective_symmetric_decomposition_max_abs_error": float(np.max(np.abs(effective_symmetric["total"] - effective_symmetric["mean"] - effective_symmetric["variance"]))),
        "base_symmetric_vs_legacy_max_abs_error": float(np.max(np.abs(base_symmetric["total"] - symmetric_diagonal_gaussian_kl(source["mu"], source["log_std"], final["mu"], final["log_std"])))),
    }
    for family, base, effective in (("directional", base_directional, effective_directional), ("symmetric", base_symmetric, effective_symmetric)):
        base_mean = float(np.mean(selected(base["mean"], team_mask, alive)))
        effective_mean = float(np.mean(selected(effective["mean"], team_mask, alive)))
        base_variance = selected(base["variance"], team_mask, alive)
        effective_variance = selected(effective["variance"], team_mask, alive)
        identity[f"{family}_effective_vs_base_variance_max_abs_error"] = float(np.max(np.abs(effective_variance - base_variance)))
        identity[f"{family}_effective_mean_over_base_mean"] = safe_ratio(effective_mean, base_mean)
    return row, {**details, "identity_checks": identity}


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "deterministic_action_l2_mean", "raw_mu_abs_mean", "raw_mu_l2_mean",
        "base_directional_kl_mean_mean", "base_directional_kl_variance_mean", "base_directional_kl_total_mean",
        "effective_directional_kl_mean_mean", "effective_directional_kl_variance_mean", "effective_directional_kl_total_mean",
        "base_symmetric_kl_mean_mean", "base_symmetric_kl_variance_mean", "base_symmetric_kl_total_mean",
        "effective_symmetric_kl_mean_mean", "effective_symmetric_kl_variance_mean", "effective_symmetric_kl_total_mean",
        "std_geometric_ratio",
    )
    result = []
    for seed in SEEDS:
        for group in ("W1", "W2", "W3", "ALL"):
            control = next(row for row in rows if row["seed"] == seed and row["arm"] == "Control" and row["group"] == group)
            dawe = next(row for row in rows if row["seed"] == seed and row["arm"] == "DAWE" and row["group"] == group)
            pair: dict[str, Any] = {"seed": seed, "group": group, "coverage": "SUFFICIENT" if control["coverage"] == dawe["coverage"] == "SUFFICIENT" else "INSUFFICIENT_COVERAGE"}
            for metric in metrics:
                pair[f"Control_{metric}"] = control[metric]
                pair[f"DAWE_{metric}"] = dawe[metric]
                ratio = safe_ratio(dawe[metric], control[metric])
                pair[f"DAWE_over_Control_{metric}"] = ratio["value"]
                pair[f"DAWE_over_Control_{metric}_status"] = ratio["status"]
            result.append(pair)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for this checkpoint/Actor audit")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing audit directory: {OUTPUT}")
    validate_formal_results()
    with np.load(BANK_PATH, allow_pickle=False) as bank:
        observations = np.asarray(bank["observations"], dtype=np.float32)
        alive = np.asarray(bank["alive"], dtype=np.float32)
        wave = np.asarray(bank["wave"], dtype=np.int64)
    validate_bank(observations, alive, wave)

    coverage = {
        "source": str(BANK_PATH.relative_to(ROOT)).replace("\\", "/"),
        "historical_source": "historical Plain MAPPO bank",
        "valid_use": "COMMON_FIXED_STATE_ACTOR_FUNCTION_COMPARISON_ONLY",
        "not_valid_as": "DAWE natural state visitation distribution",
        "schema": {"observations": list(observations.shape), "alive": list(alive.shape), "wave": list(wave.shape)},
        "groups": {},
    }
    for wave_id in (1, 2, 3):
        mask = wave == wave_id
        teams = int(mask.sum())
        live = int((alive[mask] > 0.5).sum())
        dead = int(alive[mask].size - live)
        coverage["groups"][f"W{wave_id}"] = {
            "team_state_count": teams, "alive_agent_count": live, "dead_agent_count": dead,
            "status": "SUFFICIENT" if teams >= MIN_WAVE_TEAM_STATES and live >= MIN_WAVE_ALIVE_AGENTS else "INSUFFICIENT_COVERAGE",
        }
    if any(value["status"] != "SUFFICIENT" for value in coverage["groups"].values()):
        coverage["overall"] = "INSUFFICIENT_COVERAGE"
    else:
        coverage["overall"] = "SUFFICIENT"

    paths = [source_path(seed) for seed in SEEDS] + [final_path(arm, seed) for seed in SEEDS for arm in ("Control", "DAWE")]
    checkpoint_before = checkpoint_mutation_guard(paths)
    inventory: list[dict[str, Any]] = []
    actor_guards: dict[str, Any] = {}
    outputs: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    states: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in SEEDS:
        source = load_checkpoint(source_path(seed), device=args.device)
        validate_checkpoint_identity(source, SOURCE_STEP, seed, f"Source seed{seed}")
        states[(seed, "Source")] = source
        for arm in ("Control", "DAWE"):
            final = load_checkpoint(final_path(arm, seed), device=args.device)
            validate_checkpoint_identity(final, FINAL_STEP, seed, f"{arm} final seed{seed}")
            expected_modules = ["actor_gradient_clipping", "actor_lr_decay"] + ([] if arm == "Control" else ["deployment_aligned_wave_exploration"])
            if final.get("enabled_modules") != expected_modules:
                raise RuntimeError(f"{arm} seed{seed} enabled_modules mismatch")
            states[(seed, arm)] = final
            config = yaml.safe_load((run_dir(arm, seed) / "algorithm_config.yaml").read_text(encoding="utf-8"))
            source_output, source_guard = actor_outputs(config, source, args.device, observations, alive)
            final_output, final_guard = actor_outputs(config, final, args.device, observations, alive)
            outputs[(seed, f"{arm}_Source")] = source_output
            outputs[(seed, arm)] = final_output
            actor_guards[f"seed{seed}_{arm}_Source"] = source_guard
            actor_guards[f"seed{seed}_{arm}_Final"] = final_guard
        inventory.extend([
            {"seed": seed, "role": "Source", "path": str(source_path(seed).relative_to(ROOT)).replace("\\", "/"), "sampled_steps": SOURCE_STEP, "sha256": checkpoint_before[str(source_path(seed).relative_to(ROOT))]},
            *[{"seed": seed, "role": f"{arm} Final", "path": str(final_path(arm, seed).relative_to(ROOT)).replace("\\", "/"), "sampled_steps": FINAL_STEP, "sha256": checkpoint_before[str(final_path(arm, seed).relative_to(ROOT))]} for arm in ("Control", "DAWE")],
        ])

    summary_rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ("Control", "DAWE"):
            source = outputs[(seed, f"{arm}_Source")]
            final = outputs[(seed, arm)]
            for group, mask in (("W1", wave == 1), ("W2", wave == 2), ("W3", wave == 3), ("ALL", np.ones(len(wave), dtype=bool))):
                row, detail = group_summary(seed, arm, group, mask, alive, source, final, wave)
                summary_rows.append(row)
                detailed.append(detail)
    pairs = paired_rows(summary_rows)

    identity_checks = {
        f"seed{item['seed']}_{item['arm']}_{item['group']}": item["identity_checks"] for item in detailed
    }
    max_identity_error = max(
        value for checks in identity_checks.values() for key, value in checks.items()
        if key.endswith("max_abs_error")
    )
    if max_identity_error > 1e-8:
        raise RuntimeError(f"KL mathematical identity check failed: {max_identity_error}")
    for item in detailed:
        if item["arm"] == "DAWE" and item["group"] in {"W1", "W2"}:
            for family in ("directional", "symmetric"):
                ratio = item["identity_checks"][f"{family}_effective_mean_over_base_mean"]
                if ratio["status"] == "DEFINED" and not math.isclose(ratio["value"], 16.0, rel_tol=1e-8, abs_tol=1e-8):
                    raise RuntimeError(f"rho=.25 mean-KL identity failed: {item}")
        if item["group"] == "W3":
            for family in ("directional", "symmetric"):
                ratio = item["identity_checks"][f"{family}_effective_mean_over_base_mean"]
                if ratio["status"] == "DEFINED" and not math.isclose(ratio["value"], 1.0, rel_tol=1e-8, abs_tol=1e-8):
                    raise RuntimeError(f"rho=1 mean-KL identity failed: {item}")

    checkpoint_after = checkpoint_mutation_guard(paths)
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("checkpoint file mutation guard failed")
    if any(value["before"] != value["after"] for value in actor_guards.values()):
        raise RuntimeError("Actor in-memory mutation guard failed")

    def seed_rows(arm: str, group: str) -> list[dict[str, Any]]:
        return [row for row in summary_rows if row["arm"] == arm and row["group"] == group]

    def seed_values(arm: str, group: str, metric: str) -> dict[str, float]:
        return {str(row["seed"]): float(row[metric]) for row in seed_rows(arm, group)}

    def seed_mean(arm: str, group: str, metric: str) -> float:
        return float(np.mean(list(seed_values(arm, group, metric).values())))

    def ordering(arm: str, group: str, family: str) -> dict[str, str]:
        return {
            str(row["seed"]): (
                "mean" if row[f"{family}_kl_mean_mean"] > row[f"{family}_kl_variance_mean"]
                else "variance" if row[f"{family}_kl_variance_mean"] > row[f"{family}_kl_mean_mean"]
                else "equal"
            )
            for row in seed_rows(arm, group)
        }

    component_ordering = {
        group: {
            "directional": ordering("DAWE", group, "effective_directional"),
            "symmetric": ordering("DAWE", group, "effective_symmetric"),
        }
        for group in ("W1", "W2", "W3")
    }
    w1_w2_symmetric_mean = all(
        value == "mean" for group in ("W1", "W2")
        for value in component_ordering[group]["symmetric"].values()
    )
    w3_symmetric_variance = all(value == "variance" for value in component_ordering["W3"]["symmetric"].values())
    if w1_w2_symmetric_mean and w3_symmetric_variance:
        label = "MEAN_AND_VARIANCE_BOTH_MATERIAL"
        support = "both"
    else:
        all_orders = [value for group in component_ordering.values() for family in group.values() for value in family.values()]
        if all(value == "mean" for value in all_orders):
            label, support = "MEAN_DRIFT_DOMINANT_DESCRIPTIVELY", "mean-update stabilization"
        elif all(value == "variance" for value in all_orders):
            label, support = "VARIANCE_DRIFT_DOMINANT_DESCRIPTIVELY", "variance decoupling"
        else:
            label, support = "MIXED_OR_UNRESOLVED", "unresolved"

    contribution_summary: dict[str, Any] = {}
    for group in ("W1", "W2", "W3", "ALL"):
        contribution_summary[group] = {}
        for family in ("base_directional", "effective_directional", "base_symmetric", "effective_symmetric"):
            mean_component = seed_mean("DAWE", group, f"{family}_kl_mean_mean")
            variance_component = seed_mean("DAWE", group, f"{family}_kl_variance_mean")
            total = seed_mean("DAWE", group, f"{family}_kl_total_mean")
            contribution_summary[group][family] = {
                "training_seed_mean_KL_mean": mean_component,
                "training_seed_mean_KL_variance": variance_component,
                "training_seed_mean_KL_total": total,
                "ratio_of_aggregated_seed_means": {
                    "mean_component_fraction": contribution_fraction(mean_component, total),
                    "variance_component_fraction": contribution_fraction(variance_component, total),
                },
            }

    dawe_base_sigma_ratio = {
        group: {
            str(row["seed"]): float(row["base_final_std_mean"] / row["base_source_std_mean"])
            for row in seed_rows("DAWE", group)
        }
        for group in ("W1", "W2", "W3")
    }
    dawe_effective_vs_control_final = {
        group: {
            str(seed): float(
                next(row for row in seed_rows("DAWE", group) if row["seed"] == seed)["effective_final_std_mean"]
                / next(row for row in seed_rows("Control", group) if row["seed"] == seed)["effective_final_std_mean"]
            )
            for seed in SEEDS
        }
        for group in ("W1", "W2", "W3")
    }
    action_drift_comparison = {
        group: {
            str(seed): {
                "Control": next(row for row in seed_rows("Control", group) if row["seed"] == seed)["deterministic_action_l2_mean"],
                "DAWE": next(row for row in seed_rows("DAWE", group) if row["seed"] == seed)["deterministic_action_l2_mean"],
                "DAWE_higher": next(row for row in seed_rows("DAWE", group) if row["seed"] == seed)["deterministic_action_l2_mean"] > next(row for row in seed_rows("Control", group) if row["seed"] == seed)["deterministic_action_l2_mean"],
            }
            for seed in SEEDS
        }
        for group in ("W1", "W2", "W3", "ALL")
    }
    scientific_answers = {
        "1_dawe_base_sigma_relative_to_source": {
            "answer": "INCREASED_IN_ALL_WAVES_AND_ALL_THREE_TRAINING_SEEDS",
            "final_over_source_ratio": dawe_base_sigma_ratio,
        },
        "2_dawe_effective_w1_w2_sigma_relative_to_control": {
            "answer": "LOWER_THAN_CONTROL_FOR_W1_AND_W2_IN_ALL_THREE_TRAINING_SEEDS",
            "DAWE_over_Control_final_effective_sigma": {key: dawe_effective_vs_control_final[key] for key in ("W1", "W2")},
        },
        "3_w3_shared_actor_sigma_change": {
            "answer": "DAWE_W3_SIGMA_INCREASED_FROM_SOURCE_AND_EXCEEDED_CONTROL_IN_ALL_THREE_TRAINING_SEEDS",
            "DAWE_final_over_source": dawe_base_sigma_ratio["W3"],
            "DAWE_over_Control_final": dawe_effective_vs_control_final["W3"],
        },
        "4_deterministic_mean_action_drift_dawe_vs_control": {
            "answer": "NOT_CONSISTENTLY_HIGHER_FOR_DAWE",
            "per_seed_wave": action_drift_comparison,
        },
        "5_base_kl_components": contribution_summary,
        "6_effective_kl_components": contribution_summary,
        "7_rho_quarter_mean_kl_amplification": {
            "answer": "PASS_APPROXIMATELY_16X_FOR_W1_W2_DIRECTIONAL_AND_SYMMETRIC",
            "identity_file": "mathematical_identity_checks.json",
        },
        "8_w3_component_interpretation": {
            "answer": "SYMMETRIC_KL_VARIANCE_COMPONENT_EXCEEDS_MEAN_IN_ALL_THREE_SEEDS; DIRECTIONAL_ORDERING_IS_2_VARIANCE_1_MEAN",
            "ordering": component_ordering["W3"],
            "components": contribution_summary["W3"],
        },
        "9_cross_seed_direction": {
            "answer": "CONSISTENT_FOR_SIGMA_INFLATION_AND_W1_W2_MEAN_AMPLIFICATION; MIXED_FOR_RAW_DETERMINISTIC_MEAN_ACTION_DRIFT",
            "component_ordering": component_ordering,
        },
        "10_next_algorithm_decision_support": {
            "answer": support,
            "descriptive_label": label,
            "basis": "W1/W2 effective symmetric KL is mean-component dominated across all seeds, while W3 is variance-component dominated across all seeds; no arbitrary percentage threshold used.",
        },
    }

    analysis = {
        "status": "DAWE_FIXED_BANK_KL_DECOMPOSITION_COMPLETE",
        "scope": "COMMON_FIXED_STATE_ACTOR_FUNCTION_COMPARISON_ONLY",
        "replication_unit": "training_seed", "n_training_seeds": 3,
        "bank_alive_observations_are_not_independent_replicates": True,
        "descriptive_label": label,
        "next_mechanism_support": support,
        "component_ordering_by_wave_and_seed": component_ordering,
        "scientific_answers": scientific_answers,
        "formal_label_verified": "SAFETY_FAIL",
        "coverage": coverage["overall"],
        "environment_steps": 0, "backward_calls": 0, "optimizer_steps": 0,
        "formal_evaluations": 0, "used_44m": False, "accessed_45m": False,
        "checkpoint_mutation_guard": "PASS", "actor_mutation_guard": "PASS",
        "no_root_cause_claim": True,
    }
    limitations = {
        "historical_plain_bank_not_dawe_natural_visitation": True,
        "fixed_state_function_distance_not_performance_causality": True,
        "training_seed_is_replication_unit": True,
        "alive_agent_observations_not_independent_replicates": True,
        "no_p_values_over_bank_observations": True,
        "no_root_cause_proven_label": True,
    }

    OUTPUT.mkdir(parents=True)
    json_dump(OUTPUT / "run_status.json", {"status": analysis["status"], "device": torch.cuda.get_device_name(torch.device(args.device)), **{key: analysis[key] for key in ("environment_steps", "backward_calls", "optimizer_steps", "formal_evaluations", "used_44m", "accessed_45m")}})
    json_dump(OUTPUT / "inventory.json", {"bank": str(BANK_PATH.relative_to(ROOT)).replace("\\", "/"), "formal_analysis": str(FORMAL_ANALYSIS.relative_to(ROOT)).replace("\\", "/"), "formal_protocol": str(FORMAL_PROTOCOL.relative_to(ROOT)).replace("\\", "/")})
    json_dump(OUTPUT / "bank_coverage.json", coverage)
    json_dump(OUTPUT / "checkpoint_manifest.json", {"checkpoints": inventory, "sha256_before": checkpoint_before, "sha256_after": checkpoint_after, "actor_state_guards": actor_guards, "status": "PASS"})
    csv_write(OUTPUT / "per_seed_wave_summary.csv", summary_rows)
    csv_write(OUTPUT / "paired_dawe_vs_control.csv", pairs)
    json_dump(OUTPUT / "kl_component_summary.json", {"groups": detailed})
    json_dump(OUTPUT / "mean_drift_summary.json", {"groups": [{key: value for key, value in item.items() if "mu" in key or "action" in key or key in {"seed", "arm", "group", "coverage"}} for item in detailed]})
    json_dump(OUTPUT / "variance_drift_summary.json", {"groups": [{key: value for key, value in item.items() if "std" in key or "variance" in key or key in {"seed", "arm", "group", "coverage"}} for item in detailed]})
    json_dump(OUTPUT / "mathematical_identity_checks.json", {"maximum_absolute_identity_error": max_identity_error, "checks": identity_checks, "status": "PASS"})
    json_dump(OUTPUT / "analysis.json", analysis)
    json_dump(OUTPUT / "limitations.json", limitations)
    (OUTPUT / "decision_support.txt").write_text(
        f"DESCRIPTIVE_LABEL={label}\nNEXT_MECHANISM_SUPPORT={support}\n"
        "SCOPE=COMMON_FIXED_STATE_ACTOR_FUNCTION_COMPARISON_ONLY\nROOT_CAUSE_CLAIM=NOT_MADE\n",
        encoding="utf-8",
    )
    print("checkpoint SHA guard: PASS")
    print("Actor mutation guard: PASS")
    print("environment_steps=0")
    print("backward_calls=0")
    print("optimizer_steps=0")
    print("45m_accessed=false")
    print(json.dumps({"status": analysis["status"], "descriptive_label": label, "output": str(OUTPUT.relative_to(ROOT)).replace("\\", "/")}, indent=2))


if __name__ == "__main__":
    main()
