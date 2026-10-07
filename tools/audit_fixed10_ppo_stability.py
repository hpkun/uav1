#!/usr/bin/env python3
"""Strict offline audit of Fixed10 PPO optimization stability.

This tool reads completed logs and CPU checkpoint metadata only.  It never
constructs an environment, trainer, rollout, model inference, or optimizer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/fixed10_ppo_stability_audit"
SEEDS = (5301, 5302, 5303)
SOURCE_STEP, TARGET_STEP = 1_505_280, 1_805_280
EVALUATION_STEPS = (1_603_584, 1_701_888, 1_800_192, 1_805_280)
RUNS = {
    "Control05": lambda seed: ROOT / f"outputs/dev_actor_clip05_fixed10_seed{seed}_300k",
    "Treatment10": lambda seed: ROOT / f"outputs/dev_actor_clip10_fixed10_seed{seed}_300k",
}
WINDOWS = {
    "EARLY": (1_505_280, 1_603_584),
    "MIDDLE": (1_603_584, 1_701_888),
    "LATE": (1_701_888, 1_805_280),
    "SEGMENT_1": (1_505_280, 1_603_584),
    "SEGMENT_2": (1_603_584, 1_701_888),
    "SEGMENT_3": (1_701_888, 1_800_192),
    "SEGMENT_4": (1_800_192, 1_805_280),
}
KL_THRESHOLDS = (.03, .05, .10, .20)
TAIL_THRESHOLDS = (5., 10., 20., 50.)
PAIR_METRICS = {
    "kl": "approx_kl", "clip_fraction": "clip_fraction", "entropy": "entropy",
    "actor_preclip_norm": "actor_grad_clip_preclip_norm",
    "actor_exact_scale": "actor_grad_clip_exact_scale",
    "actor_clip_pressure": "actor_grad_clip_pressure_fraction",
    "critic_preclip_norm": "actor_grad_clip_critic_preclip_norm",
    "critic_exact_scale": "actor_grad_clip_critic_exact_scale",
    "ratio_std": "ratio_std", "max_abs_log_ratio": "max_abs_log_ratio",
    "underflow_fraction": "ratio_underflow_fraction",
}
EVAL_FIELDS = {
    "AW": "average_waves_cleared", "W1": "clear_wave_1_probability",
    "W2": "clear_wave_2_probability", "W3": "clear_wave_3_probability",
    "Return": "average_return", "RedLoss": "average_red_loss",
    "Boundary": "average_red_boundary_exits", "Ground": "average_red_ground_losses",
    "EpisodeLength": "average_episode_length",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict]):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, q):
    values = np.asarray(list(values), dtype=np.float64)
    return None if not values.size else float(np.quantile(values, q))


def basic_stats(values, quantiles=(.10, .90, .95), include_max=True):
    values = np.asarray(list(values), dtype=np.float64)
    if not values.size:
        return {"n": 0, "mean": None, "median": None, **{f"p{int(q*100)}": None for q in quantiles}, **({"max": None} if include_max else {})}
    result = {"n": int(values.size), "mean": float(values.mean()), "median": float(np.median(values))}
    result.update({f"p{int(q*100)}": float(np.quantile(values, q)) for q in quantiles})
    if include_max:
        result["max"] = float(values.max())
    return result


def threshold_fractions(values, thresholds=KL_THRESHOLDS):
    values = np.asarray(list(values), dtype=np.float64)
    return {f"gt_{threshold:g}": (float(np.mean(values > threshold)) if values.size else None) for threshold in thresholds}


def ratio_tail_stats(rows):
    result = {}
    for key in ("clip_fraction", "ratio_mean", "ratio_std", "ratio_p1", "ratio_p50", "ratio_p99", "ratio_min", "ratio_max"):
        stats = basic_stats((float(row[key]) for row in rows), quantiles=(.90, .95))
        result.update({f"{key}_{name}": value for name, value in stats.items()})
    for key in ("log_ratio_min", "log_ratio_max", "max_abs_log_ratio"):
        values = [float(row[key]) for row in rows]
        stats = basic_stats(values, quantiles=(.90, .95))
        for name in ("median", "p90", "p95", "max"):
            result[f"{key}_{name}"] = stats[name]
    tails = np.asarray([float(row["max_abs_log_ratio"]) for row in rows], dtype=np.float64)
    for threshold in TAIL_THRESHOLDS:
        result[f"max_abs_log_ratio_gt_{threshold:g}_fraction"] = float(np.mean(tails > threshold)) if tails.size else None
    return result


def underflow_stats(rows):
    counts = sum(int(row["ratio_underflow_count"]) for row in rows)
    samples = sum(int(row["ratio_sample_count"]) for row in rows)
    fractions = [float(row["ratio_underflow_fraction"]) for row in rows]
    return {
        "ratio_underflow_fraction_mean": statistics.mean(fractions) if fractions else None,
        "ratio_underflow_fraction_median": statistics.median(fractions) if fractions else None,
        "ratio_underflow_fraction_max": max(fractions) if fractions else None,
        "ratio_underflow_count_sum": counts, "ratio_sample_count_sum": samples,
        "pooled_underflow_count_over_sample_count": counts / samples if samples else None,
    }


def select_window(rows, bounds):
    lower, upper = bounds
    return [row for row in rows if lower < int(row["sampled_steps"]) <= upper]


def exact_pairs(control, treatment):
    c = {int(row["sampled_steps"]): row for row in control}
    t = {int(row["sampled_steps"]): row for row in treatment}
    if len(c) != len(control) or len(t) != len(treatment):
        raise RuntimeError("duplicate optimization sampled_steps")
    return {
        "control_only": sorted(set(c) - set(t)), "treatment_only": sorted(set(t) - set(c)),
        "intersection": sorted(set(c) & set(t)), "pairs": [(step, c[step], t[step]) for step in sorted(set(c) & set(t))],
    }


def evaluation_rows_exact(rows):
    by_step = {int(row["sampled_steps"]): row for row in rows}
    missing = [step for step in EVALUATION_STEPS if step not in by_step]
    extra = sorted(set(by_step) - set(EVALUATION_STEPS))
    if missing or extra:
        raise RuntimeError(f"evaluation grid mismatch; missing={missing}, extra={extra}")
    return [by_step[step] for step in EVALUATION_STEPS]


def eval_metrics(row):
    result = {key: float(row[column]) for key, column in EVAL_FIELDS.items()}
    result["Q2"] = None if result["W1"] == 0 else result["W2"] / result["W1"]
    result["Q3"] = None if result["W2"] == 0 else result["W3"] / result["W2"]
    return result


def treatment_excess_label(seed_summaries):
    flags = {}
    for seed, values in seed_summaries.items():
        flags[str(seed)] = values["median_delta_kl"] > .005 and values["delta_fraction_kl_gt_05"] >= .10
    count = sum(flags.values())
    return ("STRONG" if count >= 2 else "WEAK" if count == 0 else "MIXED"), flags


def kl_instability_label(rows_by_seed):
    flags = {str(seed): sum(float(row["approx_kl"]) > .05 for row in rows) >= 2 for seed, rows in rows_by_seed.items()}
    return ("YES" if sum(flags.values()) >= 2 else "NO_OR_MIXED"), flags


def flatten(prefix, stats):
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def run_window_row(method, seed, name, rows):
    kl = [float(row["approx_kl"]) for row in rows]
    result = {"method": method, "seed": seed, "window": name, "row_count": len(rows)}
    result.update(flatten("kl", basic_stats(kl)))
    result.update({f"kl_fraction_{key}": value for key, value in threshold_fractions(kl).items()})
    for threshold in (.05, .10, .20):
        result[f"kl_ge_{threshold:g}_steps"] = json.dumps([int(row["sampled_steps"]) for row in rows if float(row["approx_kl"]) >= threshold])
    result.update(ratio_tail_stats(rows))
    result.update(underflow_stats(rows))
    for prefix, key in (
        ("actor_preclip", "actor_grad_clip_preclip_norm"), ("actor_postclip", "actor_grad_clip_postclip_norm"),
        ("actor_exact_scale", "actor_grad_clip_exact_scale"), ("actor_pressure", "actor_grad_clip_pressure_fraction"),
        ("critic_preclip", "actor_grad_clip_critic_preclip_norm"), ("critic_postclip", "actor_grad_clip_critic_postclip_norm"),
        ("critic_exact_scale", "actor_grad_clip_critic_exact_scale"), ("critic_pressure", "actor_grad_clip_critic_pressure_fraction"),
        ("entropy", "entropy"), ("value_loss", "value_loss"), ("actor_loss", "actor_loss"),
    ):
        result.update(flatten(prefix, basic_stats((float(row[key]) for row in rows), include_max=False)))
    return result


def protocol_validate(method, seed, path, source_state, source_sha):
    required = ("run_config.json", "run_summary.json", "branch_from.json", "algorithm_config.yaml", "evaluation_history.csv", "optimization_metrics.jsonl", "train.log", "final.pt")
    missing = [name for name in required if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"missing files for {path}: {missing}")
    run = read_json(path / "run_config.json")
    summary = read_json(path / "run_summary.json")
    branch = read_json(path / "branch_from.json")
    config = yaml.safe_load((path / "algorithm_config.yaml").read_text(encoding="utf-8"))
    optimization = read_jsonl(path / "optimization_metrics.jsonl")
    evaluation = evaluation_rows_exact(read_csv(path / "evaluation_history.csv"))
    final_state = torch.load(path / "final.pt", map_location="cpu", weights_only=False)
    expected_method = "actor_grad_clip_05_control" if method == "Control05" else "actor_grad_clip_10"
    checks = {
        "seed": int(run["seed"]) == seed,
        "development_method": run["development_method"] == expected_method,
        "parent_checkpoint_sha": branch["parent_checkpoint_sha256"] == source_sha,
        "source_sampled_steps": int(branch["source_sampled_steps"]) == SOURCE_STEP == int(source_state["sampled_steps"]),
        "target_run_config": int(run["total_sampled_steps"]) == TARGET_STEP,
        "target_run_summary": int(summary["sampled_steps"]) == TARGET_STEP,
        "target_final_checkpoint": int(final_state["sampled_steps"]) == TARGET_STEP,
        "environment": run["environment_variant"] == "persistent_wave_v2",
        "num_envs": int(run["num_envs"]) == 24,
        "rollout": int(config["training"]["rollout_steps"]) == 256,
        "minibatch": int(config["training"]["minibatch_size"]) == 512,
        "ppo_epochs": int(config["training"]["ppo_epochs"]) == 10,
        "critic_lr_config": float(config["training"]["critic_learning_rate"]) == 3e-4,
        "actor_lr_runtime": all(float(row["actor_learning_rate"]) == 1e-4 for row in optimization),
        "critic_lr_runtime": all(float(row["critic_learning_rate"]) == 3e-4 for row in optimization),
        "actor_clip": float(run["actor_max_grad_norm"]) == (.5 if method == "Control05" else 1.0),
        "critic_clip": float(run["critic_max_grad_norm"]) == .5,
        "optimization_rows": len(optimization) == 49,
        "evaluation_episodes": all(int(row["evaluation_episodes"]) == 50 for row in evaluation),
        "evaluation_44m": all((int(row["evaluation_seed_base"]), int(row["evaluation_seed_end"])) == (44_000_000, 44_000_049) for row in evaluation),
        "no_45m": all(not (45_000_000 <= int(row["evaluation_seed_base"]) <= 45_000_199 or 45_000_000 <= int(row["evaluation_seed_end"]) <= 45_000_199) for row in evaluation),
        "ten_epochs_each_row": all(int(row["ppo_epochs_executed"]) == 10 for row in optimization),
        "minibatch_arithmetic_each_row": all(int(row["ppo_minibatches_executed"]) == 10 * int(row["ppo_minibatches_per_epoch"]) for row in optimization),
        "actor_steps_each_row": all(int(row["actor_optimizer_steps_this_update"]) == int(row["ppo_minibatches_executed"]) for row in optimization),
        "critic_steps_each_row": all(int(row["critic_optimizer_steps_this_update"]) == int(row["ppo_minibatches_executed"]) for row in optimization),
        "actor_steps_added": int(final_state["actor_updates"]) - int(source_state["actor_updates"]) == 5860,
        "critic_steps_added": int(final_state["critic_updates"]) - int(source_state["critic_updates"]) == 5860,
        "ppo_updates_added": int(final_state["ppo_updates"]) - int(source_state["ppo_updates"]) == 49,
    }
    if not all(checks.values()):
        raise RuntimeError(f"protocol failure {method} seed{seed}: {[key for key, value in checks.items() if not value]}")
    manifest = {row["path"]: row["sha256"] for row in run["runtime_source_manifest_files"]}
    return {
        "status": "PASS", "checks": checks, "source_checkpoint_sha256": source_sha,
        "runtime_source_manifest_sha256": run["runtime_source_manifest_sha256"],
        "trainer_sha256": manifest["algorithm/modular_mappo/trainer.py"],
        "optimization_rows": len(optimization), "actor_steps_added": 5860,
        "critic_steps_added": 5860, "ppo_updates_added": 49,
        "checkpoint_metadata_read_on_cpu": True,
    }, optimization, evaluation


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    data, evaluations, protocol = {}, {}, {}
    source_states = {}
    for seed in SEEDS:
        source = ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt"
        source_states[seed] = torch.load(source, map_location="cpu", weights_only=False)
        source_sha = sha256(source)
        for method, path_fn in RUNS.items():
            key = (method, seed)
            protocol[f"{method}_seed{seed}"], data[key], evaluations[key] = protocol_validate(method, seed, path_fn(seed), source_states[seed], source_sha)
    runtime_shas = {row["runtime_source_manifest_sha256"] for row in protocol.values()}
    trainer_shas = {row["trainer_sha256"] for row in protocol.values()}
    if len(runtime_shas) != 1 or len(trainer_shas) != 1:
        raise RuntimeError("six runs do not share one runtime/trainer SHA")

    step_grid = {}
    paired_steps = []
    for seed in SEEDS:
        pairing = exact_pairs(data[("Control05", seed)], data[("Treatment10", seed)])
        step_grid[str(seed)] = {key: value for key, value in pairing.items() if key != "pairs"}
        step_grid[str(seed)]["paired_optimization_rows"] = len(pairing["pairs"])
        if pairing["control_only"] or pairing["treatment_only"] or len(pairing["pairs"]) != 49:
            raise RuntimeError(f"exact optimization alignment failed for seed{seed}")
        for step, control, treatment in pairing["pairs"]:
            row = {"seed": seed, "sampled_steps": step}
            for short, field in PAIR_METRICS.items():
                row[f"control_{short}"] = float(control[field])
                row[f"treatment_{short}"] = float(treatment[field])
                row[f"delta_{short}"] = float(treatment[field]) - float(control[field])
            paired_steps.append(row)

    stability_rows = []
    for (method, seed), rows in data.items():
        stability_rows.append(run_window_row(method, seed, "ALL", rows))
        for name, bounds in WINDOWS.items():
            stability_rows.append(run_window_row(method, seed, name, select_window(rows, bounds)))

    paired_window_rows = []
    for seed in SEEDS:
        seed_rows = [row for row in paired_steps if row["seed"] == seed]
        for name, bounds in {"ALL": (SOURCE_STEP, TARGET_STEP), **WINDOWS}.items():
            selected = [row for row in seed_rows if bounds[0] < row["sampled_steps"] <= bounds[1]]
            result = {"seed": seed, "window": name, "row_count": len(selected)}
            for short in PAIR_METRICS:
                values = [row[f"delta_{short}"] for row in selected]
                result.update({f"delta_{short}_{key}": value for key, value in basic_stats(values, include_max=False).items()})
                result[f"delta_{short}_positive_fraction"] = float(np.mean(np.asarray(values) > 0)) if values else None
            paired_window_rows.append(result)

    evaluation_rows = []
    eval_lookup = {}
    for seed in SEEDS:
        control = {int(row["sampled_steps"]): eval_metrics(row) for row in evaluations[("Control05", seed)]}
        treatment = {int(row["sampled_steps"]): eval_metrics(row) for row in evaluations[("Treatment10", seed)]}
        for step in EVALUATION_STEPS:
            row = {"seed": seed, "sampled_steps": step}
            for method, metrics in (("control", control[step]), ("treatment", treatment[step])):
                row.update({f"{method}_{key}": value for key, value in metrics.items()})
            for key in (*EVAL_FIELDS, "Q2", "Q3"):
                left, right = treatment[step][key], control[step][key]
                row[f"delta_{key}"] = None if left is None or right is None else left - right
            evaluation_rows.append(row)
            eval_lookup[(seed, step)] = row

    alignment_rows = []
    performance_drop_intervals = []
    counterexamples = []
    for seed in SEEDS:
        for method in RUNS:
            metrics = {int(row["sampled_steps"]): eval_metrics(row) for row in evaluations[(method, seed)]}
            for start, end in zip(EVALUATION_STEPS[:-1], EVALUATION_STEPS[1:]):
                optimization = [row for row in data[(method, seed)] if start < int(row["sampled_steps"]) <= end]
                changes = {key: metrics[end][key] - metrics[start][key] for key in ("AW", "W1", "W2", "W3", "Boundary", "Ground")}
                kl_values = [float(row["approx_kl"]) for row in optimization]
                drop = changes["AW"] <= -.20 or changes["W3"] <= -.10
                aligned = {
                    "seed": seed, "method": method, "start_step": start, "end_step": end,
                    **{f"delta_{key}": value for key, value in changes.items()},
                    "performance_drop_interval": drop, "optimization_row_count": len(optimization),
                    "kl_mean": statistics.mean(kl_values), "kl_median": statistics.median(kl_values),
                    "kl_max": max(kl_values), "kl_gt_05_fraction": float(np.mean(np.asarray(kl_values) > .05)),
                    "clip_fraction_median": statistics.median(float(row["clip_fraction"]) for row in optimization),
                    "ratio_std_median": statistics.median(float(row["ratio_std"]) for row in optimization),
                    "max_abs_log_ratio_max": max(float(row["max_abs_log_ratio"]) for row in optimization),
                    "actor_preclip_median": statistics.median(float(row["actor_grad_clip_preclip_norm"]) for row in optimization),
                    "actor_exact_scale_median": statistics.median(float(row["actor_grad_clip_exact_scale"]) for row in optimization),
                    "critic_preclip_median": statistics.median(float(row["actor_grad_clip_critic_preclip_norm"]) for row in optimization),
                    "critic_exact_scale_median": statistics.median(float(row["actor_grad_clip_critic_exact_scale"]) for row in optimization),
                    "entropy_median": statistics.median(float(row["entropy"]) for row in optimization),
                    "value_loss_median": statistics.median(float(row["value_loss"]) for row in optimization),
                }
                alignment_rows.append(aligned)
                if drop:
                    performance_drop_intervals.append(aligned)
                    if aligned["kl_gt_05_fraction"] == 0:
                        counterexamples.append({"type": "PERFORMANCE_DROP_WITHOUT_KL_GT_05", **aligned})
                if changes["AW"] >= .20 and aligned["kl_gt_05_fraction"] > 0:
                    counterexamples.append({"type": "PERFORMANCE_RECOVERY_DESPITE_KL_GT_05", **aligned})

    all_window = {(row["method"], row["seed"]): row for row in stability_rows if row["window"] == "ALL"}
    all_pair = {(row["seed"], row["window"]): row for row in paired_window_rows}
    excess_inputs = {
        seed: {
            "median_delta_kl": all_pair[(seed, "ALL")]["delta_kl_median"],
            "delta_fraction_kl_gt_05": all_window[("Treatment10", seed)]["kl_fraction_gt_0.05"] - all_window[("Control05", seed)]["kl_fraction_gt_0.05"],
        } for seed in SEEDS
    }
    treatment_excess, excess_flags = treatment_excess_label(excess_inputs)
    control_instability, control_flags = kl_instability_label({seed: data[("Control05", seed)] for seed in SEEDS})
    treatment_instability, treatment_flags = kl_instability_label({seed: data[("Treatment10", seed)] for seed in SEEDS})

    # Explicit cross-branch counterexample: Treatment is worse at the endpoint in
    # two seeds, while the predefined excess-KL rule may still be absent.
    worse_final = [seed for seed in SEEDS if eval_lookup[(seed, TARGET_STEP)]["delta_AW"] <= -.20 or eval_lookup[(seed, TARGET_STEP)]["delta_W3"] <= -.10]
    if worse_final and treatment_excess != "STRONG":
        counterexamples.append({"type": "WORSE_TREATMENT_ENDPOINT_WITHOUT_STRONG_TREATMENT_EXCESS_KL", "seeds": worse_final})

    both_instability = control_instability == "YES" and treatment_instability == "YES"
    if counterexamples:
        diagnosis = "CASE_C_KL_NOT_ALIGNED_WITH_PERFORMANCE"
    elif treatment_excess == "STRONG":
        diagnosis = "CASE_A_TREATMENT_EXCESS_KL"
    elif both_instability:
        diagnosis = "CASE_B_BOTH_HAVE_KL_INSTABILITY"
    else:
        diagnosis = "CASE_D_INSUFFICIENT_EVIDENCE"

    seed_cases = {}
    for seed in SEEDS:
        case = {
            "seed": seed,
            "evaluations": [row for row in evaluation_rows if row["seed"] == seed],
            "alignment": [row for row in alignment_rows if row["seed"] == seed],
            "control_kl_ge_05_steps": [int(row["sampled_steps"]) for row in data[("Control05", seed)] if float(row["approx_kl"]) >= .05],
            "treatment_kl_ge_05_steps": [int(row["sampled_steps"]) for row in data[("Treatment10", seed)] if float(row["approx_kl"]) >= .05],
            "paired_all": all_pair[(seed, "ALL")],
        }
        if seed == 5302:
            control_max = max(data[("Control05", seed)], key=lambda row: float(row["approx_kl"]))
            case["control_max_kl_exact_row"] = {key: control_max[key] for key in (
                "sampled_steps", "approx_kl", "clip_fraction", "ratio_underflow_fraction",
                "log_ratio_min", "log_ratio_max", "max_abs_log_ratio",
                "actor_grad_clip_exact_scale", "actor_grad_clip_critic_exact_scale")}
        if seed == 5303:
            case["final_tail_note"] = "SEGMENT_4 contains one optimization row; it is a single observation, not a stable distribution."
        seed_cases[seed] = case

    semantics = {
        "approx_kl": "Pooled alive-sample mean of (ratio - 1) - log_ratio across every minibatch in one complete PPO update (10 epochs). It is a PPO old-vs-current log-prob shift proxy, not a deployment-distribution KL bound.",
        "clip_fraction": "Pooled alive-sample fraction with abs(ratio-1)>0.2 across the complete PPO update.",
        "ratio_summary": "ratio_mean/std/p1/p50/p99/min/max are computed over the pooled alive-sample ratio values from all minibatches and all 10 epochs in one PPO update.",
        "log_ratio_extrema": "log_ratio_min/max and max_abs_log_ratio are pooled-sample extrema within one PPO update; one extreme may represent very few samples.",
        "ratio_underflow_fraction": "Pooled fraction of samples where exp(log_ratio) is exactly zero while log_ratio remains finite.",
        "actor_gradient": "actor_grad_clip_preclip_norm is clip_grad_norm_'s total preclip norm per minibatch, aggregated as an unweighted minibatch mean; postclip is the measured norm after clipping; exact_scale is post/pre per minibatch then averaged; pressure is the fraction of minibatches whose preclip norm exceeded the Actor limit.",
        "critic_gradient": "Same meanings as Actor diagnostics, with the Critic limit fixed at 0.5 in both branches. Later branch differences can reflect different collected data.",
        "entropy": "Monte Carlo estimate: negative squashed log-probability of a reparameterized sample, alive-sample weighted across minibatches; it is not analytic Gaussian entropy and may be negative.",
        "loss_aggregation": "Actor/value loss, entropy, KL and clip fraction are alive-sample weighted or pooled. Gradient diagnostics are unweighted minibatch means. One JSONL row is one complete PPO update, not one epoch or minibatch.",
        "ppo_ratio_clipping_vs_gradient_clipping": "PPO clip_ratio=0.2 constrains the surrogate objective; Actor norm clipping constrains each minibatch gradient norm; Critic norm clipping independently constrains each Critic minibatch gradient. None directly bounds cumulative 10-epoch KL.",
        "opt_warn": "Runner emits OPT_WARN when core metrics are nonfinite, aggregate approx_kl>=0.05, or aggregate underflow_fraction>0.",
    }
    availability = {
        "optimization_step_level": "AVAILABLE", "evaluation_checkpoint_level": "AVAILABLE",
        "per_epoch_KL": "UNAVAILABLE", "PER_EPOCH_KL_TRAJECTORY": "UNAVAILABLE",
        "per_minibatch_KL": "UNAVAILABLE", "per_epoch_actor_critic_norm": "UNAVAILABLE",
        "per_minibatch_clipped_fraction_distribution": "UNAVAILABLE",
        "per_wave_KL": "UNAVAILABLE", "per_wave_actor_gradient": "UNAVAILABLE",
        "note": "The 10-epoch pooled aggregates cannot recover which epoch or minibatch first became abnormal.",
    }
    limitations = [
        "Only three training seeds are independent repeats; 49 rows are autocorrelated within-run observations.",
        "Only four 50-episode evaluation checkpoints exist; episodes are not independent training repeats.",
        "Optimization metrics pool ten epochs, so epoch-level onset and minibatch distributions are unrecoverable.",
        "Extreme ratio/log-ratio values can arise from a tiny number of samples and do not alone prove global collapse.",
        "Evaluation/optimization alignment is descriptive and cannot establish causality.",
        "Gradient norm clipping constrains individual updates, not cumulative Adam displacement or KL.",
    ]
    report = {
        "status": "FIXED10_PPO_STABILITY_AUDIT_COMPLETE", "protocol": "6/6 PASS",
        "exact_step_alignment": "49/49 per seed PASS", "replication_unit": "training_seed", "n": 3,
        "runtime_source_manifest_sha256": next(iter(runtime_shas)), "trainer_sha256": next(iter(trainer_shas)),
        "kl_instability_present": {"Control05": control_instability, "Treatment10": treatment_instability,
            "control_seed_flags": control_flags, "treatment_seed_flags": treatment_flags},
        "treatment_excess_kl": treatment_excess, "treatment_excess_seed_flags": excess_flags,
        "treatment_excess_inputs": excess_inputs, "performance_drop_intervals": performance_drop_intervals,
        "counterexamples": counterexamples, "PPO_INSTABILITY_AS_CAUSAL_ROOT": "NOT_IDENTIFIED",
        "NEXT_PPO_DIAGNOSIS": diagnosis, "metric_semantics": semantics,
        "available_granularity": availability, "limitations": limitations,
        "per_run_all_window": {f"{method}_seed{seed}": all_window[(method, seed)] for method in RUNS for seed in SEEDS},
        "paired_all_window": {f"seed{seed}": all_pair[(seed, "ALL")] for seed in SEEDS},
        "offline_guards": {"training": False, "evaluation": False, "environment_created": False,
            "rollout_created": False, "policy_inference": False, "backward": False,
            "optimizer_step": False, "cuda": False, "uses_45m": False,
            "checkpoint_metadata_cpu_only": True},
    }

    OUT.mkdir(parents=True)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "protocol_validation.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (OUT / "metric_semantics.json").write_text(json.dumps(semantics, indent=2), encoding="utf-8")
    (OUT / "metric_availability.json").write_text(json.dumps(availability, indent=2), encoding="utf-8")
    (OUT / "optimization_step_grid.json").write_text(json.dumps(step_grid, indent=2), encoding="utf-8")
    write_csv(OUT / "ppo_stability_windows.csv", stability_rows)
    write_csv(OUT / "paired_step_diagnostics.csv", paired_steps)
    write_csv(OUT / "paired_window_summary.csv", paired_window_rows)
    write_csv(OUT / "evaluation_dynamics.csv", evaluation_rows)
    write_csv(OUT / "evaluation_optimization_alignment.csv", alignment_rows)
    for seed, case in seed_cases.items():
        (OUT / f"seed{seed}_case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    decision = [
        "FIXED10 PPO OPTIMIZATION STABILITY AUDIT",
        "Protocol: 6/6 PASS", "Exact pairing: 49/49 per seed PASS",
        f"KL_INSTABILITY_PRESENT Control05={control_instability} Treatment10={treatment_instability}",
        f"TREATMENT_EXCESS_KL={treatment_excess}", f"NEXT_PPO_DIAGNOSIS={diagnosis}",
        "PPO_INSTABILITY_AS_CAUSAL_ROOT=NOT_IDENTIFIED",
        "PER_EPOCH_KL_TRAJECTORY=UNAVAILABLE",
        "No training, evaluation, environment, rollout, inference, backward, optimizer step, CUDA, or 45M access occurred.",
    ]
    (OUT / "decision_support.txt").write_text("\n".join(decision) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "protocol": report["protocol"],
        "exact_step_alignment": report["exact_step_alignment"],
        "KL_INSTABILITY_PRESENT": report["kl_instability_present"],
        "TREATMENT_EXCESS_KL": treatment_excess, "NEXT_PPO_DIAGNOSIS": diagnosis,
        "output": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
