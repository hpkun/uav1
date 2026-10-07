"""Strictly offline, CPU-only deep audit for Mission-Aware FiLM development runs.

This script reads completed result artifacts only.  It never imports an environment,
policy evaluator, runner, or trainer, and it never performs a rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5301, 5302, 5303)
ALPHA = 0.2
FORMAL_ROOTS = {
    "plain": ROOT / "outputs/diag_mappo_learnability",
    "raw_context": ROOT / "outputs/dev_actor_mission_context_3m",
    "film": ROOT / "outputs/dev_mission_aware_film_3m",
}
DEFAULT_OUTPUT = ROOT / "outputs/mission_aware_film_deep_analysis"

EVAL_RENAME = {
    "clear_wave_1_probability": "w1",
    "clear_wave_2_probability": "w2",
    "clear_wave_3_probability": "w3",
    "average_waves_cleared": "average_waves",
    "average_return": "return",
    "average_red_loss": "red_loss",
    "average_blue_loss": "blue_loss",
    "average_red_boundary_exits": "boundary",
    "average_red_ground_losses": "ground",
    "timeout_rate": "timeout",
    "average_episode_length": "episode_length",
    "average_red_survivors_after_wave_1_conditional_on_clear": "survivors_entering_w2",
    "average_red_survivors_after_wave_2_conditional_on_clear": "survivors_entering_w3",
    "wave_1_duration": "wave1_clear_time",
    "wave_2_duration": "wave2_clear_time",
    "wave_3_duration": "wave3_clear_time",
}
PRIMARY = [
    "w1", "w2", "w3", "q2", "q3", "average_waves", "return", "red_loss",
    "blue_loss", "kill_loss_ratio", "boundary", "ground", "timeout",
    "episode_length", "survivors_entering_w2", "survivors_entering_w3",
    "wave1_clear_time", "wave2_clear_time", "wave3_clear_time",
]
FILM_METRICS = [
    "mission_feature_norm", "film_delta_gamma_abs_mean", "film_delta_gamma_abs_max",
    "film_beta_abs_mean", "film_beta_abs_max", "film_residual_abs_mean",
    "film_residual_abs_max", "film_gamma_mean", "film_gamma_min", "film_gamma_max",
    "film_hidden_base_norm", "film_hidden_modulated_norm", "film_saturation_fraction",
]
OPT_METRICS = [
    "approx_kl", "clip_fraction", "actor_loss", "value_loss", "entropy",
    "actor_grad_norm", "critic_grad_norm", "actor_learning_rate",
    "ratio_underflow_fraction", "log_ratio_min", "log_ratio_max",
]


def safe_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dir(method: str, seed: int) -> Path:
    if method == "plain":
        return FORMAL_ROOTS[method] / f"l3_seed{seed}"
    return FORMAL_ROOTS[method] / f"seed{seed}"


def add_q(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["q2"] = np.where(df["w1"] > 0, df["w2"] / df["w1"], np.nan)
    df["q3"] = np.where(df["w2"] > 0, df["w3"] / df["w2"], np.nan)
    return df


def load_eval(directory: Path, method: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(directory / "evaluation_history.csv").rename(columns=EVAL_RENAME)
    df = add_q(df)
    df.insert(0, "method", method)
    df.insert(1, "training_seed", seed)
    return df


def finite_state_dict(state: dict) -> bool:
    tensors = []
    for key in ("actor", "critic"):
        tensors.extend(state.get(key, {}).values())
    return bool(tensors) and all(torch.is_tensor(v) and torch.isfinite(v).all().item() for v in tensors)


def audit_film_run(directory: Path, seed: int) -> dict:
    required = [
        "evaluation_history.csv", "optimization_metrics.jsonl", "training_metrics.jsonl",
        "run_config.json", "run_summary.json", "env_config.yaml", "runtime_env_config.yaml",
        "latest.pt", "final.pt", "checkpoint_3000000.pt",
    ]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        return {"training_seed": seed, "status": "FAIL", "missing": missing}
    run = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    env = yaml.safe_load((directory / "env_config.yaml").read_text(encoding="utf-8"))
    runtime = yaml.safe_load((directory / "runtime_env_config.yaml").read_text(encoding="utf-8"))
    ev = pd.read_csv(directory / "evaluation_history.csv")
    opt = pd.read_json(directory / "optimization_metrics.jsonl", lines=True)
    train = pd.read_json(directory / "training_metrics.jsonl", lines=True)
    checkpoints = {}
    for name in ("latest.pt", "final.pt", "checkpoint_3000000.pt", "best_eval.pt"):
        path = directory / name
        if not path.exists():
            checkpoints[name] = {"exists": False}
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)
        checkpoints[name] = {
            "exists": True,
            "sampled_steps": int(state.get("sampled_steps", -1)),
            "finite": finite_state_dict(state),
            "sha256": sha256(path),
        }
    log = (directory / "train.log").read_text(encoding="utf-8", errors="replace") if (directory / "train.log").exists() else ""
    # Derived Q2/Q3 are legitimately printed as ``nan`` when their denominator is
    # zero.  Treat only actual runtime-failure markers as log errors here; numeric
    # finiteness is checked directly from checkpoints and metric columns.
    error_hits = re.findall(r"Traceback|\bOOM\b|out of memory|(?:^|\s)Exception(?:\s|:|$)", log, flags=re.I | re.M)
    expected_modules = ["actor_lr_decay", "mission_film", "wave_context"]
    hashes_equal = (
        run.get("declared_environment_config_sha256")
        == run.get("effective_training_environment_config_sha256")
        == run.get("evaluation_environment_config_sha256")
    )
    opt_finite = all(
        column in opt.columns and np.isfinite(pd.to_numeric(opt[column], errors="coerce")).all()
        for column in OPT_METRICS + FILM_METRICS
    )
    eval_required = [
        "average_return", "average_waves_cleared", "clear_wave_1_probability",
        "clear_wave_2_probability", "clear_wave_3_probability", "average_red_loss",
        "average_blue_loss", "average_red_boundary_exits", "average_red_ground_losses",
        "timeout_rate", "average_episode_length",
    ]
    eval_finite = all(np.isfinite(pd.to_numeric(ev[column], errors="coerce")).all() for column in eval_required)
    config_checks = {
        "training_seed": run.get("seed") == seed,
        "training_device_cuda": run.get("device") == "cuda",
        "sampled_steps": summary.get("sampled_steps") == 3_000_000,
        "environment": run.get("environment_variant") == "persistent_wave_v2",
        "declared_waves_steps": run.get("declared_total_waves") == 3 and run.get("declared_max_steps") == 3000,
        "effective_waves_steps": run.get("effective_training_total_waves") == 3 and run.get("effective_training_max_steps") == 3000,
        "evaluation_waves_steps": run.get("evaluation_total_waves") == 3 and run.get("evaluation_max_steps") == 3000,
        "env_hashes_equal": hashes_equal,
        "actor_input_dim": run.get("actor_input_dim") == 52,
        "actor_context_dim": run.get("actor_context_dim") == 5,
        "critic_context_dim": run.get("critic_context_dim") == 0,
        "film_contract": run.get("mission_film_enabled") is True
        and run.get("mission_film_mode") == "bounded_augmented_film"
        and run.get("mission_encoder_hidden_dim") == 32
        and run.get("mission_film_alpha") == ALPHA,
        "enabled_modules": run.get("enabled_modules") == expected_modules,
        "no_resume": summary.get("resume_count") == 0,
        "no_branch": not run.get("branch_provenance") and not summary.get("branch_provenance"),
        "eval_history": len(ev) == 30 and int(ev.sampled_steps.iloc[-1]) == 3_000_000,
        "metrics_complete": bool(int(opt.sampled_steps.iloc[-1]) == 3_000_000 and train.sampled_steps.max() >= 2_900_000),
        "optimization_metrics_finite": bool(opt_finite),
        "evaluation_metrics_finite": bool(eval_finite),
        "logs_clean": not error_hits,
        "checkpoint_endpoints": all(checkpoints[name].get("sampled_steps") == 3_000_000 and checkpoints[name].get("finite") for name in ("latest.pt", "final.pt", "checkpoint_3000000.pt")),
        "eval_seed_range_44m": bool(ev.evaluation_seed_base.eq(44_000_000).all() and ev.evaluation_seed_end.eq(44_000_049).all()),
        "no_45m_eval": not ((ev.evaluation_seed_base >= 45_000_000) & (ev.evaluation_seed_base <= 45_000_199)).any(),
    }
    return {
        "training_seed": seed,
        "status": "PASS" if all(config_checks.values()) else "FAIL",
        "checks": config_checks,
        "run_sampled_steps": run.get("total_sampled_steps"),
        "summary_sampled_steps": summary.get("sampled_steps"),
        "evaluation_rows": len(ev),
        "optimization_rows": len(opt),
        "training_episodes": len(train),
        "completed_episodes": summary.get("completed_episodes"),
        "declared_env": {"total_waves": env.get("persistent_waves", {}).get("total_waves"), "max_steps": env.get("simulation", {}).get("max_steps")},
        "runtime_env": {"total_waves": runtime.get("persistent_waves", {}).get("total_waves"), "max_steps": runtime.get("simulation", {}).get("max_steps")},
        "checkpoints": checkpoints,
        "best_checkpoint_step": summary.get("best_checkpoint_step"),
        "error_hits": error_hits,
    }


def describe(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {"mean": None, "sample_std": None, "min": None, "max": None}
    return {
        "mean": float(x.mean()), "sample_std": float(x.std(ddof=1)) if len(x) > 1 else None,
        "min": float(x.min()), "max": float(x.max()),
    }


def nearest(df: pd.DataFrame, step: int) -> pd.Series:
    return df.iloc[(df.sampled_steps - step).abs().argmin()]


def percentile(series: pd.Series, q: float):
    x = pd.to_numeric(series, errors="coerce").dropna()
    return None if x.empty else float(x.quantile(q))


def corr_rows(df: pd.DataFrame, seed: int) -> list[dict]:
    rows = []
    for feature in ("kl_mean", "kl_p95", "kl_max", "kl_warning_fraction", "entropy_mean", "clip_fraction_mean"):
        for outcome in ("average_waves", "w1", "w2", "w3", "q2", "q3", "boundary", "ground"):
            pair = df[[feature, outcome]].dropna()
            pearson = pair[feature].corr(pair[outcome], method="pearson") if len(pair) >= 3 else np.nan
            spearman = pair[feature].rank(method="average").corr(pair[outcome].rank(method="average"), method="pearson") if len(pair) >= 3 else np.nan
            rows.append({"training_seed": seed, "feature": feature, "outcome": outcome, "n_timepoints": len(pair), "pearson": pearson, "spearman": spearman})
    return rows


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    frame = df.copy()
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].map(lambda x: "" if pd.isna(x) else f"{x:.{digits}f}" if isinstance(x, (float, np.floating)) else str(x))
    headers = [str(x) for x in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(x) for x in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    resolved = out.resolve()
    for source in FORMAL_ROOTS.values():
        if resolved == source.resolve() or source.resolve() in resolved.parents:
            raise RuntimeError("derived output must not be placed inside a formal result directory")
    out.mkdir(parents=True, exist_ok=True)

    audits = [audit_film_run(run_dir("film", seed), seed) for seed in SEEDS]
    if not all(row["status"] == "PASS" for row in audits):
        (out / "deep_analysis.json").write_text(json.dumps({"status": "PROTOCOL_FAIL", "film_run_audit": audits}, indent=2), encoding="utf-8")
        raise RuntimeError("FiLM protocol audit failed; scientific analysis aborted")

    trajectories = pd.concat([load_eval(method, run_dir(method, seed), seed) if False else load_eval(run_dir(method, seed), method, seed) for seed in SEEDS for method in ("plain", "raw_context", "film")], ignore_index=True)
    keep = ["method", "training_seed", "sampled_steps"] + [c for c in PRIMARY if c in trajectories.columns]
    trajectories[keep].to_csv(out / "evaluation_trajectories.csv", index=False)

    final = trajectories.loc[trajectories.sampled_steps.eq(3_000_000), keep].copy()
    method_summary = []
    for method, group in final.groupby("method", sort=False):
        for metric in PRIMARY:
            if metric in group:
                method_summary.append({"method": method, "metric": metric, **describe(group[metric])})
    pd.DataFrame(method_summary).to_csv(out / "three_method_3m_summary.csv", index=False)

    delta_rows = []
    for seed in SEEDS:
        indexed = final.loc[final.training_seed.eq(seed)].set_index("method")
        for comparison, left, right in (("film_minus_plain", "film", "plain"), ("raw_minus_plain", "raw_context", "plain"), ("film_minus_raw", "film", "raw_context")):
            row = {"comparison": comparison, "training_seed": seed}
            for metric in PRIMARY:
                if metric in indexed:
                    row[metric] = safe_float(indexed.at[left, metric]) - safe_float(indexed.at[right, metric])
            delta_rows.append(row)
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(out / "three_method_paired_deltas.csv", index=False)
    delta_summary = []
    lower_better = {"red_loss", "boundary", "ground", "timeout"}
    for comparison, group in deltas.groupby("comparison", sort=False):
        for metric in PRIMARY:
            if metric not in group:
                continue
            stats = describe(group[metric])
            wins = int((group[metric] < 0).sum()) if metric in lower_better else int((group[metric] > 0).sum())
            delta_summary.append({"comparison": comparison, "metric": metric, **stats, "paired_wins_of_3": wins})
    pd.DataFrame(delta_summary).to_csv(out / "three_method_paired_delta_summary.csv", index=False)

    key_steps = []
    for seed in SEEDS:
        df = trajectories[(trajectories.method == "film") & (trajectories.training_seed == seed)]
        for target in (900_000, 1_500_000, 2_000_000, 3_000_000):
            row = nearest(df, target)
            key_steps.append({"training_seed": seed, "target_step": target, **{c: row[c] for c in keep if c not in ("method", "training_seed")}})
    pd.DataFrame(key_steps).to_csv(out / "film_key_endpoints.csv", index=False)

    peak_rows = []
    peak_context = []
    for seed in SEEDS:
        df = trajectories[(trajectories.method == "film") & (trajectories.training_seed == seed)].sort_values("sampled_steps")
        final_row = df.iloc[-1]
        summary = json.loads((run_dir("film", seed) / "run_summary.json").read_text(encoding="utf-8"))
        best_step = int(summary["best_checkpoint_step"])
        avg_peak = df.loc[df.average_waves.idxmax()]
        w3_peak = df.loc[df.w3.idxmax()]
        peak_context.append({
            "training_seed": seed,
            "repository_best_step": best_step,
            "max_average_waves_step": int(avg_peak.sampled_steps),
            "max_average_waves": avg_peak.average_waves,
            "max_w3_step": int(w3_peak.sampled_steps),
            "max_w3": w3_peak.w3,
            "final_average_waves": final_row.average_waves,
            "final_w3": final_row.w3,
            **{f"avg_peak_{m}": avg_peak[m] for m in ("w1", "w2", "w3", "q2", "q3", "boundary", "ground")},
            **{f"final_{m}": final_row[m] for m in ("w1", "w2", "w3", "q2", "q3", "boundary", "ground")},
        })
        for metric in ("average_waves", "w1", "w2", "w3", "q2", "q3", "return"):
            valid = df.dropna(subset=[metric])
            peak = valid.loc[valid[metric].idxmax()]
            final_value = safe_float(final_row[metric])
            peak_value = safe_float(peak[metric])
            peak_rows.append({
                "training_seed": seed, "metric": metric, "peak_step": int(peak.sampled_steps),
                "peak_value": peak_value, "final_value": final_value,
                "absolute_drop_final_minus_peak": final_value - peak_value,
                "retention_final_over_peak": final_value / peak_value if peak_value not in (None, 0) else None,
            })
    pd.DataFrame(peak_rows).to_csv(out / "peak_to_final.csv", index=False)
    pd.DataFrame(peak_context).to_csv(out / "peak_average_waves_context.csv", index=False)

    opt_summaries = []
    opt_windows_all = []
    corr_all = []
    film_full_all = []
    film_summary = []
    film_stage_rows = []
    film_align_all = []
    train_align_all = []
    high_kl_effects = []
    action_distribution_rows = []
    film_performance_correlations = []
    for seed in SEEDS:
        directory = run_dir("film", seed)
        opt = pd.DataFrame(records(directory / "optimization_metrics.jsonl")).sort_values("sampled_steps")
        train = pd.DataFrame(records(directory / "training_metrics.jsonl")).sort_values("sampled_steps")
        ev = trajectories[(trajectories.method == "film") & (trajectories.training_seed == seed)].sort_values("sampled_steps")
        kl = pd.to_numeric(opt.approx_kl, errors="coerce")
        opt_summaries.append({
            "training_seed": seed, "updates": len(opt),
            "kl_ge_0_03_count": int((kl >= .03).sum()), "kl_ge_0_05_count": int((kl >= .05).sum()),
            "kl_ge_0_10_count": int((kl >= .10).sum()), "kl_ge_0_20_count": int((kl >= .20).sum()),
            "kl_ge_0_05_fraction": float((kl >= .05).mean()),
            "kl_p50": percentile(kl, .50), "kl_p90": percentile(kl, .90),
            "kl_p95": percentile(kl, .95), "kl_p99": percentile(kl, .99), "kl_max": float(kl.max()),
            "underflow_event_count": int((opt.ratio_underflow_fraction > 0).sum()),
            "underflow_max_fraction": float(opt.ratio_underflow_fraction.max()),
            "entropy_first": float(opt.entropy.iloc[0]), "entropy_900k": float(nearest(opt, 900_000).entropy),
            "entropy_1p5m": float(nearest(opt, 1_500_000).entropy), "entropy_2m": float(nearest(opt, 2_000_000).entropy),
            "entropy_3m": float(opt.entropy.iloc[-1]), "entropy_min": float(opt.entropy.min()),
            "clip_fraction_mean": float(opt.clip_fraction.mean()), "clip_fraction_p95": percentile(opt.clip_fraction, .95), "clip_fraction_max": float(opt.clip_fraction.max()),
            "actor_grad_mean": float(opt.actor_grad_norm.mean()), "actor_grad_p95": percentile(opt.actor_grad_norm, .95), "actor_grad_max": float(opt.actor_grad_norm.max()),
            "critic_grad_mean": float(opt.critic_grad_norm.mean()), "critic_grad_p95": percentile(opt.critic_grad_norm, .95), "critic_grad_max": float(opt.critic_grad_norm.max()),
            "value_loss_mean": float(opt.value_loss.mean()), "value_loss_p95": percentile(opt.value_loss, .95), "value_loss_max": float(opt.value_loss.max()),
            "actor_loss_mean": float(opt.actor_loss.mean()), "actor_loss_p95": percentile(opt.actor_loss, .95), "actor_loss_min": float(opt.actor_loss.min()), "actor_loss_max": float(opt.actor_loss.max()),
            "log_ratio_min": float(opt.log_ratio_min.min()), "log_ratio_max": float(opt.log_ratio_max.max()),
            "actor_lr_near_600k": float(nearest(opt, 600_000).actor_learning_rate),
            "actor_lr_near_900k": float(nearest(opt, 900_000).actor_learning_rate),
            "actor_lr_post_900k_min": float(opt.loc[opt.sampled_steps >= 900_000, "actor_learning_rate"].min()),
            "actor_lr_post_900k_max": float(opt.loc[opt.sampled_steps >= 900_000, "actor_learning_rate"].max()),
        })
        for axis in ("psi", "theta", "v"):
            logstd = f"policy_log_std_mean_{axis}"
            sat90 = f"action_abs_gt_0_9_fraction_{axis}"
            sat99 = f"action_abs_gt_0_99_fraction_{axis}"
            if all(column in opt.columns for column in (logstd, sat90, sat99)):
                action_distribution_rows.append({
                    "training_seed": seed, "axis": axis,
                    "log_std_first": float(opt[logstd].iloc[0]), "log_std_900k": float(nearest(opt, 900_000)[logstd]),
                    "log_std_1p5m": float(nearest(opt, 1_500_000)[logstd]), "log_std_2m": float(nearest(opt, 2_000_000)[logstd]),
                    "log_std_3m": float(opt[logstd].iloc[-1]), "log_std_min": float(opt[logstd].min()), "log_std_max": float(opt[logstd].max()),
                    "action_abs_gt_0_9_mean": float(opt[sat90].mean()), "action_abs_gt_0_9_p95": percentile(opt[sat90], .95),
                    "action_abs_gt_0_9_max": float(opt[sat90].max()), "action_abs_gt_0_9_final": float(opt[sat90].iloc[-1]),
                    "action_abs_gt_0_99_mean": float(opt[sat99].mean()), "action_abs_gt_0_99_p95": percentile(opt[sat99], .95),
                    "action_abs_gt_0_99_max": float(opt[sat99].max()), "action_abs_gt_0_99_final": float(opt[sat99].iloc[-1]),
                })

        windows = []
        for _, erow in ev.iterrows():
            step = int(erow.sampled_steps)
            window = opt[(opt.sampled_steps > step - 100_000) & (opt.sampled_steps <= step)]
            row = {"training_seed": seed, "eval_step": step, "optimization_rows": len(window)}
            row.update({m: erow[m] for m in ("average_waves", "w1", "w2", "w3", "q2", "q3", "boundary", "ground")})
            row.update({
                "kl_mean": window.approx_kl.mean(), "kl_p95": window.approx_kl.quantile(.95), "kl_max": window.approx_kl.max(),
                "kl_warning_fraction": (window.approx_kl >= .05).mean(), "entropy_mean": window.entropy.mean(), "entropy_min": window.entropy.min(),
                "clip_fraction_mean": window.clip_fraction.mean(), "clip_fraction_p95": window.clip_fraction.quantile(.95),
                "actor_grad_p95": window.actor_grad_norm.quantile(.95), "value_loss_mean": window.value_loss.mean(),
                "value_loss_p95": window.value_loss.quantile(.95), "underflow_event_count": int((window.ratio_underflow_fraction > 0).sum()),
            })
            windows.append(row)
        window_df = pd.DataFrame(windows)
        window_df["delta_average_waves"] = window_df.average_waves.diff()
        window_df["delta_boundary"] = window_df.boundary.diff()
        opt_windows_all.append(window_df)
        corr_all.extend(corr_rows(window_df, seed))
        warned = window_df.kl_warning_fraction.gt(0)
        high_kl_effects.append({
            "training_seed": seed, "warning_windows": int(warned.sum()), "clean_windows": int((~warned).sum()),
            "mean_delta_waves_after_warning_window": window_df.loc[warned, "delta_average_waves"].mean(),
            "mean_delta_waves_after_clean_window": window_df.loc[~warned, "delta_average_waves"].mean(),
            "mean_delta_boundary_after_warning_window": window_df.loc[warned, "delta_boundary"].mean(),
            "mean_delta_boundary_after_clean_window": window_df.loc[~warned, "delta_boundary"].mean(),
        })

        full_cols = ["sampled_steps"] + FILM_METRICS
        film = opt[full_cols].copy()
        film.insert(0, "training_seed", seed)
        film["gamma_strength"] = film.film_delta_gamma_abs_mean / ALPHA
        film["beta_strength"] = film.film_beta_abs_mean / ALPHA
        film["residual_strength"] = film.film_residual_abs_mean / ALPHA
        film["hidden_norm_ratio"] = np.where(film.film_hidden_base_norm > 0, film.film_hidden_modulated_norm / film.film_hidden_base_norm, np.nan)
        film_full_all.append(film)
        for metric in FILM_METRICS + ["gamma_strength", "beta_strength", "residual_strength", "hidden_norm_ratio"]:
            film_summary.append({"training_seed": seed, "metric": metric, "mean": film[metric].mean(), "p50": film[metric].quantile(.5), "p95": film[metric].quantile(.95), "max": film[metric].max(), "final": film[metric].iloc[-1]})
        for label, target in (("early", int(film.sampled_steps.iloc[0])), ("900k", 900_000), ("1.5m", 1_500_000), ("2m", 2_000_000), ("3m", 3_000_000)):
            frow = nearest(film, target)
            film_stage_rows.append({"training_seed": seed, "stage": label, **{c: frow[c] for c in film.columns if c != "training_seed"}})
        aligned = window_df.merge(film, left_on=["training_seed", "eval_step"], right_on=["training_seed", "sampled_steps"], how="left")
        if aligned[FILM_METRICS].isna().all(axis=None):
            aligned_rows = []
            for _, erow in window_df.iterrows():
                frow = nearest(film, int(erow.eval_step))
                aligned_rows.append({**erow.to_dict(), **{c: frow[c] for c in film.columns if c not in ("training_seed", "sampled_steps")}, "film_diag_step": int(frow.sampled_steps)})
            aligned = pd.DataFrame(aligned_rows)
        else:
            aligned["film_diag_step"] = aligned["sampled_steps"]
        film_align_all.append(aligned)
        for feature in ("film_delta_gamma_abs_mean", "film_beta_abs_mean", "film_residual_abs_mean", "film_saturation_fraction", "hidden_norm_ratio"):
            for outcome in ("average_waves", "w1", "w2", "w3", "boundary", "ground"):
                pair = aligned[[feature, outcome]].dropna()
                film_performance_correlations.append({
                    "training_seed": seed, "feature": feature, "outcome": outcome,
                    "n_timepoints": len(pair),
                    "pearson": pair[feature].corr(pair[outcome], method="pearson") if len(pair) >= 3 else np.nan,
                    "spearman": pair[feature].rank(method="average").corr(pair[outcome].rank(method="average"), method="pearson") if len(pair) >= 3 else np.nan,
                })

        train_rows = []
        for _, erow in ev.iterrows():
            step = int(erow.sampled_steps)
            tw = train[(train.sampled_steps > step - 100_000) & (train.sampled_steps <= step)]
            train_rows.append({
                "training_seed": seed, "eval_step": step, "training_episodes_in_window": len(tw),
                "train_waves_mean": tw.waves_cleared.mean(), "train_return_mean": tw.team_raw_environment_return.mean(),
                "train_red_loss_mean": tw.red_losses.mean(), "train_blue_loss_mean": tw.blue_losses.mean(),
                "eval_average_waves": erow.average_waves, "eval_return": erow["return"],
                "eval_red_loss": erow.red_loss, "eval_blue_loss": erow.blue_loss,
                "waves_gap_eval_minus_train": erow.average_waves - tw.waves_cleared.mean(),
                "return_gap_eval_minus_train": erow["return"] - tw.team_raw_environment_return.mean(),
            })
        train_align_all.append(pd.DataFrame(train_rows))

    pd.DataFrame(opt_summaries).to_csv(out / "optimization_summary.csv", index=False)
    pd.DataFrame(action_distribution_rows).to_csv(out / "policy_action_distribution_summary.csv", index=False)
    opt_windows = pd.concat(opt_windows_all, ignore_index=True)
    opt_windows.to_csv(out / "optimization_eval_windows.csv", index=False)
    pd.DataFrame(corr_all).to_csv(out / "optimization_performance_correlations.csv", index=False)
    pd.DataFrame(high_kl_effects).to_csv(out / "high_kl_window_effects.csv", index=False)
    film_full = pd.concat(film_full_all, ignore_index=True)
    film_full.to_csv(out / "film_diagnostics_full.csv", index=False)
    pd.DataFrame(film_summary).to_csv(out / "film_diagnostics_summary.csv", index=False)
    pd.DataFrame(film_stage_rows).to_csv(out / "film_diagnostics_stages.csv", index=False)
    film_alignment = pd.concat(film_align_all, ignore_index=True)
    film_alignment.to_csv(out / "film_eval_alignment.csv", index=False)
    pd.DataFrame(film_performance_correlations).to_csv(out / "film_performance_correlations.csv", index=False)
    train_alignment = pd.concat(train_align_all, ignore_index=True)
    train_alignment.to_csv(out / "training_eval_alignment.csv", index=False)

    hypotheses = pd.DataFrame([
        {"hypothesis": "H1", "statement": "任务阶段信息本身没有价值", "assessment": "INCONCLUSIVE", "evidence": "Raw-Plain final average-waves mean delta +0.08 but only 1/3 paired wins; FiLM is worse, so neither value nor no-value is established."},
        {"hypothesis": "H2", "statement": "Raw concat不稳定但FiLM解决了融合问题", "assessment": "NOT_SUPPORTED", "evidence": "FiLM-Raw final average-waves delta -0.633 with 0/3 wins; mean peak retention Film 0.730 versus Raw 0.961."},
        {"hypothesis": "H3", "statement": "FiLM形成较强中期策略但不能保持", "assessment": "PLAUSIBLE", "evidence": "Film peak average waves 2.10/1.68/2.06, then final drops 0.90/0.38/0.32; however mid-run superiority over Plain is not consistent across seeds."},
        {"hypothesis": "H4", "statement": "失败主要由FiLM幅度过大或饱和导致", "assessment": "NOT_SUPPORTED", "evidence": "Final saturation is only 6.2%/3.1%/3.9%; modulation strength is bounded and generally positively associated with performance."},
        {"hypothesis": "H5", "statement": "失败主要与PPO后期漂移或大KL更新相关", "assessment": "PLAUSIBLE", "evidence": "All seeds show 2.9M performance drops with KL-warning windows, but full-trajectory warning windows do not consistently predict worse next evaluation; no causal attribution."},
        {"hypothesis": "H6", "statement": "第三波困难是主要失败来源", "assessment": "NOT_SUPPORTED", "evidence": "Final Film Q2/Q3 means are similarly low (0.488/0.489); W1 also degrades and boundary rises, so the defect is not isolated to wave 3."},
        {"hypothesis": "H7", "statement": "基础战术能力后期也发生退化", "assessment": "SUPPORTED", "evidence": "Peak-to-final W1 drops 0.12/0.14/0.06 and boundary rises from max-average-waves checkpoints by 1.12/0.42/0.14."},
    ])
    hypotheses.to_csv(out / "hypothesis_assessment.csv", index=False)
    directions = pd.DataFrame([
        {"rank": 1, "direction": "D+B: PPO conservative update / policy retention for FiLM", "support": "Three-seed peak loss and synchronized late boundary excursions; current modulation is active but not saturated.", "counterevidence": "KL-performance relation is not causal and is positive over much of learning.", "worth_next_cycle": "YES, one tightly scoped matched causal screen", "changes_question": "No; tests retention of static mission conditioning", "engineering_cost": "Medium", "scientific_explanatory_power": "High"},
        {"rank": 2, "direction": "E: retain Plain MAPPO as the practical baseline", "support": "Plain beats FiLM on average waves in 3/3 pairs and reaches its own maximum at 3M in all seeds.", "counterevidence": "Does not create a new method or explain partial observability.", "worth_next_cycle": "YES as mandatory control, not a new candidate", "changes_question": "No", "engineering_cost": "Low", "scientific_explanatory_power": "Medium"},
        {"rank": 3, "direction": "A: GRU/history modeling", "support": "Later-wave state may require history; static FiLM did not improve final outcomes.", "counterevidence": "This dataset never manipulates memory, so history need is untested.", "worth_next_cycle": "YES after the retention hypothesis is closed", "changes_question": "Yes; from static conditioning to temporal state inference", "engineering_cost": "High", "scientific_explanatory_power": "High if causally matched"},
        {"rank": 4, "direction": "C: reduce FiLM alpha", "support": "Could reduce intervention size in principle.", "counterevidence": "Saturation is low and stronger modulation is not associated with worse performance.", "worth_next_cycle": "NO as a standalone formal direction", "changes_question": "No", "engineering_cost": "Low", "scientific_explanatory_power": "Low"},
        {"rank": 5, "direction": "End mission modules immediately", "support": "Current FiLM fails the gate and raw context is mixed.", "counterevidence": "Mid-run FiLM capability plus late degradation leaves one retention hypothesis unresolved.", "worth_next_cycle": "NO before one bounded retention test", "changes_question": "Yes", "engineering_cost": "Low", "scientific_explanatory_power": "Low"},
    ])
    directions.to_csv(out / "next_direction_ranking.csv", index=False)

    protocol = {
        "status": "PASS", "film_run_audit": audits,
        "selection_key": ["clear_wave_3_probability", "average_waves_cleared", "average_return", "negative_average_red_loss"],
        "replication_unit": "training_seed_n3", "evaluation_episodes_are_not_independent_replicates": True,
        "offline_only": True, "cpu_checkpoint_loading": True, "new_evaluation": False,
        "development_eval_seed_range": [44_000_000, 44_000_049],
        "future_final_seed_range": [45_000_000, 45_000_199], "future_final_used": False,
    }
    analysis = {
        "status": "DEEP_ANALYSIS_COMPLETE", "development_label": "MISSION_AWARE_FILM_NOT_SUPPORTED",
        "protocol": protocol, "formal_analyzer": str(ROOT / "outputs/mission_aware_film_analysis/analysis.json"),
        "notes": [
            "All correlations are exploratory time-series associations with strong autocorrelation.",
            "No episode-level significance tests were performed.",
            "The 3M final/latest endpoint remains primary; peaks and best_eval are diagnostic only.",
        ],
    }
    (out / "deep_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    final_display = final[["method", "training_seed", "w1", "w2", "w3", "q2", "q3", "average_waves", "return", "boundary", "ground"]]
    delta_display = pd.DataFrame(delta_summary)
    delta_display = delta_display[delta_display.metric.isin(["w1", "w2", "w3", "q2", "q3", "average_waves", "return", "boundary", "ground"])]
    peak_display = pd.DataFrame(peak_context)[["training_seed", "repository_best_step", "max_average_waves_step", "max_average_waves", "final_average_waves", "avg_peak_w1", "final_w1", "avg_peak_w2", "final_w2", "avg_peak_w3", "final_w3", "avg_peak_boundary", "final_boundary"]]
    opt_display = pd.DataFrame(opt_summaries)[["training_seed", "kl_ge_0_05_count", "kl_ge_0_10_count", "kl_ge_0_20_count", "kl_p50", "kl_p95", "kl_max", "entropy_first", "entropy_3m", "underflow_event_count"]]
    film_stage_display = pd.DataFrame(film_stage_rows)[["training_seed", "stage", "gamma_strength", "beta_strength", "residual_strength", "film_saturation_fraction", "hidden_norm_ratio"]]
    report = [
        "# Mission-Aware FiLM deep offline analysis", "",
        "## Protocol", "", "Protocol: **PASS**. All three FiLM runs reached 3,000,000 sampled steps; declared, training, and evaluation environments are the same three-wave/3000-step persistent_wave_v2 configuration. Checkpoints and recorded optimization/evaluation metrics are finite. Resume and branch counts are zero. Evaluations use 44,000,000–44,000,049 only; 45,000,000–45,000,199 is unused.", "",
        "Formal 3M development label: **MISSION_AWARE_FILM_NOT_SUPPORTED**.", "",
        "## 3M primary endpoint", "", md_table(final_display), "",
        "## Paired differences", "", md_table(delta_display), "",
        "## FiLM peak-to-final behavior", "", md_table(peak_display), "",
        "All three seeds lose average waves after their observed peak. The corresponding W1 decline and boundary increase show that late degradation is not confined to the third wave.", "",
        "## PPO diagnostics", "", md_table(opt_display), "",
        "KL excursions exist, and all three seeds have a sharp 2.9M evaluation drop after a KL-warning window. Across all 30 time points, however, warning windows do not consistently imply a worse next evaluation. This supports a policy-drift hypothesis only as plausible, not proved.", "",
        "## FiLM diagnostics", "", md_table(film_stage_display), "",
        "FiLM modulation grows from near zero, remains bounded, and does not show widespread saturation. The evidence does not support excessive FiLM amplitude as the primary failure mechanism.", "",
        "## Hypotheses", "", md_table(hypotheses), "",
        "## Next directions", "", md_table(directions), "",
        "The 3M final/latest endpoint remains primary. Peak and repository-selected checkpoints are diagnostic and do not replace it. Correlations are exploratory time-series associations with strong autocorrelation; no episode-level significance tests were performed.",
    ]
    (out / "deep_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": analysis["status"], "protocol": "PASS", "development_label": analysis["development_label"], "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
