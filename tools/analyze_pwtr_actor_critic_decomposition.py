"""Read-only paired/factorial analysis for PWTR actor-vs-critic decomposition."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5301, 5302, 5303)
TARGET = 1_805_280
MILESTONES = (1_603_584, 1_701_888, 1_800_192, TARGET)
METHODS = {
    "Stratified": "stratified", "CurrentActor": "current_actor_only",
    "CurrentCritic": "current_critic_only", "CurrentBoth": "current_extra",
    "RecentActor": "recent_actor_only", "RecentCritic": "recent_critic_only",
    "RecentBoth": "uniform_recent",
}
NEW = {"CurrentActor", "CurrentCritic", "RecentActor", "RecentCritic"}
EXPECTED = {
    "Stratified": ("pwtr_stratified", True, False, "recent_uniform", False, False),
    "CurrentActor": ("pwtr_current_actor_only", True, True, "current", True, False),
    "CurrentCritic": ("pwtr_current_critic_only", True, True, "current", False, True),
    "CurrentBoth": ("pwtr_current_extra", True, True, "current", True, True),
    "RecentActor": ("pwtr_recent_actor_only", True, True, "recent_uniform", True, False),
    "RecentCritic": ("pwtr_recent_critic_only", True, True, "recent_uniform", False, True),
    "RecentBoth": ("pwtr_uniform_recent", True, True, "recent_uniform", True, True),
}
METRICS = {"W1": "clear_wave_1_probability", "W2": "clear_wave_2_probability",
    "W3": "clear_wave_3_probability", "AW": "average_waves_cleared",
    "Return": "average_return", "RedLoss": "average_red_loss",
    "Boundary": "average_red_boundary_exits", "Ground": "average_red_ground_losses",
    "EpisodeLength": "average_episode_length"}
WINDOWS = ((1_505_280, 1_603_584), (1_603_584, 1_701_888), (1_701_888, TARGET))


def interaction(control: float, actor: float, critic: float, both: float) -> float:
    return both - actor - critic + control


def main_effects(control: float, actor: float, critic: float, both: float) -> tuple[float, float]:
    return .5 * ((actor - control) + (both - critic)), .5 * ((critic - control) + (both - actor))


def direction_label(deltas: list[float]) -> str:
    avg = statistics.mean(deltas)
    if sum(value > 0 for value in deltas) >= 2 and avg > 0: return "POSITIVE"
    if sum(value < 0 for value in deltas) >= 2 and avg < 0: return "NEGATIVE"
    return "MIXED"


def qmetrics(values: dict) -> dict:
    values["Q2"] = None if values["W1"] == 0 else values["W2"] / values["W1"]
    values["Q3"] = None if values["W2"] == 0 else values["W3"] / values["W2"]
    return values


def numeric(row: dict, column: str) -> float:
    value = float(row[column])
    if not math.isfinite(value): raise RuntimeError(f"non-finite {column}")
    return value


def run_path(method: str, seed: int) -> Path:
    return ROOT / f"outputs/dev_pwtr_{METHODS[method]}_seed{seed}_300k"


def read_run(method: str, seed: int) -> dict:
    path = run_path(method, seed)
    required = ("run_summary.json", "run_config.json", "algorithm_config.yaml", "evaluation_history.csv",
                "training_metrics.jsonl", "optimization_metrics.jsonl", "latest.pt", "final.pt", "train.log")
    missing = [name for name in required if not (path / name).is_file()]
    if missing: raise RuntimeError(f"{path}: missing {missing}")
    summary = json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
    if int(summary.get("sampled_steps", -1)) != TARGET: raise RuntimeError(f"{path}: incomplete")
    config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    algorithm = yaml.safe_load((path / "algorithm_config.yaml").read_text(encoding="utf-8"))
    module = algorithm["modules"]["persistent_wave_trajectory_replay"]
    expected = EXPECTED[method]
    actual = (algorithm.get("development_method"), bool(module.get("fresh_wave_stratification")),
              bool(module.get("replay_enabled")), module.get("replay_source"),
              bool(module.get("actor_replay")), bool(module.get("critic_replay")))
    if actual != expected or bool(module.get("priority_enabled")) or bool(module.get("bridge_enabled")):
        raise RuntimeError(f"{path}: method identity mismatch {actual}")
    if (int(config.get("seed", -1)), config.get("environment_variant")) != (seed, "persistent_wave_v2"):
        raise RuntimeError(f"{path}: runtime identity mismatch")
    with (path / "evaluation_history.csv").open(newline="", encoding="utf-8-sig") as stream:
        evaluation = list(csv.DictReader(stream))
    exact = [row for row in evaluation if int(float(row["sampled_steps"])) == TARGET]
    if len(exact) != 1: raise RuntimeError(f"{path}: exact endpoint unavailable")
    if (int(float(exact[0]["evaluation_episodes"])), int(float(exact[0]["evaluation_seed_base"])),
            int(float(exact[0]["evaluation_seed_end"]))) != (50, 44_000_000, 44_000_049):
        raise RuntimeError(f"{path}: evaluation protocol mismatch")
    endpoint = qmetrics({key: numeric(exact[0], column) for key, column in METRICS.items()})
    checkpoints = {}
    for name in ("latest.pt", "final.pt"):
        state = torch.load(path / name, map_location="cuda", weights_only=False)
        if int(state.get("sampled_steps", -1)) != TARGET or state.get("extra", {}).get("development_method") != expected[0]:
            raise RuntimeError(f"{path}/{name}: checkpoint mismatch")
        checkpoints[name] = int(state["sampled_steps"])
    optimization = []
    for line in (path / "optimization_metrics.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if any(not math.isfinite(float(v)) for v in row.values() if isinstance(v, (int, float))):
                raise RuntimeError(f"{path}: non-finite optimization row")
            optimization.append(row)
    if not optimization: raise RuntimeError(f"{path}: no optimization rows")
    log = (path / "train.log").read_text(encoding="utf-8", errors="replace")
    if any(marker in log for marker in ("Traceback", "CUDA out of memory", "protocol mismatch")):
        raise RuntimeError(f"{path}: failure marker")
    return {"path": str(path), "endpoint": endpoint, "evaluation": evaluation,
            "optimization": optimization, "config": config, "summary": summary}


def delta(high: dict, low: dict) -> dict:
    return {key: None if high.get(key) is None or low.get(key) is None else high[key] - low[key]
            for key in (*METRICS, "Q2", "Q3")}


def paired_metric_deltas(data: dict, method: str, control: str, metric: str,
                         seeds=SEEDS) -> dict:
    """Aggregate only mathematically defined paired deltas and report omissions."""
    values, undefined_seeds = [], []
    for seed in seeds:
        high = data[method, seed]["endpoint"].get(metric)
        low = data[control, seed]["endpoint"].get(metric)
        if high is None or low is None:
            undefined_seeds.append(seed)
        else:
            values.append(high - low)
    return {"values": values, "n_defined": len(values), "undefined_seeds": undefined_seeds}


def factorial_metric_rows(data: dict, source: str, actor: str, critic: str,
                          both: str, metric: str, seeds=SEEDS) -> tuple[list[dict], dict]:
    """Return defined per-seed factorial effects plus explicit coverage metadata."""
    rows, undefined_seeds = [], []
    for seed in seeds:
        values = [data[name, seed]["endpoint"].get(metric)
                  for name in ("Stratified", actor, critic, both)]
        if any(value is None for value in values):
            undefined_seeds.append(seed)
            continue
        effect_actor, effect_critic = main_effects(*values)
        rows.append({"source": source, "seed": seed, "metric": metric,
            "actor_main_effect": effect_actor, "critic_main_effect": effect_critic,
            "interaction": interaction(*values)})
    return rows, {"n_defined": len(rows), "undefined_seeds": undefined_seeds}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def sample_stats(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values)}


def distribution(values: list[float]) -> dict:
    ordered = sorted(values); index = max(0, math.ceil(.95 * len(ordered)) - 1)
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p95": ordered[index], "max": max(values)}


def mechanism_decision(labels: dict, interactions: dict) -> dict:
    actor = [labels["CurrentActor"], labels["RecentActor"]]
    critic = [labels["CurrentCritic"], labels["RecentCritic"]]
    actor_signal = "SUPPORTED_AS_PRIMARY_CANDIDATE" if actor == ["NEGATIVE", "NEGATIVE"] and critic != ["NEGATIVE", "NEGATIVE"] else "NOT_ISOLATED" if critic == ["NEGATIVE", "NEGATIVE"] else "MIXED"
    critic_signal = "SUPPORTED_AS_PRIMARY_CANDIDATE" if critic == ["NEGATIVE", "NEGATIVE"] and actor != ["NEGATIVE", "NEGATIVE"] else "NOT_ISOLATED" if actor == ["NEGATIVE", "NEGATIVE"] else "MIXED"
    negative = []
    for source in ("current", "recent"):
        rows = interactions[source]
        negative.append(sum(value < 0 for value in rows) >= 2 and statistics.mean(rows) < 0)
    interaction_signal = "SUPPORTED_DESCRIPTIVE" if any(negative) else "NOT_SUPPORTED"
    if all(label == "NEGATIVE" for label in actor + critic): case = "CASE_D_BOTH_COMPONENTS_NEGATIVE_STOP_PWTR_REPLAY"
    elif actor == ["NEGATIVE", "NEGATIVE"] and all(label != "NEGATIVE" for label in critic): case = "CASE_A_ACTOR_REPLAY_PRIMARY_CANDIDATE"
    elif critic == ["NEGATIVE", "NEGATIVE"] and all(label != "NEGATIVE" for label in actor): case = "CASE_B_CRITIC_REPLAY_PRIMARY_CANDIDATE"
    elif all(label != "NEGATIVE" for label in actor + critic) and interaction_signal == "SUPPORTED_DESCRIPTIVE": case = "CASE_C_ACTOR_CRITIC_COUPLING_PRIMARY_CANDIDATE"
    elif labels["RecentActor"] == "NEGATIVE" and labels["CurrentActor"] != "NEGATIVE": case = "CASE_E_HISTORICAL_OFF_POLICY_ACTOR_REUSE_CANDIDATE"
    elif labels["RecentCritic"] == "NEGATIVE" and labels["CurrentCritic"] != "NEGATIVE": case = "CASE_F_HISTORICAL_CRITIC_TARGET_REUSE_CANDIDATE"
    else: case = "MIXED_NO_PREREGISTERED_CASE"
    return {"actor_replay_signal": actor_signal, "critic_replay_signal": critic_signal,
            "actor_critic_negative_interaction": interaction_signal, "decision_case": case}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="outputs/pwtr_actor_critic_decomposition_300k_analysis")
    args = parser.parse_args(); output = ROOT / args.output_dir
    if output.exists(): raise FileExistsError(output)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is mandatory for checkpoint audit")
    data = {(method, seed): read_run(method, seed) for method in METHODS for seed in SEEDS}
    endpoint_rows = [{"method": method, "seed": seed, "sampled_steps": TARGET, **data[method, seed]["endpoint"]}
                     for method in METHODS for seed in SEEDS]
    comparisons = {
        "CurrentActor-Stratified": ("CurrentActor", "Stratified"), "CurrentCritic-Stratified": ("CurrentCritic", "Stratified"),
        "CurrentBoth-Stratified": ("CurrentBoth", "Stratified"), "CurrentBoth-CurrentActor": ("CurrentBoth", "CurrentActor"),
        "CurrentBoth-CurrentCritic": ("CurrentBoth", "CurrentCritic"), "RecentActor-Stratified": ("RecentActor", "Stratified"),
        "RecentCritic-Stratified": ("RecentCritic", "Stratified"), "RecentBoth-Stratified": ("RecentBoth", "Stratified"),
        "RecentBoth-RecentActor": ("RecentBoth", "RecentActor"), "RecentBoth-RecentCritic": ("RecentBoth", "RecentCritic")}
    paired = []
    for comparison, (high, low) in comparisons.items():
        for seed in SEEDS: paired.append({"comparison": comparison, "seed": seed, **{f"delta_{k}": v for k, v in delta(data[high, seed]["endpoint"], data[low, seed]["endpoint"]).items()}})
    factorial, interaction_aw = [], {"current": [], "recent": []}
    for source, actor, critic, both in (("current", "CurrentActor", "CurrentCritic", "CurrentBoth"), ("recent", "RecentActor", "RecentCritic", "RecentBoth")):
        for metric in (*METRICS, "Q2", "Q3"):
            rows, coverage = factorial_metric_rows(data, source, actor, critic, both, metric)
            factorial.extend(rows)
            if metric == "AW": interaction_aw[source].extend(row["interaction"] for row in rows)
            summary = {"source": source, "seed": "mean_sd", "metric": metric, **coverage}
            for field in ("actor_main_effect", "critic_main_effect", "interaction"):
                values = [row[field] for row in rows]
                stats = sample_stats(values) if values else {"mean": None, "sample_sd": None, "min": None, "max": None}
                summary.update({f"{field}_{stat}": value for stat, value in stats.items()})
            factorial.append(summary)
    dynamics = []
    for method in NEW:
        for seed in SEEDS:
            rows = data[method, seed]["evaluation"]
            by_step = {int(float(row["sampled_steps"])): row for row in rows}
            for step in MILESTONES:
                if step not in by_step: raise RuntimeError(f"{method}/seed{seed}: missing milestone {step}")
                values = qmetrics({key: numeric(by_step[step], column) for key, column in METRICS.items()})
                dynamics.append({"method": method, "seed": seed, "sampled_steps": step, **values})
    opt_fields = {"actor": ("pwtr_actor_replay_loss", "pwtr_actor_replay_grad_norm_preclip", "pwtr_actor_eligible_fraction", "pwtr_vtrace_rho_mean", "actor_grad_norm", "approx_kl"),
                  "critic": ("pwtr_critic_replay_loss", "pwtr_critic_replay_grad_norm_preclip", "critic_grad_norm", "value_loss")}
    optimization = []
    for method in NEW:
        mode = "actor" if method.endswith("Actor") else "critic"
        for seed in SEEDS:
            rows = data[method, seed]["optimization"]
            for start, end in WINDOWS:
                window = [row for row in rows if start < int(row.get("sampled_steps", -1)) <= end]
                if not window: raise RuntimeError(f"{method}/seed{seed}: empty optimization window")
                for key in opt_fields[mode]:
                    values = [float(row[key]) for row in window if isinstance(row.get(key), (int, float))]
                    if values: optimization.append({"method": method, "seed": seed, "window_start": start,
                        "window_end": end, "metric": key, "count": len(values), **distribution(values)})
    direct = {method: [data[method, seed]["endpoint"]["AW"] - data["Stratified", seed]["endpoint"]["AW"] for seed in SEEDS] for method in NEW}
    labels = {method: direction_label(values) for method, values in direct.items()}
    direct_secondary = {method: {metric: paired_metric_deltas(data, method, "Stratified", metric)
                        for metric in ("W3", "Q2", "Q3")} for method in NEW}
    safety = {method: "SAFETY_SIGNAL" if any(
        data[method, seed]["endpoint"]["AW"] - data["Stratified", seed]["endpoint"]["AW"] <= -.50 or
        data[method, seed]["endpoint"]["W1"] - data["Stratified", seed]["endpoint"]["W1"] <= -.25 for seed in SEEDS) else "NO_SAFETY_SIGNAL" for method in NEW}
    report = {"status": "PWTR_ACTOR_CRITIC_DECOMPOSITION_ANALYSIS_COMPLETE", "primary_endpoint": TARGET,
        "primary_metric": "AverageWaves", "training_seed_replication_n": 3,
        "direct_component_labels": {method: {"label": labels[method], "delta_AW": sample_stats(direct[method]),
            **{f"mean_delta_{metric}": (statistics.mean(direct_secondary[method][metric]["values"])
                                        if direct_secondary[method][metric]["values"] else None)
               for metric in ("W3", "Q2", "Q3")},
            **{f"n_defined_{metric}": direct_secondary[method][metric]["n_defined"] for metric in ("Q2", "Q3")},
            **{f"undefined_{metric}_seeds": direct_secondary[method][metric]["undefined_seeds"] for metric in ("Q2", "Q3")},
            "safety": safety[method]} for method in NEW},
        "interaction_AW": {source: {"values": values, **sample_stats(values)} for source, values in interaction_aw.items()},
        "mechanism": mechanism_decision(labels, interaction_aw),
        "old_reference_methods": ["Stratified", "CurrentBoth", "RecentBoth"],
        "new_decomposition_methods": sorted(NEW),
        "core_implementation_claim": "registration-only difference; preflight separately verifies frozen optimizer/replay/environment SHA",
        "evaluation_seed_range": [44_000_000, 44_000_049], "reserved_45m_used": False,
        "no_inferential_significance_claim": True, "formal_reevaluation_performed": False}
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(output / "factorial_endpoint_table.csv", endpoint_rows)
    write_csv(output / "paired_component_effects.csv", paired)
    write_csv(output / "factorial_interactions.csv", factorial)
    write_csv(output / "learning_dynamics.csv", dynamics)
    write_csv(output / "optimization_summary.csv", optimization)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
