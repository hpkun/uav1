#!/usr/bin/env python3
"""Strictly offline W1SG/WSAI mechanism evidence audit."""
from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5301, 5302, 5303)
SOURCE_STEP = 1_505_280
END_STEP = 1_805_280
EPSILON = 1e-30
OUTPUT = ROOT / "outputs/partial_sharing_evidence_analysis"
WINDOWS = {
    "EARLY": (SOURCE_STEP, 1_603_584),
    "MIDDLE": (1_603_584, 1_701_888),
    "LATE": (1_701_888, END_STEP),
}
EVALUATION_STEPS = (1_603_584, 1_701_888, 1_800_192, END_STEP)
SPECIALIZATION_TARGETS = {
    "EARLY_ENDPOINT": 1_603_584,
    "MIDDLE_ENDPOINT": 1_701_888,
    "LATE_ENDPOINT": 1_800_192,
    "FINAL": END_STEP,
}
FAMILIES = {
    "backbone": "backbone.",
    "mean_head": "mean.",
    "log_std_head": "log_std.",
}
REQUIRED_FILES = (
    "run_config.json", "run_summary.json", "evaluation_history.csv",
    "optimization_metrics.jsonl", "latest.pt", "final.pt",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def window_for_step(step: int) -> str | None:
    for name, (lower, upper) in WINDOWS.items():
        if lower < int(step) <= upper:
            return name
    return None


def quantile(values: Iterable[float], probability: float) -> float | None:
    values = list(values)
    return None if not values else float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    values = [float(value) for value in values]
    if not values:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {
        "n": len(values), "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p10": quantile(values, .10), "p90": quantile(values, .90),
    }


def family_counts(actor_state: dict[str, torch.Tensor]) -> dict[str, dict[str, float | int]]:
    counts = {family: 0 for family in FAMILIES}
    unknown = []
    for name, value in actor_state.items():
        matches = [family for family, prefix in FAMILIES.items() if name.startswith(prefix)]
        if len(matches) != 1:
            unknown.append(name)
        else:
            counts[matches[0]] += int(value.numel())
    if unknown:
        raise RuntimeError(f"unclassified Actor parameters: {unknown}")
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("empty source Actor state")
    return {family: {"parameter_count": count, "parameter_fraction": count / total}
            for family, count in counts.items()}


def importance_derived(importance_share: float, parameter_fraction: float,
                       family_mean: float, global_mean: float) -> tuple[float, float]:
    return (float(importance_share) / max(float(parameter_fraction), EPSILON),
            float(family_mean) / max(float(global_mean), EPSILON))


def wsai_proxies(row: dict[str, Any]) -> dict[str, float]:
    pre = [float(row[f"wsai_wave{wave}_actor_grad_norm_preclip"]) for wave in (1, 2, 3)]
    post = [float(row[f"wsai_wave{wave}_actor_grad_norm_postclip"]) for wave in (1, 2, 3)]
    global_pre = float(row["wsai_global_actor_grad_norm_preclip"])
    global_post = float(row["wsai_global_actor_grad_norm_postclip"])
    later = math.sqrt(pre[1] ** 2 + pre[2] ** 2)
    total = sum(pre)
    return {
        "global_clip_scale_proxy": global_post / max(global_pre, EPSILON),
        "later_preclip_norm": later,
        "w1_to_later_grad_ratio": pre[0] / max(later, EPSILON),
        "wave1_norm_fraction_proxy": pre[0] / max(total, EPSILON),
        "wave2_norm_fraction_proxy": pre[1] / max(total, EPSILON),
        "wave3_norm_fraction_proxy": pre[2] / max(total, EPSILON),
        "wave2_clip_retention_proxy": post[1] / max(pre[1], EPSILON),
        "wave3_clip_retention_proxy": post[2] / max(pre[2], EPSILON),
    }


def last_not_after(rows: list[dict[str, Any]], target: int) -> dict[str, Any] | None:
    eligible = [row for row in rows if int(row["sampled_steps"]) <= int(target)]
    return None if not eligible else max(eligible, key=lambda row: int(row["sampled_steps"]))


def exact_evaluation(rows: list[dict[str, str]], target: int) -> dict[str, str] | None:
    matches = [row for row in rows if int(row["sampled_steps"]) == int(target)]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate evaluation step {target}")
    return None if not matches else matches[0]


def family_pattern(rankings: dict[int, list[str]]) -> str:
    first = [ranking[0] for ranking in rankings.values()]
    if sum(name in {"mean_head", "log_std_head"} for name in first) >= 2:
        return "HEAD_ENRICHED"
    if first.count("backbone") >= 2:
        return "BACKBONE_ENRICHED"
    return "MIXED"


def clip_coupling_class(per_seed: dict[int, dict[str, float]]) -> tuple[str, dict[int, bool]]:
    flags = {
        seed: (values["row_level_clip_pressure_fraction"] >= .80
               and values["median_w1_to_later_grad_ratio"] >= 2.0
               and (values["median_wave2_clip_retention_proxy"] <= .25
                    or values["median_wave3_clip_retention_proxy"] <= .25))
        for seed, values in per_seed.items()
    }
    count = sum(flags.values())
    return ("STRONG_SIGNAL" if count >= 2 else "WEAK_SIGNAL" if count == 0 else "MIXED"), flags


def validate_run(path: Path, seed: int, method: str) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"{path}: missing {missing}")
    run = load_json(path / "run_config.json")
    summary = load_json(path / "run_summary.json")
    evaluations = load_csv(path / "evaluation_history.csv")
    endpoint = exact_evaluation(evaluations, END_STEP)
    if endpoint is None:
        raise RuntimeError(f"{path}: exact endpoint missing")
    identity = (run.get("development_method"), int(run.get("seed", -1)),
                int(run.get("total_sampled_steps", -1)))
    if identity != (method, seed, END_STEP):
        raise RuntimeError(f"{path}: run identity mismatch {identity}")
    if (int(endpoint["evaluation_episodes"]), int(endpoint["evaluation_seed_base"]),
            int(endpoint["evaluation_seed_end"])) != (50, 44_000_000, 44_000_049):
        raise RuntimeError(f"{path}: endpoint evaluation protocol mismatch")
    if any(45_000_000 <= int(row["evaluation_seed_base"]) <= 45_000_199 for row in evaluations):
        raise RuntimeError(f"{path}: forbidden 45M evaluation evidence")
    sampled = int(summary.get("sampled_steps", summary.get("total_sampled_steps", -1)))
    if sampled != END_STEP:
        raise RuntimeError(f"{path}: summary endpoint mismatch {sampled}")
    return {"path": str(path.relative_to(ROOT)), "method": method, "seed": seed,
            "exact_endpoint": END_STEP, "evaluation_episodes": 50,
            "evaluation_seed_start": 44_000_000, "evaluation_seed_end": 44_000_049,
            "uses_45m": False, "status": "PASS"}


def summarize_w1sg(rows: list[dict[str, Any]], fractions: dict[str, float],
                   seed: int, family_csv: list[dict[str, Any]],
                   gating_csv: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    active = [row for row in rows if float(row["w1sg_active"]) > .5]
    if not rows:
        raise RuntimeError(f"W1SG seed{seed}: empty optimization history")
    for row in rows:
        step = int(row["sampled_steps"]); window = window_for_step(step)
        gating_csv.append({
            "seed": seed, "sampled_steps": step, "window": window,
            "active": int(float(row["w1sg_active"]) > .5),
            "w1_alive_samples": float(row["w1sg_w1_alive_samples"]),
            "gate_mean": float(row["w1sg_gate_mean"]), "gate_min": float(row["w1sg_gate_min"]),
            "gate_p10": float(row["w1sg_gate_p10"]), "gate_p50": float(row["w1sg_gate_p50"]),
            "gate_p90": float(row["w1sg_gate_p90"]),
            "cosine": float(row["w1sg_raw_protected_gradient_cosine"]),
            "renorm_scale": float(row["w1sg_renorm_scale"]),
            "norm_error": float(row["w1sg_norm_preservation_relative_error"]),
        })
        if float(row["w1sg_active"]) <= .5:
            continue
        for family in FAMILIES:
            share = float(row[f"w1sg_{family}_importance_share"])
            mean = float(row[f"w1sg_{family}_importance_mean"])
            enrichment, relative = importance_derived(
                share, fractions[family], mean, float(row["w1sg_importance_raw_mean"]))
            family_csv.append({
                "seed": seed, "sampled_steps": step, "window": window, "family": family,
                "parameter_fraction": fractions[family], "importance_share": share,
                "importance_enrichment": enrichment, "importance_mean": mean,
                "relative_global_mean": relative,
            })
    def block(selected: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"active_rows": len(selected)}
        for family in FAMILIES:
            entries = [row for row in family_csv if row["seed"] == seed and row["family"] == family
                       and any(int(source["sampled_steps"]) == row["sampled_steps"] for source in selected)]
            result[family] = {
                "importance_share": distribution(row["importance_share"] for row in entries),
                "importance_enrichment": distribution(row["importance_enrichment"] for row in entries),
                "relative_global_mean": distribution(row["relative_global_mean"] for row in entries),
            }
        result["gating"] = {
            key: distribution(float(row[field]) for row in selected)
            for key, field in {
                "raw_protected_gradient_cosine": "w1sg_raw_protected_gradient_cosine",
                "renorm_scale": "w1sg_renorm_scale", "gate_mean": "w1sg_gate_mean",
                "gate_min": "w1sg_gate_min", "gate_p10": "w1sg_gate_p10",
                "gate_p50": "w1sg_gate_p50", "gate_p90": "w1sg_gate_p90",
                "norm_preservation_relative_error": "w1sg_norm_preservation_relative_error",
            }.items()
        }
        return result
    overall = block(active)
    overall.update({
        "row_count": len(rows), "active_row_count": len(active),
        "active_fraction": len(active) / len(rows),
        "identity_fallback_final_cumulative_count": int(rows[-1]["w1sg_identity_fallback_count"]),
        "replay_gradient_step_final_cumulative_count": int(rows[-1]["w1sg_replay_gradient_step_count"]),
        "active_update_final_cumulative_count": int(rows[-1]["w1sg_active_update_count"]),
    })
    windowed = {window: block([row for row in active if window_for_step(int(row["sampled_steps"])) == window])
                for window in WINDOWS}
    return overall, windowed


def summarize_wsai(rows: list[dict[str, Any]], seed: int,
                   gradient_csv: list[dict[str, Any]],
                   specialization_csv: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not rows:
        raise RuntimeError(f"WSAI seed{seed}: empty optimization history")
    for row in rows:
        step = int(row["sampled_steps"]); proxy = wsai_proxies(row)
        gradient_csv.append({
            "seed": seed, "sampled_steps": step, "window": window_for_step(step),
            **{f"wave{wave}_alive_fraction": float(row[f"wsai_wave{wave}_alive_fraction"]) for wave in (1, 2, 3)},
            **{f"wave{wave}_preclip_norm": float(row[f"wsai_wave{wave}_actor_grad_norm_preclip"]) for wave in (1, 2, 3)},
            **{f"wave{wave}_postclip_norm": float(row[f"wsai_wave{wave}_actor_grad_norm_postclip"]) for wave in (1, 2, 3)},
            "global_preclip_norm": float(row["wsai_global_actor_grad_norm_preclip"]),
            "global_postclip_norm": float(row["wsai_global_actor_grad_norm_postclip"]), **proxy,
        })
        specialization_csv.append({
            "seed": seed, "sampled_steps": step,
            **{f"actor{pair}_relative_l2": float(row[f"wsai_actor{pair}_relative_l2"]) for pair in ("12", "13", "23")},
            **{f"wave{wave}_optimizer_steps": int(row[f"wsai_wave{wave}_optimizer_steps"]) for wave in (1, 2, 3)},
            **{f"wave{wave}_routed_alive_samples": int(row[f"wsai_wave{wave}_routed_alive_samples"]) for wave in (1, 2, 3)},
            **{f"wave{wave}_actor_lr": float(row[f"wsai_wave{wave}_actor_lr"]) for wave in (1, 2, 3)},
        })
    seed_gradient = [row for row in gradient_csv if row["seed"] == seed]
    final = rows[-1]
    counts = [float(final[f"wsai_wave{wave}_routed_alive_samples"]) for wave in (1, 2, 3)]
    total = sum(counts)
    def block(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "row_count": len(selected),
            "global_preclip_norm": distribution(row["global_preclip_norm"] for row in selected),
            "global_postclip_norm": distribution(row["global_postclip_norm"] for row in selected),
            "global_clip_scale_proxy": distribution(row["global_clip_scale_proxy"] for row in selected),
            "w1_to_later_grad_ratio": distribution(row["w1_to_later_grad_ratio"] for row in selected),
            **{f"wave{wave}_norm_fraction_proxy": distribution(row[f"wave{wave}_norm_fraction_proxy"] for row in selected)
               for wave in (1, 2, 3)},
            "wave2_clip_retention_proxy": distribution(row["wave2_clip_retention_proxy"] for row in selected),
            "wave3_clip_retention_proxy": distribution(row["wave3_clip_retention_proxy"] for row in selected),
            "row_level_clip_pressure_fraction": (None if not selected else
                sum(row["global_preclip_norm"] > .5 for row in selected) / len(selected)),
        }
    overall = block(seed_gradient)
    overall.update({
        "cumulative_routed_alive_counts": {f"W{wave}": int(counts[wave - 1]) for wave in (1, 2, 3)},
        "cumulative_routed_alive_fractions": {f"W{wave}": counts[wave - 1] / max(total, EPSILON) for wave in (1, 2, 3)},
        "final_optimizer_steps": {f"W{wave}": int(final[f"wsai_wave{wave}_optimizer_steps"]) for wave in (1, 2, 3)},
        "final_actor_lrs": {f"W{wave}": float(final[f"wsai_wave{wave}_actor_lr"]) for wave in (1, 2, 3)},
    })
    windowed = {window: block([row for row in seed_gradient if row["window"] == window]) for window in WINDOWS}
    return overall, windowed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis directory: {OUTPUT}")
    protocol: dict[str, Any] = {"W1SG": {}, "WSAI": {}, "references": {}}
    methods = {
        "W1SG": ("dev_w1sg_current_actor_seed{seed}_300k", "w1sg_current_actor"),
        "WSAI": ("dev_wsai_seed{seed}_300k", "wsai_mappo"),
        "Plain": ("dev_pwtr_plain_seed{seed}_300k", "pwtr_plain_matched_control"),
        "Stratified": ("dev_pwtr_stratified_seed{seed}_300k", "pwtr_stratified"),
        "CurrentActor": ("dev_pwtr_current_actor_only_seed{seed}_300k", "pwtr_current_actor_only"),
    }
    for label, (template, method) in methods.items():
        target = protocol[label] if label in {"W1SG", "WSAI"} else protocol["references"].setdefault(label, {})
        for seed in SEEDS:
            target[seed] = validate_run(ROOT / "outputs" / template.format(seed=seed), seed, method)

    family_topologies = {}
    for seed in SEEDS:
        source = ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt"
        state = torch.load(source, map_location="cpu", weights_only=False)
        if int(state["sampled_steps"]) != SOURCE_STEP or int(state["extra"]["training_seed"]) != seed:
            raise RuntimeError(f"source checkpoint identity mismatch: seed{seed}")
        family_topologies[seed] = family_counts(state["actor"])
    if any(family_topologies[seed] != family_topologies[SEEDS[0]] for seed in SEEDS[1:]):
        raise RuntimeError("source Actor family topology differs across seeds")
    families = family_topologies[SEEDS[0]]
    fractions = {family: float(values["parameter_fraction"]) for family, values in families.items()}

    family_csv: list[dict[str, Any]] = []
    gating_csv: list[dict[str, Any]] = []
    gradient_csv: list[dict[str, Any]] = []
    specialization_csv: list[dict[str, Any]] = []
    alignment_csv: list[dict[str, Any]] = []
    w1sg_per_seed = {}; w1sg_windowed = {}; wsai_per_seed = {}; wsai_windowed = {}
    wsai_rows: dict[int, list[dict[str, Any]]] = {}
    for seed in SEEDS:
        w1_rows = load_jsonl(ROOT / f"outputs/dev_w1sg_current_actor_seed{seed}_300k/optimization_metrics.jsonl")
        w1sg_per_seed[seed], w1sg_windowed[seed] = summarize_w1sg(
            w1_rows, fractions, seed, family_csv, gating_csv)
        current_wsai = load_jsonl(ROOT / f"outputs/dev_wsai_seed{seed}_300k/optimization_metrics.jsonl")
        wsai_rows[seed] = current_wsai
        wsai_per_seed[seed], wsai_windowed[seed] = summarize_wsai(
            current_wsai, seed, gradient_csv, specialization_csv)

    rankings = {}
    for seed in SEEDS:
        medians = {family: w1sg_per_seed[seed][family]["importance_enrichment"]["median"] for family in FAMILIES}
        rankings[seed] = sorted(medians, key=lambda family: (-float(medians[family]), family))
    pattern = family_pattern(rankings)

    coupling_inputs = {
        seed: {
            "row_level_clip_pressure_fraction": float(wsai_per_seed[seed]["row_level_clip_pressure_fraction"]),
            "median_w1_to_later_grad_ratio": float(wsai_per_seed[seed]["w1_to_later_grad_ratio"]["median"]),
            "median_wave2_clip_retention_proxy": float(wsai_per_seed[seed]["wave2_clip_retention_proxy"]["median"]),
            "median_wave3_clip_retention_proxy": float(wsai_per_seed[seed]["wave3_clip_retention_proxy"]["median"]),
        } for seed in SEEDS
    }
    coupling, coupling_flags = clip_coupling_class(coupling_inputs)

    specialization_endpoints = {}
    eval_alignment = {}
    for seed in SEEDS:
        specialization_endpoints[seed] = {}
        for label, target in SPECIALIZATION_TARGETS.items():
            row = last_not_after(wsai_rows[seed], target)
            specialization_endpoints[seed][label] = None if row is None else {
                "requested_sampled_steps": target, "actual_sampled_steps": int(row["sampled_steps"]),
                "step_delta": target - int(row["sampled_steps"]),
                **{f"actor{pair}_relative_l2": float(row[f"wsai_actor{pair}_relative_l2"]) for pair in ("12", "13", "23")},
            }
        evaluations = load_csv(ROOT / f"outputs/dev_wsai_seed{seed}_300k/evaluation_history.csv")
        eval_alignment[seed] = {}
        for step in EVALUATION_STEPS:
            evaluation = exact_evaluation(evaluations, step)
            if evaluation is None:
                eval_alignment[seed][step] = {"status": "MISSING", "evaluation_step": step}
                continue
            optimization = last_not_after(wsai_rows[seed], step)
            if optimization is None:
                raise RuntimeError(f"seed{seed}: no optimization row before evaluation {step}")
            proxy = wsai_proxies(optimization)
            w1 = float(evaluation["clear_wave_1_probability"]); w2 = float(evaluation["clear_wave_2_probability"])
            row = {
                "seed": seed, "status": "EXACT", "evaluation_step": step,
                "optimization_step": int(optimization["sampled_steps"]),
                "step_gap": step - int(optimization["sampled_steps"]),
                "AW": float(evaluation["average_waves_cleared"]), "W1": w1, "W2": w2,
                "W3": float(evaluation["clear_wave_3_probability"]),
                "Q2": None if w1 == 0 else w2 / w1,
                "Q3": None if w2 == 0 else float(evaluation["clear_wave_3_probability"]) / w2,
                "Return": float(evaluation["average_return"]),
                "Boundary": float(evaluation["average_red_boundary_exits"]),
                "Ground": float(evaluation["average_red_ground_losses"]),
                "global_clip_scale_proxy": proxy["global_clip_scale_proxy"],
                "w1_to_later_grad_ratio": proxy["w1_to_later_grad_ratio"],
                **{f"actor{pair}_relative_l2": float(optimization[f"wsai_actor{pair}_relative_l2"]) for pair in ("12", "13", "23")},
            }
            alignment_csv.append(row); eval_alignment[seed][step] = row

    evidence = []
    if pattern == "HEAD_ENRICHED":
        evidence.append("wave-specific head/residual specialization has direct sensitivity support.")
    elif pattern == "BACKBONE_ENRICHED":
        evidence.append("head-only specialization may miss a substantial sensitivity-bearing shared subspace.")
    else:
        evidence.append("current family evidence does not isolate the next sharing boundary.")
    if coupling == "STRONG_SIGNAL":
        evidence.append("WSAI efficacy is confounded by strong shared global-clipping pressure; do not attribute WSAI failure solely to loss of cross-wave transfer.")
    elif coupling == "WEAK_SIGNAL":
        evidence.append("shared global clipping is unlikely to explain most WSAI failure.")
    else:
        evidence.append("current clipping evidence does not isolate the next sharing boundary.")
    next_evidence = {"W1SG_FAMILY_PATTERN": pattern, "WSAI_GLOBAL_CLIP_COUPLING": coupling,
                     "statements": evidence}
    report = {
        "analysis": "PARTIAL_SHARING_EVIDENCE_AUDIT", "strictly_offline": True,
        "training_seed_is_replication_unit": True, "inferential_statistics": False,
        "protocol_validation": protocol,
        "w1sg_parameter_families": {"topology_identical_across_seeds": True, "families": families},
        "w1sg_per_seed": w1sg_per_seed, "w1sg_windowed": w1sg_windowed,
        "w1sg_family_enrichment_rankings": rankings,
        "wsai_per_seed": wsai_per_seed, "wsai_windowed": wsai_windowed,
        "wsai_specialization_endpoints": specialization_endpoints,
        "wsai_eval_alignment": eval_alignment,
        "wsai_clip_coupling_inputs": coupling_inputs,
        "wsai_clip_coupled_seed_flags": coupling_flags,
        "W1SG_FAMILY_PATTERN": pattern,
        "WSAI_GLOBAL_CLIP_COUPLING": coupling,
        "WSAI_GLOBAL_CLIP_COUPLING_NOTE": "mechanism-screen heuristic, not a statistical test",
        "NEXT_STRUCTURE_EVIDENCE": next_evidence,
        "uses_45m": False, "new_training": False, "new_policy_evaluation": False,
    }
    OUTPUT.mkdir(parents=True)
    write_csv(OUTPUT / "w1sg_family_importance.csv", family_csv)
    write_csv(OUTPUT / "w1sg_gating_dynamics.csv", gating_csv)
    write_csv(OUTPUT / "wsai_gradient_budget.csv", gradient_csv)
    write_csv(OUTPUT / "wsai_specialization_dynamics.csv", specialization_csv)
    write_csv(OUTPUT / "wsai_eval_alignment.csv", alignment_csv)
    (OUTPUT / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["PARTIAL_SHARING_EVIDENCE_AUDIT_COMPLETE", f"W1SG_FAMILY_PATTERN={pattern}",
             f"WSAI_GLOBAL_CLIP_COUPLING={coupling}", *evidence]
    (OUTPUT / "decision_support.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": lines[0], "W1SG_FAMILY_PATTERN": pattern,
                      "WSAI_GLOBAL_CLIP_COUPLING": coupling,
                      "analysis_json": str(OUTPUT / "analysis.json")}, indent=2))


if __name__ == "__main__":
    main()
