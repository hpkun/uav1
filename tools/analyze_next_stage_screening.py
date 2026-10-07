"""Post-run analysis for matched All-Off/M5 and M6+M8 anchor screening.

This tool never trains.  It discovers completed formal runs from their metadata,
uses diagnostic seeds disjoint from the 20M holdout, and keeps every raw Pareto
metric instead of constructing a composite score.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "next_stage_screening"
CACHE = OUT / "evaluation_cache"
PW_ENV = ROOT / "configs" / "persistent_wave_v2_environment.yaml"
DIRECT_ENV = ROOT / "configs" / "combat_environment.yaml"
HOLDOUt = range(20_000_000, 20_000_200)
ANCHOR_COEFFICIENTS = (0.001, 0.003, 0.01, 0.03, 0.10)

SUMMARY_METRICS = {
    "W1": "clear_wave_1_probability", "W2": "clear_wave_2_probability",
    "W3": "clear_wave_3_probability", "average_waves": "average_waves_cleared",
    "return": "average_return", "red_loss": "average_red_loss",
    "blue_loss": "average_blue_loss",
    "K_L": "kill_loss_ratio", "boundary": "average_red_boundary_exits",
    "ground": "average_red_ground_losses",
    "timeout": "timeout_rate", "episode_length": "average_episode_length",
}


def validate_diagnostic_seeds(seeds) -> list[int]:
    values = [int(seed) for seed in seeds]
    if not values or len(values) != len(set(values)):
        raise ValueError("diagnostic seeds must be non-empty and unique")
    if set(values).intersection(HOLDOUt):
        raise ValueError("20M formal holdout seeds are forbidden")
    return values


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(path: str | Path) -> dict[str, Any]:
    import torch
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    state = torch.load(source, map_location="cpu", weights_only=False)
    extra = state.get("extra", {})
    return {
        "path": str(source), "sha256": sha256_file(source),
        "sampled_steps": int(state.get("sampled_steps", 0)),
        "training_seed": extra.get("training_seed"),
        "environment_variant": extra.get("environment_variant", "direct_v2_3"),
    }


def validate_same_source_checkpoints(warm_start: str | Path, reference: str | Path) -> dict[str, Any]:
    warm, reference_meta = source_identity(warm_start), source_identity(reference)
    fields = ("sha256", "sampled_steps", "training_seed", "environment_variant")
    mismatches = [field for field in fields if warm[field] != reference_meta[field]]
    if mismatches:
        raise RuntimeError(f"warm-start/reference checkpoint mismatch: {', '.join(mismatches)}")
    return warm


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_snapshot(directory: Path) -> dict[str, Any]:
    snapshot = directory / "algorithm_config.yaml"
    if not snapshot.is_file():
        return {}
    return yaml.safe_load(snapshot.read_text(encoding="utf-8"))


def discover_runs(outputs: Path = ROOT / "outputs") -> dict[str, Any]:
    """Discover the eight required runs using protocol metadata, not names."""
    buckets: dict[str, list[Path]] = {"All-Off": [], "M5": [], "M6 control": [], "Direct source": []}
    anchors: dict[float, list[Path]] = {coefficient: [] for coefficient in ANCHOR_COEFFICIENTS}
    for directory in outputs.iterdir():
        config_path, summary_path = directory / "run_config.json", directory / "run_summary.json"
        if not directory.is_dir() or not config_path.is_file() or not summary_path.is_file():
            continue
        try:
            run, summary = _read_json(config_path), _read_json(summary_path)
        except (OSError, json.JSONDecodeError):
            continue
        if int(run.get("seed", -1)) != 2023 or bool(run.get("smoke", False)):
            continue
        algorithm, variant = run.get("algorithm"), run.get("environment_variant")
        steps, modules = int(summary.get("sampled_steps", -1)), set(run.get("enabled_modules", []))
        snapshot = _resolved_snapshot(directory)
        gamma = run.get("training_gamma", snapshot.get("training", {}).get("gamma"))
        if int(run.get("num_envs", -1)) != 24 or gamma is None or not np.isclose(float(gamma), .999):
            continue
        if algorithm == "MAPPO" and variant == "direct_v2_3" and steps == 3_000_000:
            buckets["Direct source"].append(directory)
        elif algorithm == "modular_mappo" and variant == "persistent_wave_v2" and steps == 1_500_000:
            if modules == set(): buckets["All-Off"].append(directory)
            elif modules == {"wave_balancing"}: buckets["M5"].append(directory)
        elif algorithm == "modular_mappo" and variant == "persistent_wave_v2" and steps == 300_000:
            if modules == {"warm_start"}:
                buckets["M6 control"].append(directory)
            elif modules == {"warm_start", "policy_anchor"}:
                coefficient = float(snapshot.get("modules", {}).get("policy_anchor", {}).get("coefficient", -1))
                for expected in ANCHOR_COEFFICIENTS:
                    if np.isclose(coefficient, expected, rtol=0, atol=1e-12):
                        anchors[expected].append(directory)
                        break
    ambiguous = {label: paths for label, paths in buckets.items() if len(paths) != 1}
    ambiguous.update({f"anchor {coefficient:g}": paths for coefficient, paths in anchors.items() if len(paths) != 1})
    if ambiguous:
        raise RuntimeError(f"required run discovery failed or ambiguous: {ambiguous}")
    return {**{label: paths[0] for label, paths in buckets.items()},
            "anchors": {coefficient: paths[0] for coefficient, paths in anchors.items()}}


def validate_run_provenance(runs: dict[str, Any]) -> dict[str, Any]:
    direct = source_identity(runs["Direct source"] / "best_eval.pt")
    candidates = {"M6 control": runs["M6 control"], **{f"anchor {c:g}": p for c, p in runs["anchors"].items()}}
    for label, directory in candidates.items():
        run = _read_json(directory / "run_config.json")
        warm, anchor = run.get("warm_start_provenance", {}), run.get("policy_anchor_provenance", {})
        for key, expected in (("source_checkpoint_sha256", direct["sha256"]),
                              ("source_sampled_steps", direct["sampled_steps"]),
                              ("source_training_seed", direct["training_seed"]),
                              ("source_environment_variant", direct["environment_variant"])):
            if warm.get(key) != expected:
                raise RuntimeError(f"{label}: warm-start provenance mismatch for {key}")
        if label.startswith("anchor"):
            for key, expected in (("source_checkpoint_sha256", direct["sha256"]),
                                  ("source_sampled_steps", direct["sampled_steps"]),
                                  ("source_training_seed", direct["training_seed"]),
                                  ("source_environment_variant", direct["environment_variant"])):
                if anchor.get(key) != expected:
                    raise RuntimeError(f"{label}: policy-anchor provenance mismatch for {key}")
    return direct


def checkpoint_step(path: Path) -> int:
    import torch
    return int(torch.load(path, map_location="cpu", weights_only=False).get("sampled_steps", 0))


def checkpoint_roles(directory: Path) -> list[dict[str, Any]]:
    numeric = []
    for path in directory.glob("checkpoint_*.pt"):
        try: numeric.append((checkpoint_step(path), path))
        except (ValueError, OSError): continue
    if not numeric:
        raise RuntimeError(f"no periodic checkpoints in {directory}")
    rows = []
    for requested in (100_000, 200_000, 300_000):
        actual, path = min(numeric, key=lambda item: (abs(item[0] - requested), item[0]))
        rows.append({"checkpoint_role": f"{requested // 1000}k", "requested_step": requested,
                     "actual_step": actual, "checkpoint": path})
    for role, name in (("best", "best_eval.pt"), ("latest", "latest.pt")):
        path = directory / name
        rows.append({"checkpoint_role": role, "requested_step": None,
                     "actual_step": checkpoint_step(path), "checkpoint": path})
    return rows


def _evaluation_key(checkpoint: Path, env: Path, seed_base: int, episodes: int) -> tuple[str, str, int, int]:
    return str(checkpoint.resolve()), str(env.resolve()), int(seed_base), int(episodes)


def build_evaluation_plan(runs: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from tools.modular_1p5m_screening import task
    mappings, unique = [], {}
    def add(group, candidate, coefficient, role, requested, actual, checkpoint, env, seed, episodes, cross=False):
        key = _evaluation_key(checkpoint, env, seed, episodes)
        if key not in unique:
            name = f"next_{len(unique):03d}"
            unique[key] = task(name, candidate, checkpoint, env, seed, episodes, cross)
        mappings.append({"group": group, "candidate": candidate, "coefficient": coefficient,
                         "checkpoint_role": role, "requested_step": requested, "actual_step": actual,
                         "evaluation": unique[key]["name"]})
    for candidate in ("All-Off", "M5"):
        for role, name in (("best", "best_eval.pt"), ("latest", "latest.pt")):
            checkpoint = runs[candidate] / name
            add("matched", candidate, None, role, None, checkpoint_step(checkpoint), checkpoint,
                PW_ENV, 35_000_000, 50)
    source = runs["Direct source"] / "best_eval.pt"
    source_step = checkpoint_step(source)
    for env, group, seed, cross in ((PW_ENV, "anchor_pw", 35_100_000, True),
                                    (DIRECT_ENV, "anchor_direct", 35_200_000, False)):
        add(group, "Direct source", None, "source", None, source_step, source, env, seed, 30, cross)
    candidates = [("M6 control", 0.0, runs["M6 control"])] + [
        (f"anchor {coefficient:g}", coefficient, directory)
        for coefficient, directory in sorted(runs["anchors"].items())]
    for candidate, coefficient, directory in candidates:
        for row in checkpoint_roles(directory):
            for env, group, seed in ((PW_ENV, "anchor_pw", 35_100_000),
                                     (DIRECT_ENV, "anchor_direct", 35_200_000)):
                add(group, candidate, coefficient, row["checkpoint_role"], row["requested_step"],
                    row["actual_step"], row["checkpoint"], env, seed, 30, env == DIRECT_ENV)
    return list(unique.values()), mappings


def _evaluate_worker(spec: dict[str, Any], cache_dir: str) -> str:
    import tools.modular_1p5m_screening as common
    common.CACHE = Path(cache_dir)
    return common.evaluate_task(spec)


def run_evaluations(tasks: list[dict[str, Any]], workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_evaluate_worker, spec, str(CACHE)) for spec in tasks]
        for future in as_completed(futures):
            print(future.result(), flush=True)


def _payload(name: str) -> dict[str, Any]:
    return _read_json(CACHE / f"{name}.json")


def matched_episode_delta(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, float]:
    if candidate.seed.duplicated().any() or baseline.seed.duplicated().any():
        raise ValueError("duplicate seed in matched evaluation")
    merged = candidate.merge(baseline, on="seed", suffixes=("_candidate", "_baseline"), validate="one_to_one")
    if len(merged) != len(candidate) or len(merged) != len(baseline):
        raise ValueError("matched evaluation seed mismatch")
    episode_fields = {"W1":"clear_wave_1", "W2":"clear_wave_2", "W3":"clear_wave_3",
        "average_waves":"waves_cleared", "return":"episode_return", "red_loss":"red_losses",
        "blue_loss":"blue_losses", "K_L":"episode_kill_loss_ratio", "boundary":"red_boundary_exits",
        "ground":"red_ground_losses", "timeout":"timeout", "episode_length":"episode_length"}
    return {f"delta_{label}": float((merged[f"{field}_candidate"] - merged[f"{field}_baseline"]).mean())
            for label, field in episode_fields.items()}


def classify_candidate(delta_direct_win: float, persistent_deltas: dict[str, float]) -> str:
    """Descriptive Pareto label; deliberately returns no scalar score."""
    if float(delta_direct_win) < -0.10:
        return "FORGETTING"
    improvements = sum(float(persistent_deltas[field]) > 0 for field in
                       ("W3", "average_waves", "return", "K_L"))
    improvements += int(float(persistent_deltas["red_loss"]) < 0)
    progress = float(persistent_deltas["W3"]) > 0 or float(persistent_deltas["average_waves"]) > 0
    return "ADAPTATION_CANDIDATE" if improvements >= 3 and progress else "PRESERVATION_ONLY"


def _optimization_at(directory: Path, step: int) -> dict[str, Any]:
    path = directory / "optimization_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    row = min(rows, key=lambda item: (abs(int(item["sampled_steps"]) - int(step)), int(item["sampled_steps"])))
    return {name: row.get(name) for name in ("anchor_kl", "anchor_loss", "anchor_effective_coefficient",
        "approx_kl", "log_ratio_min", "log_ratio_max", "max_abs_log_ratio")}


def _read_optimization(directory: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in (directory / "optimization_metrics.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows).sort_values("sampled_steps")


def build_anchor_regularization_diagnostics(runs: dict[str, Any]) -> pd.DataFrame:
    candidates = [("M6 control", 0.0, runs["M6 control"])] + [
        (f"anchor {coefficient:g}", coefficient, directory)
        for coefficient, directory in sorted(runs["anchors"].items())]
    rows=[]
    for candidate, coefficient, directory in candidates:
        frame = _read_optimization(directory)
        for metric in ("anchor_kl", "anchor_loss", "anchor_effective_coefficient"):
            values = pd.to_numeric(frame[metric], errors="coerce").dropna().to_numpy(float)
            record={"candidate":candidate,"coefficient":coefficient,"metric":metric,
                    "mean":float(np.mean(values)),"median":float(np.median(values)),
                    "p90":float(np.quantile(values,.90)),"p95":float(np.quantile(values,.95)),
                    "max":float(np.max(values)),"finite":bool(np.isfinite(values).all())}
            for requested in (100_000,200_000,300_000):
                index=(frame.sampled_steps.astype(int)-requested).abs().idxmin()
                record[f"requested_{requested//1000}k_step"]=int(frame.loc[index,"sampled_steps"])
                record[f"value_near_{requested//1000}k"]=float(frame.loc[index,metric])
            rows.append(record)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"anchor_regularization_diagnostics.csv",index=False)
    return result


def build_anchor_training_stability(runs: dict[str, Any]) -> pd.DataFrame:
    candidates = [("M6 control", 0.0, runs["M6 control"])] + [
        (f"anchor {coefficient:g}", coefficient, directory)
        for coefficient, directory in sorted(runs["anchors"].items())]
    rows=[]
    for candidate, coefficient, directory in candidates:
        evaluation=pd.read_csv(directory/"evaluation_history.csv").sort_values("sampled_steps")
        optimization=_read_optimization(directory)
        for _, item in evaluation.iterrows():
            step=int(item.sampled_steps)
            index=(optimization.sampled_steps.astype(int)-step).abs().idxmin()
            opt=optimization.loc[index]
            rows.append({"candidate":candidate,"coefficient":coefficient,"sampled_steps":step,
                "clear_wave_1_probability":item.get("clear_wave_1_probability"),
                "clear_wave_2_probability":item.get("clear_wave_2_probability"),
                "clear_wave_3_probability":item.get("clear_wave_3_probability"),
                "average_waves_cleared":item.get("average_waves_cleared"),
                "average_return":item.get("average_return"),"average_red_loss":item.get("average_red_loss"),
                "average_red_boundary_exits":item.get("average_red_boundary_exits"),
                "optimization_actual_step":int(opt.sampled_steps),"anchor_kl":opt.get("anchor_kl"),
                "anchor_loss":opt.get("anchor_loss"),"anchor_effective_coefficient":opt.get("anchor_effective_coefficient"),
                "approx_kl":opt.get("approx_kl"),"log_ratio_min":opt.get("log_ratio_min"),
                "log_ratio_max":opt.get("log_ratio_max"),"max_abs_log_ratio":opt.get("max_abs_log_ratio")})
    result=pd.DataFrame(rows).sort_values(["coefficient","sampled_steps"])
    result["best_W3_so_far"]=result.groupby("candidate")["clear_wave_3_probability"].cummax()
    result["W3_drop_from_best"]=result["best_W3_so_far"]-result["clear_wave_3_probability"]
    result["descriptive_collapse_from_peak"]=(result["W3_drop_from_best"]>=.20)
    result.to_csv(OUT/"anchor_training_stability.csv",index=False)
    return result


def _summary_row(mapping: dict[str, Any]) -> dict[str, Any]:
    summary = _payload(mapping["evaluation"])["summary"]
    return {**mapping, "win_rate": summary["win_rate"],
            **{label: summary[field] for label, field in SUMMARY_METRICS.items()}}


def build_matched_tables(mappings: list[dict[str, Any]]) -> None:
    selected = [row for row in mappings if row["group"] == "matched"]
    summary = pd.DataFrame([_summary_row(row) for row in selected])
    episodes = []
    for row in selected:
        for episode in _payload(row["evaluation"])["episodes"]:
            episodes.append({"candidate": row["candidate"], "checkpoint_role": row["checkpoint_role"], **episode})
    episode_frame = pd.DataFrame(episodes)
    deltas = []
    for role in ("best", "latest"):
        baseline = episode_frame[(episode_frame.candidate == "All-Off") & (episode_frame.checkpoint_role == role)]
        candidate = episode_frame[(episode_frame.candidate == "M5") & (episode_frame.checkpoint_role == role)]
        deltas.append({"row_type":"paired_delta", "candidate":f"M5 - All-Off ({role})",
                       "checkpoint_role":role, **matched_episode_delta(candidate, baseline)})
    summary.insert(0, "row_type", "summary")
    pd.concat([summary, pd.DataFrame(deltas)], ignore_index=True, sort=False).to_csv(
        OUT / "matched_alloff_vs_m5_50ep.csv", index=False)
    episode_frame.to_csv(OUT / "matched_alloff_vs_m5_per_episode.csv", index=False)


def build_anchor_tables(runs: dict[str, Any], mappings: list[dict[str, Any]]) -> pd.DataFrame:
    run_dirs = {"M6 control": runs["M6 control"], **{f"anchor {c:g}": p for c, p in runs["anchors"].items()}}
    frames = {}
    for group, filename in (("anchor_pw", "anchor_screen_pw.csv"), ("anchor_direct", "anchor_screen_direct.csv")):
        rows=[]
        for mapping in [row for row in mappings if row["group"] == group]:
            record = _summary_row(mapping)
            if mapping["candidate"] != "Direct source":
                record.update(_optimization_at(run_dirs[mapping["candidate"]], mapping["actual_step"]))
            rows.append(record)
        frames[group] = pd.DataFrame(rows)
        frames[group].to_csv(OUT / filename, index=False)
    keys = ["candidate", "coefficient", "checkpoint_role", "requested_step", "actual_step"]
    pw = frames["anchor_pw"].rename(columns={column:f"PW_{column}" for column in SUMMARY_METRICS})
    direct_names={column:f"Direct_{column}" for column in ("win_rate",*SUMMARY_METRICS.keys())}
    direct_names["win_rate"]="Direct_win"
    direct = frames["anchor_direct"].rename(columns=direct_names)
    diagnostic = [c for c in ("anchor_kl","anchor_loss","anchor_effective_coefficient","approx_kl",
                               "log_ratio_min","log_ratio_max","max_abs_log_ratio") if c in direct]
    direct = direct.drop(columns=diagnostic)
    joint = pw.merge(direct, on=keys, validate="one_to_one")
    source = joint[joint.candidate == "Direct source"].iloc[0]
    joint["delta_Direct_win"] = joint["Direct_win"] - source["Direct_win"]
    joint["delta_Direct_return"] = joint["Direct_return"] - source["Direct_return"]
    for field in ("W3", "average_waves", "return", "red_loss", "K_L"):
        joint[f"delta_PW_{field}"] = joint[f"PW_{field}"] - source[f"PW_{field}"]
    def label(row):
        if row.candidate == "Direct source": return "SOURCE"
        return classify_candidate(row.delta_Direct_win, {
            field:row[f"delta_PW_{field}"] for field in ("W3","average_waves","return","red_loss","K_L")})
    joint["classification"] = joint.apply(label, axis=1)
    joint.to_csv(OUT / "anchor_screen_joint.csv", index=False)
    return joint


def _plot_pareto(joint: pd.DataFrame) -> None:
    plot = joint[(joint.candidate != "Direct source") & (joint.checkpoint_role.isin(["300k","best","latest"]))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for _, row in plot.iterrows():
        label = f"c={row.coefficient:g} {row.checkpoint_role}"
        axes[0].scatter(row.Direct_win, row.PW_W3); axes[0].annotate(label,(row.Direct_win,row.PW_W3),fontsize=7)
        axes[1].scatter(row.Direct_win, row.PW_average_waves); axes[1].annotate(label,(row.Direct_win,row.PW_average_waves),fontsize=7)
    source = joint[joint.candidate == "Direct source"].iloc[0]
    for axis, y, ylabel in ((axes[0],source.PW_W3,"PW W3"),(axes[1],source.PW_average_waves,"PW average waves")):
        axis.axvline(source.Direct_win - .10, color="tab:red", linestyle="--", label="source win - 0.10")
        axis.axhline(y, color="tab:gray", linestyle=":", label="untouched source")
        axis.set(xlabel="Direct win (preservation)", ylabel=ylabel); axis.grid(alpha=.2); axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "anchor_pareto.png", dpi=180); plt.close(fig)


def write_manifest(runs: dict[str, Any], tasks: list[dict[str, Any]], joint: pd.DataFrame | None = None) -> None:
    payload = {"runs": {key: ({str(c):str(p) for c,p in value.items()} if isinstance(value,dict) else str(value))
                        for key,value in runs.items()},
               "fresh_seed_ranges": {"matched":"35000000-35000049", "persistent":"35100000-35100029",
                                     "direct":"35200000-35200029"},
               "formal_holdout_used": False, "composite_score_used": False,
               "direct_source": source_identity(runs["Direct source"] / "best_eval.pt"),
               "evaluation_task_count": len(tasks),
               "tasks": tasks}
    if joint is not None:
        payload["classifications"] = joint[["candidate","coefficient","checkpoint_role","actual_step","classification"]].to_dict("records")
    def safe(value):
        if isinstance(value,dict): return {str(key):safe(item) for key,item in value.items()}
        if isinstance(value,list): return [safe(item) for item in value]
        if isinstance(value,np.generic): value=value.item()
        if isinstance(value,float) and not np.isfinite(value): return None
        return value
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("evaluate","report","all"), default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--preflight-sources", nargs=2, metavar=("WARM_START", "REFERENCE"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        _plot_pareto(pd.read_csv(OUT / "anchor_screen_joint.csv")); return
    if args.preflight_sources:
        print(json.dumps(validate_same_source_checkpoints(*args.preflight_sources), indent=2)); return
    runs = discover_runs(); validate_run_provenance(runs)
    tasks, mappings = build_evaluation_plan(runs)
    if args.mode in ("evaluate", "all"): run_evaluations(tasks, args.workers)
    if args.mode in ("report", "all"):
        missing = [spec["name"] for spec in tasks if not (CACHE / f"{spec['name']}.json").is_file()]
        if missing: raise RuntimeError(f"missing evaluation caches: {missing}; run --mode evaluate first")
        OUT.mkdir(parents=True, exist_ok=True)
        build_matched_tables(mappings); joint = build_anchor_tables(runs, mappings)
        build_anchor_regularization_diagnostics(runs)
        build_anchor_training_stability(runs)
        write_manifest(runs, tasks, joint)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--plot-only"], check=True)
    else: write_manifest(runs, tasks)


if __name__ == "__main__":
    main()
