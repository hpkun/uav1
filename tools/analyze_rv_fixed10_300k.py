#!/usr/bin/env python3
"""Fail-closed, read-only exact-endpoint analyzer for the six-run RV screen."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256, runtime_source_manifest
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
from algorithm.modular_mappo.trainer import MODULAR_MAPPO_IMPL_VERSION
from algorithm.train_modular_mappo import load_config
from tools.preflight_rv_fixed10_300k import validate_plain_feedforward_architecture

SEEDS = (5301, 5302, 5303)
SOURCE_STEP = 1_505_280
TARGET = 1_805_280
OUT = ROOT / "outputs/rv_fixed10_300k_analysis"
ENVIRONMENT = ROOT / "configs/persistent_wave_v2_environment.yaml"
CONFIGS = {
    "Control": ROOT / "configs/dev_rv_fixed10_control_300k.yaml",
    "RV": ROOT / "configs/dev_rv_mappo_v1_300k.yaml",
}
RUNS = {
    "Control": lambda seed: ROOT / f"outputs/dev_rv_control_seed{seed}_300k",
    "RV": lambda seed: ROOT / f"outputs/dev_rv_v1_seed{seed}_300k",
}
METHODS = {"Control": "rv_fixed10_control", "RV": "rv_mappo_v1"}
ENABLED = {
    "Control": ["actor_gradient_clipping", "actor_lr_decay"],
    "RV": ["actor_gradient_clipping", "actor_lr_decay", "reference_variance"],
}
KEY_RUNTIME_FILES = {
    "algorithm/modular_mappo/trainer.py",
    "algorithm/modules/reference_variance.py",
    "algorithm/modular_mappo/runner.py",
    "algorithm/train_modular_mappo.py",
}
FIELDS = {
    "AW": "average_waves_cleared", "W1": "clear_wave_1_probability",
    "W2": "clear_wave_2_probability", "W3": "clear_wave_3_probability",
    "Return": "average_return", "RedLoss": "average_red_loss",
    "Boundary": "average_red_boundary_exits", "Ground": "average_red_ground_losses",
    "EpisodeLength": "average_episode_length",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def actor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_rv_frozen_actor_state(endpoint_state: dict[str, Any], source_state: dict[str, Any],
                                   source_sha: str, seed: int, label: str,
                                   branch_parent_sha: str | None = None) -> dict[str, bool]:
    """Fail closed on every endpoint invariant that defines RV V1."""
    required=("reference_variance_actor_state","reference_variance_actor_sha256",
              "reference_variance_source_checkpoint_sha256","reference_variance_source_sampled_steps",
              "reference_variance_source_training_seed","reference_variance_state")
    missing=[key for key in required if endpoint_state.get(key) is None]
    if missing:raise RuntimeError(f"seed{seed} {label} RV checkpoint lacks: {missing}")
    for name in ("log_std.weight","log_std.bias"):
        source=source_state["actor"][name];endpoint=endpoint_state["actor"][name]
        if not torch.equal(endpoint,source):
            maximum=float((endpoint-source).abs().max())
            raise RuntimeError(f"seed{seed} {label} {name} is not bitwise frozen; max_abs_diff={maximum}")
    embedded_sha=actor_state_sha256(endpoint_state["reference_variance_actor_state"])
    source_actor_sha=actor_state_sha256(source_state["actor"])
    stored_sha=endpoint_state["reference_variance_actor_sha256"]
    if stored_sha!=embedded_sha or embedded_sha!=source_actor_sha:
        raise RuntimeError(f"seed{seed} {label} RV reference actor/source SHA mismatch: stored={stored_sha}, embedded={embedded_sha}, source={source_actor_sha}")
    reference_source_sha=endpoint_state["reference_variance_source_checkpoint_sha256"]
    expected_parent=source_sha if branch_parent_sha is None else branch_parent_sha
    if reference_source_sha!=source_sha or reference_source_sha!=expected_parent:
        raise RuntimeError(f"seed{seed} {label} RV source checkpoint SHA mismatch")
    if int(endpoint_state["reference_variance_source_sampled_steps"])!=SOURCE_STEP:
        raise RuntimeError(f"seed{seed} {label} RV source sampled step mismatch")
    if int(endpoint_state["reference_variance_source_training_seed"])!=seed:
        raise RuntimeError(f"seed{seed} {label} RV source training seed mismatch")
    return {"reference_actor_matches_source":True,"log_std_weight_bitwise_frozen":True,
            "log_std_bias_bitwise_frozen":True,"reference_source_checkpoint_sha_match":True}


def validate_rv_optimization_rows(rows: list[dict[str, Any]], label: str) -> dict[str, bool]:
    exact={"rv_enabled":1.0,"rv_log_std_head_requires_grad":0.0,
           "rv_log_std_head_optimizer_membership":1.0,"rv_log_std_head_grad_norm":0.0,
           "rv_reference_any_grad_present":0.0,"rv_reference_mutation_detected":0.0}
    for index,row in enumerate(rows):
        for key,expected in exact.items():
            if key not in row:raise RuntimeError(f"{label} row{index} missing {key}")
            if float(row[key])!=expected:raise RuntimeError(f"{label} row{index} {key}={row[key]}, expected {expected}")
        for wave in (1,2,3):
            count_key=f"rv_wave{wave}_alive_sample_count";error_key=f"rv_wave{wave}_behavior_vs_reference_std_max_abs_error"
            if count_key not in row or error_key not in row:raise RuntimeError(f"{label} row{index} missing wave{wave} RV diagnostic")
            count=float(row[count_key]);error=row[error_key]
            if count>0:
                if error in (None,""):raise RuntimeError(f"{label} row{index} wave{wave} has samples but undefined sigma error")
                if abs(float(error))>1e-7:raise RuntimeError(f"{label} row{index} wave{wave} behavior/reference sigma error={error}")
            elif error not in (None,"") and abs(float(error))>1e-7:
                raise RuntimeError(f"{label} row{index} wave{wave} empty-sample sigma error is nonzero")
    return {"rv_training_diagnostics_pass":True,"reference_mutation_detected_any":False,
            "reference_grad_present_any":False,"current_log_std_grad_present_any":False,
            "current_log_std_optimizer_membership_all":True,"behavior_reference_sigma_match_all":True}


def validate_control_optimization_rows(rows: list[dict[str, Any]], label: str) -> None:
    for index,row in enumerate(rows):
        if "rv_enabled" not in row:raise RuntimeError(f"{label} row{index} missing rv_enabled")
        if float(row["rv_enabled"])!=0.0:raise RuntimeError(f"{label} row{index} Control rv_enabled must be 0")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, data: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in data:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def metrics(row: dict[str, Any]) -> dict[str, float | None]:
    result = {key: float(row[field]) for key, field in FIELDS.items()}
    result["Q2"] = None if result["W1"] == 0 else result["W2"] / result["W1"]
    result["Q3"] = None if result["W2"] == 0 else result["W3"] / result["W2"]
    return result


def delta(high: dict[str, Any], low: dict[str, Any], key: str) -> float | None:
    return None if high[key] is None or low[key] is None else high[key] - low[key]


def evaluation_interval(row: dict[str, Any]) -> tuple[int, int, int]:
    start_value = row.get("seed_start", row.get("evaluation_seed_base"))
    if start_value in (None, ""):
        raise RuntimeError("evaluation row lacks seed_start/evaluation_seed_base")
    start = int(float(start_value))
    episodes_value = row.get("evaluation_episodes", row.get("episodes"))
    if episodes_value in (None, ""):
        raise RuntimeError("evaluation row lacks episode count")
    episodes = int(float(episodes_value))
    end_value = row.get("seed_end", row.get("evaluation_seed_end"))
    end = start + episodes - 1 if end_value in (None, "") else int(float(end_value))
    if episodes <= 0 or end != start + episodes - 1:
        raise RuntimeError(f"inconsistent evaluation interval: {start}..{end}, episodes={episodes}")
    return start, end, episodes


def intervals_intersect(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1])


def validate_evaluation_history(history: list[dict[str, Any]]) -> dict[str, Any]:
    endpoints = []
    for row in history:
        start, end, episodes = evaluation_interval(row)
        if intervals_intersect((start, end), (45_000_000, 45_000_199)):
            raise RuntimeError(f"evaluation interval intersects reserved 45M: {start}..{end}")
        if int(float(row["sampled_steps"])) == TARGET:
            endpoints.append(row)
    if len(endpoints) != 1:
        raise RuntimeError(f"expected exactly one exact endpoint row, got {len(endpoints)}")
    if evaluation_interval(endpoints[0]) != (44_000_000, 44_000_049, 50):
        raise RuntimeError("exact endpoint is not the registered 44M 50-episode interval")
    return endpoints[0]


def validate_run_config(run: dict[str, Any], method: str, seed: int, environment_hash: str,
                        algorithm_hash: str, module_hash: str) -> None:
    expected = {
        "algorithm": "modular_mappo", "seed": seed, "total_sampled_steps": TARGET,
        "num_envs": 24, "smoke": False, "device": "cuda",
        "environment_variant": "persistent_wave_v2",
        "environment_config_sha256": environment_hash,
        "algorithm_config_sha256": algorithm_hash,
        "development_method": METHODS[method],
        "module_config_sha256": module_hash,
        "modular_mappo_impl_version": MODULAR_MAPPO_IMPL_VERSION,
        "baseline_mappo_impl_version": MAPPO_IMPL_VERSION,
    }
    mismatches = {key: (run.get(key), value) for key, value in expected.items() if run.get(key) != value}
    if mismatches:
        raise RuntimeError(f"run_config mismatch: {mismatches}")
    if run.get("enabled_modules") != ENABLED[method]:
        raise RuntimeError(f"run_config enabled_modules mismatch: {run.get('enabled_modules')}")
    validate_plain_feedforward_architecture(run.get("network_architecture", {}))


def validate_checkpoint(state: dict[str, Any], run: dict[str, Any], seed: int,
                        source_state: dict[str, Any], source_sha: str, label: str,
                        branch_parent_sha: str | None = None) -> dict[str, bool]:
    extra = state.get("extra", {})
    expected = {
        "algorithm": "modular_mappo",
        "modular_mappo_impl_version": MODULAR_MAPPO_IMPL_VERSION,
        "baseline_mappo_impl_version": MAPPO_IMPL_VERSION,
        "sampled_steps": TARGET,
        "module_config_sha256": run["module_config_sha256"],
    }
    mismatches = {key: (state.get(key), value) for key, value in expected.items() if state.get(key) != value}
    if mismatches:
        raise RuntimeError(f"{label} checkpoint mismatch: {mismatches}")
    extra_expected = {
        "training_seed": seed,
        "training_total_sampled_steps": TARGET,
        "training_num_envs": 24,
        "training_smoke": False,
        "environment_variant": "persistent_wave_v2",
        "environment_config_sha256": run["environment_config_sha256"],
        "algorithm_config_sha256": run["algorithm_config_sha256"],
        "runtime_source_manifest_sha256": run["runtime_source_manifest_sha256"],
    }
    extra_mismatches = {key: (extra.get(key), value) for key, value in extra_expected.items() if extra.get(key) != value}
    if extra_mismatches:
        raise RuntimeError(f"{label} checkpoint extra mismatch: {extra_mismatches}")
    if state.get("enabled_modules") != run["enabled_modules"]:
        raise RuntimeError(f"{label} checkpoint module identity mismatch")
    rv_enabled="reference_variance" in run["enabled_modules"]
    if rv_enabled:
        integrity=validate_rv_frozen_actor_state(state,source_state,source_sha,seed,label,branch_parent_sha)
    elif state.get("reference_variance_actor_state") is not None:
        raise RuntimeError(f"{label} Control unexpectedly contains an RV reference actor")
    else:integrity={}
    validate_plain_feedforward_architecture(extra.get("network_architecture", {}))
    deltas = {
        "ppo_updates": int(state.get("ppo_updates", -1)) - int(source_state.get("ppo_updates", -1)),
        "actor_updates": int(state.get("actor_updates", -1)) - int(source_state.get("actor_updates", -1)),
        "critic_updates": int(state.get("critic_updates", -1)) - int(source_state.get("critic_updates", -1)),
    }
    if deltas != {"ppo_updates": 49, "actor_updates": 5860, "critic_updates": 5860}:
        raise RuntimeError(f"{label} checkpoint update deltas mismatch: {deltas}")
    return integrity


def validate_optimization_metrics(rows: list[dict[str, Any]]) -> tuple[int, int]:
    if len(rows) != 49:
        raise RuntimeError(f"expected 49 optimization rows, got {len(rows)}")
    actor_steps = sum(int(row.get("actor_optimizer_steps_this_update", -1)) for row in rows)
    critic_steps = sum(int(row.get("critic_optimizer_steps_this_update", -1)) for row in rows)
    if (actor_steps, critic_steps) != (5860, 5860):
        raise RuntimeError(f"optimization step sums mismatch: {(actor_steps, critic_steps)}")
    if any(int(row.get("ppo_epochs_executed", -1)) != 10 for row in rows):
        raise RuntimeError("not every optimization row executed 10 PPO epochs")
    return actor_steps, critic_steps


def validate_run_summary(summary: dict[str, Any], run: dict[str, Any], method: str) -> None:
    expected = {
        "algorithm": "modular_mappo", "sampled_steps": TARGET,
        "latest_step": TARGET, "development_method": METHODS[method],
    }
    mismatches = {key: (summary.get(key), value) for key, value in expected.items() if summary.get(key) != value}
    if mismatches:
        raise RuntimeError(f"run_summary mismatch: {mismatches}")
    protocol = summary.get("protocol", {})
    protocol_expected = {
        "environment_config_sha256": run["environment_config_sha256"],
        "algorithm_config_sha256": run["algorithm_config_sha256"],
        "module_config_sha256": run["module_config_sha256"],
    }
    protocol_mismatches = {key: (protocol.get(key), value) for key, value in protocol_expected.items() if protocol.get(key) != value}
    if protocol_mismatches:
        raise RuntimeError(f"run_summary protocol mismatch: {protocol_mismatches}")
    if protocol.get("enabled_modules") != ENABLED[method]:
        raise RuntimeError("run_summary enabled_modules mismatch")
    latest = summary.get("latest_evaluation")
    if latest is not None:
        if int(latest.get("sampled_steps", -1)) != TARGET:
            raise RuntimeError("run_summary latest_evaluation is not exact endpoint")
        if evaluation_interval(latest) != (44_000_000, 44_000_049, 50):
            raise RuntimeError("run_summary latest_evaluation protocol mismatch")


def runtime_file_map(run: dict[str, Any]) -> dict[str, str]:
    rows = run.get("runtime_source_manifest_files")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("run lacks complete runtime_source_manifest_files")
    result = {str(row["path"]).replace("\\", "/"): row["sha256"] for row in rows}
    missing = sorted(KEY_RUNTIME_FILES - result.keys())
    if missing:
        raise RuntimeError(f"runtime manifest lacks key files: {missing}")
    if not any(path.startswith("env/") and path.endswith(".py") for path in result):
        raise RuntimeError("runtime manifest lacks env Python sources")
    return result


def validate_formal_run_artifacts(path: Path) -> None:
    """Explain absent/not-yet-complete formal runs without masking the protocol failure."""
    required=("run_config.json","run_summary.json","branch_from.json","evaluation_history.csv",
              "optimization_metrics.jsonl","latest.pt","final.pt")
    if not path.is_dir():
        raise RuntimeError(f"formal run has not been started or migrated: {path}; run tools/run_rv_fixed10_300k.sh first")
    missing=[name for name in required if not (path/name).is_file()]
    if missing:
        raise RuntimeError(f"formal run is incomplete: {path}; missing={missing}; wait for the six-run launcher to finish before analysis")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {OUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory for checkpoint audit")

    environment = load_config(ENVIRONMENT)
    environment_hash = config_sha256(environment)
    configurations = {method: load_config(path) for method, path in CONFIGS.items()}
    algorithm_hashes = {method: config_sha256(config) for method, config in configurations.items()}
    module_hashes = {method: config_sha256(config["modules"]) for method, config in configurations.items()}
    endpoint: dict[int, dict[str, dict[str, Any]]] = {}
    paired: list[dict[str, Any]] = []
    mechanism: list[dict[str, Any]] = []
    protocol: dict[str, Any] = {}
    training_runtime_sha: str | None = None
    training_manifest: dict[str, str] | None = None
    rv_integrity={"reference_actor_matches_source_all_runs":True,
                  "current_log_std_bitwise_frozen_all_runs":True,
                  "reference_mutation_detected_any":False,"reference_grad_present_any":False,
                  "current_log_std_grad_present_any":False,
                  "current_log_std_optimizer_membership_all":True,
                  "behavior_reference_sigma_match_all":True}

    for seed in SEEDS:
        endpoint[seed] = {}
        source_path = ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt"
        source_sha = file_sha256(source_path)
        source_state = torch.load(source_path, map_location="cuda", weights_only=False)
        if int(source_state.get("sampled_steps", -1)) != SOURCE_STEP:
            raise RuntimeError(f"source checkpoint step mismatch: seed{seed}")
        for method, path_fn in RUNS.items():
            path = path_fn(seed)
            validate_formal_run_artifacts(path)
            run = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
            summary = json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
            branch = json.loads((path / "branch_from.json").read_text(encoding="utf-8"))
            history = csv_rows(path / "evaluation_history.csv")
            optimization = jsonl_rows(path / "optimization_metrics.jsonl")

            validate_run_config(run, method, seed, environment_hash, algorithm_hashes[method], module_hashes[method])
            if branch.get("parent_checkpoint_sha256") != source_sha or int(branch.get("source_training_seed", -1)) != seed:
                raise RuntimeError(f"parent checkpoint provenance mismatch: {path}")
            final_row = validate_evaluation_history(history)
            actor_steps, critic_steps = validate_optimization_metrics(optimization)
            validate_run_summary(summary, run, method)
            final_state = torch.load(path / "final.pt", map_location="cuda", weights_only=False)
            latest_state = torch.load(path / "latest.pt", map_location="cuda", weights_only=False)
            final_integrity=validate_checkpoint(final_state,run,seed,source_state,source_sha,"final.pt",branch.get("parent_checkpoint_sha256"))
            latest_integrity=validate_checkpoint(latest_state,run,seed,source_state,source_sha,"latest.pt",branch.get("parent_checkpoint_sha256"))
            if method=="RV":
                diagnostic_integrity=validate_rv_optimization_rows(optimization,f"RV seed{seed}")
                if final_state["reference_variance_actor_sha256"]!=latest_state["reference_variance_actor_sha256"]:
                    raise RuntimeError(f"seed{seed} final/latest RV reference actor SHA mismatch")
                rv_protocol={**final_integrity,**diagnostic_integrity}
            else:
                validate_control_optimization_rows(optimization,f"Control seed{seed}");rv_protocol={}
            if any(final_state.get(key) != latest_state.get(key) for key in ("sampled_steps", "ppo_updates", "actor_updates", "critic_updates")):
                raise RuntimeError(f"final.pt/latest.pt endpoint counters differ: {path}")

            manifest = runtime_file_map(run)
            run_runtime_sha = run.get("runtime_source_manifest_sha256")
            if not isinstance(run_runtime_sha, str):
                raise RuntimeError(f"missing runtime source SHA: {path}")
            if training_runtime_sha is None:
                training_runtime_sha, training_manifest = run_runtime_sha, manifest
            elif run_runtime_sha != training_runtime_sha or manifest != training_manifest:
                raise RuntimeError("six runs do not share one complete runtime source manifest")

            endpoint[seed][method] = metrics(final_row)
            protocol[f"{method}_seed{seed}"] = {
                "status": "PASS", "runtime_source_sha": run_runtime_sha,
                "parent_sha": source_sha, "optimization_rows": 49,
                "actor_steps": actor_steps, "critic_steps": critic_steps,
                "final_latest_same_endpoint": True, "uses_45m": False,**rv_protocol,
            }
            windows = {"early": optimization[:10], "late": optimization[-16:], "final5": optimization[-5:]}
            for window, selected in windows.items():
                for wave in (1, 2, 3):
                    def mean(key: str) -> float | None:
                        values = [float(row[key]) for row in selected if row.get(key) not in (None, "")]
                        return statistics.mean(values) if values else None
                    mechanism.append({
                        "method": method, "seed": seed, "window": window, "wave": wave,
                        "reference_behavior_std_mean": mean(f"rv_wave{wave}_reference_behavior_std_mean"),
                        "current_unused_std_mean": mean(f"rv_wave{wave}_current_unused_std_mean"),
                        "current_unused_over_reference_std_ratio": mean(f"rv_wave{wave}_current_unused_over_reference_std_ratio"),
                        "behavior_vs_reference_std_max_abs_error": mean(f"rv_wave{wave}_behavior_vs_reference_std_max_abs_error"),
                        "reference_log_std_mean": mean(f"rv_wave{wave}_reference_log_std_mean"),
                        "current_unused_log_std_mean": mean(f"rv_wave{wave}_current_unused_log_std_mean"),
                        "reference_mutation_detected": mean("rv_reference_mutation_detected"),
                        "log_std_head_grad_norm": mean("rv_log_std_head_grad_norm"),
                        "approx_kl": mean("approx_kl"),"clip_fraction":mean("clip_fraction"),
                        "entropy":mean("entropy"),"actor_gradient_norm":mean("actor_grad_norm"),
                    })
        del source_state
        for key in (*FIELDS, "Q2", "Q3"):
            paired.append({
                "seed": seed, "metric": key,
                "Control": endpoint[seed]["Control"][key], "RV": endpoint[seed]["RV"][key],
                "delta": delta(endpoint[seed]["RV"], endpoint[seed]["Control"], key),
            })

    aggregate: dict[str, Any] = {}
    for key in (*FIELDS, "Q2", "Q3"):
        rows = [row for row in paired if row["metric"] == key]
        values = [row["delta"] for row in rows if row["delta"] is not None]
        aggregate[key] = {
            "mean_delta": statistics.mean(values) if values else None,
            "wins": sum(value > 0 for value in values), "defined_n": len(values),
            "per_seed": {str(row["seed"]): row["delta"] for row in rows},
        }
    safety = all(aggregate["AW"]["per_seed"][str(seed)] > -0.50 and aggregate["W1"]["per_seed"][str(seed)] > -0.25 for seed in SEEDS) and aggregate["W1"]["mean_delta"] >= -0.05
    efficacy = aggregate["AW"]["wins"] >= 2 and aggregate["AW"]["mean_delta"] > 0 and aggregate["W3"]["wins"] >= 2 and aggregate["W3"]["mean_delta"] > 0
    label = "SAFETY_FAIL" if not safety else "PROMISING" if efficacy else "NOT_SUPPORTED"

    current_manifest = runtime_source_manifest(ROOT)
    current_sha = current_manifest["runtime_source_manifest_sha256"]
    runtime_matches = current_sha == training_runtime_sha
    report = {
        "status": "RV_FIXED10_300K_ANALYSIS_COMPLETE",
        "replication_unit": "training_seed", "n": 3, "primary_endpoint": TARGET,
        "protocol": protocol, "endpoint": endpoint,
        "paired_RV_minus_Control": aggregate, "mechanism_diagnostics": mechanism,
        "rv_integrity":rv_integrity,
        "gates": {"safety": safety, "efficacy": efficacy}, "RV_SCREEN": label,
        "evaluation_rerun": False, "uses_45m": False,
        "runtime_source_provenance": {
            "training_runtime_source_sha256": training_runtime_sha,
            "current_runtime_source_sha256": current_sha,
            "runtime_source_matches_current": runtime_matches,
            "warning": None if runtime_matches else "CURRENT_SOURCE_DIFFERS_FROM_TRAINING_SOURCE",
            "six_run_manifests_identical": True,
            "key_training_source_files": {key: training_manifest[key] for key in sorted(KEY_RUNTIME_FILES)},
        },
    }
    OUT.mkdir(parents=True)
    (OUT / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "protocol_validation.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    write_csv(OUT / "paired_endpoint.csv", paired)
    write_csv(OUT / "mechanism_diagnostics.csv", mechanism)
    (OUT / "decision_support.txt").write_text(f"RV_SCREEN={label}\nSafety={safety}\nEfficacy={efficacy}\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "label": label, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()

