"""Offline-only preregistered analysis of six matched team-credit branches."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from algorithm.common.protocol import aggregate_runtime_source_manifest

SEEDS = (5301, 5302, 5303)
SOURCE = 1_505_280
TARGET = 1_805_280
ADDITIONAL = 300_000
EVALUATION_SEED_START = 44_000_000
EVALUATION_SEED_END = 44_000_049
EVALUATION_EPISODES = 50
MIN_W2_ENTRY_COUNT_PER_BRANCH = 30

_TEXT_FAILURE_MARKERS = (
    "traceback", "out of memory", "cuda error", "floatingpointerror",
)
_EXPECTED_RUNTIME = {
    "control": {
        "development_method": "team_credit_matched_control",
        "enabled_modules": ["actor_lr_decay"],
        "intervention": "team_mean_credit_control",
    },
    "teammean": {
        "development_method": "team_credit_matched_teammean",
        "enabled_modules": ["actor_lr_decay", "team_mean_credit"],
        "intervention": "team_mean_credit",
    },
}


def assert_finite_numeric_tree(value: Any, source: str, path: str = "$") -> None:
    """Reject numeric NaN/Inf without treating ordinary strings as evidence."""
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise RuntimeError(f"{source}: non-finite numeric field {path}={value!r}")
        return
    if isinstance(value, dict):
        step = value.get("sampled_steps")
        row_source = source if step is None else f"{source} sampled_steps={step}"
        for key, item in value.items():
            assert_finite_numeric_tree(item, row_source, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite_numeric_tree(item, source, f"{path}[{index}]")


def check_text_failure_markers(text: str, source: str) -> None:
    lowered = text.lower()
    hits = [marker for marker in _TEXT_FAILURE_MARKERS if marker in lowered]
    if hits:
        raise RuntimeError(f"{source}: failure marker(s): {hits}")


def rows_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        assert_finite_numeric_tree(row, str(path))
    return rows


def endpoint(path: Path) -> dict[str, float | int | None]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    matches = [row for row in rows if int(row["sampled_steps"]) == TARGET]
    if len(matches) != 1:
        raise RuntimeError(f"{path}: exact {TARGET} evaluation missing/duplicated")
    row = matches[0]

    def number(key: str) -> float:
        value = float(row[key])
        if not math.isfinite(value):
            raise RuntimeError(f"{path}: non-finite endpoint field {key} at sampled_steps={TARGET}")
        return value

    w1, w2, w3 = (number(f"clear_wave_{wave}_probability") for wave in (1, 2, 3))
    result: dict[str, float | int | None] = {
        "W1": w1, "W2": w2, "W3": w3,
        "Q2": None if w1 == 0 else w2 / w1,
        "Q3": None if w2 == 0 else w3 / w2,
        "AverageWaves": number("average_waves_cleared"),
        "Return": number("average_return"),
        "RedLoss": number("average_red_loss"),
        "BlueLoss": number("average_blue_loss"),
        "Boundary": number("average_red_boundary_exits"),
        "Ground": number("average_red_ground_losses"),
        "EpisodeLength": number("average_episode_length"),
        "evaluation_episodes": int(number("evaluation_episodes")),
        "evaluation_seed_base": int(number("evaluation_seed_base")),
        "evaluation_seed_end": int(number("evaluation_seed_end")),
    }
    assert_finite_numeric_tree(result, f"{path} endpoint sampled_steps={TARGET}")
    return result


def optional_delta(treatment: float | None, control: float | None) -> float | None:
    return None if treatment is None or control is None else treatment - control


def w2_entry_sample_status(control_count: float, treatment_count: float) -> str:
    if control_count >= MIN_W2_ENTRY_COUNT_PER_BRANCH and treatment_count >= MIN_W2_ENTRY_COUNT_PER_BRANCH:
        return "VALID"
    return "INSUFFICIENT_ENTRY_SAMPLES"


def verify_six_branch_runtime_source_consistency(
    run_configs: list[tuple[str, dict[str, Any]]]
) -> dict[str, Any]:
    if len(run_configs) != 6:
        raise RuntimeError(f"expected six formal runtime manifests, got {len(run_configs)}")
    identities: list[tuple[str, int]] = []
    for label, run_config in run_configs:
        manifest_sha = run_config.get("runtime_source_manifest_sha256")
        file_count = run_config.get("runtime_source_manifest_file_count")
        files = run_config.get("runtime_source_manifest_files")
        if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
            raise RuntimeError(f"{label}: runtime source manifest SHA missing")
        if not isinstance(file_count, int) or isinstance(file_count, bool) or file_count <= 0:
            raise RuntimeError(f"{label}: runtime source manifest file_count invalid")
        if not isinstance(files, list) or not files or len(files) != file_count:
            raise RuntimeError(f"{label}: runtime source manifest files incomplete")
        paths = [item.get("path") for item in files if isinstance(item, dict)]
        if len(paths) != file_count or paths != sorted(paths) or len(set(paths)) != file_count:
            raise RuntimeError(f"{label}: runtime source manifest paths are not canonical")
        if aggregate_runtime_source_manifest(files) != manifest_sha:
            raise RuntimeError(f"{label}: runtime source manifest aggregate mismatch")
        identities.append((manifest_sha, file_count))
    if len(set(identities)) != 1:
        raise RuntimeError("FORMAL_RUNTIME_SOURCE_MISMATCH")
    manifest_sha, file_count = identities[0]
    return {
        "runtime_source_manifest_sha256": manifest_sha,
        "runtime_source_manifest_file_count": file_count,
        "runtime_source_consistency_verified": True,
    }


def verify_runtime_intervention(
    branch: str,
    run_config: dict[str, Any],
    provenance: dict[str, Any],
    optimization_rows: list[dict[str, Any]],
    source: str,
) -> dict[str, float | bool]:
    expected = _EXPECTED_RUNTIME[branch]
    if run_config.get("development_method") != expected["development_method"]:
        raise RuntimeError(f"{source}: development_method mismatch")
    if run_config.get("enabled_modules") != expected["enabled_modules"]:
        raise RuntimeError(f"{source}: enabled_modules mismatch: {run_config.get('enabled_modules')!r}")
    if provenance.get("intervention") != expected["intervention"]:
        raise RuntimeError(f"{source}: intervention mismatch")
    budget = (
        int(provenance.get("source_sampled_steps", -1)),
        int(provenance.get("target_sampled_steps", -1)),
        int(provenance.get("additional_sampled_steps", -1)),
    )
    if budget != (SOURCE, TARGET, ADDITIONAL) or int(provenance.get("parent_sampled_steps", -1)) != SOURCE:
        raise RuntimeError(f"{source}: branch provenance budget mismatch: {budget}")
    if not optimization_rows:
        raise RuntimeError(f"{source}: no optimization rows in formal branch window")

    expected_enabled = 0 if branch == "control" else 1
    for row in optimization_rows:
        if float(row.get("team_credit_enabled", -1)) != expected_enabled:
            raise RuntimeError(f"{source}: team_credit_enabled mismatch at sampled_steps={row.get('sampled_steps')}")
    if branch == "control":
        return {
            "runtime_intervention_verified": True,
            "max_team_credit_sum_abs_error": 0.0,
            "mean_team_credit_mean_abs_redistribution": 0.0,
        }

    expected_identity = {
        "team_mean_credit_enabled": True,
        "team_mean_credit_version": 1,
        "team_mean_credit_mode": "alive_sum_preserving_mean",
        "team_mean_credit_reward_scope": "training_credit_only",
        "team_mean_credit_sum_preserving": True,
    }
    for key, expected_value in expected_identity.items():
        if run_config.get(key) != expected_value:
            raise RuntimeError(f"{source}: method identity mismatch for {key}")

    for row in optimization_rows:
        step = row.get("sampled_steps")
        if float(row.get("team_credit_live_sample_count", 0)) <= 0:
            raise RuntimeError(f"{source}: no live team-credit samples at sampled_steps={step}")
        for key in (
            "team_credit_original_live_reward_sum",
            "team_credit_transformed_live_reward_sum",
            "team_credit_sum_abs_error_max",
            "team_credit_mean_abs_redistribution",
        ):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RuntimeError(f"{source}: invalid {key} at sampled_steps={step}: {value!r}")
    redistributions = [float(row["team_credit_mean_abs_redistribution"]) for row in optimization_rows]
    if not any(value > 0 for value in redistributions):
        raise RuntimeError(f"{source}: team-credit redistribution is zero throughout the formal window")
    return {
        "runtime_intervention_verified": True,
        "max_team_credit_sum_abs_error": max(float(row["team_credit_sum_abs_error_max"]) for row in optimization_rows),
        "mean_team_credit_mean_abs_redistribution": sum(redistributions) / len(redistributions),
    }


def _load_branch(seed: int, branch: str) -> dict[str, Any]:
    directory = ROOT / f"outputs/dev_team_credit_{branch}_seed{seed}_300k"
    required = [directory / name for name in (
        "run_summary.json", "run_config.json", "branch_from.json", "evaluation_history.csv",
        "optimization_metrics.jsonl", "training_metrics.jsonl", "train.log", "latest.pt", "final.pt",
    )]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"incomplete branch {directory}: {missing}")

    summary = json.loads((directory / "run_summary.json").read_text(encoding="utf-8"))
    run_config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    provenance = json.loads((directory / "branch_from.json").read_text(encoding="utf-8"))
    for name, value in (("run_summary", summary), ("run_config", run_config), ("branch_from", provenance)):
        assert_finite_numeric_tree(value, f"{directory}/{name}.json")
    if int(summary["sampled_steps"]) != TARGET or int(run_config["total_sampled_steps"]) != TARGET:
        raise RuntimeError(f"{directory}: target mismatch")
    if int(provenance.get("source_training_seed", -1)) != seed:
        raise RuntimeError(f"{directory}: source training seed mismatch")
    if provenance.get("source_checkpoint_unchanged") is not True:
        raise RuntimeError(f"{directory}: source checkpoint unchanged evidence missing")
    destination_sha = run_config.get("runtime_source_manifest_sha256")
    if provenance.get("destination_runtime_source_manifest_sha256") != destination_sha:
        raise RuntimeError(f"{directory}: destination runtime source provenance mismatch")
    source_runtime_sha = provenance.get("source_runtime_source_manifest_sha256")
    source_runtime_available = provenance.get("source_runtime_source_provenance_available")
    if source_runtime_sha is None:
        if source_runtime_available is not False:
            raise RuntimeError(f"{directory}: historical source provenance availability mismatch")
    elif source_runtime_available is not True:
        raise RuntimeError(f"{directory}: source runtime source provenance availability mismatch")

    endpoint_metrics = endpoint(directory / "evaluation_history.csv")
    if endpoint_metrics["evaluation_episodes"] != EVALUATION_EPISODES or (
        endpoint_metrics["evaluation_seed_base"], endpoint_metrics["evaluation_seed_end"]
    ) != (EVALUATION_SEED_START, EVALUATION_SEED_END):
        raise RuntimeError(f"{directory}: evaluation protocol mismatch")
    train_log = directory / "train.log"
    check_text_failure_markers(train_log.read_text(encoding="utf-8", errors="ignore"), str(train_log))

    optimization_rows = [row for row in rows_jsonl(directory / "optimization_metrics.jsonl")
                         if SOURCE < int(row["sampled_steps"]) <= TARGET]
    training_rows = [row for row in rows_jsonl(directory / "training_metrics.jsonl")
                     if SOURCE < int(row["sampled_steps"]) <= TARGET]
    intervention = verify_runtime_intervention(branch, run_config, provenance, optimization_rows, str(directory))

    def total(key: str) -> float:
        return sum(float(row.get(key, 0)) for row in optimization_rows)

    entries: dict[int, dict[str, float | None]] = {}
    for wave in (2, 3):
        count = total(f"natural_entry_count_wave{wave}")
        survivors = total(f"natural_entry_survivor_sum_wave{wave}")
        entries[wave] = {"count": count, "survivors": survivors,
                         "mean": None if count == 0 else survivors / count}
    episode_means = {
        key: (sum(float(row[key]) for row in training_rows) / len(training_rows) if training_rows else None)
        for key in ("red_losses", "red_boundary_exits", "red_ground_losses", "waves_cleared")
    }
    return {
        "endpoint": endpoint_metrics, "entries": entries,
        "training_episode_means": episode_means,
        "source_sha256": provenance["parent_checkpoint_sha256"],
        "run": run_config, "provenance": provenance, "credit_provenance": intervention,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/team_credit_300k_analysis")
    args = parser.parse_args()
    output = ROOT / args.output_dir
    records: list[dict[str, Any]] = []
    mechanism: list[dict[str, Any]] = []
    runtime_configs: list[tuple[str, dict[str, Any]]] = []
    for seed in SEEDS:
        pair = {branch: _load_branch(seed, branch) for branch in ("control", "teammean")}
        runtime_configs.extend((f"seed{seed}/{branch}", pair[branch]["run"]) for branch in ("control", "teammean"))
        if pair["control"]["source_sha256"] != pair["teammean"]["source_sha256"]:
            raise RuntimeError(f"seed {seed}: unmatched source SHA")
        control_endpoint = pair["control"]["endpoint"]
        treatment_endpoint = pair["teammean"]["endpoint"]
        metrics: dict[str, Any] = {
            key: treatment_endpoint[key] - control_endpoint[key]
            for key in ("W1", "W2", "W3", "AverageWaves", "Return", "RedLoss", "BlueLoss",
                        "Boundary", "Ground", "EpisodeLength")
        }
        metrics.update({
            "seed": seed,
            "control_source_sha256": pair["control"]["source_sha256"],
            "delta_Q2": optional_delta(treatment_endpoint["Q2"], control_endpoint["Q2"]),
            "delta_Q3": optional_delta(treatment_endpoint["Q3"], control_endpoint["Q3"]),
        })
        for wave in (2, 3):
            control_entry = pair["control"]["entries"][wave]
            treatment_entry = pair["teammean"]["entries"][wave]
            metrics.update({
                f"W{wave}_entry_count_control": control_entry["count"],
                f"W{wave}_entry_count_treatment": treatment_entry["count"],
                f"W{wave}_entry_survivor_mean_control": control_entry["mean"],
                f"W{wave}_entry_survivor_mean_treatment": treatment_entry["mean"],
                f"delta_W{wave}_entry_survivor_mean": optional_delta(treatment_entry["mean"], control_entry["mean"]),
            })
        metrics["W2_entry_sample_status"] = w2_entry_sample_status(
            float(metrics["W2_entry_count_control"]), float(metrics["W2_entry_count_treatment"])
        )
        records.append(metrics)
        mechanism.append({"seed": seed, **pair})

    average_waves_wins = sum(row["AverageWaves"] > 0 for row in records)
    red_loss_wins = sum(row["RedLoss"] < 0 for row in records)
    valid_entries = [row for row in records if row["W2_entry_sample_status"] == "VALID"]
    entry_wins = sum(row["delta_W2_entry_survivor_mean"] > 0 for row in valid_entries)

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in records) / len(records)

    gate_a = average_waves_wins >= 2 and mean("AverageWaves") > 0
    gate_b = red_loss_wins >= 2 and mean("RedLoss") < 0
    gate_c = len(valid_entries) == 3 and entry_wins >= 2
    gate_d = all(row["AverageWaves"] > -0.50 and row["W1"] > -0.25 for row in records)
    decision = ("SAFETY_FAIL" if not gate_d else "NOT_SUPPORTED" if not gate_a
                else "MECHANISM_INCONCLUSIVE" if not (gate_b and gate_c) else "PROMISING")
    runtime_source = verify_six_branch_runtime_source_consistency(runtime_configs)
    result = {
        "TEAM_CREDIT_300K_SCREEN": decision,
        "protocol": {
            "source_sampled_steps": SOURCE, "target_sampled_steps": TARGET,
            "additional_sampled_steps": ADDITIONAL,
            "evaluation_seed_start": EVALUATION_SEED_START,
            "evaluation_seed_end": EVALUATION_SEED_END,
            "evaluation_episodes": EVALUATION_EPISODES,
            "min_W2_entry_count_per_branch": MIN_W2_ENTRY_COUNT_PER_BRANCH,
            "training_seed_is_replication_unit": True,
            **runtime_source,
        },
        "runtime_intervention_verified": True,
        "primary_endpoint": TARGET, "training_seed_replication_n": 3,
        "paired_deltas": records, "mechanism_records": mechanism,
        "gates": {
            "A_performance": gate_a, "B_survival": gate_b, "C_entry": gate_c,
            "D_catastrophic_guard": gate_d, "average_waves_wins": average_waves_wins,
            "red_loss_wins": red_loss_wins, "W2_entry_wins": entry_wins,
            "W2_entry_valid_seeds": len(valid_entries),
            "mean_delta_AverageWaves": mean("AverageWaves"),
            "mean_delta_RedLoss": mean("RedLoss"),
        },
        "interpretation": "TEAM_LEVEL_CREDIT_ASSIGNMENT_BENEFIT only if positive; teammate-loss-only causation is not identified",
    }
    assert_finite_numeric_tree(result, "analysis result")
    output.mkdir(parents=True, exist_ok=False)
    (output / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "paired_endpoint.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in records for key in row}))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
