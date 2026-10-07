"""Fixed-observation Actor function audit for the Fixed10 branches.

This tool performs checkpoint reads and ``torch.no_grad`` Actor forward passes
only.  It deliberately does not step an environment.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
try:
    from tools.fixed10_wave_state_drift_common import (
        AUDIT_DIR, BANK_PATH, INVENTORY_PATH, ROOT, checkpoint_mutation_guard,
        comparison_specifications, csv_write, distribution_stats, json_dump,
        load_checkpoint, policy_specifications, run_dir,
        symmetric_diagonal_gaussian_kl, validate_bank,
    )
except ModuleNotFoundError:  # direct ``python tools/...py`` execution
    from fixed10_wave_state_drift_common import (
        AUDIT_DIR, BANK_PATH, INVENTORY_PATH, ROOT, checkpoint_mutation_guard,
        comparison_specifications, csv_write, distribution_stats, json_dump,
        load_checkpoint, policy_specifications, run_dir,
        symmetric_diagonal_gaussian_kl, validate_bank,
    )

# Exploratory descriptive thresholds declared before reading policy outputs.
MIN_WAVE_TEAM_STATES = 30
MIN_WAVE_ALIVE_AGENTS = 60
DRIFT_ACTION_L2_MEDIAN = 0.05
DRIFT_SYMMETRIC_KL_MEDIAN = 0.01


def load_policy_outputs(device: str, observations: np.ndarray, alive: np.ndarray) -> tuple[dict, list[Path]]:
    outputs: dict[str, dict[str, np.ndarray]] = {}
    paths: list[Path] = []
    for spec in policy_specifications():
        path = spec["path"]
        paths.append(path)
        config = yaml.safe_load((run_dir(spec["arm"], spec["seed"]) / "algorithm_config.yaml").read_text(encoding="utf-8"))
        trainer = build_modular_mappo_trainer(config, device=device, hidden_dim=256, total_sampled_steps=1_805_280)
        checkpoint = load_checkpoint(path, device="cpu")
        trainer.actor.load_state_dict(checkpoint["actor"], strict=True)
        trainer.actor.eval()
        before = {name: value.detach().cpu().clone() for name, value in trainer.actor.state_dict().items()}
        mus, log_stds = [], []
        with torch.no_grad():
            for start in range(0, len(observations), 256):
                obs = torch.as_tensor(observations[start:start + 256], dtype=torch.float32, device=device)
                mask = torch.as_tensor(alive[start:start + 256], dtype=torch.float32, device=device)
                distribution, _ = trainer.actor.distribution_step(obs, None, None, None, mask)
                mus.append(distribution.mean.detach().cpu().numpy())
                log_stds.append(torch.log(distribution.scale).detach().cpu().numpy())
        if any(not torch.equal(value.detach().cpu(), before[name]) for name, value in trainer.actor.state_dict().items()):
            raise RuntimeError(f"Actor parameters mutated during no-grad inference: {spec['id']}")
        outputs[spec["id"]] = {"mu": np.concatenate(mus), "log_std": np.concatenate(log_stds)}
        del trainer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return outputs, paths


def pair_rows(comparison: dict[str, str], output_a: dict, output_b: dict,
              alive: np.ndarray, wave: np.ndarray) -> list[dict[str, Any]]:
    delta_mu = output_b["mu"] - output_a["mu"]
    delta_log_std = output_b["log_std"] - output_a["log_std"]
    delta_action = np.tanh(output_b["mu"]) - np.tanh(output_a["mu"])
    action_l2 = np.linalg.vector_norm(delta_action, axis=-1)
    abs_mu = np.mean(np.abs(delta_mu), axis=-1)
    abs_log_std = np.mean(np.abs(delta_log_std), axis=-1)
    symmetric_kl = symmetric_diagonal_gaussian_kl(
        output_a["mu"], output_a["log_std"], output_b["mu"], output_b["log_std"]
    )
    rows = []
    for group, team_mask in [("W1", wave == 1), ("W2", wave == 2), ("W3", wave == 3),
                             ("ALL", np.ones(len(wave), dtype=bool))]:
        live = alive[team_mask] > 0.5
        if not np.any(live):
            raise RuntimeError(f"{comparison['id']} {group} has no alive observations")
        values = {
            "action_l2": action_l2[team_mask][live],
            "symmetric_kl": symmetric_kl[team_mask][live],
            "heading_action_abs_delta": np.abs(delta_action[team_mask, :, 0][live]),
            "pitch_action_abs_delta": np.abs(delta_action[team_mask, :, 1][live]),
            "speed_action_abs_delta": np.abs(delta_action[team_mask, :, 2][live]),
            "abs_delta_mu": abs_mu[team_mask][live],
            "abs_delta_log_std": abs_log_std[team_mask][live],
        }
        row: dict[str, Any] = {
            **comparison,
            "group": group,
            "team_state_count": int(np.sum(team_mask)),
            "alive_agent_count": int(np.sum(live)),
            "dead_agent_observations_excluded": int(live.size - np.sum(live)),
            "coverage": "SUFFICIENT" if group == "ALL" or (
                int(np.sum(team_mask)) >= MIN_WAVE_TEAM_STATES and int(np.sum(live)) >= MIN_WAVE_ALIVE_AGENTS
            ) else "INSUFFICIENT_COVERAGE",
        }
        for metric, metric_values in values.items():
            stats = distribution_stats(metric_values, include_p95=metric in {"action_l2", "symmetric_kl"})
            for stat, value in stats.items():
                if stat != "n":
                    row[f"{metric}_{stat}"] = value
        rows.append(row)
    return rows


def ratio(numerator: float | None, denominator: float | None, coverage: str) -> dict[str, Any]:
    if coverage != "SUFFICIENT":
        return {"value": None, "status": "INSUFFICIENT_COVERAGE"}
    if denominator is None or numerator is None or abs(denominator) <= 1e-12:
        return {"value": None, "status": "UNDEFINED"}
    return {"value": float(numerator / denominator), "status": "DEFINED"}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for the Fixed10 checkpoint/Actor audit")
    if AUDIT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite existing audit directory: {AUDIT_DIR}")
    if not INVENTORY_PATH.is_file():
        raise FileNotFoundError("run preflight_fixed10_wave_state_drift.py first")
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if inventory.get("status") != "PASS":
        raise RuntimeError("checkpoint inventory did not pass")
    with np.load(BANK_PATH, allow_pickle=False) as bank:
        observations = np.asarray(bank["observations"], dtype=np.float32)
        alive = np.asarray(bank["alive"], dtype=np.float32)
        wave = np.asarray(bank["wave"], dtype=np.int64)
    validate_bank(observations, alive, wave)
    all_paths = [spec["path"] for spec in policy_specifications()]
    before = checkpoint_mutation_guard(all_paths)
    AUDIT_DIR.mkdir(parents=True)
    shutil.copy2(INVENTORY_PATH, AUDIT_DIR / "inventory.json")
    coverage = {
        "source": str(BANK_PATH.relative_to(ROOT)),
        "source_experiment": "historical Plain MAPPO",
        "valid_use": "COMMON_FIXED_STATE_ACTOR_FUNCTION_COMPARISON_ONLY",
        "invalid_use": "Fixed10 natural entry distribution",
        "schema": {"observations": list(observations.shape), "alive": list(alive.shape), "wave": list(wave.shape)},
        "thresholds": {"minimum_team_states": MIN_WAVE_TEAM_STATES, "minimum_alive_agents": MIN_WAVE_ALIVE_AGENTS},
        "groups": {},
        "near_boundary": {"status": "UNAVAILABLE", "reason": "bank has no verified raw metre-scale position metadata; observation indices are not inferred"},
    }
    for wave_id in (1, 2, 3):
        mask = wave == wave_id
        teams, live = int(mask.sum()), int((alive[mask] > 0.5).sum())
        coverage["groups"][f"W{wave_id}"] = {
            "team_state_count": teams, "alive_agent_count": live,
            "status": "SUFFICIENT" if teams >= MIN_WAVE_TEAM_STATES and live >= MIN_WAVE_ALIVE_AGENTS else "INSUFFICIENT_COVERAGE",
        }
    json_dump(AUDIT_DIR / "fixed_bank_coverage.json", coverage)
    outputs, _ = load_policy_outputs("cuda:0", observations, alive)
    rows: list[dict[str, Any]] = []
    for comparison in comparison_specifications():
        rows.extend(pair_rows(comparison, outputs[comparison["policy_a"]], outputs[comparison["policy_b"]], alive, wave))
    csv_write(AUDIT_DIR / "fixed_bank_policy_drift.csv", rows)
    summaries = []
    for comparison in comparison_specifications():
        group_rows = {row["group"]: row for row in rows if row["id"] == comparison["id"]}
        summaries.append({
            **comparison,
            "groups": group_rows,
            "w3_over_w1_action_l2_mean": ratio(group_rows["W3"]["action_l2_mean"], group_rows["W1"]["action_l2_mean"], group_rows["W3"]["coverage"]),
            "w3_over_w1_action_l2_median": ratio(group_rows["W3"]["action_l2_median"], group_rows["W1"]["action_l2_median"], group_rows["W3"]["coverage"]),
            "w3_over_w1_symmetric_kl_mean": ratio(group_rows["W3"]["symmetric_kl_mean"], group_rows["W1"]["symmetric_kl_mean"], group_rows["W3"]["coverage"]),
            "w3_over_w1_symmetric_kl_median": ratio(group_rows["W3"]["symmetric_kl_median"], group_rows["W1"]["symmetric_kl_median"], group_rows["W3"]["coverage"]),
        })
    primary = [item for item in summaries if item["class"] == "primary_final"]
    observed_count = sum(
        item["groups"]["ALL"]["action_l2_median"] >= DRIFT_ACTION_L2_MEDIAN
        or item["groups"]["ALL"]["symmetric_kl_median"] >= DRIFT_SYMMETRIC_KL_MEDIAN
        for item in primary
    )
    fixed_status = "OBSERVED" if observed_count >= 2 else "SMALL_OR_MIXED"
    fixed_summary = {
        "status": fixed_status,
        "exploratory_descriptive_thresholds": {
            "not_statistical_significance": True,
            "observed_if_at_least_two_of_three_primary_pairs": True,
            "median_action_l2_at_least": DRIFT_ACTION_L2_MEDIAN,
            "or_median_symmetric_kl_at_least": DRIFT_SYMMETRIC_KL_MEDIAN,
        },
        "primary_pairs_exceeding_threshold": observed_count,
        "comparisons": summaries,
        "checkpoint_selection_warning": "Treatment seed5303 peak 1701888 was selected on the 44M development evaluation and is selection-biased.",
        "ppo_approx_kl_distinction": "symmetric_kl here is pre-tanh fixed-state policy-function distance, not PPO training approx_kl",
    }
    json_dump(AUDIT_DIR / "fixed_bank_policy_drift_summary.json", fixed_summary)
    by_id = {item["id"]: item for item in summaries}
    json_dump(AUDIT_DIR / "seed5303_peak_final_case.json", by_id["seed5303_treatment_peak_vs_final"])
    json_dump(AUDIT_DIR / "seed5303_last_update_case.json", {
        "treatment": by_id["seed5303_treatment_last5088"],
        "control": by_id["seed5303_control_last5088"],
        "interval_samples": 5_088,
    })
    natural = {"status": "NOT_RUN", "reason": "Fixed10 natural-entry collection requires the separate user-launched 80-episode diagnostic"}
    cross = {"status": "NOT_RUN", "reason": "cross-entry replay requires valid Fixed10 natural-entry snapshots"}
    json_dump(AUDIT_DIR / "natural_entry_distribution_summary.json", natural)
    json_dump(AUDIT_DIR / "entry_snapshot_manifest.json", {"status": "NOT_RUN", "snapshot_count_w2": 0, "snapshot_count_w3": 0})
    json_dump(AUDIT_DIR / "cross_entry_replay_summary.json", cross)
    limitations = {
        "training_seed_is_replication_unit": True,
        "historical_bank_is_not_fixed10_natural_distribution": True,
        "near_boundary_stratification": "UNAVAILABLE",
        "peak_checkpoint_selection_bias": True,
        "fixed_state_distance_is_not_performance_causality": True,
        "natural_entry_conditioning_selection_bias": "conditioning on reaching W3 introduces selection bias",
        "no_root_cause_claim": True,
    }
    json_dump(AUDIT_DIR / "limitations.json", limitations)
    after = checkpoint_mutation_guard(all_paths)
    if before != after:
        raise RuntimeError("input checkpoint mutation guard failed")
    analysis = {
        "status": "FIXED_BANK_COMPLETE_NATURAL_ENTRY_AND_CROSS_NOT_RUN",
        "fixed_state_policy_drift": fixed_status,
        "natural_entry_distribution_shift": "INSUFFICIENT_COVERAGE",
        "cross_entry_policy_effect": "INSUFFICIENT_COVERAGE",
        "cross_entry_source_effect": "INSUFFICIENT_COVERAGE",
        "mechanism_decision": "H1_SUPPORTED_DESCRIPTIVELY" if fixed_status == "OBSERVED" else "INSUFFICIENT_EVIDENCE",
        "environment_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "formal_evaluations": 0,
        "forbidden_45m_used": False,
        "mutation_guard": {"status": "PASS", "checkpoint_sha256_before": before, "checkpoint_sha256_after": after},
        "stages": {"checkpoint_inventory": "PASS", "fixed_bank": "PASS", "natural_entry": "NOT_RUN", "cross_entry": "NOT_RUN"},
    }
    json_dump(AUDIT_DIR / "analysis.json", analysis)
    (AUDIT_DIR / "decision_support.txt").write_text(
        "FIXED_STATE_POLICY_DRIFT=" + fixed_status + "\n"
        "NATURAL_ENTRY_DISTRIBUTION_SHIFT=INSUFFICIENT_COVERAGE (NOT_RUN)\n"
        "CROSS_ENTRY_POLICY_EFFECT=INSUFFICIENT_COVERAGE (NOT_RUN)\n"
        "CROSS_ENTRY_SOURCE_EFFECT=INSUFFICIENT_COVERAGE (NOT_RUN)\n"
        f"MECHANISM_DECISION={analysis['mechanism_decision']}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": analysis["status"], "fixed_state_policy_drift": fixed_status,
                      "output": str(AUDIT_DIR.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
