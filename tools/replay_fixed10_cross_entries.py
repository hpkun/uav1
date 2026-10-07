"""Run the bounded Peak/Final 2x2 downstream-policy cross-entry diagnostic."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from env.persistent_env import PersistentWaveCombatEnv
try:
    from tools.fixed10_wave_state_drift_common import (
        AUDIT_DIR, ROOT, assert_safe_diagnostic_seeds, checkpoint_mutation_guard,
        csv_write, json_dump, load_checkpoint, run_dir,
    )
except ModuleNotFoundError:  # direct ``python tools/...py`` execution
    from fixed10_wave_state_drift_common import (
        AUDIT_DIR, ROOT, assert_safe_diagnostic_seeds, checkpoint_mutation_guard,
        csv_write, json_dump, load_checkpoint, run_dir,
    )

MIN_CELL_CONTINUATIONS = 4
SUCCESS_RATE_EFFECT_THRESHOLD = 0.15


def policy_specs():
    return {
        "Peak": (run_dir("10", 5303) / "best_eval.pt", 1_701_888),
        "Final": (run_dir("10", 5303) / "final.pt", 1_805_280),
    }


def future_seed(snapshot_index: int, replicate: int) -> int:
    value = 88_430_000 + 2 * int(snapshot_index) + int(replicate)
    assert_safe_diagnostic_seeds([value])
    return value


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for checkpoint diagnostic rollouts")
    output = AUDIT_DIR / "cross_entry_replay.csv"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    snapshot_path = AUDIT_DIR / "entry_snapshot_bank.pt"
    if not snapshot_path.is_file():
        raise FileNotFoundError("natural-entry snapshot bank is required")
    snapshots = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    selected = [item for item in snapshots if item["policy_id"] in {"Treatment10_seed5303_peak", "Treatment10_seed5303_final"}]
    policies = policy_specs()
    paths = [value[0] for value in policies.values()]
    before = checkpoint_mutation_guard(paths)
    run = run_dir("10", 5303)
    config = yaml.safe_load((run / "algorithm_config.yaml").read_text(encoding="utf-8"))
    env_config = yaml.safe_load((run / "runtime_env_config.yaml").read_text(encoding="utf-8"))
    rows = []
    for policy_label, (checkpoint_path, expected_step) in policies.items():
        checkpoint = load_checkpoint(checkpoint_path)
        if int(checkpoint["sampled_steps"]) != expected_step:
            raise RuntimeError(f"{policy_label} checkpoint metadata mismatch")
        trainer = build_modular_mappo_trainer(config, device="cuda:0", hidden_dim=256, total_sampled_steps=1_805_280)
        trainer.actor.load_state_dict(checkpoint["actor"], strict=True)
        trainer.actor.eval()
        for snapshot_index, item in enumerate(selected):
            source_label = "Peak" if item["policy_id"].endswith("_peak") else "Final"
            entry_wave = int(item["entry_wave"])
            for replicate in (0, 1):
                rng_seed = future_seed(snapshot_index, replicate)
                env = PersistentWaveCombatEnv(deepcopy(env_config))
                env.reset(88_499_999)
                restored = env.restore_curriculum_state(item["snapshot"])
                env.rng = np.random.default_rng(rng_seed)
                observation = restored["observation"]
                alive = restored["red_alive_mask"]
                start_step = int(env.steps)
                success = False
                done = False
                while not done and int(env.wave_index) == entry_wave:
                    with torch.no_grad():
                        actions, _ = trainer.act(observation[None], alive[None], deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(actions[0])
                    alive = info["red_alive_mask"]
                    success = int(info.get("waves_cleared", 0)) >= entry_wave
                    done = bool(terminated or truncated or success)
                rows.append({
                    "snapshot_index": snapshot_index, "source_entry_policy": source_label,
                    "source_episode_seed": item["episode_seed"], "entry_wave": entry_wave,
                    "entry_survivors": item["features"]["red_survivors"],
                    "downstream_policy": policy_label, "downstream_checkpoint_step": expected_step,
                    "future_rng_replicate": replicate, "future_rng_seed": rng_seed,
                    "success": int(success), "continuation_steps": int(env.steps - start_step),
                })
        del trainer
        torch.cuda.empty_cache()
    csv_write(output, rows)
    cells = {}
    for wave in (2, 3):
        for source in ("Peak", "Final"):
            for policy in ("Peak", "Final"):
                group = [row for row in rows if row["entry_wave"] == wave and row["source_entry_policy"] == source and row["downstream_policy"] == policy]
                cells[f"W{wave}_{source}Entry_{policy}Policy"] = {"n": len(group), "success_rate": float(np.mean([row["success"] for row in group])) if group else None}
    interactions = {}
    for wave in (2, 3):
        a = cells[f"W{wave}_PeakEntry_PeakPolicy"]["success_rate"]
        b = cells[f"W{wave}_PeakEntry_FinalPolicy"]["success_rate"]
        c = cells[f"W{wave}_FinalEntry_PeakPolicy"]["success_rate"]
        d = cells[f"W{wave}_FinalEntry_FinalPolicy"]["success_rate"]
        interactions[f"W{wave}"] = None if any(value is None for value in (a, b, c, d)) else float(a - b - c + d)
    policy_effects = {}
    source_effects = {}
    policy_observed = False
    source_observed = False
    any_policy_sufficient = False
    any_source_sufficient = False
    for wave in (2, 3):
        for source in ("Peak", "Final"):
            peak_cell = cells[f"W{wave}_{source}Entry_PeakPolicy"]
            final_cell = cells[f"W{wave}_{source}Entry_FinalPolicy"]
            sufficient = min(peak_cell["n"], final_cell["n"]) >= MIN_CELL_CONTINUATIONS
            delta = None if not sufficient else final_cell["success_rate"] - peak_cell["success_rate"]
            any_policy_sufficient |= sufficient
            policy_observed |= bool(sufficient and abs(delta) >= SUCCESS_RATE_EFFECT_THRESHOLD)
            policy_effects[f"W{wave}_{source}Entry"] = {"coverage": "SUFFICIENT" if sufficient else "INSUFFICIENT_COVERAGE",
                                                         "final_minus_peak_policy": delta}
        for policy in ("Peak", "Final"):
            peak_cell = cells[f"W{wave}_PeakEntry_{policy}Policy"]
            final_cell = cells[f"W{wave}_FinalEntry_{policy}Policy"]
            sufficient = min(peak_cell["n"], final_cell["n"]) >= MIN_CELL_CONTINUATIONS
            delta = None if not sufficient else final_cell["success_rate"] - peak_cell["success_rate"]
            any_source_sufficient |= sufficient
            source_observed |= bool(sufficient and abs(delta) >= SUCCESS_RATE_EFFECT_THRESHOLD)
            source_effects[f"W{wave}_{policy}Policy"] = {"coverage": "SUFFICIENT" if sufficient else "INSUFFICIENT_COVERAGE",
                                                         "final_minus_peak_entry_source": delta}
    policy_status = "OBSERVED" if policy_observed else "SMALL_OR_MIXED" if any_policy_sufficient else "INSUFFICIENT_COVERAGE"
    source_status = "OBSERVED" if source_observed else "SMALL_OR_MIXED" if any_source_sufficient else "INSUFFICIENT_COVERAGE"
    summary = {
        "status": "COMPLETE", "formal_evaluation": False, "training": False,
        "cells": cells, "source_by_policy_interaction": interactions,
        "cross_entry_policy_effect": policy_status, "policy_effects": policy_effects,
        "cross_entry_source_effect": source_status, "source_effects": source_effects,
        "exploratory_success_rate_effect_threshold": SUCCESS_RATE_EFFECT_THRESHOLD,
        "minimum_cell_continuations": MIN_CELL_CONTINUATIONS,
        "threshold_is_not_a_significance_test": True,
        "future_rng_replicates_per_snapshot": 2,
        "interpretation_limit": "The 2x2 table is descriptive and cannot exactly decompose whole-task AW causality.",
    }
    json_dump(AUDIT_DIR / "cross_entry_replay_summary.json", summary)
    after = checkpoint_mutation_guard(paths)
    if before != after:
        raise RuntimeError("input checkpoint mutation guard failed")
    analysis_path = AUDIT_DIR / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["stages"]["cross_entry"] = "COMPLETE"
    analysis["cross_entry_policy_effect"] = policy_status
    analysis["cross_entry_source_effect"] = source_status
    fixed = analysis["fixed_state_policy_drift"]
    natural = analysis["natural_entry_distribution_shift"]
    if fixed == "OBSERVED" and natural == "OBSERVED":
        analysis["mechanism_decision"] = "H1_H2_BOTH_PLAUSIBLE"
    elif fixed == "OBSERVED":
        analysis["mechanism_decision"] = "H1_SUPPORTED_DESCRIPTIVELY"
    elif natural == "OBSERVED":
        analysis["mechanism_decision"] = "H2_SUPPORTED_DESCRIPTIVELY"
    else:
        analysis["mechanism_decision"] = "INSUFFICIENT_EVIDENCE"
    analysis["cross_entry_mutation_guard"] = {"status": "PASS", "before": before, "after": after}
    json_dump(analysis_path, analysis)
    (AUDIT_DIR / "decision_support.txt").write_text(
        f"FIXED_STATE_POLICY_DRIFT={fixed}\n"
        f"NATURAL_ENTRY_DISTRIBUTION_SHIFT={natural}\n"
        f"CROSS_ENTRY_POLICY_EFFECT={policy_status}\n"
        f"CROSS_ENTRY_SOURCE_EFFECT={source_status}\n"
        f"MECHANISM_DECISION={analysis['mechanism_decision']}\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "COMPLETE", "continuations": len(rows), "interactions": interactions}, indent=2))


if __name__ == "__main__":
    main()
