import ast
from pathlib import Path

import pytest

from tools.audit_fixed10_ppo_stability import (
    EVALUATION_STEPS, ROOT, basic_stats, evaluation_rows_exact, exact_pairs,
    kl_instability_label, ratio_tail_stats, select_window, threshold_fractions,
    treatment_excess_label, underflow_stats,
)


def opt(step, kl=0.01, tail=1.0, underflow=0, samples=100):
    return {
        "sampled_steps": step, "approx_kl": kl, "clip_fraction": .1,
        "ratio_mean": 1., "ratio_std": .2, "ratio_p1": .5, "ratio_p50": 1.,
        "ratio_p99": 1.5, "ratio_min": .1, "ratio_max": 2.,
        "log_ratio_min": -tail, "log_ratio_max": tail / 2,
        "max_abs_log_ratio": tail, "ratio_underflow_fraction": underflow / samples,
        "ratio_underflow_count": underflow, "ratio_sample_count": samples,
    }


def test_kl_threshold_fractions_are_row_fractions():
    values = [.01, .04, .06, .11, .21]
    result = threshold_fractions(values)
    assert result == {"gt_0.03": .8, "gt_0.05": .6, "gt_0.1": .4, "gt_0.2": .2}


def test_ratio_tail_statistics_and_underflow_pooling():
    rows = [opt(1, tail=4, underflow=0, samples=90), opt(2, tail=12, underflow=2, samples=10)]
    tails = ratio_tail_stats(rows)
    underflow = underflow_stats(rows)
    assert tails["max_abs_log_ratio_gt_5_fraction"] == .5
    assert tails["max_abs_log_ratio_gt_10_fraction"] == .5
    assert underflow["ratio_underflow_count_sum"] == 2
    assert underflow["ratio_sample_count_sum"] == 100
    assert underflow["pooled_underflow_count_over_sample_count"] == .02


def test_exact_pairing_requires_same_seed_step_grid():
    paired = exact_pairs([opt(1), opt(2)], [opt(1), opt(2)])
    assert len(paired["pairs"]) == 2 and not paired["control_only"] and not paired["treatment_only"]
    mismatch = exact_pairs([opt(1), opt(2)], [opt(1), opt(3)])
    assert mismatch["control_only"] == [2] and mismatch["treatment_only"] == [3]


def test_window_boundaries_are_left_open_right_closed_and_tail_one_row():
    rows = [opt(1_800_192), opt(1_805_280)]
    assert [row["sampled_steps"] for row in select_window(rows, (1_701_888, 1_800_192))] == [1_800_192]
    assert [row["sampled_steps"] for row in select_window(rows, (1_800_192, 1_805_280))] == [1_805_280]
    assert basic_stats([.2])["p95"] == pytest.approx(.2)


def test_evaluation_grid_is_exact_and_nearest_is_forbidden():
    exact = [{"sampled_steps": str(step)} for step in EVALUATION_STEPS]
    assert [int(row["sampled_steps"]) for row in evaluation_rows_exact(exact)] == list(EVALUATION_STEPS)
    bad = [{"sampled_steps": str(step + (1 if step == EVALUATION_STEPS[0] else 0))} for step in EVALUATION_STEPS]
    with pytest.raises(RuntimeError):
        evaluation_rows_exact(bad)


def test_missing_values_return_unavailable_shape():
    stats = basic_stats([])
    assert stats["n"] == 0 and stats["mean"] is None and stats["p95"] is None


def test_treatment_excess_rule():
    strong = {1: {"median_delta_kl": .006, "delta_fraction_kl_gt_05": .10}, 2: {"median_delta_kl": .02, "delta_fraction_kl_gt_05": .20}, 3: {"median_delta_kl": 0., "delta_fraction_kl_gt_05": 0.}}
    assert treatment_excess_label(strong)[0] == "STRONG"
    weak = {seed: {"median_delta_kl": 0., "delta_fraction_kl_gt_05": 0.} for seed in (1, 2, 3)}
    assert treatment_excess_label(weak)[0] == "WEAK"


def test_kl_instability_uses_seed_as_replication_unit():
    rows = {1: [opt(1, .06), opt(2, .07)], 2: [opt(1, .08), opt(2, .09)], 3: [opt(i, .01) for i in range(49)]}
    label, flags = kl_instability_label(rows)
    assert label == "YES" and flags == {"1": True, "2": True, "3": False}
    assert len(rows[3]) == 49  # rows are not promoted to independent seeds


def test_different_seed_pairing_is_explicitly_separate():
    seed1 = exact_pairs([opt(1)], [opt(1)])
    seed2 = exact_pairs([opt(2)], [opt(2)])
    assert seed1["intersection"] != seed2["intersection"]


def test_script_is_strictly_offline_and_does_not_reference_45m():
    path = ROOT / "tools/audit_fixed10_ppo_stability.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.append(node.func.id)
    forbidden = {"evaluate", "collect_rollout", "step", "backward", "reset"}
    assert forbidden.isdisjoint(calls)
    assert not any(isinstance(node, ast.Name) and node.id in {"trainer", "environment", "optimizer"} for node in ast.walk(tree))
    # Constants may appear only in a rejection guard; no 45M file/range may be
    # used as an input source or evaluation request.
    assert "outputs/45" not in source and "evaluation_seed_base = 45" not in source
    assert "map_location=\"cpu\"" in source
