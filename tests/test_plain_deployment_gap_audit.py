from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from tools import audit_plain_deployment_gap as audit


def write_history(path: Path, rows: list[tuple[int, float]]) -> list[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [
        "sampled_steps", "average_waves_cleared", "clear_wave_1_probability",
        "clear_wave_2_probability", "clear_wave_3_probability", "average_return",
        "average_red_boundary_exits", "average_red_ground_losses",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names); writer.writeheader()
        for step, waves in rows:
            writer.writerow({
                "sampled_steps": step, "average_waves_cleared": waves,
                "clear_wave_1_probability": min(waves, 1.0),
                "clear_wave_2_probability": max(0.0, min(waves - 1.0, 1.0)),
                "clear_wave_3_probability": max(0.0, waves - 2.0),
                "average_return": waves * 10, "average_red_boundary_exits": 0,
                "average_red_ground_losses": 0,
            })
    return audit.read_history(path)


def fake_run(tmp_path: Path, best_step: int = 1_200_000):
    run = tmp_path / "run"; run.mkdir()
    steps = (300_000, 900_000, 1_200_000, 1_500_000, 3_000_000)
    waves = {300_000: 2.9, 900_000: 1.0, 1_200_000: 2.5,
             1_500_000: 0.4, 3_000_000: 1.8}
    rows = write_history(run / "evaluation_history.csv", [(s, waves[s]) for s in steps])
    metadata = {}
    for step in steps:
        path = run / f"checkpoint_{step}.pt"; path.touch()
        metadata[path.resolve()] = {"sampled_steps": step}
    best = run / "best_eval.pt"; best.touch()
    metadata[best.resolve()] = {"sampled_steps": best_step}
    return run, rows, lambda path: metadata[path.resolve()]


def test_strong_uses_exact_best_eval(tmp_path):
    run, rows, reader = fake_run(tmp_path)
    result = audit.select_checkpoints(run, rows, reader)
    assert result["stages"]["STRONG"]["path"].name == "best_eval.pt"
    assert result["stages"]["STRONG"]["sampled_steps"] == 1_200_000


def test_strong_falls_back_to_best_exact_periodic(tmp_path):
    run, rows, reader = fake_run(tmp_path, best_step=1_100_000)
    result = audit.select_checkpoints(run, rows, reader)
    assert result["stages"]["STRONG"]["sampled_steps"] == 1_200_000
    assert result["stages"]["STRONG"]["path"].name == "checkpoint_1200000.pt"


def test_collapse_is_lowest_distinct_exact_post_900k(tmp_path):
    run, rows, reader = fake_run(tmp_path)
    result = audit.select_checkpoints(run, rows, reader)
    assert result["stages"]["COLLAPSE"]["sampled_steps"] == 1_500_000


def test_final_is_fixed_exact_3m(tmp_path):
    run, rows, reader = fake_run(tmp_path)
    result = audit.select_checkpoints(run, rows, reader)
    assert result["stages"]["FINAL"]["path"].name == "checkpoint_3000000.pt"
    assert result["stages"]["FINAL"]["sampled_steps"] == 3_000_000


def test_duplicate_stage_checkpoint_is_evaluated_once(tmp_path):
    run, rows, reader = fake_run(tmp_path, best_step=3_000_000)
    result = audit.select_checkpoints(run, rows, reader)
    assert len(result["unique"]) == 2
    final = next(row for row in result["unique"] if row["sampled_steps"] == 3_000_000)
    labels = final["stage_labels"]
    assert labels == ["STRONG", "FINAL"]
    assert final["path"].name == "checkpoint_3000000.pt"


def test_pre_900k_checkpoint_cannot_be_strong_or_collapse(tmp_path):
    run, rows, reader = fake_run(tmp_path, best_step=1_100_000)
    result = audit.select_checkpoints(run, rows, reader)
    assert result["stages"]["STRONG"]["sampled_steps"] >= 900_000
    assert result["stages"]["COLLAPSE"]["sampled_steps"] >= 900_000


def test_checkpoint_requires_exact_evaluation_row(tmp_path):
    run, rows, reader = fake_run(tmp_path, best_step=1_100_000)
    result = audit.select_checkpoints(run, rows, reader)
    assert result["stages"]["STRONG"]["sampled_steps"] != 1_100_000


def test_policy_rng_is_fixed_and_repeatable():
    audit.set_policy_rng(770001, "cpu")
    first = torch.randn(8)
    audit.set_policy_rng(770001, "cpu")
    second = torch.randn(8)
    assert torch.equal(first, second)


def test_seed_range_is_strictly_44m_and_45m_rejected():
    assert audit.validate_evaluation_seeds(range(44_000_000, 44_000_050)) == audit.EVALUATION_SEEDS
    with pytest.raises(RuntimeError):
        audit.validate_evaluation_seeds(range(45_000_000, 45_000_050))


def test_cli_cannot_accept_evaluation_seed_or_checkpoint():
    parser = audit.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--evaluation-seed-base", "45000000"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "anything.pt"])


def test_overall_strong_needs_at_least_two_of_three():
    assert audit.overall_label({"5301": True, "5302": True, "5303": False}) == \
        "PLAIN_DETERMINISTIC_DEPLOYMENT_GAP_STRONG"


def test_one_seed_only_is_weak_or_mixed():
    assert audit.overall_label({"5301": True, "5302": False, "5303": False}) == \
        "PLAIN_DETERMINISTIC_DEPLOYMENT_GAP_WEAK_OR_MIXED"


def test_deterministic_reproduction_mismatch_fails_closed():
    actual = {key: 0.0 for key in audit.REPRODUCTION_METRICS}
    history = {
        "clear_wave_1_probability": 1.0, "clear_wave_2_probability": 0.0,
        "clear_wave_3_probability": 0.0, "average_waves_cleared": 0.0,
        "average_return": 0.0, "average_red_boundary_exits": 0.0,
        "average_red_ground_losses": 0.0,
    }
    with pytest.raises(audit.DeterministicReproductionMismatch, match="PLAIN_DETERMINISTIC_REPRODUCTION_MISMATCH"):
        audit.assert_deterministic_reproduction(actual, history)


def test_existing_output_refuses_overwrite_by_default(tmp_path):
    output = tmp_path / "audit"; output.mkdir()
    with pytest.raises(FileExistsError):
        audit.prepare_output_paths(output, overwrite=False)
    json_path, text_path = audit.prepare_output_paths(output, overwrite=True)
    assert json_path.name == audit.JSON_NAME and text_path.name == audit.TEXT_NAME


def test_json_schema_fields_are_complete():
    report = {
        "provenance": {}, "selected_checkpoints": {}, "selection_reason": {},
        "deterministic_results": {}, "stochastic_repeat_results": {},
        "stochastic_summary": {}, "gaps": {}, "per_seed_labels": {},
        "overall_label": "PLAIN_DEPLOYMENT_AUDIT_INCONCLUSIVE",
        "evaluation_seed_range": [44_000_000, 44_000_049],
        "policy_rng_seeds": list(audit.POLICY_RNG_SEEDS), "45m_untouched": True,
    }
    audit.validate_report_schema(report)
    json.dumps(report)


def test_gap_threshold_requires_all_three_repeats():
    assert audit.seed_deployment_gap(0.5, [0.9, 1.0, 1.1])
    assert not audit.seed_deployment_gap(0.5, [0.79, 1.2, 1.2])
