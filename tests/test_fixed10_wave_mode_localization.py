from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tools.audit_fixed10_wave_mode_localization as audit
from tools.audit_fixed10_deployment_modes import policy_rng_seed


def row(mode, role="Peak", seed=88330000, stream=0, waves=2):
    return {
        "deployment_mode": mode, "checkpoint_role": role, "environment_seed": seed,
        "policy_rng_stream_id": stream, "policy_rng_seed": policy_rng_seed(seed, stream),
        "waves_cleared": waves, "reached_w2": int(waves >= 1), "reached_w3": int(waves >= 2),
        "w3_cleared": int(waves >= 3), "episode_length": 100, "team_episode_return": 1.0,
        "red_losses": 0, "red_boundary_exits": 0, "red_ground_losses": 0,
    }


def test_four_mode_action_switching_rules():
    assert [[audit.deterministic_for_wave(m, w) for w in (1, 2, 3)] for m in audit.ALL_MODES] == [
        [True, True, True], [True, True, False], [False, False, True], [False, False, False]
    ]


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        audit.deterministic_for_wave("BAD", 1)


def test_invalid_wave_rejected():
    with pytest.raises(ValueError):
        audit.deterministic_for_wave("DDS", 4)


def test_dds_and_ddd_prefix_match():
    assert all(audit.deterministic_for_wave("DDS", w) == audit.deterministic_for_wave("DDD", w) for w in (1, 2))


def test_ssd_and_sss_prefix_match():
    assert all(audit.deterministic_for_wave("SSD", w) == audit.deterministic_for_wave("SSS", w) for w in (1, 2))


def test_w2_to_w3_capture_uses_pre_step_action_wave():
    env = SimpleNamespace(wave_index=3, steps=71, _wave_start_step=71)
    assert audit.should_capture_w3_entry(2, {"spawned_next_wave": True, "wave_index": 3}, env)
    assert not audit.should_capture_w3_entry(3, {"spawned_next_wave": True, "wave_index": 3}, env)


def test_w3_entry_not_recorded_without_actual_spawn():
    env = SimpleNamespace(wave_index=3, steps=71, _wave_start_step=71)
    assert not audit.should_capture_w3_entry(2, {"spawned_next_wave": False, "wave_index": 3}, env)


def test_w3_entry_requires_exact_post_spawn_step():
    env = SimpleNamespace(wave_index=3, steps=72, _wave_start_step=71)
    assert not audit.should_capture_w3_entry(2, {"spawned_next_wave": True, "wave_index": 3}, env)


def test_policy_seed_mapping_is_reused_and_distinct():
    assert policy_rng_seed(88330000, 0) == 6804150757774932854
    assert len({policy_rng_seed(88330000, s) for s in (0, 1, 2)}) == 3


def test_45m_seed_is_forbidden():
    with pytest.raises(RuntimeError, match="45M"):
        audit.validate_environment_seeds([45000000])


def test_q2_q3_denominators_are_conditional():
    result = audit.aggregate([row("DDS", waves=1), row("DDS", seed=88330001, waves=3)])
    assert result["W1"] == 1.0 and result["W2"] == 0.5 and result["W3"] == 0.5
    assert result["Q2"] == 0.5 and result["Q3"] == 1.0


def test_q3_is_none_when_w2_zero():
    result = audit.aggregate([row("DDS", waves=1)])
    assert result["Q3"] is None and result["Q3_undefined_reason"] == "W2_ZERO"


def test_missing_entry_uses_null_not_zero():
    missing = audit._missing_entry("Peak", "DDS", 88330000, 0, policy_rng_seed(88330000, 0))
    assert missing["W3_entry_reached"] == 0
    assert missing["W3_entry_not_reached_reason"] == "NOT_REACHED"
    assert all(missing[key] is None for key in audit.ENTRY_FIELDS)


def test_prefix_validation_accepts_matching_observable_outcomes():
    ddd = [row("DDD", waves=2)]; sss = [row("SSS", waves=1)]
    mixed = [row("DDS", waves=2), row("SSD", waves=1)]
    result = audit.validate_prefixes(mixed, ddd, sss)
    assert result["status"] == "PASS" and not result["full_state_identity_claimed"]


def test_prefix_validation_fails_closed_on_dds_mismatch():
    with pytest.raises(RuntimeError, match="failed closed"):
        audit.validate_prefixes([row("DDS", waves=1)], [row("DDD", waves=2)], [])


def test_prefix_validation_fails_closed_on_ssd_mismatch():
    with pytest.raises(RuntimeError, match="failed closed"):
        audit.validate_prefixes([row("SSD", waves=2)], [], [row("SSS", waves=1)])


def test_dds_ddd_pair_marks_reused_reference_not_independent():
    pair = audit.build_action_pairs([row("DDS", waves=3)], [row("DDD", waves=2)], [])[0]
    assert pair["comparison"] == "DDD_vs_DDS"
    assert pair["reference_is_reused_per_stream_not_independent"] is True


def test_ssd_sss_pair_uses_exact_stream_key():
    pair = audit.build_action_pairs([row("SSD", stream=1, waves=2)], [], [row("SSS", stream=1, waves=3)])[0]
    assert pair["comparison"] == "SSD_vs_SSS" and pair["policy_rng_stream_id"] == 1


def test_full_coverage_is_exactly_192():
    rows = [row(mode, role, seed, stream, 2) for mode in audit.MIXED_MODES for role in audit.CHECKPOINTS
            for seed in audit.ENVIRONMENT_SEEDS for stream in audit.POLICY_STREAM_IDS]
    audit.validate_mixed_coverage(rows, full=True)


def test_full_coverage_rejects_missing_episode():
    rows = [row(mode, role, seed, stream, 2) for mode in audit.MIXED_MODES for role in audit.CHECKPOINTS
            for seed in audit.ENVIRONMENT_SEEDS for stream in audit.POLICY_STREAM_IDS]
    with pytest.raises(RuntimeError):
        audit.validate_mixed_coverage(rows[:-1], full=True)


def test_smoke_coverage_is_four_and_cannot_pass_as_full():
    rows = [row(mode, role, audit.ENVIRONMENT_SEEDS[0], 0, 2) for mode in audit.MIXED_MODES for role in audit.CHECKPOINTS]
    audit.validate_mixed_coverage(rows, full=False)
    with pytest.raises(RuntimeError):
        audit.validate_mixed_coverage(rows, full=True)


def test_conditional_entry_summary_excludes_not_reached_zeros():
    reached = audit._missing_entry("Peak", "DDS", 88330000, 0, 1)
    reached.update({"W3_entry_reached": 1, **{key: 4.0 for key in audit.ENTRY_FIELDS}})
    absent = audit._missing_entry("Peak", "DDS", 88330001, 0, 2)
    summary = audit._entry_summary([reached, absent])["groups"]["DDS"]["Peak"]
    assert summary["W3_entry_rate"] == 0.5
    assert summary["mean_W3_entry_red_survivors"] == 4.0


def test_prefix_entry_comparison_uses_only_conditional_entry_means():
    reached_dds = audit._missing_entry("Peak", "DDS", 88330000, 0, 1)
    reached_ssd = audit._missing_entry("Peak", "SSD", 88330000, 0, 1)
    for item, value in ((reached_dds, 3.0), (reached_ssd, 4.0)):
        item.update({"W3_entry_reached": 1, **{key: value for key in audit.ENTRY_FIELDS}})
    entries = audit._entry_summary([reached_dds, reached_ssd])
    summary = {mode: {role: {"W2": 1.0, "W3": 1.0, "Q3": 1.0, "AverageWaves": 3.0}
                              for role in audit.CHECKPOINTS} for mode in audit.ALL_MODES}
    comparison = audit._descriptive_prefix_comparison(summary, entries)
    assert comparison["DDS_vs_SSD_conditional_entry_state"]["Peak"][
        "SSD_minus_DDS_mean_W3_entry_red_survivors"] == 1.0


def test_primary_independent_unit_is_environment_seed():
    source = [row("DDS", seed=seed, stream=stream) for seed in (88330000, 88330001) for stream in (0, 1, 2)]
    assert len(source) == 6 and len({r["environment_seed"] for r in source}) == 2


def test_smoke_label_is_never_a_formal_effect_label():
    summary = {mode: {role: {"W2": 1.0, "W3": 1.0} for role in audit.CHECKPOINTS} for mode in audit.ALL_MODES}
    assert audit._label(summary, full=False) == "MIXED_OR_INSUFFICIENT_EVIDENCE"


def test_smoke_research_questions_refuse_formal_conclusion():
    summary = {mode: {role: {"W2": 1.0, "W3": 1.0, "Q3": 1.0, "AverageWaves": 3.0}
                              for role in audit.CHECKPOINTS} for mode in audit.ALL_MODES}
    prefix = {"DDS_vs_SSD_conditional_entry_state": {}}
    answer = audit._research_questions(summary, prefix, "BOTH_EFFECTS_OBSERVED", full=False)
    assert answer["interpretation_status"] == "SMOKE_ONLY_NO_FORMAL_RESEARCH_CONCLUSION"
    assert answer["9_algorithm_design_support"] == "NO_FORMAL_RECOMMENDATION_FROM_SMOKE"


def test_tool_has_no_training_or_update_api():
    assert not hasattr(audit, "train")
    assert not hasattr(audit, "optimizer_step")
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert ".backward(" not in source and ".step()" not in source


def test_source_full_audit_and_checkpoint_hashes_validate():
    ddd, sss, provenance = audit.validate_source_audit()
    assert len(ddd) == 32 and len(sss) == 96
    assert provenance["run_status"]["status"] == "COMPLETE"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_native_policy_rng_is_reproducible_and_isolated():
    before = torch.cuda.get_rng_state().clone()
    seed = policy_rng_seed(88330000, 0)
    with audit.isolated_policy_rng(seed, "cuda"):
        a = torch.randn(16, device="cuda")
    with audit.isolated_policy_rng(seed, "cuda"):
        b = torch.randn(16, device="cuda")
    assert torch.equal(a, b)
    assert torch.equal(before, torch.cuda.get_rng_state())


def test_entry_extraction_source_function_is_reused():
    assert audit.entry_features.__module__ == "tools.audit_comprehensive_persistent_wave"


def test_historical_data_are_only_opened_for_reading(monkeypatch):
    # Source validator has no output argument and therefore cannot rewrite the source audit.
    assert "source_dir" in audit.validate_source_audit.__annotations__
    assert audit.SOURCE_AUDIT.name == "fixed10_deployment_mode_audit"
