from pathlib import Path

import pytest
import torch

from tools.analyze_partial_sharing_evidence import (
    EPSILON, clip_coupling_class, exact_evaluation, family_counts,
    family_pattern, importance_derived, last_not_after, window_for_step,
    wsai_proxies,
)


def test_family_parameter_count_is_exact():
    state = {
        "backbone.0.weight": torch.zeros(2, 3),
        "backbone.0.bias": torch.zeros(2),
        "mean.weight": torch.zeros(3, 2),
        "mean.bias": torch.zeros(3),
        "log_std.weight": torch.zeros(3, 2),
        "log_std.bias": torch.zeros(3),
    }
    result = family_counts(state)
    assert result["backbone"]["parameter_count"] == 8
    assert result["mean_head"]["parameter_count"] == 9
    assert result["log_std_head"]["parameter_count"] == 9
    assert sum(row["parameter_fraction"] for row in result.values()) == pytest.approx(1)


def test_importance_enrichment_formula():
    enrichment, relative = importance_derived(.6, .2, .03, .01)
    assert enrichment == pytest.approx(3)
    assert relative == pytest.approx(3)


def test_active_filter_contract_is_explicit_in_source():
    source = Path("tools/analyze_partial_sharing_evidence.py").read_text(encoding="utf-8")
    assert 'float(row["w1sg_active"]) > .5' in source
    assert 'float(row["w1sg_active"]) <= .5' in source


def test_window_boundaries():
    assert window_for_step(1_505_280) is None
    assert window_for_step(1_505_281) == "EARLY"
    assert window_for_step(1_603_584) == "EARLY"
    assert window_for_step(1_603_585) == "MIDDLE"
    assert window_for_step(1_701_888) == "MIDDLE"
    assert window_for_step(1_701_889) == "LATE"
    assert window_for_step(1_805_280) == "LATE"


def test_wsai_proxy_formulas_and_zero_safety():
    row = {
        "wsai_wave1_actor_grad_norm_preclip": 4,
        "wsai_wave2_actor_grad_norm_preclip": 3,
        "wsai_wave3_actor_grad_norm_preclip": 0,
        "wsai_wave1_actor_grad_norm_postclip": .4,
        "wsai_wave2_actor_grad_norm_postclip": .3,
        "wsai_wave3_actor_grad_norm_postclip": 0,
        "wsai_global_actor_grad_norm_preclip": 5,
        "wsai_global_actor_grad_norm_postclip": .5,
    }
    proxy = wsai_proxies(row)
    assert proxy["global_clip_scale_proxy"] == pytest.approx(.1)
    assert proxy["w1_to_later_grad_ratio"] == pytest.approx(4 / 3)
    assert proxy["wave2_clip_retention_proxy"] == pytest.approx(.1)
    assert proxy["wave3_clip_retention_proxy"] == 0
    zero = {key: 0 for key in row}
    assert all(value == 0 for value in wsai_proxies(zero).values())
    assert EPSILON > 0


def test_exact_evaluation_never_uses_nearest():
    rows = [{"sampled_steps": "99"}, {"sampled_steps": "101"}]
    assert exact_evaluation(rows, 100) is None
    assert exact_evaluation(rows, 101) == rows[1]


def test_specialization_alignment_is_not_after_target():
    rows = [{"sampled_steps": 90}, {"sampled_steps": 110}]
    chosen = last_not_after(rows, 100)
    assert chosen["sampled_steps"] == 90
    assert 100 - chosen["sampled_steps"] == 10


@pytest.mark.parametrize(("rankings", "expected"), [
    ({1: ["mean_head"], 2: ["log_std_head"], 3: ["backbone"]}, "HEAD_ENRICHED"),
    ({1: ["backbone"], 2: ["backbone"], 3: ["mean_head"]}, "BACKBONE_ENRICHED"),
    ({1: ["backbone"], 2: ["mean_head"], 3: ["other"]}, "MIXED"),
])
def test_family_ranking_rule(rankings, expected):
    assert family_pattern(rankings) == expected


def test_clip_coupling_rule_strong_weak_mixed():
    coupled = {"row_level_clip_pressure_fraction": .9, "median_w1_to_later_grad_ratio": 3,
               "median_wave2_clip_retention_proxy": .2, "median_wave3_clip_retention_proxy": .4}
    uncoupled = {"row_level_clip_pressure_fraction": .7, "median_w1_to_later_grad_ratio": 1,
                 "median_wave2_clip_retention_proxy": .5, "median_wave3_clip_retention_proxy": .5}
    assert clip_coupling_class({1: coupled, 2: coupled, 3: uncoupled})[0] == "STRONG_SIGNAL"
    assert clip_coupling_class({1: uncoupled, 2: uncoupled, 3: uncoupled})[0] == "WEAK_SIGNAL"
    assert clip_coupling_class({1: coupled, 2: uncoupled, 3: uncoupled})[0] == "MIXED"


def test_analyzer_is_strictly_offline():
    source = Path("tools/analyze_partial_sharing_evidence.py").read_text(encoding="utf-8")
    forbidden = ("evaluate_" + "modular", "make_" + "combat_environment",
                 "optimizer" + ".step", "back" + "ward", "trainer" + ".update")
    assert all(token not in source for token in forbidden)
