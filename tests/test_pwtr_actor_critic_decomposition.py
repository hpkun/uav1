from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from algorithm.modular_mappo.protocol import validate_pwtr_branch
from algorithm.train_modular_mappo import load_config
from tools.analyze_pwtr_actor_critic_decomposition import (
    direction_label, factorial_metric_rows, interaction, main_effects,
    paired_metric_deltas, qmetrics,
)
from tools.preflight_pwtr_actor_critic_decomposition import CORE, OLD, SEEDS, sha, validate_old_reference

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"
NAMES = ("current_actor_only", "current_critic_only", "recent_actor_only", "recent_critic_only")
EXPECTED = {
    "current_actor_only": (True, True, "current", False, False, True, False),
    "current_critic_only": (True, True, "current", False, False, False, True),
    "recent_actor_only": (True, True, "recent_uniform", False, False, True, False),
    "recent_critic_only": (True, True, "recent_uniform", False, False, False, True),
}


def mode(config):
    module = config["modules"]["persistent_wave_trajectory_replay"]
    return (module["fresh_wave_stratification"], module["replay_enabled"], module["replay_source"],
            module["priority_enabled"], module["bridge_enabled"], module["actor_replay"], module["critic_replay"])


def test_four_configs_are_exact_matched_decomposition_protocols():
    configs = {name: load_config(ROOT / f"configs/dev_pwtr_{name}_300k.yaml") for name in NAMES}
    for name, config in configs.items():
        branch = config["development_branch"]
        assert mode(config) == EXPECTED[name]
        assert (branch["source_sampled_steps"], branch["additional_sampled_steps"],
                branch["target_sampled_steps"], config["training"]["total_sampled_steps"]) == (1_505_280, 300_000, 1_805_280, 1_805_280)
        assert branch["actor_optimizer_restore"] and branch["critic_optimizer_restore"] and branch["rng_restore"]
        enabled = sorted(key for key, value in config["modules"].items() if isinstance(value, dict) and value.get("enabled", False))
        assert enabled == ["actor_lr_decay", "persistent_wave_trajectory_replay"]
        module = config["modules"]["persistent_wave_trajectory_replay"]
        assert tuple(module[key] for key in ("sequence_length", "bridge_half_length", "min_segment_length", "partition_capacity", "actor_max_age_updates")) == (128, 64, 32, 32, 2)


@pytest.mark.skipif(not SOURCE.is_file(), reason="formal source checkpoint unavailable")
def test_validator_accepts_exact_modes_and_rejects_one_wrong_switch():
    state = torch.load(SOURCE, map_location="cpu", weights_only=False)
    env = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    runtime = {"training_seed": 5301, "training_num_envs": 24, "training_smoke": False}
    for name in NAMES:
        config = load_config(ROOT / f"configs/dev_pwtr_{name}_300k.yaml")
        result = validate_pwtr_branch(state, env, config, runtime)
        assert result["intervention"] == f"pwtr_{name}"
    bad = deepcopy(load_config(ROOT / "configs/dev_pwtr_current_actor_only_300k.yaml"))
    bad["modules"]["persistent_wave_trajectory_replay"]["critic_replay"] = True
    with pytest.raises(RuntimeError, match="ablation mode mismatch"):
        validate_pwtr_branch(state, env, bad, runtime)


def test_factorial_interaction_and_main_effect_formulas():
    assert interaction(1.0, 1.2, .9, .8) == pytest.approx(-.3)
    actor, critic = main_effects(1.0, 1.2, .9, .8)
    assert actor == pytest.approx(.05)
    assert critic == pytest.approx(-.25)
    assert direction_label([.1, .2, -.1]) == "POSITIVE"
    assert direction_label([-.1, -.2, .1]) == "NEGATIVE"
    assert direction_label([.1, -.2, 0.0]) == "MIXED"


def test_undefined_conditional_ratio_and_paired_aggregation_are_explicit():
    values = qmetrics({"W1": .12, "W2": 0.0, "W3": 0.0})
    assert values["Q2"] == 0.0
    assert values["Q3"] is None
    data = {
        ("Method", 1): {"endpoint": {"Q3": .5}}, ("Control", 1): {"endpoint": {"Q3": .2}},
        ("Method", 2): {"endpoint": {"Q3": None}}, ("Control", 2): {"endpoint": {"Q3": .4}},
        ("Method", 3): {"endpoint": {"Q3": .7}}, ("Control", 3): {"endpoint": {"Q3": .3}},
    }
    result = paired_metric_deltas(data, "Method", "Control", "Q3", seeds=(1, 2, 3))
    assert result["values"] == pytest.approx([.3, .4])
    assert result["n_defined"] == 2
    assert result["undefined_seeds"] == [2]


def test_factorial_undefined_q3_skips_only_that_seed_and_reports_coverage():
    data = {}
    for seed in (1, 2, 3):
        for method, value in (("Stratified", .2), ("Actor", .3), ("Critic", .4), ("Both", .45)):
            data[method, seed] = {"endpoint": {"Q3": value}}
    data["Actor", 2]["endpoint"]["Q3"] = None
    rows, coverage = factorial_metric_rows(data, "current", "Actor", "Critic", "Both", "Q3", seeds=(1, 2, 3))
    assert [row["seed"] for row in rows] == [1, 3]
    assert coverage == {"n_defined": 2, "undefined_seeds": [2]}
    assert all(row["interaction"] == pytest.approx(-.05) for row in rows)


def test_aw_primary_label_uses_all_three_seeds_independent_of_q3_coverage():
    aw_deltas = [.1, .2, -.05]
    assert direction_label(aw_deltas) == "POSITIVE"
    # Undefined Q3 is intentionally irrelevant to the pre-registered AW label.
    assert paired_metric_deltas({
        ("M", 1): {"endpoint": {"Q3": .5}}, ("C", 1): {"endpoint": {"Q3": .4}},
        ("M", 2): {"endpoint": {"Q3": None}}, ("C", 2): {"endpoint": {"Q3": .4}},
        ("M", 3): {"endpoint": {"Q3": .5}}, ("C", 3): {"endpoint": {"Q3": .4}},
    }, "M", "C", "Q3", seeds=(1, 2, 3))["n_defined"] == 2


def test_launcher_is_serial_exact_and_analyzer_does_not_use_45m():
    launcher = (ROOT / "tools/run_pwtr_actor_critic_decomposition_300k.sh").read_text(encoding="utf-8")
    assert "branches=(current_actor_only current_critic_only recent_actor_only recent_critic_only)" in launcher
    assert "for seed in 5301 5302 5303" in launcher
    assert "nohup" not in launcher and " &\n" not in launcher
    analyzer = (ROOT / "tools/analyze_pwtr_actor_critic_decomposition.py").read_text(encoding="utf-8")
    assert "45_000_000" not in analyzer
    assert "evaluate" not in analyzer


def test_preflight_checks_both_environment_sources_and_all_reference_parent_shas_and_identities():
    assert "env/combat_env.py" in CORE
    assert "env/persistent_env.py" in CORE
    rows = []
    for seed in SEEDS:
        source = ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt"
        expected_sha = sha(source)
        for branch in OLD:
            run = ROOT / f"outputs/dev_pwtr_{branch}_seed{seed}_300k"
            rows.append(validate_old_reference(run, branch, seed, expected_sha))
    assert len(rows) == 9
    assert all(row["parent_checkpoint_sha256_match"] and row["branch_identity_match"] for row in rows)


def test_preflight_rejects_wrong_reference_parent_sha_and_branch_identity():
    run = ROOT / "outputs/dev_pwtr_stratified_seed5301_300k"
    with pytest.raises(RuntimeError, match="parent checkpoint SHA mismatch"):
        validate_old_reference(run, "stratified", 5301, "0" * 64)
    source = ROOT / "outputs/diag_mappo_learnability/l3_seed5301/checkpoint_1505280.pt"
    with pytest.raises(RuntimeError, match="branch identity mismatch"):
        validate_old_reference(run, "uniform_recent", 5301, sha(source))
