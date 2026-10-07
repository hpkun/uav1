from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Normal

from env.factory import make_combat_environment
from tools import diagnose_mappo_fixed_state_bank as diagnostic


ROOT = Path(__file__).resolve().parents[1]


class DummyActor(nn.Module):
    def distribution(self, observations):
        mean = observations[..., :3] * 0.1
        log_std = observations[..., 3:6].clamp(-5, 2) * 0.1
        return Normal(mean, log_std.exp())


class DummyCritic(nn.Module):
    def forward(self, observations, alive):
        return observations[..., :2].sum(-1) * alive


class DummyTrainer:
    def __init__(self):
        self.device = torch.device("cpu")
        self.actor = DummyActor()
        self.critic = DummyCritic()


def record(wave=1, episode=0, step=0, agent=0, value=1.0):
    local = np.full(52, value, np.float32)
    team = np.stack([local + index for index in range(4)])
    return {
        "episode_seed": 88_700_000 + episode, "episode_index": episode,
        "global_step": step, "wave_index": wave, "waves_remaining": 3 - wave,
        "focal_agent_index": agent, "local_observation": team[agent].copy(),
        "team_observation": team, "alive_mask": np.ones(4, np.float32),
        "red_survivors": 4, "blue_survivors": 4,
        "remaining_horizon_steps": 3000 - step, "own_fire_ready": True,
        "team_fire_readiness": np.ones(4, bool), "entry_wave": None,
        "checkpoint": "FreshStrong", "entry_global_step": step,
        "distance_to_boundary": 1000.0, "altitude": 3000.0,
        "speed": 225.0, "heading": 0.0, "pitch": 0.0,
        "formation_dispersion": 300.0, "nearest_blue_distance": 4000.0,
        "nearest_blue_ata": 0.1, "nearest_blue_aa": 0.2,
        "nearest_blue_closing_velocity": 10.0,
    }


def outputs(bank):
    base = diagnostic.evaluate_fixed_bank(DummyTrainer(), bank)
    return {name: copy.deepcopy(base) for name in diagnostic.CHECKPOINTS}


def test_checkpoint_identity_and_best_step_are_validated():
    identity = diagnostic.checkpoint_identity(
        "FreshStrong", diagnostic.CHECKPOINTS["FreshStrong"]
    )
    assert identity["algorithm"] == "MAPPO"
    assert identity["best_evaluation_sampled_steps"] == identity["sampled_steps"]
    assert (identity["observation_dim"], identity["action_dim"], identity["num_agents"]) == (52, 3, 4)


def test_generator_is_unique_fresh_strong_and_pairs_are_fixed():
    assert diagnostic.BANK_GENERATOR == "FreshStrong"
    assert [label for _, _, label in diagnostic.PAIR_SPECS] == [
        "FRESH_STRONG_VS_FINAL", "CONTROL_VS_TREATMENT"
    ]


def test_seed_ranges_are_disjoint_and_avoid_44m_to_47m():
    ranges = diagnostic.validate_seed_protocol(88_700_000, 12, 6)
    assert ranges == {"bank": [88_700_000, 88_700_011], "deployment": [88_710_000, 88_710_005]}
    assert not any(44_000_000 <= value < 48_000_000 for bounds in ranges.values() for value in bounds)


def test_wrong_seed_base_and_overlapping_subranges_fail():
    with pytest.raises(RuntimeError):
        diagnostic.validate_seed_protocol(44_000_000, 12, 6)
    with pytest.raises(RuntimeError):
        diagnostic.validate_seed_protocol(88_700_000, 10_001, 6)


def test_output_refuses_nonempty_but_accepts_empty(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    diagnostic.ensure_output_available(empty)
    (empty / "evidence.txt").write_text("x")
    with pytest.raises(FileExistsError):
        diagnostic.ensure_output_available(empty)


def test_bank_record_is_live_focal_only_and_time_aligned():
    env = make_combat_environment(diagnostic.load_yaml(diagnostic.ENV_CONFIG))
    observation, _ = env.reset(89_000_001)
    env.red[2].alive = False
    observation = env._observations()
    rows = diagnostic.make_bank_records(env, observation, 89_000_001, 0)
    assert {row["focal_agent_index"] for row in rows} == {0, 1, 3}
    assert all(row["wave_index"] == 1 and row["global_step"] == 0 for row in rows)
    assert all(row["local_observation"].shape == (52,) for row in rows)
    assert all(row["team_observation"].shape == (4, 52) for row in rows)


def test_all_zero_focal_observations_are_excluded(monkeypatch):
    env = make_combat_environment(diagnostic.load_yaml(diagnostic.ENV_CONFIG))
    env.reset(89_000_010)
    zeros = np.zeros((4, 52), np.float32)
    monkeypatch.setattr(env, "_observations", lambda: zeros.copy())
    assert diagnostic.make_bank_records(env, zeros, 89_000_010, 0) == []


def test_time_alignment_rejects_observation_from_other_instant():
    env = make_combat_environment(diagnostic.load_yaml(diagnostic.ENV_CONFIG))
    observation, _ = env.reset(89_000_002)
    observation = observation.copy()
    observation[0, 0] += 1
    with pytest.raises(RuntimeError, match="time alignment"):
        diagnostic.make_bank_records(env, observation, 89_000_002, 0)


def test_entry_is_first_post_spawn_decision_state_with_fire_context():
    env = make_combat_environment(diagnostic.load_yaml(diagnostic.ENV_CONFIG))
    _, _ = env.reset(89_000_003)
    for state in env.blue:
        state.alive = False
    observation, _, terminated, truncated, info = env.step(np.zeros((4, 3), np.float32))
    assert info["spawned_next_wave"] and not terminated and not truncated
    rows = diagnostic.make_bank_records(env, observation, 89_000_003, 0, entry_wave=2)
    assert rows and all(row["entry_wave"] == row["wave_index"] == 2 for row in rows)
    assert all(row["own_fire_ready"] for row in rows)
    assert all(np.array_equal(row["team_fire_readiness"], np.ones(4, bool)) for row in rows)


def test_fixed_bank_evaluation_has_focal_value_indexing_and_no_environment_dependency():
    bank = [record(wave=1, agent=2)]
    result = diagnostic.evaluate_fixed_bank(DummyTrainer(), bank)
    expected = bank[0]["team_observation"][2, :2].sum()
    assert result[0]["critic_value"] == pytest.approx(expected)
    assert set(result[0]) == {"mean_raw_action", "deterministic_action", "log_std", "critic_value"}


def test_all_checkpoints_can_share_exact_same_bank_object():
    bank = [record(wave=wave, episode=wave, value=float(wave)) for wave in (1, 2, 3)]
    evaluated = {name: diagnostic.evaluate_fixed_bank(DummyTrainer(), bank) for name in diagnostic.CHECKPOINTS}
    assert all(len(rows) == len(bank) for rows in evaluated.values())


def test_log_std_quantiles_and_clamp_hit_fractions():
    bank = [record(wave=1), record(wave=1, step=1)]
    evaluated = outputs(bank)
    evaluated["FreshStrong"][0]["log_std"] = np.array([-5.0, 2.0, 0.0])
    evaluated["FreshStrong"][1]["log_std"] = np.array([-4.98, 1.98, 1.0])
    rows = diagnostic.log_std_table(evaluated, bank)
    psi = next(row for row in rows if row["checkpoint"] == "FreshStrong" and row["wave"] == 1 and row["axis"] == "psi")
    theta = next(row for row in rows if row["checkpoint"] == "FreshStrong" and row["wave"] == 1 and row["axis"] == "theta")
    assert psi["median"] == pytest.approx(-4.99) and psi["fraction_le_neg4_99"] == .5
    assert theta["fraction_ge_1_99"] == .5


def test_empty_quantiles_remain_explicitly_undefined():
    result = diagnostic.quantiles([])
    assert result["n"] == 0 and result["mean"] is None and result["p99"] is None


def test_mean_action_and_log_std_drift_are_separate_quantities():
    bank = [record(wave=1)]
    evaluated = outputs(bank)
    evaluated["FreshFinal"][0]["mean_raw_action"] += 1
    evaluated["FreshFinal"][0]["log_std"] += 2
    rows = diagnostic.drift_table(evaluated, bank)
    names = {row["quantity"] for row in rows if row["comparison"] == "FRESH_STRONG_VS_FINAL" and row["wave"] == 1}
    assert names == {"mean_raw_action_l2", "deterministic_action_l2", "log_std_l2"}


def test_critic_table_calls_difference_surface_not_error():
    bank = [record(wave=1)]
    rows = diagnostic.critic_table(outputs(bank), bank)
    assert {row["row_type"] for row in rows} == {"surface", "absolute_surface_difference"}
    assert all("error" not in row["row_type"] for row in rows)


def test_nearest_raw_and_standardized_l2():
    source = np.array([0.0, 0.0])
    candidates = np.array([[1.0, 2.0], [3.0, 0.0]])
    assert diagnostic.nearest_index(source, candidates) == (0, pytest.approx(np.sqrt(5)))
    index, distance = diagnostic.nearest_index(source, candidates, np.array([10.0, 1.0]))
    assert index == 1 and distance == pytest.approx(.3)


def test_alias_pairs_are_cross_wave_and_cross_episode_filter_is_real():
    bank = [
        record(1, 0, 1, value=1), record(2, 0, 2, value=1.01),
        record(2, 1, 3, value=1.02), record(3, 2, 4, value=2),
    ]
    rows = diagnostic.cross_wave_alias_rows(bank, outputs(bank))
    assert rows and all(row["source_wave"] != row["target_wave"] for row in rows)
    assert all(not row["same_episode"] for row in rows if row["restriction"] == "cross_episode_only")
    assert {row["distance_kind"] for row in rows} == {"raw_normalized_observation_l2", "bank_standardized_l2"}


def test_exact_alias_is_only_elementwise_equality():
    bank = [record(1, 0, value=1), record(2, 1, value=1)]
    bank[1]["local_observation"] = bank[0]["local_observation"].copy()
    rows = diagnostic.cross_wave_alias_rows(bank, outputs(bank))
    assert rows and all(row["exact_alias"] for row in rows)
    bank[1]["local_observation"][0] += 1e-7
    rows = diagnostic.cross_wave_alias_rows(bank, outputs(bank))
    assert rows and all(not row["exact_alias"] for row in rows)
    assert all(row["pair_class"] == "cross_wave_nearest_neighbor" for row in rows)
    assert all(row["direction"] == "W1_TO_W2" for row in rows)


def test_alias_rows_preserve_context_and_checkpoint_surface_differences():
    bank = [record(1, 0, value=1), record(2, 1, value=2)]
    rows = diagnostic.cross_wave_alias_rows(bank, outputs(bank))
    row = rows[0]
    assert all(key in row for key in (
        "source_waves_remaining", "target_remaining_horizon_steps",
        "source_own_fire_ready", "target_red_survivors",
        "freshstrong_deterministic_action_l2", "freshstrong_log_std_l2",
        "freshstrong_critic_value_abs_diff",
    ))


def test_insufficient_w3_coverage_warns_without_switching_generator():
    bank = [record(1, 0), record(2, 1)]
    coverage = diagnostic.bank_coverage(bank, [], 12)
    assert coverage["coverage_status"] == "INSUFFICIENT_W3_BANK_COVERAGE"
    assert coverage["warnings"] and diagnostic.BANK_GENERATOR == "FreshStrong"


def test_entry_coverage_separates_events_live_records_and_episode_counts():
    entries = []
    for episode, survivors in ((0, 4), (1, 2)):
        for agent in range(survivors):
            row = record(2, episode, 100 + episode, agent)
            row["entry_wave"] = 2
            entries.append(row)
    coverage = diagnostic.bank_coverage(entries, entries, 2)
    assert coverage["w2_entry_events"] == 2
    assert coverage["w2_entry_live_focal_records"] == 6
    assert coverage["episodes_reaching_w2_entry"] == 2


def test_low_w3_episode_diversity_has_separate_warning():
    bank = [record(3, 0, step, step % 4) for step in range(30)]
    coverage = diagnostic.bank_coverage(bank, [], 12)
    assert "LOW_W3_EPISODE_DIVERSITY" in coverage["warnings"]
    assert coverage["minimum_recommended_w3_episodes"] == 3


def test_coverage_reports_per_wave_per_agent_counts():
    bank = [record(wave=wave, episode=wave, agent=agent) for wave in (1, 2, 3) for agent in range(4)]
    coverage = diagnostic.bank_coverage(bank, [], 1)
    assert all(coverage[f"wave_{wave}_agent_{agent}_records"] == 1 for wave in (1, 2, 3) for agent in range(4))


def test_entry_summary_contains_required_metrics():
    entry = record(2, 0)
    entry["entry_wave"] = 2
    rows = diagnostic.entry_summary([entry])
    metrics = {row["metric"] for row in rows if row["entry_wave"] == 2}
    assert {"red_survivors", "distance_to_boundary", "altitude", "speed", "formation_dispersion", "nearest_blue_distance", "remaining_horizon_steps", "own_fire_ready"} <= metrics


def test_entry_summary_deduplicates_team_events_but_keeps_live_focals():
    entries = []
    for agent in range(4):
        row = record(2, 0, 100, agent)
        row["entry_wave"] = 2
        entries.append(row)
    rows = diagnostic.entry_summary(entries)
    team = next(row for row in rows if row["entry_wave"] == 2 and row["metric"] == "red_survivors")
    focal = next(row for row in rows if row["entry_wave"] == 2 and row["metric"] == "altitude")
    assert team["aggregation_level"] == "entry_event" and team["n"] == 1
    assert focal["aggregation_level"] == "live_focal_agent" and focal["n"] == 4


def test_checkpoint_entry_collection_uses_all_policies_and_common_environment_seeds(monkeypatch):
    calls = []
    def fake_episode(policy, config, seed, deterministic, checkpoint=None, entry_sink=None, **kwargs):
        calls.append((policy, seed, deterministic, checkpoint))
        if deterministic and entry_sink is not None:
            row = record(2, seed - 88_710_000, 100)
            row.update({"checkpoint": checkpoint, "entry_wave": 2, "entry_global_step": 100, "episode_seed": seed})
            entry_sink.append(row)
        return {"environment_seed": seed, "deterministic": deterministic, "waves_cleared": 1,
                "return": 0.0, "red_loss": 0, "ground": 0, "boundary": 0}
    monkeypatch.setattr(diagnostic, "run_deployment_episode", fake_episode)
    policies = {name: object() for name in ("FreshStrong", "FreshFinal", "ControlFinal", "TreatmentFinal")}
    _, entries = diagnostic.deployment_summary(policies, {}, [88_710_000, 88_710_001])
    assert {row["checkpoint"] for row in entries} == set(policies)
    assert all({row["episode_seed"] for row in entries if row["checkpoint"] == name} == {88_710_000, 88_710_001} for name in policies)
    assert all(policy is policies[name] for policy, _, _, name in calls)


def test_policy_episode_rng_is_scenario_local_and_reproducible():
    first = diagnostic.policy_episode_seed(89_710_000, 88_710_001)
    assert first == diagnostic.policy_episode_seed(89_710_000, 88_710_001)
    assert first != diagnostic.policy_episode_seed(89_710_000, 88_710_002)


def test_deterministic_and_stochastic_deployment_paths_preserve_global_rng(monkeypatch):
    class FakeEnv:
        red_alive_mask = np.ones(4, np.float32)
        def reset(self, seed):
            return np.zeros((4, 52), np.float32), {}
        def step(self, action):
            return np.zeros((4, 52), np.float32), np.ones(4, np.float32), True, False, {
                "waves_cleared": 1, "red_losses": 0,
                "red_ground_losses": 0, "red_boundary_exits": 0,
            }
    class Policy:
        device = torch.device("cpu")
        def act(self, observation, alive, deterministic):
            if deterministic:
                return np.zeros((4, 3), np.float32)
            return torch.rand(4, 3).numpy()
    monkeypatch.setattr(diagnostic, "make_combat_environment", lambda config: FakeEnv())
    before = torch.get_rng_state().clone()
    deterministic = diagnostic.run_deployment_episode(Policy(), {}, 88_710_000, True)
    stochastic = diagnostic.run_deployment_episode(Policy(), {}, 88_710_000, False)
    assert deterministic["deterministic"] and not stochastic["deterministic"]
    assert torch.equal(before, torch.get_rng_state())


def test_recalibrated_root_cause_labels_are_present():
    text = (ROOT / "outputs/mappo_root_cause_audit/root_cause_report_recalibrated.md").read_text(encoding="utf-8")
    for label in (
        "IMPLEMENTATION_CORRECT", "PLAUSIBLE_NOT_ESTABLISHED",
        "SUPPORTED_ASSOCIATION", "SUPPORTED_PHENOMENON",
        "STRONGLY_SUPPORTED_PHENOMENON",
        "UNDERLYING_MECHANISM_NOT_FULLY_LOCALIZED",
    ):
        assert label in text
    assert "update-level mean" in text and "root-cause winner" in text


def test_required_output_names_are_wired_in_source():
    source = (ROOT / "tools/diagnose_mappo_fixed_state_bank.py").read_text(encoding="utf-8")
    for name in (
        "diagnostic_summary.json", "bank_metadata.json", "bank_coverage.json",
        "policy_std_by_wave.csv", "policy_mean_drift_by_wave.csv",
        "critic_value_by_wave.csv", "cross_wave_alias_pairs.csv",
        "cross_wave_alias_summary.csv", "entry_state_by_checkpoint.csv",
        "entry_state_summary.csv", "bank_context.jsonl",
        "deployment_mode_comparison.csv", "diagnostic_report.md",
    ):
        assert name in source


def test_tool_does_not_emit_causal_winner():
    source = (ROOT / "tools/diagnose_mappo_fixed_state_bank.py").read_text(encoding="utf-8")
    assert '"root_cause_winner": None' in source
    assert "WAVE_CONTEXT_CONFIRMED" not in source


def test_bank_context_persistence_aligns_record_index(tmp_path):
    bank = [record(1, 0, 10), record(2, 1, 20)]
    context = tmp_path / "bank_context.jsonl"
    archive = tmp_path / "bank_states.npz"
    diagnostic.save_bank_context(context, bank)
    diagnostic.save_bank(archive, bank)
    rows = [json.loads(line) for line in context.read_text().splitlines()]
    arrays = np.load(archive)
    assert [row["record_index"] for row in rows] == arrays["record_index"].tolist() == [0, 1]
    assert rows[1]["episode_seed"] == int(arrays["episode_seed"][1])
    assert len(rows[0]["team_fire_readiness"]) == 4


def test_environment_sha_mismatch_fails_closed(tmp_path):
    state = {"extra": {"environment_config_sha256": "wrong"}}
    with pytest.raises(RuntimeError, match="environment SHA"):
        diagnostic.validate_checkpoint_provenance(
            "FreshFinal", tmp_path / "checkpoint.pt", state, "wrong", "sha"
        )


def test_fresh_strong_sha_mismatch_fails_closed(tmp_path):
    state = {"extra": {"environment_config_sha256": diagnostic.EXPECTED_ENVIRONMENT_SHA256}}
    with pytest.raises(RuntimeError, match="FreshStrong frozen"):
        diagnostic.validate_checkpoint_provenance(
            "FreshStrong", tmp_path / "best_eval.pt", state,
            diagnostic.EXPECTED_ENVIRONMENT_SHA256, "wrong",
        )


def test_continuation_parent_source_mismatch_fails_closed(tmp_path):
    run = {"parent_checkpoint_sha256": "wrong", "source_sampled_steps": 1_001_472,
           "seed": 5303, "environment_config_sha256": diagnostic.EXPECTED_ENVIRONMENT_SHA256}
    branch = {"parent_checkpoint_sha256": diagnostic.EXPECTED_COMMON_SOURCE_SHA256,
              "source_sampled_steps": 1_001_472, "source_training_seed": 5303,
              "environment_config_sha256": diagnostic.EXPECTED_ENVIRONMENT_SHA256}
    (tmp_path / "run_config.json").write_text(json.dumps(run))
    (tmp_path / "branch_from.json").write_text(json.dumps(branch))
    state = {"extra": {"environment_config_sha256": diagnostic.EXPECTED_ENVIRONMENT_SHA256}}
    with pytest.raises(RuntimeError, match="parent checkpoint SHA"):
        diagnostic.validate_checkpoint_provenance(
            "ControlFinal", tmp_path / "final.pt", state,
            diagnostic.EXPECTED_ENVIRONMENT_SHA256, "irrelevant",
        )
