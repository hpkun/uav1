from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modules import (MilestoneAwareRetentionCreditModule, continuation_coefficients,
                               successful_wave_from_transition, tempered_wave_weights)
from algorithm.train_modular_mappo import load_config

ROOT = Path(__file__).resolve().parents[1]


def v2_config(**overrides):
    config = {"enabled": True, "version": 2, "max_waves": 3,
              "continuation_alpha": 1.0, "wave_balance_temperature": .5,
              "wave_weight_min": .5, "wave_weight_max": 2.0,
              "deployment_distill_coefficient": .05, "elite_segments_per_wave": 8,
              "elite_rows_per_segment": 64, "retention_samples_per_wave": 32,
              "retention_min_rows_per_wave": 32, "retention_stride": 8}
    config.update(overrides)
    return config


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = nn.Linear(52, 3, bias=False)
        nn.init.zeros_(self.mean.weight)
        self.log_std = nn.Parameter(torch.zeros(3))

    def distribution(self, observation):
        return torch.distributions.Normal(self.mean(observation), self.log_std.exp())


def segment(wave=1, survivors=3, duration=100, rows=40, action=.25):
    values = np.arange(rows, dtype=np.float32)
    observations = np.repeat(values[:, None], 52, axis=1)
    actions = np.full((rows, 3), action, dtype=np.float32)
    return {"wave": wave, "observations": observations, "target_actions": actions,
            "red_survivors_after_clear": survivors, "wave_duration_steps": duration}


def ingest(module, *segments):
    module.ingest_success_segments(list(segments), TinyActor(), torch.device("cpu"))


def test_v1_default_and_explicit_version_remain_fifo_gaussian():
    default = MilestoneAwareRetentionCreditModule({"enabled": True})
    explicit = MilestoneAwareRetentionCreditModule({"enabled": True, "version": 1})
    assert default.version == explicit.version == 1
    assert isinstance(default.banks[1], type(explicit.banks[1]))


def test_v2_checkpoint_version_and_strict_cross_version_rejection():
    v1 = MilestoneAwareRetentionCreditModule({"enabled": True, "version": 1})
    v2 = MilestoneAwareRetentionCreditModule(v2_config())
    assert v2.state_dict()["version"] == 2
    with pytest.raises(RuntimeError, match="version mismatch"):
        v1.load_state_dict(v2.state_dict())
    with pytest.raises(RuntimeError, match="version mismatch"):
        v2.load_state_dict(v1.state_dict())


def test_trainer_strict_loader_reports_v1_v2_version_mismatch(tmp_path):
    v1_config = load_config(ROOT / "configs/dev_marc_mappo_v1_1m.yaml")
    v2_algorithm_config = load_config(ROOT / "configs/dev_marc_mappo_v2_1m.yaml")
    v1 = build_modular_mappo_trainer(v1_config, "cpu", total_sampled_steps=1_000_000)
    v2 = build_modular_mappo_trainer(v2_algorithm_config, "cpu", total_sampled_steps=1_000_000)
    v1_path, v2_path = tmp_path / "v1.pt", tmp_path / "v2.pt"
    v1.save(v1_path); v2.save(v2_path)
    with pytest.raises(RuntimeError, match="MARC version mismatch"):
        v2.load(v1_path, strict_protocol=True)
    with pytest.raises(RuntimeError, match="MARC version mismatch"):
        v1.load(v2_path, strict_protocol=True)


def test_elite_insert_fill_reject_replace_and_equal_quality():
    module = MilestoneAwareRetentionCreditModule(v2_config(elite_segments_per_wave=2))
    ingest(module, segment(survivors=2, duration=100), segment(survivors=3, duration=200))
    assert len(module.elite_segments[1]) == 2
    old_ids = [row["insertion_id"] for row in module.elite_segments[1]]
    ingest(module, segment(survivors=1, duration=1))
    assert [row["insertion_id"] for row in module.elite_segments[1]] == old_ids
    ingest(module, segment(survivors=2, duration=100))
    assert [row["insertion_id"] for row in module.elite_segments[1]] == old_ids
    ingest(module, segment(survivors=4, duration=999))
    assert sorted(row["quality"] for row in module.elite_segments[1]) == [(3, -200), (4, -999)]
    assert module.segment_insertions[1] == 2
    assert module.segment_rejections[1] == 2
    assert module.segment_replacements[1] == 1


def test_quality_is_survivors_first_then_shorter_duration():
    module = MilestoneAwareRetentionCreditModule(v2_config(elite_segments_per_wave=1))
    ingest(module, segment(survivors=2, duration=1))
    ingest(module, segment(survivors=3, duration=1000))
    assert module.elite_segments[1][0]["quality"] == (3, -1000)
    ingest(module, segment(survivors=3, duration=100))
    assert module.elite_segments[1][0]["quality"] == (3, -100)


def test_uniform_subsampling_is_deterministic_and_covers_whole_segment():
    module = MilestoneAwareRetentionCreditModule(v2_config())
    small = module.uniform_subsample_indices(10, 64)
    large = module.uniform_subsample_indices(100, 64)
    assert np.array_equal(small, np.arange(10))
    assert len(large) == 64 and large[0] == 0 and large[-1] == 99
    assert np.array_equal(large, module.uniform_subsample_indices(100, 64))


def test_runner_v2_collects_executed_action_aligned_and_alive_only():
    runner = ModularMAPPOTrainingRunner.__new__(ModularMAPPOTrainingRunner)
    module = MilestoneAwareRetentionCreditModule(v2_config(retention_stride=1))
    runner.trainer = type("Trainer", (), {"milestone_aware_retention_credit": module})()
    runner.num_envs = 1
    runner.episode_steps = np.asarray([0])
    runner.marc_wave_start_steps = np.asarray([0])
    runner.marc_pending_episode = [{1: [], 2: [], 3: []}]
    observations = np.arange(4 * 52, dtype=np.float32).reshape(1, 4, 52)
    actions = np.arange(12, dtype=np.float32).reshape(1, 4, 3) / 12
    runner._marc_collect_candidates(observations, np.asarray([[1, 0, 1, 0]]),
                                    np.asarray([1]), actions)
    rows = runner.marc_pending_episode[0][1]
    assert len(rows) == 2
    assert np.array_equal(rows[0]["observation"], observations[0, 0])
    assert np.array_equal(rows[0]["target_action"], actions[0, 0])
    assert np.array_equal(rows[1]["observation"], observations[0, 2])
    assert np.array_equal(rows[1]["target_action"], actions[0, 2])


def test_runner_v2_milestones_duration_and_reset_are_exact():
    runner = ModularMAPPOTrainingRunner.__new__(ModularMAPPOTrainingRunner)
    module = MilestoneAwareRetentionCreditModule(v2_config())
    runner.trainer = type("Trainer", (), {"milestone_aware_retention_credit": module})()
    runner.num_envs = 1
    runner.episode_steps = np.asarray([9])
    runner.marc_wave_start_steps = np.asarray([0])
    row = {"observation": np.ones(52), "target_action": np.ones(3)}
    runner.marc_pending_episode = [{1: [row], 2: [], 3: []}]
    completed = runner._marc_finalize_step(np.asarray([1]), np.asarray([False]),
        [{"spawned_next_wave": True, "red_success": False, "episode_length": 10,
          "red_alive_mask": [1, 1, 1, 0]}])
    assert completed[0]["wave_duration_steps"] == 10
    assert completed[0]["red_survivors_after_clear"] == 3
    assert runner.marc_wave_start_steps[0] == 10
    runner.marc_pending_episode[0][2] = [row]
    runner._marc_finalize_step(np.asarray([2]), np.asarray([False]),
        [{"spawned_next_wave": True, "red_success": False, "episode_length": 17,
          "red_alive_mask": [1, 1, 0, 0]}])
    assert runner.marc_wave_start_steps[0] == 17
    runner._marc_finalize_step(np.asarray([3]), np.asarray([True]),
        [{"spawned_next_wave": False, "red_success": False, "episode_length": 20,
          "red_alive_mask": [0, 0, 0, 0]}])
    assert runner.marc_wave_start_steps[0] == 0


def test_milestone_semantics_reject_failed_waves():
    assert successful_wave_from_transition(1, spawned_next_wave=False,
                                           episode_done=True, red_success=False) is None
    assert successful_wave_from_transition(2, spawned_next_wave=False,
                                           episode_done=True, red_success=False) is None
    assert successful_wave_from_transition(3, spawned_next_wave=False,
                                           episode_done=True, red_success=True) == 3


def test_distillation_uses_detached_actions_and_not_log_std():
    module = MilestoneAwareRetentionCreditModule(v2_config(retention_min_rows_per_wave=1,
                                                           retention_samples_per_wave=1))
    actor = TinyActor()
    target = np.zeros((1, 3), dtype=np.float32)
    candidate = {**segment(rows=1, action=0), "target_actions": target,
                 "observations": np.ones((1, 52), dtype=np.float32)}
    module.ingest_success_segments([candidate],
                                   actor, torch.device("cpu"))
    loss, metrics = module.retention_loss(actor, torch.device("cpu"))
    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    with torch.no_grad():
        actor.mean.weight.fill_(.1)
    loss, _ = module.retention_loss(actor, torch.device("cpu"))
    loss.backward()
    assert loss.item() > 0 and actor.mean.weight.grad.abs().sum() > 0
    assert actor.log_std.grad is None
    assert metrics["marc_v2_deployment_distill_loss"] == pytest.approx(0.0)


def test_ready_waves_are_equal_weighted():
    module = MilestoneAwareRetentionCreditModule(v2_config(retention_min_rows_per_wave=1,
                                                           retention_samples_per_wave=1))
    actor = TinyActor()
    ingest(module, segment(wave=1, rows=1, action=0), segment(wave=2, rows=1, action=1))
    loss, metrics = module.retention_loss(actor, torch.device("cpu"))
    # W1 MSE=0, W2 MSE=1; equal wave mean then coefficient .05.
    assert loss.item() == pytest.approx(.025, abs=2e-5)
    assert metrics["marc_v2_action_mse_wave1"] == pytest.approx(0, abs=1e-8)
    assert metrics["marc_v2_action_mse_wave2"] == pytest.approx(1, abs=2e-4)


def test_marc_rng_does_not_touch_numpy_or_torch_global_rng():
    module = MilestoneAwareRetentionCreditModule(v2_config(retention_min_rows_per_wave=1,
                                                           retention_samples_per_wave=1))
    ingest(module, segment(rows=2))
    np.random.seed(7); torch.manual_seed(7)
    np_before = np.random.get_state(); torch_before = torch.random.get_rng_state().clone()
    module.sample_elite_balanced()
    np_after = np.random.get_state(); torch_after = torch.random.get_rng_state()
    assert np.array_equal(np_before[1], np_after[1])
    assert torch.equal(torch_before, torch_after)


def test_v2_state_roundtrip_is_bit_equivalent_and_preserves_counters():
    source = MilestoneAwareRetentionCreditModule(v2_config(elite_segments_per_wave=1))
    ingest(source, segment(rows=70), segment(survivors=1, rows=2))
    saved = source.state_dict()
    restored = MilestoneAwareRetentionCreditModule(v2_config(elite_segments_per_wave=1))
    restored.load_state_dict(deepcopy(saved))
    assert restored.state_dict()["insertion_counter"] == saved["insertion_counter"]
    assert restored.segment_rejections == source.segment_rejections
    assert np.array_equal(restored.elite_segments[1][0]["observations"],
                          source.elite_segments[1][0]["observations"])
    assert restored.rng.bit_generator.state == source.rng.bit_generator.state


def test_resume_discards_pending_but_preserves_elite_and_marc_rng():
    module = MilestoneAwareRetentionCreditModule(v2_config())
    ingest(module, segment(rows=40))
    elite_before = deepcopy(module.state_dict()["elite_segments"])
    rng_before = deepcopy(module.rng.bit_generator.state)
    runner = ModularMAPPOTrainingRunner.__new__(ModularMAPPOTrainingRunner)
    runner.num_envs = 2
    runner.trainer = type("Trainer", (), {"milestone_aware_retention_credit": module})()
    runner._marc_discard_pending_after_resume({
        "marc_pending_dropped_on_resume": 4,
        "marc_pending_candidate_counts": [{"1": 3, "2": 2, "3": 0},
                                          {"1": 1, "2": 0, "3": 0}],
    })
    assert runner.marc_pending_dropped_on_resume == 10
    assert all(not rows[wave] for rows in runner.marc_pending_episode for wave in (1, 2, 3))
    assert np.array_equal(runner.marc_wave_start_steps, np.zeros(2, dtype=np.int64))
    after = module.state_dict()
    assert after["elite_segments"]["1"][0]["quality"] == elite_before["1"][0]["quality"]
    assert np.array_equal(after["elite_segments"]["1"][0]["observations"],
                          elite_before["1"][0]["observations"])
    assert module.rng.bit_generator.state == rng_before


def test_v2_protocol_keeps_plain_feed_forward_actor_and_v1_math():
    config = load_config(ROOT / "configs/dev_marc_mappo_v2_1m.yaml")
    trainer = build_modular_mappo_trainer(config, "cpu", total_sampled_steps=1_000_000)
    architecture = checkpoint_architecture(trainer)
    assert architecture["actor_input_dim"] == 52
    assert architecture["actor_context_dim"] == 0
    assert architecture["actor_gru_hidden_dim"] == 0
    assert architecture["critic_context_dim"] == 0
    assert architecture["marc_version"] == 2
    assert architecture["log_std_retention"] is False
    assert continuation_coefficients(torch.tensor([1, 2, 3])).tolist() == pytest.approx([1/3, 1/2, 1])
    waves = torch.tensor([[1], [1], [2], [3]])
    alive = torch.ones(4, 1, 4)
    direct, _ = tempered_wave_weights(waves, alive)
    via_module, _ = trainer.milestone_aware_retention_credit.wave_weights(waves, alive)
    assert torch.equal(direct, via_module)
    state = trainer.checkpoint_state()
    assert state["development_feature_versions"]["milestone_aware_retention_credit"] == 2
    assert state["milestone_aware_retention_credit_state"]["version"] == 2
