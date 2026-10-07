from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from algorithm.modules.persistent_wave_trajectory_replay import (
    BRIDGE_12, BRIDGE_23, PWTR_PARTITION_CAPACITY, W2_INTERNAL, W3_INTERNAL,
    PersistentWaveTrajectoryReplayModule, clipped_importance_weights,
    normalized_ess_and_freshness, replay_batch_budget, replay_budget_fill_fraction,
    transition_actor_age_mask, valid_collection_generations,
    vtrace_targets_and_advantages, wave_stratified_permutation,
)
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from tools.analyze_pwtr_ablation import classify_full_gate, descriptive, endpoint, TARGET


ROOT = Path(__file__).resolve().parents[1]


def cfg(**overrides):
    value = {
        "enabled": True, "fresh_wave_stratification": True,
        "replay_enabled": True, "replay_source": "recent_uniform",
        "priority_enabled": False, "bridge_enabled": True,
        "actor_replay": True, "critic_replay": True,
        "sequence_length": 128, "bridge_half_length": 64,
        "min_segment_length": 32, "partition_capacity": 32,
        "actor_max_age_updates": 2,
    }
    value.update(overrides)
    return value


def rollout(waves, transitions=None, dones=None, alive=None, next_alive=None):
    waves = np.asarray(waves, dtype=np.int64).reshape(-1, 1)
    t = len(waves)
    transitions = np.zeros((t, 1), dtype=np.float32) if transitions is None else np.asarray(transitions, dtype=np.float32).reshape(t, 1)
    dones = np.zeros((t, 1), dtype=np.float32) if dones is None else np.asarray(dones, dtype=np.float32).reshape(t, 1)
    alive = np.ones((t, 1, 4), dtype=np.float32) if alive is None else np.asarray(alive, dtype=np.float32).reshape(t, 1, 4)
    next_alive = alive.copy() if next_alive is None else np.asarray(next_alive, dtype=np.float32).reshape(t, 1, 4)
    marker = np.arange(t, dtype=np.float32).reshape(t, 1, 1, 1)
    obs = np.broadcast_to(marker, (t, 1, 4, 52)).copy()
    return SimpleNamespace(
        observations=obs, next_observations=obs + 0.5,
        raw_actions=np.zeros((t, 1, 4, 3), np.float32),
        old_log_probs=np.zeros((t, 1, 4), np.float32),
        rewards=np.ones((t, 1, 4), np.float32), dones=dones,
        alive_masks=alive, next_alive_masks=next_alive,
        wave_indices=waves, wave_transition_flags=transitions,
    )


def test_stratification_is_exact_and_handles_remainder_and_missing_waves():
    waves = np.array([1] * 47 + [2] * 35 + [3] * 18)
    order = wave_stratified_permutation(waves, 17, np.random.default_rng(7))
    assert sorted(order.tolist()) == list(range(100))
    assert np.bincount(waves[order], minlength=4).tolist() == np.bincount(waves, minlength=4).tolist()
    global_fraction = np.bincount(waves, minlength=4)[1:] / len(waves)
    strat_error = np.mean([np.abs(np.bincount(waves[order[i:i+17]], minlength=4)[1:] / len(order[i:i+17]) - global_fraction).sum() for i in range(0, 85, 17)])
    random = np.random.default_rng(19).permutation(len(waves))
    random_error = np.mean([np.abs(np.bincount(waves[random[i:i+17]], minlength=4)[1:] / len(random[i:i+17]) - global_fraction).sum() for i in range(0, 85, 17)])
    assert strat_error < random_error
    assert len(wave_stratified_permutation(np.ones(23), 8, np.random.default_rng(1))) == 23
    no_w3 = np.array([1] * 9 + [2] * 8)
    assert sorted(wave_stratified_permutation(no_w3, 6, np.random.default_rng(2)).tolist()) == list(range(17))


def test_internal_extraction_cross_rollout_remainders_fifo_and_partitions():
    module = PersistentWaveTrajectoryReplayModule(cfg(), 3)
    module.ingest_rollout(rollout([2] * 80), 1)
    assert not module.partitions[W2_INTERNAL]
    module.ingest_rollout(rollout([2] * 48), 1)
    assert module.partitions[W2_INTERNAL][-1]["actual_length"] == 128
    module.ingest_rollout(rollout([3] * 40, dones=[0] * 39 + [1]), 2)
    assert module.partitions[W3_INTERNAL][-1]["actual_length"] == 40
    before = len(module.partitions[W2_INTERNAL])
    module.ingest_rollout(rollout([2] * 31, dones=[0] * 30 + [1]), 3)
    assert len(module.partitions[W2_INTERNAL]) == before
    for generation in range(40):
        module.ingest_rollout(rollout([2] * 32, dones=[0] * 31 + [1]), 10 + generation)
    assert len(module.partitions[W2_INTERNAL]) == PWTR_PARTITION_CAPACITY
    assert len(module.partitions[W3_INTERNAL]) == 1


def test_bridge_12_23_cross_rollout_padding_and_boundary_semantics():
    module = PersistentWaveTrajectoryReplayModule(cfg(), 4)
    # Transition flag is carried by the final source-wave transition; its next obs is post-spawn.
    module.ingest_rollout(rollout([1] * 70, transitions=[0] * 69 + [1]), 1)
    assert 1 in module.pending_bridges[0]
    module.ingest_rollout(rollout([2] * 64), 1)
    b12 = module.partitions[BRIDGE_12][-1]
    assert b12["actual_length"] == 128
    assert np.all(b12["wave_indices"][:64] == 1) and np.all(b12["wave_indices"][64:128] == 2)
    assert b12["dones"][63] == 0 and b12["next_alive_masks"][63].sum() == 4
    module.ingest_rollout(rollout([2] * 70, transitions=[0] * 69 + [1]), 2)
    module.ingest_rollout(rollout([3] * 20, dones=[0] * 19 + [1]), 2)
    b23 = module.partitions[BRIDGE_23][-1]
    assert b23["actual_length"] == 84
    assert b23["valid_time_mask"].sum() == 84 and b23["valid_time_mask"][84:].sum() == 0


def test_importance_freshness_vtrace_terminal_death_bootstrap_and_finite():
    new = torch.tensor([[[0.0, -1.0, 1000.0, -1000.0]]])
    old = torch.zeros_like(new)
    alive = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    valid = torch.ones(1, 1)
    _, individual, joint_log, joint = clipped_importance_weights(new, old, alive, valid)
    assert individual[0, 0, 0] == 1 and 0 < individual[0, 0, 1] < 1
    assert joint_log.item() == -1 and torch.allclose(joint, torch.tensor([[np.exp(-1)]], dtype=torch.float32))
    ess, divergence, freshness = normalized_ess_and_freshness(torch.tensor([[0.0, -2.0]]), torch.ones(1, 2))
    assert 0 < ess <= 1 and divergence > 0 and 0 < freshness <= 1
    rewards = torch.tensor([[[1.0]], [[2.0]]]).transpose(0, 1)
    values = torch.zeros_like(rewards)
    # For contiguous transitions V(next_s_t) equals V(s_{t+1}); the final
    # terminal transition's next value is intentionally irrelevant.
    next_values = torch.tensor([[[0.0], [20.0]]])
    dones = torch.tensor([[0.0, 1.0]])
    alive1 = torch.ones_like(rewards)
    target, advantage = vtrace_targets_and_advantages(rewards, values, next_values, dones, alive1, alive1, torch.ones(1, 2), torch.ones(1, 2), .9)
    assert torch.allclose(target[0, 1], torch.tensor([2.0]))
    assert torch.allclose(target[0, 0], torch.tensor([1 + .9 * 2]))
    # Death truncates even when episode is not done.
    dead_next = alive1.clone(); dead_next[:, 0] = 0
    death_target, _ = vtrace_targets_and_advantages(rewards, values, next_values, torch.zeros_like(dones), alive1, dead_next, torch.ones(1, 2), torch.ones(1, 2), .9)
    assert torch.allclose(death_target[0, 0], torch.tensor([1.0]))
    # A nonterminal last segment step uses stored next-state bootstrap.
    one_target, _ = vtrace_targets_and_advantages(torch.ones(1, 1, 1), torch.zeros(1, 1, 1), torch.full((1, 1, 1), 5.0), torch.zeros(1, 1), torch.ones(1, 1, 1), torch.ones(1, 1, 1), torch.ones(1, 1), torch.ones(1, 1), .9)
    assert torch.allclose(one_target, torch.tensor([[[5.5]]]))
    assert torch.isfinite(individual).all() and torch.isfinite(target).all() and torch.isfinite(advantage).all()


def test_priority_formula_has_no_wave_or_success_bonus_and_budget_is_matched():
    def priority(error, freshness):
        return (error + 1e-6) * freshness
    assert priority(2, .8) > priority(1, .8)
    assert priority(1, .8) > priority(1, .4)
    # Wave and outcome labels are intentionally absent from the frozen formula.
    assert priority(1, .5) == priority(1, .5)
    assert [replay_batch_budget(n) for n in (0, 1, 512, 513, 6144)] == [0, 1, 1, 2, 12]


def test_memory_rng_checkpoint_roundtrip_and_sampling_identity():
    module = PersistentWaveTrajectoryReplayModule(cfg(), 123)
    module.ingest_rollout(rollout([2] * 128), 1)
    module.ingest_rollout(rollout([3] * 128), 1)
    state = module.state_dict()
    restored = PersistentWaveTrajectoryReplayModule(cfg(), 999)
    restored.load_state_dict(deepcopy(state))
    assert restored.state_dict()["segment_id_counter"] == state["segment_id_counter"]
    left = [s["segment_id"] for s in module.sample_batch(2)]
    right = [s["segment_id"] for s in restored.sample_batch(2)]
    assert left == right


def test_six_branch_configs_are_exact_and_matched():
    names = ("plain", "stratified", "current_extra", "uniform_recent", "priority_recent", "full")
    configs = {name: load_config(ROOT / f"configs/dev_pwtr_{name}_300k.yaml") for name in names}
    for config in configs.values():
        assert config["training"]["total_sampled_steps"] == 1_805_280
        assert config["development_branch"]["source_sampled_steps"] == 1_505_280
        assert config["development_branch"]["additional_sampled_steps"] == 300_000
        assert config["implementation"]["evaluation_seed_base"] == 44_000_000
    assert not configs["plain"]["modules"].get("persistent_wave_trajectory_replay", {}).get("enabled", False)
    assert configs["full"]["modules"]["persistent_wave_trajectory_replay"]["bridge_enabled"] is True


def test_disabled_module_does_not_consume_rng_or_change_permutation():
    disabled = PersistentWaveTrajectoryReplayModule({"enabled": False}, 1)
    a = np.random.default_rng(88); b = np.random.default_rng(88)
    assert np.array_equal(disabled.fresh_epoch_permutation(np.array([1, 2, 3]), 2, a), b.permutation(3))


def test_analyzer_frozen_gates_and_descriptive_labels():
    good = [{"AverageWaves": .1, "W3": .02, "W1": 0.0}] * 3
    assert classify_full_gate(good) == ("PROMISING", True, True, True)
    unsafe = deepcopy(good); unsafe[0] = {"AverageWaves": -.5, "W3": .02, "W1": 0.0}
    assert classify_full_gate(unsafe)[0] == "SAFETY_FAIL"
    label, _ = descriptive([{"AverageWaves": .1, "W3": .1, "Q2": .1, "Q3": .1}] * 3)
    assert label == "POSITIVE"


def test_mixed_generation_eligibility_is_transition_exact():
    module = PersistentWaveTrajectoryReplayModule(cfg(replay_source="current"), 10)
    module.ingest_rollout(rollout([2] * 30), 100)
    module.ingest_rollout(rollout([2] * 98), 101)
    segment = module.partitions[W2_INTERNAL][-1]
    assert set(valid_collection_generations(segment).tolist()) == {100, 101}
    assert module.eligible(101) == {}
    module.replay_source = "recent_uniform"
    assert module.eligible(102)[W2_INTERNAL] == [segment]
    pure = PersistentWaveTrajectoryReplayModule(cfg(replay_source="current"), 11)
    pure.ingest_rollout(rollout([2] * 128), 101)
    assert pure.eligible(101)[W2_INTERNAL][0]["segment_id"] == 0


def test_actor_age_is_transition_level_and_old_prefix_is_masked():
    ids = torch.tensor([[100, 101, 102]])
    valid = torch.ones(1, 3)
    alive = torch.ones(1, 3, 4)
    ages, mask = transition_actor_age_mask(ids, 103, valid, alive)
    assert ages.tolist() == [[3, 2, 1]]
    assert mask[0, 0].sum() == 0
    assert mask[0, 1].sum() == 4 and mask[0, 2].sum() == 4


def test_pending_discard_preserves_completed_memory_and_rng():
    module = PersistentWaveTrajectoryReplayModule(cfg(), 44)
    module.ingest_rollout(rollout([2] * 128), 1)
    completed_id = module.partitions[W2_INTERNAL][0]["segment_id"]
    module.ingest_rollout(rollout([1] * 70, transitions=[0] * 69 + [1]), 2)
    module.ingest_rollout(rollout([2] * 12), 2)
    rng_before = deepcopy(module.rng.bit_generator.state)
    dropped = module.discard_pending_after_environment_restart()
    assert dropped >= 3
    assert module.partitions[W2_INTERNAL][0]["segment_id"] == completed_id
    assert module.pending_internal == {} and module.source_tails == {}
    assert module.pending_bridges == {} and module.entry_survivors == {}
    assert module.rng.bit_generator.state == rng_before
    assert module.pending_dropped_on_resume == dropped


def test_short_w2_immediately_finalizes_bridge12_at_w3_transition():
    module = PersistentWaveTrajectoryReplayModule(cfg(), 45)
    module.ingest_rollout(rollout([1] * 70, transitions=[0] * 69 + [1]), 1)
    module.ingest_rollout(rollout([2] * 40, transitions=[0] * 39 + [1]), 1)
    assert len(module.partitions[BRIDGE_12]) == 1
    assert module.partitions[BRIDGE_12][0]["actual_length"] == 104
    assert 1 not in module.pending_bridges[0]
    assert 2 in module.pending_bridges[0]


def _write_analyzer_fixture(path: Path, final_steps: int = TARGET):
    path.mkdir()
    (path / "run_summary.json").write_text(json.dumps({"sampled_steps": TARGET}))
    (path / "run_config.json").write_text(json.dumps({"seed": 5301, "development_method": "pwtr_plain_matched_control", "enabled_modules": ["actor_lr_decay"]}))
    (path / "algorithm_config.yaml").write_text("development_method: pwtr_plain_matched_control\nmodules:\n  actor_lr_decay: {enabled: true}\n")
    columns = {"sampled_steps": TARGET, "evaluation_episodes": 50, "evaluation_seed_base": 44_000_000, "evaluation_seed_end": 44_000_049}
    for column in set(METRICS_FOR_FIXTURE.values()): columns[column] = 0.0
    with (path / "evaluation_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(columns)); writer.writeheader(); writer.writerow(columns)
    (path / "optimization_metrics.jsonl").write_text(json.dumps({"sampled_steps": TARGET, "actor_loss": 0.0}) + "\n")
    (path / "training_metrics.jsonl").write_text("{}\n")
    (path / "train.log").write_text("[DONE]\n")
    for name in ("latest.pt", "final.pt"):
        torch.save({"algorithm": "modular_mappo", "sampled_steps": final_steps,
                    "enabled_modules": ["actor_lr_decay"],
                    "extra": {"development_method": "pwtr_plain_matched_control"}}, path / name)


# Kept local so the fixture remains independent of analyzer implementation details.
METRICS_FOR_FIXTURE = {
    "W1": "clear_wave_1_probability", "W2": "clear_wave_2_probability", "W3": "clear_wave_3_probability",
    "AW": "average_waves_cleared", "Return": "average_return", "Red": "average_red_loss",
    "Blue": "average_blue_loss", "Boundary": "average_red_boundary_exits", "Ground": "average_red_ground_losses",
    "Length": "average_episode_length", "S2": "average_red_survivors_after_wave_1_conditional_on_clear",
    "S3": "average_red_survivors_after_wave_2_conditional_on_clear",
}


def test_analyzer_accepts_no_exact_step_periodic_checkpoint_and_validates_final(tmp_path):
    run = tmp_path / "run"; _write_analyzer_fixture(run)
    endpoint(run, "plain")
    assert not (run / f"checkpoint_{TARGET}.pt").exists()
    bad = tmp_path / "bad"; _write_analyzer_fixture(bad, TARGET - 1)
    with pytest.raises(RuntimeError, match="checkpoint identity/endpoint mismatch"):
        endpoint(bad, "plain")


def test_budget_fill_fraction_definition():
    assert replay_budget_fill_fraction(0, 0) == 1.0
    assert replay_budget_fill_fraction(2, 2) == 1.0
    assert replay_budget_fill_fraction(1, 2) == .5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is mandatory for runner resume integration")
def test_runner_resume_discards_pending_but_preserves_completed_memory_and_pwtr_rng(tmp_path):
    env = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    config = load_config(ROOT / "configs/dev_pwtr_full_300k.yaml")
    source = ModularMAPPOTrainingRunner(env, config, 1, 1_805_280, "cuda", 5301,
                                        tmp_path / "source", True, resume_mode=False)
    module = source.trainer.persistent_wave_trajectory_replay
    module.ingest_rollout(rollout([2] * 128), 1)
    completed_id = module.partitions[W2_INTERNAL][0]["segment_id"]
    module.ingest_rollout(rollout([1] * 70, transitions=[0] * 69 + [1]), 2)
    module.ingest_rollout(rollout([2] * 12), 2)
    saved_rng = deepcopy(module.rng.bit_generator.state)
    checkpoint = tmp_path / "resume.pt"
    source.save_checkpoint(checkpoint)
    source.vector.close()

    resumed = ModularMAPPOTrainingRunner(env, config, 1, 1_805_280, "cuda", 5301,
                                         tmp_path / "resumed", True, resume_mode=True)
    resumed.resume(checkpoint)
    restored = resumed.trainer.persistent_wave_trajectory_replay
    assert restored.partitions[W2_INTERNAL][0]["segment_id"] == completed_id
    assert restored.pending_internal == {}
    assert restored.source_tails == {}
    assert restored.pending_bridges == {}
    assert restored.entry_survivors == {}
    assert restored.pending_dropped_on_resume > 0
    assert restored.rng.bit_generator.state == saved_rng
    resumed.vector.close()
