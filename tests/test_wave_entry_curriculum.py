from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.modules.wave_entry_curriculum import WaveEntryCurriculumModule
from env.config import load_config
from env.persistent_env import PersistentWaveCombatEnv

ROOT = Path(__file__).resolve().parents[1]


def config():
    return load_config(ROOT / "configs/persistent_wave_v2_environment.yaml")


def reach_entry(env, wave):
    while env.wave_index < wave:
        for state in env.blue: state.alive = False
        _, _, terminated, truncated, info = env.step(np.zeros((4, 3), np.float32))
        assert info["spawned_next_wave"] and not terminated and not truncated
    return info


@pytest.mark.parametrize("wave", [2, 3])
def test_snapshot_round_trip_and_next_step_equivalence(wave):
    original = PersistentWaveCombatEnv(config()); original.reset(1234)
    reach_entry(original, wave)
    snapshot = original.export_curriculum_state()
    restored = PersistentWaveCombatEnv(config()); restored.reset(9999)
    metadata = restored.restore_curriculum_state(snapshot)
    assert metadata["wave_index"] == wave and metadata["steps"] == original.steps
    assert restored.steps == restored._wave_start_step
    assert np.array_equal(restored._observations(), original._observations())
    assert restored.rng.bit_generator.state == original.rng.bit_generator.state
    assert [x.armed for x in restored.red_fire_states] == [x.armed for x in original.red_fire_states]
    assert np.array_equal(restored.red_last_executed_phi, original.red_last_executed_phi)
    assert restored.fixed_policy.diagnostics() == original.fixed_policy.diagnostics()
    actions = np.full((4, 3), 0.125, np.float32)
    left, right = original.step(actions), restored.step(actions)
    for index in (0, 1): assert np.array_equal(left[index], right[index])
    assert left[2:4] == right[2:4]
    for key in ("wave_index", "waves_cleared", "red_losses", "blue_losses",
                "red_attack_kills", "blue_attack_kills", "episode_length"):
        assert left[4][key] == right[4][key]


def test_snapshot_is_deep_and_preserves_remaining_horizon():
    env = PersistentWaveCombatEnv(config()); env.reset(5); reach_entry(env, 3)
    snapshot = env.export_curriculum_state(); original_x = snapshot["red"][0].x
    env.red[0].x += 100
    assert snapshot["red"][0].x == original_x
    other = PersistentWaveCombatEnv(config()); other.reset(6); other.restore_curriculum_state(snapshot)
    assert other.max_steps - other.steps == env.max_steps - snapshot["steps"]


def test_export_rejects_non_entry_and_restore_rejects_contract_mismatch():
    env = PersistentWaveCombatEnv(config()); env.reset(8)
    with pytest.raises(RuntimeError, match="W2/W3"):
        env.export_curriculum_state()
    reach_entry(env, 2); snapshot = env.export_curriculum_state()
    env.step(np.zeros((4, 3), np.float32))
    with pytest.raises(RuntimeError, match="immediately post-spawn"):
        env.export_curriculum_state()
    bad = deepcopy(snapshot); bad["metadata"]["max_steps"] += 1
    with pytest.raises(ValueError, match="max_steps mismatch"):
        PersistentWaveCombatEnv(config()).restore_curriculum_state(bad)


def test_vector_worker_restore_synchronizes_observation_and_alive():
    source = PersistentWaveCombatEnv(config()); source.reset(7); reach_entry(source, 2)
    snapshot = source.export_curriculum_state()
    with ParallelVectorEnv(1, config(), base_seed=88_000_100) as vector:
        vector.reset(); result = vector.restore_curriculum_states({0: snapshot})[0]
        assert np.array_equal(vector.current_observations[0], result["observation"])
        assert np.array_equal(vector.current_alive_masks[0], result["red_alive_mask"])
        exported = vector.export_curriculum_states([0])[0]
        assert exported["metadata"] == snapshot["metadata"]


def module_config(**overrides):
    value = {"enabled": True, "mode": "adaptive_deficit", "recent_natural_window": 200,
        "min_natural_episodes_for_adaptation": 2, "target_reach_w2": .75,
        "target_reach_w3": .5, "max_curriculum_fraction": .5,
        "min_bank_entries": 1, "capacity_per_wave": 2, "sampling": "uniform"}
    value.update(overrides); return value


def fake_snapshot(wave, marker=0):
    return {"metadata": {"wave_index": wave}, "marker": marker}


def test_adaptive_controller_fifo_natural_only_and_rng_roundtrip():
    module = WaveEntryCurriculumModule(module_config(), 5301)
    assert module.probabilities() == (1, 0, 0)
    for marker in range(3):
        module.add_natural_entry(2, fake_snapshot(2, marker), source_sampled_steps=marker,
            source_training_seed=5301, source_env_id=0, source_episode_reset_seed=10,
            red_survivors=4)
    module.add_natural_entry(3, fake_snapshot(3), source_sampled_steps=3,
        source_training_seed=5301, source_env_id=0, source_episode_reset_seed=10,
        red_survivors=3)
    assert [row["snapshot"]["marker"] for row in module.bank[2]] == [1, 2]
    module.record_natural_episode(0); module.record_natural_episode(0)
    p1, p2, p3 = module.probabilities()
    assert p1 >= .5 and p3 > 0 and sum((p1, p2, p3)) == pytest.approx(1)
    state = module.state_dict(); clone = WaveEntryCurriculumModule(module_config(), 999)
    clone.load_state_dict(state)
    assert module.sample_reset() == clone.sample_reset()
    state["bank"][2][0]["snapshot"]["marker"] = 99
    assert module.bank[2][0]["snapshot"]["marker"] == 1


def test_targets_close_curriculum_and_rng_is_independent():
    module = WaveEntryCurriculumModule(module_config(), 5302)
    for wave in (2, 3):
        module.add_natural_entry(wave, fake_snapshot(wave), source_sampled_steps=0,
            source_training_seed=5302, source_env_id=0, source_episode_reset_seed=1,
            red_survivors=4)
    for _ in range(20): module.record_natural_episode(3)
    assert module.probabilities() == (1, 0, 0)
    trainer_rng = np.random.default_rng(42); before = deepcopy(trainer_rng.bit_generator.state)
    module.sample_reset()
    assert trainer_rng.bit_generator.state == before


def test_larger_w3_deficit_receives_larger_probability_and_bank_fallback_is_w1():
    module = WaveEntryCurriculumModule(module_config(), 5303)
    for _ in range(20): module.record_natural_episode(1)
    module.add_natural_entry(2, fake_snapshot(2), source_sampled_steps=0,
        source_training_seed=5303, source_env_id=0, source_episode_reset_seed=1,
        red_survivors=4)
    assert module.probabilities() == (1, 0, 0)  # B3 is not ready; d2 is zero.
    module.add_natural_entry(3, fake_snapshot(3), source_sampled_steps=0,
        source_training_seed=5303, source_env_id=0, source_episode_reset_seed=1,
        red_survivors=4)
    p1, p2, p3 = module.probabilities()
    assert p1 >= .5 and p3 > p2 and sum((p1, p2, p3)) == pytest.approx(1)


def test_snapshot_schema_contains_no_policy_or_ppo_replay_data():
    env = PersistentWaveCombatEnv(config()); env.reset(9); reach_entry(env, 2)
    snapshot = env.export_curriculum_state()
    forbidden = {"action", "raw_action", "old_log_prob", "advantage", "value_prediction",
                 "gae", "optimizer", "optimizer_state"}
    assert forbidden.isdisjoint(snapshot)
