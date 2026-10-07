"""Minimal persistent-wave semantics without changing the V2.3 contract."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from env.config import load_config
from env.combat_env import MultiUAVCombatEnv
from env.factory import make_combat_environment
from env.persistent_env import PersistentWaveCombatEnv
from env.weapon import FireState
from algorithm.common.checkpoint import validate_checkpoint_environment
from algorithm.common.checkpoint import evaluation_selection_key
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.common.vector_env import ParallelVectorEnv


ROOT = Path(__file__).resolve().parents[1]


def persistent_config() -> dict:
    return load_config(ROOT / "configs/persistent_wave_environment.yaml")


def clear_blue(env: MultiUAVCombatEnv) -> None:
    for state in env.blue:
        state.alive = False


def test_factory_selects_variant_without_changing_observation_or_weapon_contract():
    env = make_combat_environment(persistent_config())
    assert isinstance(env, PersistentWaveCombatEnv)
    observation, info = env.reset(71)
    assert observation.shape == (4, 52)
    assert len(env.red_fire_states) == len(env.blue_fire_states) == 4
    assert all(hasattr(state, "armed") for state in env.red_fire_states)
    assert not any(hasattr(state, "ammo_remaining") for state in env.red_fire_states)
    assert info["wave_index"] == 1
    assert info["total_waves"] == 3


def test_intermediate_clear_preserves_red_and_spawns_nearest_policy_blue_wave():
    env = PersistentWaveCombatEnv(persistent_config())
    env.reset(2025)
    env.red[1].alive = False
    env.red[0].x = 1234.0
    clear_blue(env)

    observation, _, terminated, truncated, info = env.step(
        np.zeros((4, 3), dtype=np.float32)
    )

    assert not terminated and not truncated
    assert env.wave_index == 2 and env.waves_cleared == 1
    assert info["wave_cleared_this_step"]
    assert info["spawned_next_wave"]
    assert info["termination_reason"] == "ongoing"
    assert info["blue_survivors"] == 4
    assert np.array_equal(info["blue_alive_mask"], np.ones(4, dtype=np.float32))
    assert np.array_equal(info["red_alive_mask"], env.red_alive_mask)
    assert info["blue_losses"] == 4
    assert observation.shape == (4, 52)
    assert not env.red[1].alive
    assert 0.0 < abs(env.red[0].x - 1234.0) < 30.0
    assert all(state.alive for state in env.blue)
    assert env.fixed_policy.__class__.__name__ == "NearestTargetPursuitPolicy"
    assert env._blue_wave_inside_arena(env.blue)
    assert 0 <= info["wave_spawn_candidate_index"] < 72
    assert info["minimum_spawn_distance"] == pytest.approx(
        env._minimum_red_blue_distance(env.blue)
    )


def test_clearing_step_never_lets_fresh_blue_attack_before_observation():
    env = PersistentWaveCombatEnv(persistent_config())
    env.reset(2026)
    clear_blue(env)
    attempts_before = env.combat_counts["blue"]["fire_attempts"]
    hits_before = env.combat_counts["blue"]["weapon_hits"]
    kills_before = env.combat_counts["blue"]["attack_kills"]

    observation, _, terminated, truncated, info = env.step(
        np.zeros((4, 3), dtype=np.float32)
    )

    assert not terminated and not truncated
    assert info["spawned_next_wave"]
    assert observation.shape == (4, 52)
    assert info["blue_step_fire_attempts"] == 0
    assert info["blue_step_weapon_hits"] == 0
    assert info["blue_step_attack_kills"] == 0
    assert env.combat_counts["blue"]["fire_attempts"] == attempts_before
    assert env.combat_counts["blue"]["weapon_hits"] == hits_before
    assert env.combat_counts["blue"]["attack_kills"] == kills_before


def test_intermediate_clear_rearms_both_sides_and_preserves_red_bank_state():
    env = PersistentWaveCombatEnv(persistent_config())
    env.reset(3025)
    env.red_fire_states = [FireState(armed=False) for _ in range(4)]
    env.blue_fire_states = [FireState(armed=False) for _ in range(4)]
    clear_blue(env)
    actions = np.zeros((4, 3), dtype=np.float32)
    actions[:, 0] = 1.0

    env.step(actions)

    assert all(state.armed for state in env.red_fire_states)
    assert all(state.armed for state in env.blue_fire_states)
    assert np.count_nonzero(env.red_last_executed_phi) == 4
    assert np.count_nonzero(env.blue_last_executed_phi) == 0


def test_wave_record_is_closed_before_spawn_and_matches_clearing_reward():
    env = PersistentWaveCombatEnv(persistent_config())
    env.reset(91)
    clear_blue(env)
    _, reward, _, _, info = env.step(np.zeros((4, 3), dtype=np.float32))

    record = info["per_wave_metrics"][0]
    assert record["wave_index"] == 1
    assert record["start_step"] == 0 and record["end_step"] == 1
    assert record["blue_survivors_start"] == 4
    assert record["blue_survivors_end"] == 0
    assert record["wave_completed"] and record["wave_cleared"]
    assert record["termination_reason"] == "wave_cleared"
    assert record["team_return"] == pytest.approx(float(reward.sum()))
    assert record["r1_total"] + record["r2_total"] + record["r3_total"] + record["r4_total"] == pytest.approx(record["team_return"])


def test_wave_refresh_does_not_change_the_clearing_step_reward():
    cfg = persistent_config()
    direct = MultiUAVCombatEnv(cfg)
    persistent = PersistentWaveCombatEnv(cfg)
    direct.reset(9001)
    persistent.reset(9001)
    clear_blue(direct)
    clear_blue(persistent)
    actions = np.zeros((4, 3), dtype=np.float32)

    _, direct_reward, _, _, _ = direct.step(actions)
    _, persistent_reward, _, _, _ = persistent.step(actions)

    assert np.array_equal(persistent_reward, direct_reward)


def test_final_wave_uses_original_success_termination():
    cfg = copy.deepcopy(persistent_config())
    cfg["persistent_waves"]["total_waves"] = 1
    env = PersistentWaveCombatEnv(cfg)
    env.reset(5)
    clear_blue(env)

    _, _, terminated, truncated, info = env.step(
        np.zeros((4, 3), dtype=np.float32)
    )

    assert terminated and not truncated
    assert info["red_success"] and info["red_win"]
    assert info["wave_cleared_this_step"]
    assert not info["spawned_next_wave"]
    assert info["waves_cleared"] == 1
    assert info["blue_losses"] == 4
    assert len(info["per_wave_metrics"]) == 1
    assert info["per_wave_metrics"][0]["wave_cleared"]
    assert info["per_wave_metrics"][0]["termination_reason"] == "red_win"


def test_time_limit_prevents_spawning_another_wave():
    env = PersistentWaveCombatEnv(persistent_config())
    env.reset(6)
    env.steps = env.max_steps - 1
    clear_blue(env)

    _, _, terminated, truncated, info = env.step(
        np.zeros((4, 3), dtype=np.float32)
    )

    assert not terminated and truncated
    assert info["termination_reason"] == "red_failure_timeout"
    assert info["wave_index"] == 1
    assert not info["spawned_next_wave"]
    assert len(info["per_wave_metrics"]) == 1
    assert info["per_wave_metrics"][0]["wave_cleared"]
    assert info["per_wave_metrics"][0]["termination_reason"] == "red_failure_timeout"


def test_mutual_elimination_has_priority_over_wave_spawn():
    env = PersistentWaveCombatEnv(persistent_config())
    env.reset(16)
    for state in env.red + env.blue:
        state.alive = False

    _, _, terminated, truncated, info = env.step(
        np.zeros((4, 3), dtype=np.float32)
    )

    assert terminated and not truncated
    assert info["draw"]
    assert not info["spawned_next_wave"]
    assert info["waves_cleared"] == 0
    assert len(info["per_wave_metrics"]) == 1
    assert not info["per_wave_metrics"][0]["wave_cleared"]
    assert info["per_wave_metrics"][0]["termination_reason"] == "draw_mutual_destruction"


def test_red_elimination_finishes_partial_wave_record_once():
    env = PersistentWaveCombatEnv(persistent_config())
    env.reset(116)
    for state in env.red:
        state.alive = False

    _, _, terminated, truncated, info = env.step(
        np.zeros((4, 3), dtype=np.float32)
    )

    assert terminated and not truncated
    assert len(info["per_wave_metrics"]) == 1
    record = info["per_wave_metrics"][0]
    assert not record["wave_cleared"]
    assert record["termination_reason"] == "blue_win"


def test_ordinary_timeout_finishes_partial_wave_record_once():
    env = PersistentWaveCombatEnv(persistent_config())
    env.reset(117)
    env.steps = env.max_steps - 1

    _, _, terminated, truncated, info = env.step(
        np.zeros((4, 3), dtype=np.float32)
    )

    assert not terminated and truncated
    assert len(info["per_wave_metrics"]) == 1
    record = info["per_wave_metrics"][0]
    assert not record["wave_cleared"]
    assert record["termination_reason"] == "red_failure_timeout"


def test_next_wave_generation_is_seed_deterministic():
    first = PersistentWaveCombatEnv(persistent_config())
    second = PersistentWaveCombatEnv(persistent_config())
    first.reset(12345)
    second.reset(12345)
    clear_blue(first)
    clear_blue(second)
    actions = np.zeros((4, 3), dtype=np.float32)

    first.step(actions)
    second.step(actions)

    assert np.array_equal(
        np.stack([state.as_array() for state in first.blue]),
        np.stack([state.as_array() for state in second.blue]),
    )


def test_spawn_selects_farthest_of_72_complete_formation_candidates():
    manual = PersistentWaveCombatEnv(persistent_config())
    actual = PersistentWaveCombatEnv(persistent_config())
    manual.reset(24680)
    actual.reset(24680)
    candidates = [
        manual._candidate_blue_wave(float(angle))
        for angle in np.linspace(-np.pi, np.pi, 72, endpoint=False)
    ]
    distances = [manual._minimum_red_blue_distance(row) for row in candidates]
    expected_index = int(np.argmax(distances))

    actual._spawn_next_wave()

    assert actual.last_spawn_candidate_index == expected_index
    assert actual.last_minimum_spawn_distance == pytest.approx(distances[expected_index])
    assert np.array_equal(
        np.stack([state.as_array() for state in actual.blue]),
        np.stack([state.as_array() for state in candidates[expected_index]]),
    )


def test_spawn_config_has_no_rejection_budget_or_hard_minimum_distance():
    wave_config = persistent_config()["persistent_waves"]
    assert wave_config["spawn_direction_count"] == 72
    assert "max_spawn_attempts" not in wave_config
    assert "min_red_distance" not in wave_config


def test_parallel_worker_constructs_persistent_environment_class():
    with ParallelVectorEnv(1, persistent_config(), base_seed=81_000_000) as vector:
        vector.reset()
        assert vector.worker_environment_classes == ["PersistentWaveCombatEnv"]
        assert vector.worker_environment_variants == ["persistent_wave_v1"]


@pytest.mark.parametrize(
    "checkpoint_variant,requested_variant",
    [
        ("direct_v2_3", "persistent_wave_v1"),
        ("persistent_wave_v1", "direct_v2_3"),
    ],
)
def test_checkpoint_rejects_direct_persistent_cross_loading(
    checkpoint_variant, requested_variant
):
    state = {"extra": {
        "environment_version": "2.3",
        "environment_variant": checkpoint_variant,
    }}
    with pytest.raises(RuntimeError, match="environment_variant mismatch"):
        validate_checkpoint_environment(
            state, {"environment_variant": requested_variant}
        )


def test_persistent_algorithm_configs_change_only_discount():
    import yaml

    direct = yaml.safe_load((ROOT / "configs/mappo.yaml").read_text())
    persistent = yaml.safe_load(
        (ROOT / "configs/mappo_persistent_wave.yaml").read_text()
    )
    assert direct["training"]["gamma"] == 0.99
    assert persistent["training"]["gamma"] == 0.999
    direct["training"]["gamma"] = persistent["training"]["gamma"]
    # A missing legacy field and an explicit value both select attention.
    direct["network"]["critic_type"] = direct["network"].get(
        "critic_type", "attention"
    )
    assert direct == persistent


def test_persistent_mappo_startup_loads_gamma_and_mission_identity(tmp_path):
    import yaml

    algorithm = yaml.safe_load(
        (ROOT / "configs/mappo_persistent_wave.yaml").read_text()
    )
    runner = MAPPOTrainingRunner(
        persistent_config(), algorithm, num_envs=1, total_sampled_steps=1,
        output_dir=tmp_path, smoke=True,
    )
    try:
        summary = runner.startup_summary()
        assert summary["gamma"] == 0.999
        assert summary["environment_variant"] == "persistent_wave_v1"
        assert summary["total_waves"] == 3
        assert summary["max_steps"] == 3000
        assert "gamma=0.999 | lambda=0.95 | clip=0.2 | variant=persistent_wave_v1 | waves=3 | max_steps=3000" in runner.start_log_line()
    finally:
        runner.vector.close()


def test_persistent_best_selection_prioritizes_final_clear_then_mean_waves():
    weaker = {
        "win_rate": 0.0, "average_waves_cleared": 0.5,
        "clear_wave_3_probability": 0.0, "average_return": 100.0,
        "average_red_loss": 0.0,
    }
    stronger = {
        "win_rate": 0.0, "average_waves_cleared": 1.5,
        "clear_wave_3_probability": 0.0, "average_return": -100.0,
        "average_red_loss": 4.0,
    }
    runner = MAPPOTrainingRunner.__new__(MAPPOTrainingRunner)
    runner.env_config = {"environment_variant": "persistent_wave_v1"}
    assert runner._evaluation_key(stronger) > runner._evaluation_key(weaker)
    mission_success = {
        **weaker, "clear_wave_3_probability": 0.10,
        "average_waves_cleared": 2.00,
    }
    no_success = {
        **stronger, "clear_wave_3_probability": 0.0,
        "average_waves_cleared": 2.01,
    }
    assert runner._evaluation_key(mission_success) > runner._evaluation_key(no_success)


def test_direct_best_selection_tuple_is_unchanged():
    record = {
        "win_rate": 0.4, "average_return": 5.0, "average_red_loss": 2.0,
        "average_waves_cleared": 99.0, "clear_wave_3_probability": 1.0,
    }
    assert evaluation_selection_key(record, "direct_v2_3") == (0.4, 5.0, -2.0)


def test_nonterminal_wave_transition_is_stored_in_mappo_rollout():
    old_observation = np.zeros((1, 4, 52), dtype=np.float32)
    fresh_wave_observation = np.ones((1, 4, 52), dtype=np.float32)
    alive = np.ones((1, 4), dtype=np.float32)

    class FakeTrainer:
        sampled_steps = 0
        vector_steps = 0

        @staticmethod
        def act(observations, masks, return_policy_data=False):
            actions = np.zeros((1, 4, 3), dtype=np.float32)
            raw = actions.copy()
            log_probs = np.zeros((1, 4), dtype=np.float32)
            return actions, raw, log_probs

    class FakeVector:
        current_alive_masks = alive

        @staticmethod
        def step_batch(actions):
            return SimpleNamespace(
                observations=fresh_wave_observation,
                transition_next_observations=fresh_wave_observation,
                rewards=np.zeros((1, 4), dtype=np.float32),
                terminated=np.array([False]), truncated=np.array([False]),
                infos=[{"wave_cleared_this_step": True}],
                alive_masks=alive, next_alive_masks=alive,
            )

    runner = MAPPOTrainingRunner.__new__(MAPPOTrainingRunner)
    runner.trainer = FakeTrainer()
    runner.vector = FakeVector()
    runner.num_envs = 1
    runner.rollout_steps = 1
    runner.observations = old_observation
    runner.alive_masks = alive
    runner._completed = lambda result: []
    runner._write_step_metrics = lambda result, rows: None

    rollout = runner.collect_rollout(1)

    assert rollout.dones[0, 0] == 0.0
    assert np.array_equal(rollout.observations[0], old_observation)
    assert np.array_equal(rollout.next_observations[0], fresh_wave_observation)
