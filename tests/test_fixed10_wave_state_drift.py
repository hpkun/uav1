from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from algorithm.modular_mappo.networks import ModularMAPPOActor
from env.persistent_env import PersistentWaveCombatEnv
from tools.fixed10_wave_state_drift_common import (
    FORBIDDEN_SEED_MIN, assert_safe_diagnostic_seeds, checkpoint_step,
    symmetric_diagonal_gaussian_kl, validate_bank,
)
from tools.audit_fixed10_wave_state_drift import pair_rows

ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_sampled_steps_metadata():
    assert checkpoint_step({"sampled_steps": 1_701_888}) == 1_701_888
    assert checkpoint_step({"trainer_state": {"sampled_steps": 1_800_192}}) == 1_800_192


def test_bank_schema_alive_mask_and_waves():
    observations = np.zeros((3, 4, 52), np.float32)
    alive = np.asarray([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 0, 0]], np.float32)
    wave = np.asarray([1, 2, 3], np.int64)
    validate_bank(observations, alive, wave)


def test_symmetric_kl_identical_zero_and_different_finite_nonnegative():
    mu = np.zeros((2, 4, 3))
    log_std = np.zeros_like(mu)
    same = symmetric_diagonal_gaussian_kl(mu, log_std, mu, log_std)
    assert np.array_equal(same, np.zeros((2, 4)))
    different = symmetric_diagonal_gaussian_kl(mu, log_std, mu + 0.2, log_std + 0.1)
    assert np.all(np.isfinite(different))
    assert np.all(different > 0)


def test_pair_rows_excludes_dead_agents_and_marks_w3_insufficient():
    observations = np.zeros((3, 4, 52), np.float32)
    alive = np.asarray([[1, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], np.float32)
    wave = np.asarray([1, 2, 3], np.int64)
    base = {"mu": np.zeros((3, 4, 3)), "log_std": np.zeros((3, 4, 3))}
    changed = {"mu": np.ones((3, 4, 3)) * 0.1, "log_std": np.zeros((3, 4, 3))}
    rows = pair_rows({"id": "x", "policy_a": "a", "policy_b": "b", "class": "test"}, base, changed, alive, wave)
    groups = {row["group"]: row for row in rows}
    assert groups["ALL"]["alive_agent_count"] == 4
    assert groups["ALL"]["dead_agent_observations_excluded"] == 8
    assert groups["W3"]["coverage"] == "INSUFFICIENT_COVERAGE"


def _post_spawn_snapshot():
    config = yaml.safe_load((ROOT / "configs" / "persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    env = PersistentWaveCombatEnv(deepcopy(config))
    env.reset(88_330_000)
    for state in env.blue:
        state.alive = False
    observation, _, terminated, truncated, info = env.step(np.zeros((4, 3), np.float32))
    assert not terminated and not truncated
    assert info["spawned_next_wave"] and info["wave_index"] == 2
    assert env.steps == env._wave_start_step
    return config, env.export_curriculum_state(), observation, info["red_alive_mask"]


def test_post_spawn_snapshot_roundtrip_has_no_next_wave_action_leakage():
    config, snapshot, observation, alive = _post_spawn_snapshot()
    restored_env = PersistentWaveCombatEnv(deepcopy(config))
    restored_env.reset(88_330_001)
    restored = restored_env.restore_curriculum_state(snapshot)
    assert restored_env.steps == restored_env._wave_start_step
    assert restored_env.wave_index == 2
    assert np.array_equal(restored["observation"], observation)
    assert np.array_equal(restored["red_alive_mask"], alive)
    # No W2 action has occurred: W1 ended exactly where the W2 start marker sits.
    assert restored_env.wave_records[-1]["wave_index"] == 1
    assert restored_env.wave_records[-1]["end_step"] == restored_env.steps
    assert restored_env._wave_start_step == restored_env.steps


def test_same_snapshot_actor_and_future_rng_are_one_step_reproducible():
    config, snapshot, _, _ = _post_spawn_snapshot()
    torch.manual_seed(7)
    actor = ModularMAPPOActor(52, 3, 256).eval()
    results = []
    for _ in range(2):
        env = PersistentWaveCombatEnv(deepcopy(config))
        env.reset(88_330_002)
        restored = env.restore_curriculum_state(snapshot)
        env.rng = np.random.default_rng(88_430_000)
        with torch.no_grad():
            obs = torch.as_tensor(restored["observation"])[None]
            mask = torch.as_tensor(restored["red_alive_mask"])[None]
            distribution, _ = actor.distribution_step(obs, None, None, None, mask)
            action = torch.tanh(distribution.mean)[0].numpy()
        next_obs, reward, terminated, truncated, info = env.step(action)
        results.append((next_obs, reward, terminated, truncated, info["wave_index"], env.steps))
    assert np.array_equal(results[0][0], results[1][0])
    assert np.array_equal(results[0][1], results[1][1])
    assert results[0][2:] == results[1][2:]


def test_peak_final_four_cell_pairing_and_seed_guard():
    cells = {(source, policy) for source in ("Peak", "Final") for policy in ("Peak", "Final")}
    assert cells == {("Peak", "Peak"), ("Peak", "Final"), ("Final", "Peak"), ("Final", "Final")}
    assert_safe_diagnostic_seeds([88_330_000, 88_430_000])
    try:
        assert_safe_diagnostic_seeds([FORBIDDEN_SEED_MIN])
    except ValueError:
        pass
    else:
        raise AssertionError("45M seed guard did not fail")


def test_analysis_tools_never_call_backward_or_optimizer_step():
    for name in ("audit_fixed10_wave_state_drift.py", "collect_fixed10_natural_wave_entries.py", "replay_fixed10_cross_entries.py"):
        source = (ROOT / "tools" / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        assert not any(node.func.attr == "backward" for node in calls)
        assert "optimizer.step" not in source
