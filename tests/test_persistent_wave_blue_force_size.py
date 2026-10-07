from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from env.config import load_config
from env.persistent_env import PersistentWaveCombatEnv

ROOT = Path(__file__).resolve().parents[1]
ZERO = np.zeros((4, 3), dtype=np.float32)


def config(name="persistent_wave_v2_environment.yaml"):
    return load_config(ROOT / "configs" / name)


def clear_active_blue(env):
    for state in env.blue:
        state.alive = False


def advance_wave(env):
    clear_active_blue(env)
    return env.step(ZERO)


def test_missing_blue_units_defaults_to_444():
    env = PersistentWaveCombatEnv(config())
    assert env.blue_units_per_wave == [4, 4, 4]


def test_invalid_blue_units_length_is_rejected():
    cfg = config(); cfg["persistent_waves"]["blue_units_per_wave"] = [4, 4]
    with pytest.raises(ValueError, match="length"):
        PersistentWaveCombatEnv(cfg)


@pytest.mark.parametrize("values", ([4, 4, 0], [4, 4, 5]))
def test_invalid_blue_unit_count_is_rejected(values):
    cfg = config(); cfg["persistent_waves"]["blue_units_per_wave"] = values
    with pytest.raises(ValueError, match="values"):
        PersistentWaveCombatEnv(cfg)


def test_first_wave_must_use_all_blue_slots():
    cfg = config(); cfg["persistent_waves"]["blue_units_per_wave"] = [3, 3, 3]
    with pytest.raises(ValueError, match="first persistent wave"):
        PersistentWaveCombatEnv(cfg)


def test_blue443_keeps_four_slots_and_activates_443():
    env = PersistentWaveCombatEnv(config("persistent_wave_v2_blue443_environment.yaml"))
    env.reset(7); assert len(env.blue) == 4 and int(env.blue_alive_mask.sum()) == 4
    advance_wave(env); assert len(env.blue) == 4 and int(env.blue_alive_mask.sum()) == 4
    _, _, _, _, info = advance_wave(env)
    assert len(env.blue) == 4 and int(env.blue_alive_mask.sum()) == 3
    assert info["blue_losses"] == 8


def test_blue433_keeps_four_slots_and_activates_433():
    env = PersistentWaveCombatEnv(config("persistent_wave_v2_blue433_environment.yaml"))
    env.reset(8); assert len(env.blue) == 4 and int(env.blue_alive_mask.sum()) == 4
    advance_wave(env); assert len(env.blue) == 4 and int(env.blue_alive_mask.sum()) == 3
    advance_wave(env); assert len(env.blue) == 4 and int(env.blue_alive_mask.sum()) == 3


def test_inactive_slot_is_dead_and_stays_dead_after_step():
    env = PersistentWaveCombatEnv(config("persistent_wave_v2_blue443_environment.yaml"))
    env.reset(9); advance_wave(env); advance_wave(env)
    inactive = np.flatnonzero(env.blue_alive_mask == 0)
    assert inactive.size == 1
    env.step(ZERO)
    assert not env.blue[int(inactive[0])].alive


def test_observation_remains_4_by_52_and_inactive_enemy_slot_is_zero():
    env = PersistentWaveCombatEnv(config("persistent_wave_v2_blue443_environment.yaml"))
    env.reset(10); advance_wave(env); observation, _, _, _, _ = advance_wave(env)
    inactive = int(np.flatnonzero(env.blue_alive_mask == 0)[0])
    assert observation.shape == (4, 52)
    enemy_start = 7 + 3 * 7
    assert np.array_equal(observation[:, enemy_start + 6 * inactive:enemy_start + 6 * (inactive + 1)],
                          np.zeros((4, 6), dtype=np.float32))


def test_blue443_loss_bookkeeping_is_8_then_9_then_11_and_record_starts_at_3():
    env = PersistentWaveCombatEnv(config("persistent_wave_v2_blue443_environment.yaml"))
    env.reset(11); advance_wave(env); _, _, _, _, entry = advance_wave(env)
    assert entry["blue_losses"] == 8
    next(state for state in env.blue if state.alive).alive = False
    _, _, _, _, partial = env.step(ZERO)
    assert partial["blue_losses"] == 9
    clear_active_blue(env)
    _, _, _, _, final = env.step(ZERO)
    assert final["blue_losses"] == 11
    wave3 = next(row for row in final["per_wave_metrics"] if row["wave_index"] == 3)
    assert wave3["blue_survivors_start"] == 3


def test_legacy_444_cumulative_losses_are_4_8_12():
    env = PersistentWaveCombatEnv(config()); env.reset(12)
    assert advance_wave(env)[4]["blue_losses"] == 4
    assert advance_wave(env)[4]["blue_losses"] == 8
    assert advance_wave(env)[4]["blue_losses"] == 12


def test_inactive_placeholder_creates_no_kill_or_r1_event():
    env = PersistentWaveCombatEnv(config("persistent_wave_v2_blue443_environment.yaml"))
    env.reset(13); advance_wave(env); advance_wave(env)
    # Separate both teams so no active aircraft can enter a weapon window.
    for index, red in enumerate(env.red): red.x, red.y = -4000.0, float(index * 50)
    for index, blue in enumerate(env.blue): blue.x, blue.y = 4000.0, float(index * 50)
    kills = env.combat_counts["red"]["attack_kills"]
    r1 = float(env.episode_reward_components["r1"].sum())
    _, _, _, _, info = env.step(ZERO)
    assert env.combat_counts["red"]["attack_kills"] == kills
    assert info["red_step_attack_kills"] == 0
    assert float(env.episode_reward_components["r1"].sum()) == pytest.approx(r1)


def test_legacy_and_explicit_444_spawn_are_bit_identical():
    legacy = config(); explicit = deepcopy(legacy)
    explicit["persistent_waves"]["blue_units_per_wave"] = [4, 4, 4]
    first, second = PersistentWaveCombatEnv(legacy), PersistentWaveCombatEnv(explicit)
    first.reset(1414); second.reset(1414)
    for _ in range(2):
        advance_wave(first); advance_wave(second)
        assert first.last_spawn_candidate_index == second.last_spawn_candidate_index
        assert first.rng.bit_generator.state == second.rng.bit_generator.state
        assert all(np.array_equal(a.as_array(), b.as_array()) and a.alive == b.alive
                   for a, b in zip(first.blue, second.blue))


def test_variant_configs_differ_from_legacy_only_by_force_size():
    base = config()
    for name, expected in (("persistent_wave_v2_blue443_environment.yaml", [4, 4, 3]),
                           ("persistent_wave_v2_blue433_environment.yaml", [4, 3, 3])):
        variant = config(name)
        actual = variant["persistent_waves"].pop("blue_units_per_wave")
        assert actual == expected and variant == base
