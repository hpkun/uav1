from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.train_modular_mappo import load_config
from env.factory import make_combat_environment
from env.weapon import FireState


ROOT = Path(__file__).resolve().parents[1]
LEGACY_PATH = ROOT / "configs/persistent_wave_v2_environment.yaml"
READY_PATH = ROOT / "configs/persistent_wave_v2_fire_ready_environment.yaml"


def configs():
    legacy = yaml.safe_load(LEGACY_PATH.read_text(encoding="utf-8"))
    ready = yaml.safe_load(READY_PATH.read_text(encoding="utf-8"))
    return legacy, ready


def test_legacy_is_exact_52_and_fire_ready_appends_exactly_one_feature():
    legacy_cfg, ready_cfg = configs()
    legacy = make_combat_environment(legacy_cfg)
    ready = make_combat_environment(ready_cfg)
    legacy_obs, _ = legacy.reset(7123)
    ready_obs, _ = ready.reset(7123)
    assert legacy.observation_dim == 52 and legacy_obs.shape == (4, 52)
    assert ready.observation_dim == 53 and ready_obs.shape == (4, 53)
    np.testing.assert_array_equal(ready_obs[:, :52], legacy_obs)
    np.testing.assert_array_equal(ready_obs[:, 52], np.ones(4, np.float32))


def test_ready_bit_true_false_alias_and_dead_agent_zero_contract():
    legacy_cfg, ready_cfg = configs()
    legacy = make_combat_environment(legacy_cfg)
    ready = make_combat_environment(ready_cfg)
    legacy.reset(81)
    ready.reset(81)
    before_legacy = legacy._observations().copy()
    before_ready = ready._observations().copy()
    legacy.red_fire_states[0].armed = False
    ready.red_fire_states[0].armed = False
    after_legacy = legacy._observations()
    after_ready = ready._observations()
    np.testing.assert_array_equal(before_legacy, after_legacy)
    np.testing.assert_array_equal(before_ready[:, :52], after_ready[:, :52])
    assert before_ready[0, 52] == 1.0 and after_ready[0, 52] == 0.0
    np.testing.assert_array_equal(before_ready[1:], after_ready[1:])
    ready.red[2].alive = False
    np.testing.assert_array_equal(ready._observations()[2], np.zeros(53, np.float32))


def test_attempt_rearm_and_intermediate_spawn_follow_real_fire_state():
    _, cfg = configs()
    env = make_combat_environment(cfg)
    env.reset(101)
    # Exercise the same state transition used by step(): an armed attacker in a
    # fire window consumes readiness; leaving all windows rearms it.
    for index in range(1, 4):
        env.red[index].alive = False
        env.blue[index].alive = False
    env.red[0].x = env.red[0].y = 0.0
    env.red[0].z = -3000.0
    env.red[0].psi = 0.0
    env.blue[0].x, env.blue[0].y, env.blue[0].z = 1000.0, 0.0, -3000.0
    env.blue[0].psi = np.pi
    class AlwaysMiss:
        @staticmethod
        def normal():
            return 100.0
    env.rng = AlwaysMiss()
    observation, _, terminated, truncated, info = env.step(
        np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32)
    )
    assert not terminated and not truncated and info["red_step_fire_attempts"] == 1
    assert not env.red_fire_states[0].armed and observation[0, 52] == 0.0
    env.blue[0].x = 4500.0
    observation, _, _, _, info = env.step(
        np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32)
    )
    assert info["red_step_fire_attempts"] == 0
    assert env.red_fire_states[0].armed and observation[0, 52] == 1.0

    # Intermediate wave replacement resets the surviving Red FireState.
    env = make_combat_environment(cfg)
    env.reset(102)
    env.red_fire_states = [FireState(armed=False) for _ in range(4)]
    for blue in env.blue:
        blue.alive = False
    observation, _, terminated, truncated, info = env.step(np.zeros((4, 3), np.float32))
    assert not terminated and not truncated and info["spawned_next_wave"]
    assert all(state.armed for state in env.red_fire_states)
    np.testing.assert_array_equal(observation[:, 52], env.red_alive_mask)


@pytest.mark.parametrize("ready,expected", [(False, 52), (True, 53)])
def test_parallel_vector_worker_reports_and_returns_dynamic_dimension(ready, expected):
    legacy_cfg, ready_cfg = configs()
    cfg = ready_cfg if ready else legacy_cfg
    with ParallelVectorEnv(1, cfg, base_seed=89_000_000) as vector:
        observation = vector.reset()
        assert vector.observation_dim == expected
        assert vector.worker_metadata[0]["observation_dim"] == expected
        assert observation.shape == (1, 4, expected)
        result = vector.step_batch(np.zeros((1, 4, 3), np.float32), auto_reset=False)
        assert result.transition_next_observations.shape == (1, 4, expected)


def test_fire_ready_plain_actor_and_critic_use_53d_base_and_keep_plain_protocol():
    config = load_config(ROOT / "configs/dev_fire_ready_plain_3m.yaml")
    trainer = build_modular_mappo_trainer(config, "cpu", 256, 3_000_000)
    enabled = sorted(name for name, value in config["modules"].items() if value.get("enabled"))
    assert enabled == ["actor_lr_decay"]
    assert trainer.actor.backbone[0].in_features == 53
    assert trainer.critic.embedding[0].in_features == 53
    assert trainer.actor.mean.out_features == 3
    assert not trainer.actor.entity_attention_enabled


def test_environment_configs_differ_only_by_opt_in_ready_flag():
    legacy, ready = configs()
    normalized = deepcopy(ready)
    assert normalized["observation"].pop("include_own_fire_ready") is True
    assert normalized == legacy
