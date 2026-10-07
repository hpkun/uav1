from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import pytest
import torch
import yaml

from algorithm.common.protocol import config_sha256
from algorithm.madsac.protocol import validate_madsac_checkpoint, validate_madsac_config
from algorithm.madsac.runner import FUTURE_FINAL_45M
from algorithm.madsac.trainer import MADSACTrainer
from algorithm.train_madsac import build_parser


ROOT = Path(__file__).resolve().parents[1]


def configs():
    env = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    algorithm = yaml.safe_load((ROOT / "configs/madsac_persistent_wave_v2_3m.yaml").read_text(encoding="utf-8"))
    return env, algorithm


def test_formal_config_and_paper_vs_project_metadata():
    env, algorithm = configs()
    validate_madsac_config(env, algorithm)
    assert env["environment_variant"] == "persistent_wave_v2"
    assert env["persistent_waves"]["total_waves"] == 3
    assert env["simulation"]["max_steps"] == 3000
    assert algorithm["network"] == {
        "observation_dim": 52, "action_dim": 3, "num_agents": 4,
        "actor_hidden_layers": [256, 256], "critic_hidden_layers": [256, 256],
        "attention_heads": 2,
    }
    training = algorithm["training"]
    assert training["num_train_envs"] == 24
    assert training["total_sampled_steps"] == 3_000_000
    assert training["evaluation_episodes"] == 50
    assert training["evaluation_interval_sampled_steps"] == 100_000
    assert algorithm["implementation"]["checkpoint_interval_sampled_steps"] == 300_000
    assert algorithm["metadata"]["project_implementation_choice"]["policy_delay"] == 2


def test_development_and_future_final_seed_protocol_is_static_only():
    _env, algorithm = configs()
    implementation = algorithm["implementation"]
    assert implementation["evaluation_seed_base"] == 44_000_000
    assert implementation["evaluation_policy_seed"] == 770001
    assert implementation["evaluation_mode"] == "stochastic"
    assert FUTURE_FINAL_45M == tuple(range(45_000_000, 45_000_200))
    assert not set(range(44_000_000, 44_000_050)) & set(FUTURE_FINAL_45M)


def test_cli_has_no_resume_and_runtime_overrides_are_explicit():
    parser = build_parser()
    options = {action.dest for action in parser._actions}
    assert "resume" not in options
    assert {"device", "seed", "num_envs", "total_sampled_steps", "smoke"} <= options


def test_protocol_rejects_task_and_evaluation_drift():
    env, algorithm = configs()
    bad_env = deepcopy(env)
    bad_env["persistent_waves"]["total_waves"] = 2
    with pytest.raises(RuntimeError, match="three-wave"):
        validate_madsac_config(bad_env, algorithm)
    bad_algorithm = deepcopy(algorithm)
    bad_algorithm["implementation"]["evaluation_mode"] = "deterministic"
    with pytest.raises(RuntimeError, match="primary evaluation"):
        validate_madsac_config(env, bad_algorithm)


def test_checkpoint_provenance_validation_and_runtime_seed_override(tmp_path):
    env, algorithm = configs()
    trainer = MADSACTrainer(hidden_dim=16, seed=5302)
    extra = {
        "training_seed": 5302,
        "environment_config_sha256": config_sha256(env),
        "algorithm_config_sha256": config_sha256(algorithm),
        "observation_dim": 52, "action_dim": 3, "num_agents": 4,
    }
    state = trainer.checkpoint_state(extra)
    validate_madsac_checkpoint(state, env, algorithm, expected_training_seed=5302)
    with pytest.raises(RuntimeError, match="training_seed"):
        validate_madsac_checkpoint(state, env, algorithm, expected_training_seed=5303)
    state["extra"]["algorithm_config_sha256"] = "bad"
    with pytest.raises(RuntimeError, match="algorithm_config_sha256"):
        validate_madsac_checkpoint(state, env, algorithm, expected_training_seed=5302)


def test_checkpoint_records_no_false_exact_resume_claim():
    state = MADSACTrainer(hidden_dim=16).checkpoint_state()
    assert state["formal_exact_resume_supported"] is False
    assert state["replay_buffer_included"] is False
    for key in (
        "actor", "target_actor", "critic1", "critic2", "target_critic1", "target_critic2",
        "actor_optimizer", "critic1_optimizer", "critic2_optimizer", "policy_generator_state",
    ):
        assert key in state

