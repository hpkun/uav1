from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tools.preflight_mappo_horizon_continuation import (
    EVALUATION_EPISODES,
    EVALUATION_SEED_BASE,
    EVALUATION_SEED_END,
    SOURCE,
    SOURCE_SHA256,
    SOURCE_STEP,
    TARGET_STEP,
    branch_episode_indices,
    changed_paths,
    derive_algorithm,
    derive_environment,
    ensure_output_absent,
    load_source,
    sha256,
    source_episode_indices,
    validate_algorithm_pair,
    validate_environment_pair,
    validate_source_sha,
    validate_source_state,
)


def source_artifacts():
    return load_source("cpu")


def test_wrong_source_sha_is_rejected():
    assert sha256(SOURCE) == SOURCE_SHA256
    with pytest.raises(RuntimeError, match="SHA mismatch"):
        validate_source_sha("0" * 64)


def test_wrong_source_step_is_rejected():
    env, algorithm, state, _ = source_artifacts()
    bad = deepcopy(state)
    bad["sampled_steps"] = SOURCE_STEP + 24
    with pytest.raises(RuntimeError, match="source checkpoint identity mismatch"):
        validate_source_state(bad, env, algorithm)


def test_non_mlp_source_is_rejected():
    env, algorithm, state, _ = source_artifacts()
    bad = deepcopy(state)
    bad["critic_type"] = "attention"
    with pytest.raises(RuntimeError, match="critic_type mismatch"):
        validate_source_state(bad, env, algorithm)


def test_h1000_changes_only_max_steps():
    env, _, _, _ = source_artifacts()
    h3000 = derive_environment(env, 3000)
    h1000 = derive_environment(env, 1000)
    validate_environment_pair(env, h3000, h1000)
    assert changed_paths(env, h1000) == ["simulation.max_steps"]


def test_h3000_environment_is_source_identical():
    env, _, _, _ = source_artifacts()
    assert derive_environment(env, 3000) == env


def test_training_protocol_is_unchanged():
    _, source_algorithm, _, _ = source_artifacts()
    branch = derive_algorithm(source_algorithm)
    validate_algorithm_pair(source_algorithm, branch)
    assert branch["training"] == source_algorithm["training"]
    assert changed_paths(source_algorithm, branch) == [
        "implementation.evaluation_seed_base"
    ]


def test_evaluation_bank_is_identical_for_both_branches():
    _, source_algorithm, _, _ = source_artifacts()
    control = derive_algorithm(source_algorithm)
    treatment = derive_algorithm(source_algorithm)
    assert control == treatment
    assert control["implementation"]["evaluation_seed_base"] == EVALUATION_SEED_BASE
    assert control["training"]["evaluation_episodes"] == EVALUATION_EPISODES
    assert EVALUATION_SEED_END == 49_000_049


def test_existing_output_directory_is_rejected(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        ensure_output_absent(output)


def test_source_episode_indices_restore_and_increment():
    _, _, state, _ = source_artifacts()
    previous = source_episode_indices(state)
    restored = branch_episode_indices(state)
    assert previous.shape == (24,)
    assert torch.equal(torch.as_tensor(restored), torch.as_tensor(previous + 1))


def test_target_is_exact_1p5m():
    assert SOURCE_STEP == 1_105_920
    assert TARGET_STEP == 1_500_000
    assert TARGET_STEP - SOURCE_STEP == 394_080
