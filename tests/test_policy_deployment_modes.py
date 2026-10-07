from pathlib import Path

import numpy as np
import pytest
import torch
from torch.distributions import Normal

from tools.diagnose_policy_deployment_modes import (
    NOT_FORMAL_EVIDENCE,
    ROLE,
    diagnostic_metadata,
    gauss_hermite_nodes,
    policy_noise_seed,
    prepare_output_directory,
    select_deployment_action,
    squashed_noise_action,
    squashed_normal_expectation,
    validate_checkpoint_filename,
    validate_development_seed_range,
)


def test_gauss_hermite_zero_sigma_matches_tanh_mean():
    mean = torch.tensor([-1.0, -0.2, 0.0, 0.4, 1.2], dtype=torch.float64)
    result = squashed_normal_expectation(mean, torch.zeros_like(mean), 32)
    assert torch.allclose(result, torch.tanh(mean), atol=1e-12, rtol=0)


def test_gauss_hermite_symmetric_zero_mean_is_zero():
    mean = torch.zeros(4, dtype=torch.float64)
    std = torch.tensor([0.05, 0.3, 0.8, 1.5], dtype=torch.float64)
    result = squashed_normal_expectation(mean, std, 32)
    assert torch.allclose(result, torch.zeros_like(result), atol=1e-14, rtol=0)


def test_gauss_hermite_matches_fixed_monte_carlo_reference():
    means = torch.tensor([-1.0, -0.2, 0.0, 0.4, 1.2], dtype=torch.float64)
    stds = torch.tensor([0.05, 0.3, 0.8, 1.5], dtype=torch.float64)
    grid_mean, grid_std = torch.meshgrid(means, stds, indexing="ij")
    expected = squashed_normal_expectation(grid_mean, grid_std, 32)
    generator = torch.Generator().manual_seed(20260906)
    epsilon = torch.randn(400_000, 1, 1, generator=generator, dtype=torch.float64)
    monte_carlo = torch.tanh(
        grid_mean.unsqueeze(0) + grid_std.unsqueeze(0) * epsilon
    ).mean(0)
    assert torch.allclose(expected, monte_carlo, atol=3.5e-3, rtol=0)


def test_gauss_hermite_16_32_64_converges():
    mean = torch.tensor([-1.0, -0.2, 0.0, 0.4, 1.2], dtype=torch.float64)
    std = torch.tensor([1.5, 0.8, 0.3, 1.5, 0.8], dtype=torch.float64)
    q16 = squashed_normal_expectation(mean, std, 16)
    q32 = squashed_normal_expectation(mean, std, 32)
    q64 = squashed_normal_expectation(mean, std, 64)
    assert torch.max(torch.abs(q32 - q64)) < torch.max(torch.abs(q16 - q32))
    assert torch.max(torch.abs(q32 - q64)) < 2e-4
    assert gauss_hermite_nodes(32) is gauss_hermite_nodes(32)


def test_action_modes_are_exact_and_expectation_is_rng_independent():
    mean = torch.tensor([[[-0.5, 0.0, 0.7]]])
    std = torch.tensor([[[0.2, 0.8, 1.1]]])
    distribution = Normal(mean, std)
    epsilon = torch.tensor([[[1.0, -0.25, 0.5]]])
    assert torch.equal(
        select_deployment_action(distribution, "mean"), torch.tanh(mean)
    )
    assert torch.equal(squashed_noise_action(mean, std, 0.0, epsilon), torch.tanh(mean))
    assert torch.equal(
        select_deployment_action(distribution, "noise_100", epsilon=epsilon),
        torch.tanh(mean + std * epsilon),
    )
    torch.manual_seed(1)
    first = select_deployment_action(distribution, "squashed_expectation", 32)
    torch.manual_seed(999)
    second = select_deployment_action(distribution, "squashed_expectation", 32)
    assert torch.equal(first, second)


@pytest.mark.parametrize("seed", [29_000_000, 29_000_019, 30_000_000, 30_000_199])
def test_formal_and_monitoring_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match="refuses"):
        validate_development_seed_range(seed, 1)


def test_31m_development_seeds_and_noise_pairing_are_allowed():
    assert validate_development_seed_range(31_000_000, 20) == list(range(31_000_000, 31_000_020))
    assert policy_noise_seed("noise_025", 31_000_007, 1) == 720_100_015
    assert policy_noise_seed("noise_100", 31_000_007, 1) == 720_300_015
    seeds = {
        policy_noise_seed(mode, environment_seed, replicate)
        for mode in ("noise_025", "noise_050", "noise_100")
        for environment_seed in range(31_000_000, 31_000_020)
        for replicate in (0, 1)
    }
    assert len(seeds) == 3 * 20 * 2


def test_nonempty_output_and_best_eval_are_rejected(tmp_path):
    output = tmp_path / "diagnostic"
    output.mkdir()
    (output / "existing.txt").write_text("do not overwrite")
    with pytest.raises(FileExistsError, match="non-empty"):
        prepare_output_directory(output)
    with pytest.raises(ValueError, match="best_eval"):
        validate_checkpoint_filename(Path("best_eval.pt"))


def test_metadata_is_explicitly_nonformal(tmp_path):
    env_path = tmp_path / "env.yaml"
    env_path.write_text("environment_variant: persistent_wave_v2\n")
    metadata = diagnostic_metadata(
        [], env_path, {"environment_variant": "persistent_wave_v2"},
        [31_000_000], 1, 32, "test cuda",
    )
    assert metadata["role"] == ROLE == "development_mechanism_diagnostic_only"
    assert metadata["not_formal_evidence"] is NOT_FORMAL_EVIDENCE is True
    assert metadata["paired_unit"] == "environment_seed"
    assert metadata["stochastic_repeats_are_not_independent_scenarios"] is True
