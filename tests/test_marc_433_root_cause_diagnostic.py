from __future__ import annotations

import numpy as np
import torch

from tools.analyze_marc_433_root_cause import distribution_drift, overlap, qmetrics
from tools.diagnose_marc_deployment_policy import episode_policy_seed


def test_qmetrics_preserves_conditional_probability_semantics() -> None:
    row = {"clear_wave_1_probability": 0.5, "clear_wave_2_probability": 0.25,
           "clear_wave_3_probability": 0.0}
    assert qmetrics(row) == {"Q2": 0.5, "Q3": 0.0}
    row["clear_wave_1_probability"] = 0.0
    row["clear_wave_2_probability"] = 0.0
    assert qmetrics(row) == {"Q2": None, "Q3": None}


def test_bank_overlap_distinguishes_full_rows_and_observations() -> None:
    # Reference-distribution changes must not erase observation-only overlap.
    obs = np.arange(52, dtype=np.float32)
    best = [{"observation": obs, "reference_mean": np.ones(3),
             "reference_log_std": np.zeros(3)}]
    final = [{"observation": obs.copy(), "reference_mean": np.full(3, 8.0),
              "reference_log_std": np.full(3, 9.0)}]
    assert overlap(best, final, observation_only=False)["shared_rows"] == 0
    assert overlap(best, final, observation_only=True)["shared_rows"] == 1


def test_gaussian_kl_decomposition_is_exact_and_additive() -> None:
    ref_mean = np.zeros((2, 3), dtype=np.float32)
    ref_log_std = np.zeros((2, 3), dtype=np.float32)
    cur_mean = np.ones((2, 3), dtype=np.float32)
    cur_log_std = np.full((2, 3), np.log(2.0), dtype=np.float32)
    result = distribution_drift(ref_mean, ref_log_std, cur_mean, cur_log_std)
    # KL(N(0,1)||N(1,4)): mean term=1/8 and variance term=ln2+1/8-1/2.
    expected_mean = 3.0 / 8.0
    expected_variance = 3.0 * (np.log(2.0) + 0.125 - 0.5)
    assert np.isclose(result["gaussian_kl_mean_shift_contribution"], expected_mean)
    assert np.isclose(result["gaussian_kl_variance_contribution"], expected_variance)
    assert np.isclose(result["gaussian_kl_total"], expected_mean + expected_variance)


def test_episode_policy_seed_is_stable_and_scenario_specific() -> None:
    assert episode_policy_seed(91_000_000, 49_100_017, 2) == episode_policy_seed(
        91_000_000, 49_100_017, 2
    )
    assert episode_policy_seed(91_000_000, 49_100_017, 2) != episode_policy_seed(
        91_000_000, 49_100_018, 2
    )
    assert episode_policy_seed(91_000_000, 49_100_017, 2) != episode_policy_seed(
        91_000_000, 49_100_017, 1
    )
