"""DAWE-MAPPO V1: fixed deployment-aligned wave exploration."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

DAWE_MAPPO_VERSION = 1
EXPECTED_DAWE_MULTIPLIERS = (0.25, 0.25, 1.0)


class DeploymentAlignedWaveExplorationModule:
    """Stateless fixed per-wave scaling of a diagonal Gaussian's stddev."""

    name = "deployment_aligned_wave_exploration"
    version = DAWE_MAPPO_VERSION

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = deepcopy(config or {"enabled": False})
        self.enabled = bool(self.config.get("enabled", False))
        self.mode = self.config.get("mode", "fixed_wave_std_multiplier")
        self.multipliers = tuple(float(self.config.get(f"wave{wave}_multiplier", value))
                                 for wave, value in zip((1, 2, 3), EXPECTED_DAWE_MULTIPLIERS))
        if any(value <= 0 for value in self.multipliers):
            raise ValueError("DAWE multipliers must be strictly positive")
        if self.enabled:
            expected = {
                "enabled": True, "mode": "fixed_wave_std_multiplier",
                "wave1_multiplier": 0.25, "wave2_multiplier": 0.25,
                "wave3_multiplier": 1.0,
            }
            if self.config != expected:
                raise ValueError("DAWE V1 requires exact fixed multipliers 0.25/0.25/1.0")

    def multiplier(self, wave_index: int) -> float:
        wave = int(wave_index)
        if wave not in (1, 2, 3):
            raise ValueError("DAWE wave index must be 1, 2, or 3")
        return self.multipliers[wave - 1]

    def multiplier_tensor(self, wave_indices: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        waves = torch.as_tensor(wave_indices, dtype=torch.long, device=reference.device).reshape(-1)
        if bool(((waves < 1) | (waves > 3)).any()):
            raise ValueError("DAWE wave indices must lie in 1..3")
        if reference.ndim < 2 or reference.shape[0] != waves.shape[0]:
            raise ValueError("DAWE wave indices must align with distribution batch dimension")
        table = reference.new_tensor(self.multipliers)
        return table[waves - 1].reshape(waves.shape[0], *([1] * (reference.ndim - 1)))

    def effective_distribution(self, base_distribution: torch.distributions.Normal,
                               wave_indices: torch.Tensor | None) -> torch.distributions.Normal:
        if not self.enabled:
            return base_distribution
        if wave_indices is None:
            raise ValueError("DAWE stochastic distribution requires explicit environment wave indices")
        multiplier = self.multiplier_tensor(wave_indices, base_distribution.scale)
        return torch.distributions.Normal(base_distribution.loc, base_distribution.scale * multiplier)

    def state_dict(self) -> dict[str, Any]:
        return {"version": self.version, "config": deepcopy(self.config)}

    def load_state_dict(self, state: dict[str, Any] | None, branch_from_plain: bool = False) -> None:
        if state is None:
            if branch_from_plain:
                return
            raise RuntimeError("DAWE checkpoint state missing")
        if int(state.get("version", -1)) != self.version or state.get("config") != self.config:
            raise RuntimeError("DAWE checkpoint version/config mismatch")


__all__ = ["DAWE_MAPPO_VERSION", "EXPECTED_DAWE_MULTIPLIERS",
           "DeploymentAlignedWaveExplorationModule"]
