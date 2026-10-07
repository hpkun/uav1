"""RV-MAPPO V1 fixed protocol definition (no parameters and no RNG)."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

REFERENCE_VARIANCE_VERSION = 1
EXPECTED_REFERENCE_VARIANCE_CONFIG = {
    "enabled": True,
    "mode": "frozen_source_state_dependent_variance",
    "source_sampled_steps": 1_505_280,
    "freeze_current_log_std_head": True,
    "behavior_uses_reference_mean": False,
}


class ReferenceVarianceModule:
    name = "reference_variance"
    version = REFERENCE_VARIANCE_VERSION

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = deepcopy(config or {"enabled": False})
        self.enabled = bool(self.config.get("enabled", False))
        if self.enabled and self.config != EXPECTED_REFERENCE_VARIANCE_CONFIG:
            raise ValueError("RV-MAPPO V1 requires the exact frozen-source variance protocol")

    def state_dict(self) -> dict[str, Any]:
        return {"version": self.version, "config": deepcopy(self.config)}

    def load_state_dict(self, state: dict[str, Any] | None, *, branch_from_plain: bool = False) -> None:
        if state is None:
            if branch_from_plain:
                return
            raise RuntimeError("RV checkpoint reference_variance_state missing")
        if int(state.get("version", -1)) != self.version or state.get("config") != self.config:
            raise RuntimeError("RV checkpoint version/config mismatch")


__all__ = ["REFERENCE_VARIANCE_VERSION", "EXPECTED_REFERENCE_VARIANCE_CONFIG",
           "ReferenceVarianceModule"]
