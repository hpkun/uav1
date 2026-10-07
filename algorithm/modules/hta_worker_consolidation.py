"""HTA-MAPPO V2 progressive two-timescale Worker consolidation."""
from __future__ import annotations

from typing import Any

from .base import CapabilityModule


HTA_WORKER_CONSOLIDATION_VERSION = 1


class HTAWorkerConsolidationModule(CapabilityModule):
    """Stateless, fixed Worker learning-rate multiplier for HTA-MAPPO V2."""

    name = "hta_worker_consolidation"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.version = HTA_WORKER_CONSOLIDATION_VERSION
        self.schedule = str(self.config.get("schedule", "progressive_linear"))
        self.start_step = int(self.config.get("start_step", 1_500_000))
        self.end_step = int(self.config.get("end_step", 2_500_000))
        self.final_lr_multiplier = float(self.config.get("final_lr_multiplier", 0.25))
        if self.enabled and self.schedule != "progressive_linear":
            raise ValueError("hta_worker_consolidation requires schedule=progressive_linear")
        if self.enabled and self.start_step != 1_500_000:
            raise ValueError("hta_worker_consolidation requires start_step=1500000")
        if self.enabled and self.end_step != 2_500_000:
            raise ValueError("hta_worker_consolidation requires end_step=2500000")
        if self.enabled and self.final_lr_multiplier != 0.25:
            raise ValueError("hta_worker_consolidation requires final_lr_multiplier=0.25")

    def multiplier(self, sampled_steps: int) -> float:
        if not self.enabled:
            return 1.0
        step = int(sampled_steps)
        if step <= self.start_step:
            return 1.0
        if step >= self.end_step:
            return self.final_lr_multiplier
        progress = (step - self.start_step) / (self.end_step - self.start_step)
        return 1.0 - (1.0 - self.final_lr_multiplier) * progress

    def apply(self, optimizer: Any, sampled_steps: int,
              baseline_learning_rate: float) -> dict[str, float]:
        multiplier = self.multiplier(sampled_steps)
        effective_lr = float(baseline_learning_rate) * multiplier
        if self.enabled:
            for group in optimizer.param_groups:
                group["lr"] = effective_lr
        return {"effective_lr": effective_lr, "multiplier": multiplier}


__all__ = ["HTA_WORKER_CONSOLIDATION_VERSION", "HTAWorkerConsolidationModule"]
