"""Stateless, sum-preserving alive-agent team-mean training credit."""
from __future__ import annotations

from typing import Any
import torch

from .base import CapabilityModule


TEAM_MEAN_CREDIT_VERSION = 1


class TeamMeanCreditModule(CapabilityModule):
    """Redistribute each transition's live reward sum equally over live agents."""

    name = "team_mean_credit"
    version = TEAM_MEAN_CREDIT_VERSION
    MODE = "alive_sum_preserving_mean"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.mode = str(self.config.get("mode", self.MODE))
        if self.mode != self.MODE:
            raise ValueError(f"team_mean_credit requires mode={self.MODE}")
        unknown = set(self.config) - {"enabled", "mode"}
        if unknown:
            raise ValueError(f"team_mean_credit has unsupported configuration keys: {sorted(unknown)}")

    @staticmethod
    def disabled_metrics() -> dict[str, float | str]:
        return {
            "team_credit_enabled": 0.0,
            "team_credit_version": float(TEAM_MEAN_CREDIT_VERSION),
            "team_credit_live_sample_count": 0.0,
            "team_credit_original_live_reward_sum": 0.0,
            "team_credit_transformed_live_reward_sum": 0.0,
            "team_credit_sum_abs_error_max": 0.0,
            "team_credit_mean_abs_redistribution": 0.0,
            "team_credit_max_abs_redistribution": 0.0,
            "team_credit_nonzero_redistribution_fraction": 0.0,
            "team_credit_wave1_mean_abs_redistribution": 0.0,
            "team_credit_wave2_mean_abs_redistribution": 0.0,
            "team_credit_wave3_mean_abs_redistribution": 0.0,
        }

    def transform(self, rewards: torch.Tensor, alive_masks: torch.Tensor,
                  wave_indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | str]]:
        if not self.enabled:
            return rewards, self.disabled_metrics()
        if rewards.shape != alive_masks.shape or rewards.ndim != 3:
            raise ValueError("team_mean_credit rewards/alive_masks must match [T,E,N]")
        if wave_indices.shape != rewards.shape[:2]:
            raise ValueError("team_mean_credit wave_indices must be [T,E]")
        if not (torch.isfinite(rewards).all() and torch.isfinite(alive_masks).all()):
            raise FloatingPointError("team_mean_credit received non-finite inputs")
        alive = (alive_masks > .5).to(dtype=rewards.dtype)
        live_total = (rewards * alive).sum(dim=-1, keepdim=True)
        alive_count = alive.sum(dim=-1, keepdim=True)
        transformed = alive * live_total / alive_count.clamp_min(1.0)
        transformed_total = transformed.sum(dim=-1, keepdim=True)
        error = (transformed_total - live_total).abs()
        scale = max(1.0, float(live_total.abs().max().detach().cpu()))
        tolerance = max(1e-6, 8.0 * torch.finfo(rewards.dtype).eps * scale) if rewards.dtype.is_floating_point else 1e-6
        if float(error.max().detach().cpu()) > tolerance:
            raise FloatingPointError("team_mean_credit failed live reward-sum preservation")
        delta = (transformed - rewards * alive).abs(); live = alive > .5
        live_delta = delta[live]
        metrics: dict[str, float | str] = {
            "team_credit_enabled": 1.0,
            "team_credit_version": float(self.version),
            "team_credit_mode": self.mode,
            "team_credit_live_sample_count": float(live.sum().detach().cpu()),
            "team_credit_original_live_reward_sum": float(live_total.sum().detach().cpu()),
            "team_credit_transformed_live_reward_sum": float(transformed_total.sum().detach().cpu()),
            "team_credit_sum_abs_error_max": float(error.max().detach().cpu()),
            "team_credit_mean_abs_redistribution": float(live_delta.mean().detach().cpu()) if live_delta.numel() else 0.0,
            "team_credit_max_abs_redistribution": float(live_delta.max().detach().cpu()) if live_delta.numel() else 0.0,
            "team_credit_nonzero_redistribution_fraction": float((live_delta > 0).float().mean().detach().cpu()) if live_delta.numel() else 0.0,
        }
        for wave in (1, 2, 3):
            mask = live & (wave_indices == wave).unsqueeze(-1)
            metrics[f"team_credit_wave{wave}_mean_abs_redistribution"] = (
                float(delta[mask].mean().detach().cpu()) if bool(mask.any()) else 0.0
            )
        return transformed, metrics


__all__ = ["TEAM_MEAN_CREDIT_VERSION", "TeamMeanCreditModule"]
