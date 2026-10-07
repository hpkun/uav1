"""Within-wave raw-GAE priority weights for the policy loss only."""
from __future__ import annotations

from typing import Any

import torch

from .base import CapabilityModule


ADVANTAGE_PRIORITY_VERSION = 1


def capped_mean_preserving(
    values: torch.Tensor,
    alive_mask: torch.Tensor,
    cap: float,
    iterations: int = 80,
) -> torch.Tensor:
    """Scale positive values to alive mean one while retaining a hard cap."""
    alive = alive_mask > 0.5
    if not torch.any(alive):
        return torch.zeros_like(values)
    if cap < 1.0:
        raise ValueError("a mean-one hard cap must be >= 1")
    source = values.clamp_min(0.0)
    live = source[alive]
    if not torch.any(live > 0):
        raise FloatingPointError("cannot normalize all-zero actor weights")

    def capped_mean(scale: torch.Tensor) -> torch.Tensor:
        return torch.minimum(source * scale, source.new_tensor(cap))[alive].mean()

    lo = source.new_tensor(0.0)
    hi = source.new_tensor(1.0)
    for _ in range(80):
        if float(capped_mean(hi)) >= 1.0:
            break
        hi = hi * 2.0
    else:
        raise FloatingPointError("unable to bracket mean-preserving actor scale")
    for _ in range(iterations):
        mid = (lo + hi) * 0.5
        if float(capped_mean(mid)) < 1.0:
            lo = mid
        else:
            hi = mid
    result = torch.minimum(source * hi, source.new_tensor(cap))
    return result * alive_mask


class AdvantagePriorityModule(CapabilityModule):
    name = "advantage_priority"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.source = str(self.config.get("source", "raw_gae"))
        self.scope = str(self.config.get("scope", "within_wave"))
        self.positive_only = bool(self.config.get("positive_only", True))
        self.alpha = float(self.config.get("alpha", 0.5))
        self.z_clip = float(self.config.get("z_clip", 2.0))
        self.actor_only = bool(self.config.get("actor_only", True))
        self.final_weight_cap = float(self.config.get("final_weight_cap", 4.0))
        self.epsilon = float(self.config.get("epsilon", 1e-8))
        self.mean_preserving = bool(self.config.get("mean_preserving", True))
        if self.source != "raw_gae":
            raise ValueError("advantage priority currently supports source=raw_gae only")
        if self.scope != "within_wave":
            raise ValueError("advantage priority currently supports scope=within_wave only")
        if not self.positive_only or not self.actor_only or not self.mean_preserving:
            raise ValueError("first-version advantage priority requires positive_only, actor_only and mean_preserving")
        if self.alpha < 0 or self.z_clip < 0 or self.epsilon <= 0:
            raise ValueError("invalid advantage-priority numeric configuration")
        if self.final_weight_cap < 1.0:
            raise ValueError("final_weight_cap must be >= 1")

    def compute_tensor(
        self,
        raw_advantages: torch.Tensor,
        wave_indices: torch.Tensor,
        alive_masks: torch.Tensor,
        base_actor_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Return priority, combined actor weight, and rollout diagnostics."""
        if raw_advantages.shape != alive_masks.shape:
            raise ValueError("raw advantages and alive masks must match")
        if wave_indices.shape != raw_advantages.shape[:-1]:
            raise ValueError("wave indices must match advantage prefix without agents")
        if base_actor_weights.shape == wave_indices.shape:
            base = base_actor_weights.unsqueeze(-1).expand_as(raw_advantages)
        elif base_actor_weights.shape == raw_advantages.shape:
            base = base_actor_weights
        else:
            raise ValueError("base actor weights have incompatible shape")
        if not self.enabled:
            return torch.ones_like(raw_advantages), base, {}

        alive = alive_masks > 0.5
        priority = torch.ones_like(raw_advantages)
        z_values = torch.zeros_like(raw_advantages)
        metrics: dict[str, float] = {}
        for wave in (1, 2, 3):
            selected = (wave_indices == wave).unsqueeze(-1) & alive
            values = raw_advantages[selected]
            if values.numel():
                mu = values.mean()
                sigma = values.std(unbiased=False)
                z = (values - mu) / (sigma + self.epsilon)
                positive = z.clamp_min(0.0)
                clipped = positive.clamp_max(self.z_clip)
                q = 1.0 + self.alpha * clipped
                p = q / q.mean().clamp_min(self.epsilon)
                priority[selected] = p
                z_values[selected] = z
                metrics.update({
                    f"priority_mean_wave_{wave}": float(p.mean()),
                    f"priority_max_wave_{wave}": float(p.max()),
                    f"priority_std_wave_{wave}": float(p.std(unbiased=False)),
                    f"highlight_fraction_wave_{wave}": float((z > 0).float().mean()),
                })
            else:
                metrics.update({
                    f"priority_mean_wave_{wave}": 0.0,
                    f"priority_max_wave_{wave}": 0.0,
                    f"priority_std_wave_{wave}": 0.0,
                    f"highlight_fraction_wave_{wave}": 0.0,
                })

        combined = capped_mean_preserving(
            base * priority, alive_masks, self.final_weight_cap
        )
        live_combined = combined[alive]
        metrics["combined_actor_weight_mean"] = float(live_combined.mean())
        metrics["combined_actor_weight_max"] = float(live_combined.max())
        for wave in (1, 2, 3):
            selected = (wave_indices == wave).unsqueeze(-1) & alive
            metrics[f"combined_actor_weight_wave_{wave}"] = (
                float(combined[selected].mean()) if torch.any(selected) else 0.0
            )
        return priority, combined, metrics


__all__ = [
    "ADVANTAGE_PRIORITY_VERSION", "AdvantagePriorityModule",
    "capped_mean_preserving",
]
