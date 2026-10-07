"""PopArt value normalization with output-preserving affine rescaling."""
from __future__ import annotations

from typing import Any
import torch
from torch import nn

from .base import CapabilityModule


class PopArtValueNormalizer(nn.Module, CapabilityModule):
    name = "popart"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        nn.Module.__init__(self); CapabilityModule.__init__(self, config)
        self.beta = float(self.config.get("beta", 0.999))
        self.epsilon = float(self.config.get("epsilon", 1e-5))
        self.register_buffer("mean", torch.tensor(0.0, dtype=torch.float64))
        self.register_buffer("variance", torch.tensor(1.0, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))

    @property
    def std(self) -> torch.Tensor:
        return self.variance.clamp_min(self.epsilon ** 2).sqrt()

    def normalize_targets(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values)) / self.std.to(values)

    def denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.std.to(values) + self.mean.to(values)

    @torch.no_grad()
    def update(self, targets: torch.Tensor, output_layer: nn.Linear | None = None) -> dict[str, float]:
        finite = targets.detach().double().reshape(-1)
        finite = finite[torch.isfinite(finite)]
        if finite.numel() == 0: raise ValueError("PopArt update requires finite targets")
        old_mean, old_std = self.mean.clone(), self.std.clone()
        batch_mean = finite.mean(); batch_var = finite.var(unbiased=False)
        if self.count.item() == 0:
            new_mean, new_var = batch_mean, batch_var.clamp_min(self.epsilon ** 2)
        else:
            new_mean = self.beta * self.mean + (1.0 - self.beta) * batch_mean
            second = self.beta * (self.variance + self.mean.square()) + (1.0 - self.beta) * (batch_var + batch_mean.square())
            new_var = (second - new_mean.square()).clamp_min(self.epsilon ** 2)
        self.mean.copy_(new_mean); self.variance.copy_(new_var); self.count.add_(finite.numel())
        if output_layer is not None:
            new_std = self.std
            scale = (old_std / new_std).to(output_layer.weight)
            output_layer.weight.mul_(scale)
            output_layer.bias.copy_(((old_std * output_layer.bias.double()) + old_mean - new_mean).div(new_std).to(output_layer.bias))
        return {"popart_mean": float(self.mean), "popart_std": float(self.std), "popart_count": float(self.count)}

    def metadata(self) -> dict[str, Any]:
        return {**CapabilityModule.metadata(self), "mean": float(self.mean), "variance": float(self.variance), "std": float(self.std), "count": float(self.count)}


__all__ = ["PopArtValueNormalizer"]
