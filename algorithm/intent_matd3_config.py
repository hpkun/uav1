"""Isolated Intent-Prediction MATD3 configuration extending NormalizedMATD3Config."""
from dataclasses import dataclass, field
import math

from .matd3_config import NormalizedMATD3Config


@dataclass
class IntentMATD3Config(NormalizedMATD3Config):
    intent_embed_dim: int = 64
    prediction_horizons: list[int] = field(default_factory=lambda: [5, 10, 20])
    prediction_loss_weight: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if type(self.intent_embed_dim) is not int or self.intent_embed_dim < 1:
            raise ValueError('intent_embed_dim must be a positive integer')
        if isinstance(self.prediction_horizons, (list, tuple)):
            horizons = list(self.prediction_horizons)
        else:
            raise ValueError('prediction_horizons must be a non-empty list of positive integers')
        if not horizons or any(type(h) is not int or h < 1 for h in horizons):
            raise ValueError('prediction_horizons must contain positive integers')
        if len(set(horizons)) != len(horizons) or horizons != sorted(horizons):
            raise ValueError('prediction_horizons must be strictly increasing unique integers')
        self.prediction_horizons = horizons
        if (
            type(self.prediction_loss_weight) not in (float, int)
            or not math.isfinite(self.prediction_loss_weight)
            or self.prediction_loss_weight < 0
        ):
            raise ValueError('prediction_loss_weight must be a non-negative finite number')
