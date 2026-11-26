"""Isolated Multi-Agent TD3 (MATD3) configuration extending MADDPGConfig."""
from dataclasses import dataclass
import math

from .maddpg_config import MADDPGConfig


def _check_nonnegative_finite(value, name):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a non-negative finite number')


@dataclass
class MATD3Config(MADDPGConfig):
    policy_delay: int = 2
    target_noise_std_mps2: float = 0.2
    target_noise_clip_mps2: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if type(self.policy_delay) is not int or self.policy_delay < 1:
            raise ValueError('policy_delay must be a positive integer')
        _check_nonnegative_finite(self.target_noise_std_mps2, 'target_noise_std_mps2')
        _check_nonnegative_finite(self.target_noise_clip_mps2, 'target_noise_clip_mps2')


@dataclass
class NormalizedMATD3Config(MATD3Config):
    exploration_std_speed: float = 0.3
    exploration_std_turn: float = 0.3
    target_noise_std_speed: float = 0.2
    target_noise_std_turn: float = 0.2
    target_noise_clip_speed: float = 0.5
    target_noise_clip_turn: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        for name in (
            'exploration_std_speed',
            'exploration_std_turn',
            'target_noise_std_speed',
            'target_noise_std_turn',
            'target_noise_clip_speed',
            'target_noise_clip_turn',
        ):
            _check_nonnegative_finite(getattr(self, name), name)
