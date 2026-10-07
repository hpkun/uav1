"""Paper Eq. (7)-(8) entry-triggered probabilistic attack model."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .geometry import EngagementGeometry


@dataclass
class FireState:
    armed: bool = True


class PairFireState:
    """v2.4 pair-local entry state; only the selected eligible pair disarms."""
    def __init__(self, team_size: int):
        if team_size <= 0:
            raise ValueError("PairFireState team_size must be positive")
        self.armed = np.ones((team_size, team_size), dtype=bool)

    def reset(self):
        self.armed.fill(True)


@dataclass(frozen=True)
class WeaponEnvelope:
    range_min: float
    range_max: float
    off_boresight_angle_max: float
    effective_hit_distance: float
    attack_noise_scale: float
    height_noise_scale: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.range_min < self.range_max:
            raise ValueError("weapon range must satisfy 0 <= range_min < range_max")
        if min(
            self.off_boresight_angle_max,
            self.effective_hit_distance,
        ) <= 0.0:
            raise ValueError("weapon angle and hit-distance parameters must be positive")
        if min(self.attack_noise_scale, self.height_noise_scale) < 0.0:
            raise ValueError("weapon noise scales must be non-negative")

    def in_fire_window(self, geometry: EngagementGeometry) -> bool:
        return bool(
            self.range_min <= geometry.distance <= self.range_max
            and geometry.off_boresight <= self.off_boresight_angle_max
        )

    def hit_threshold(self, distance: float) -> float:
        return float(np.pi * np.exp(-distance / self.effective_hit_distance))

    def attempt_hit(
        self, geometry: EngagementGeometry, rng: np.random.Generator
    ) -> bool:
        threshold = self.hit_threshold(geometry.distance)
        azimuth_noise = float(rng.normal())
        elevation_noise = float(rng.normal())
        return bool(
            abs(geometry.boresight_azimuth_error
                + self.attack_noise_scale * azimuth_noise) <= threshold
            and abs(geometry.boresight_elevation_error
                    + self.height_noise_scale * elevation_noise) <= threshold
        )


@dataclass(frozen=True)
class RearAspectWeaponEnvelope(WeaponEnvelope):
    target_aspect_angle_max: float

    def __post_init__(self):
        super().__post_init__()
        if not 0 < self.target_aspect_angle_max <= np.pi:
            raise ValueError("target aspect angle must be in (0, pi]")

    def qualifies(self, geometry: EngagementGeometry, attacker_speed: float,
                  target_speed: float) -> bool:
        return bool(self.in_fire_window(geometry)
                    and geometry.target_aspect <= self.target_aspect_angle_max
                    and attacker_speed >= target_speed)


__all__ = ["FireState", "PairFireState", "WeaponEnvelope", "RearAspectWeaponEnvelope"]
