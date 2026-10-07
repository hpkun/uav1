"""Li et al. Eq. (25) R3/R4 state rewards."""
from __future__ import annotations

import numpy as np

from .models import AircraftState
from .geometry import EngagementGeometry, engagement_geometry


def _tier_value(
    geometry: EngagementGeometry, tiers: list[float], values: list[float]
) -> float:
    for limit, value in zip(tiers, values):
        if abs(geometry.ata) <= limit and abs(geometry.ha) <= limit:
            return float(value)
    return 0.0


def paper_state_reward_components(
    red: list[AircraftState], blue: list[AircraftState], config: dict
) -> dict[str, np.ndarray]:
    """Compute post-transition/pre-hit R3 and R4 for each living Red UAV."""
    r3 = np.zeros(len(red), dtype=np.float32)
    r4 = np.zeros(len(red), dtype=np.float32)
    outer = float(config["angle_tiers"][-1])
    tiers = list(map(float, config["angle_tiers"]))
    advantage = list(map(float, config["advantage_rewards"]))
    threat = list(map(float, config["threat_penalties"]))
    for index, own in enumerate(red):
        if not own.alive:
            continue
        candidates = [
            (engagement_geometry(own, target).distance, target_index)
            for target_index, target in enumerate(blue) if target.alive
        ]
        if not candidates:
            continue
        _, target_index = min(candidates)
        target = blue[target_index]
        forward = engagement_geometry(own, target)
        reverse = engagement_geometry(target, own)
        if (
            abs(forward.ata) <= outer
            and abs(forward.ha) <= outer
            and forward.distance >= 4000.0
        ):
            r3[index] = float(config["approach_reward"])
        if abs(forward.aa) <= outer and forward.distance <= 4000.0:
            r4[index] = _tier_value(forward, tiers, advantage)
        elif abs(reverse.aa) <= outer and reverse.distance <= 4000.0:
            r4[index] = _tier_value(reverse, tiers, threat)
    return {"r3": r3, "r4": r4}


__all__ = ["paper_state_reward_components"]
