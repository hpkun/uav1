"""Signed air-combat geometry, including true 3-D weapon boresight error."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .math_utils import wrap_angle
from .models import AircraftState


@dataclass(frozen=True)
class EngagementGeometry:
    distance: float
    horizontal_distance: float
    line_of_sight: float
    ata: float
    aa: float
    ha: float
    hca: float
    off_boresight: float
    boresight_azimuth_error: float
    boresight_elevation_error: float

    @property
    def attack_angle(self) -> float:
        return self.off_boresight

    @property
    def target_aspect(self) -> float:
        return abs(self.aa)


def engagement_geometry(
    attacker: AircraftState, target: AircraftState
) -> EngagementGeometry:
    dx = float(target.x - attacker.x)
    dy = float(target.y - attacker.y)
    dz = float(target.z - attacker.z)
    horizontal = float(np.hypot(dx, dy))
    distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    line_of_sight = float(np.arctan2(dy, dx)) if horizontal > 0.0 else attacker.psi
    if distance > 0.0:
        los_unit = np.asarray([dx, dy, dz], dtype=float) / distance
        ct = float(np.cos(attacker.theta))
        forward_unit = np.asarray([
            ct * np.cos(attacker.psi),
            ct * np.sin(attacker.psi),
            -np.sin(attacker.theta),
        ], dtype=float)
        right_unit = np.asarray([
            -np.sin(attacker.psi), np.cos(attacker.psi), 0.0,
        ], dtype=float)
        up_unit = np.cross(right_unit, forward_unit)
        cosine = float(np.clip(np.dot(forward_unit, los_unit), -1.0, 1.0))
        off_boresight = float(np.arccos(cosine))
        los_f = float(np.dot(los_unit, forward_unit))
        los_r = float(np.dot(los_unit, right_unit))
        los_u = float(np.dot(los_unit, up_unit))
        boresight_azimuth_error = float(np.arctan2(los_r, los_f))
        boresight_elevation_error = float(
            np.arctan2(los_u, np.hypot(los_f, los_r))
        )
    else:
        # Preserve the existing zero-range boundary semantics: a coincident
        # target has no defined LOS direction and is treated as aligned.
        off_boresight = 0.0
        boresight_azimuth_error = 0.0
        boresight_elevation_error = 0.0
    return EngagementGeometry(
        distance=distance,
        horizontal_distance=horizontal,
        line_of_sight=line_of_sight,
        ata=float(wrap_angle(line_of_sight - attacker.psi)),
        aa=float(wrap_angle(target.psi - line_of_sight)),
        ha=float(np.arctan2(-dz, horizontal)),
        hca=float(wrap_angle(target.psi - attacker.psi)),
        off_boresight=off_boresight,
        boresight_azimuth_error=boresight_azimuth_error,
        boresight_elevation_error=boresight_elevation_error,
    )


__all__ = ["EngagementGeometry", "engagement_geometry"]
