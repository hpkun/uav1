"""Random-diameter 4v4 initialization for the V2.3 benchmark."""
from __future__ import annotations

import numpy as np

from .math_utils import wrap_angle
from .models import AircraftState


def random_combat_states(
    rng: np.random.Generator,
    center_radius: float,
    formation_offsets: list[float],
    altitude_center: float,
    altitude_perturbation_max: float,
    speed_center: float,
    speed_perturbation_max: float,
    heading_perturbation_max: float,
    team_size: int = 4,
) -> tuple[list[AircraftState], list[AircraftState], float]:
    """Place opposing formations at the ends of a random 8-km diameter."""
    if team_size != 4 or len(formation_offsets) != team_size:
        raise ValueError("this benchmark requires four aircraft per team")
    radial_angle = float(rng.uniform(-np.pi, np.pi))
    radial = np.array([np.cos(radial_angle), np.sin(radial_angle)])
    lateral = np.array([-radial[1], radial[0]])

    def team(side: float, nominal_heading: float) -> list[AircraftState]:
        center = side * float(center_radius) * radial
        states = []
        for offset in formation_offsets:
            position = center + float(offset) * lateral
            states.append(AircraftState(
                x=float(position[0]),
                y=float(position[1]),
                z=-float(altitude_center + rng.uniform(
                    -altitude_perturbation_max, altitude_perturbation_max
                )),
                v=float(speed_center + rng.uniform(
                    -speed_perturbation_max, speed_perturbation_max
                )),
                theta=0.0,
                psi=float(wrap_angle(nominal_heading + rng.uniform(
                    -heading_perturbation_max, heading_perturbation_max
                ))),
            ))
        return states

    return (
        team(-1.0, radial_angle),
        team(1.0, wrap_angle(radial_angle + np.pi)),
        radial_angle,
    )


__all__ = ["random_combat_states"]
