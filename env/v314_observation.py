"""Keep the authoritative 56D v3.13 layout; restore executed self control phi."""
import numpy as np
from .v313_observation import OBSERVATION_DIM, build_team_observations_v313


def build_team_observations_v314(team, opponents, config, last_executed_phi):
    phis = np.asarray(last_executed_phi, dtype=float)
    if phis.shape != (5,) or not np.isfinite(phis).all():
        raise ValueError('last_executed_phi must be finite shape (5,)')
    result = build_team_observations_v313(team, opponents, config)
    for i, own in enumerate(team):
        if own.alive:
            result[i, 4] = phis[i] / (np.pi / 2)
    if not np.isfinite(result).all():
        raise FloatingPointError('invalid v3.14 observation')
    return result
