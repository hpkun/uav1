"""49D research observation: original 45 fields unchanged, four alphas appended."""
import numpy as np
from .v338_observation import build_team_observations_v338
from .geometry import engagement_geometry

OBSERVATION_DIM = 49


def build_team_observations_v343(team, opponents, config):
    result = np.zeros((4,OBSERVATION_DIM),dtype=np.float64)
    result[:,:45] = build_team_observations_v338(team,opponents,config)
    # Append by enemy slot instead of interleaving, preserving every old index.
    for i, own in enumerate(team):
        for j, enemy in enumerate(opponents):
            if own.alive and enemy.alive:
                result[i,45+j] = engagement_geometry(own,enemy).off_boresight/np.pi
    if not np.isfinite(result).all():
        raise FloatingPointError('invalid v3.43 observation')
    return result
