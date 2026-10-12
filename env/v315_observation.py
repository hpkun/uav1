"""Unchanged 56D geometry layout, with the reserved self slot4 fixed at zero."""
from .v313_observation import OBSERVATION_DIM, build_team_observations_v313


def build_team_observations_v315(team, opponents, config):
    return build_team_observations_v313(team, opponents, config)


__all__ = ['OBSERVATION_DIM', 'build_team_observations_v315']
