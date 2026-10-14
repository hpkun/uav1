"""Eq.(25)-style hard distance gates; retained body geometry and aggregation."""
import numpy as np
from .v323_reward import body_pair_geometry, body_levels
from .v36_geometry import angle_within


def tactical_rewards_v333(red, blue, reward, weapon):
    geometry = body_pair_geometry(red, blue)
    reverse = body_pair_geometry(blue, red)
    live = np.asarray([s.alive for s in red])[:, None] & np.asarray([s.alive for s in blue])[None, :]
    guide_pairs = np.where(
        live & (geometry['distance'] >= reward['guide_distance_min'])
        & angle_within(np.abs(geometry['body_azimuth']), np.pi / 6)
        & angle_within(np.abs(geometry['body_elevation']), np.pi / 6),
        reward['guide_reward'], 0.)
    offense_pairs = np.where(
        live & (geometry['distance'] <= reward['tactical_distance_max'])
        & angle_within(np.abs(geometry['aa']), reward['tactical_aspect_max']),
        body_levels(geometry, reward, 'offense_reward'), 0.)
    defense_pairs = np.where(
        live.T & (reverse['distance'] <= reward['tactical_distance_max'])
        & angle_within(np.abs(reverse['aa']), reward['tactical_aspect_max']),
        body_levels(reverse, reward, 'threat_penalty'), 0.)
    guide = guide_pairs.max(axis=1)
    offense = offense_pairs.max(axis=1)
    defense = defense_pairs.min(axis=0)
    return dict(guide=guide, tactical_offense=offense, tactical_defense=defense,
                tactical=guide + offense + defense)
