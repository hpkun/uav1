"""Weapon-consistent body angles with unchanged horizontal aspect and weights."""
import numpy as np
from .geometry import engagement_geometry
from .v36_geometry import angle_within
from .v322_reward import distance_weights


def body_pair_geometry(attackers, targets):
    pairs = [[engagement_geometry(a, t) for t in targets] for a in attackers]
    return {name: np.asarray([[getattr(g, field) for g in row] for row in pairs])
            for name, field in (('distance', 'distance'), ('aa', 'aa'),
                                ('body_azimuth', 'boresight_azimuth_error'),
                                ('body_elevation', 'boresight_elevation_error'))}


def body_levels(geometry, reward, prefix):
    result = np.zeros_like(geometry['distance'])
    for degrees in (30, 15, 5):
        valid = (angle_within(np.abs(geometry['body_azimuth']), np.deg2rad(degrees))
                 & angle_within(np.abs(geometry['body_elevation']), np.deg2rad(degrees)))
        result = np.where(valid, reward[f'{prefix}_{degrees}deg'], result)
    return result


def tactical_rewards_v323(red, blue, reward, weapon):
    geometry = body_pair_geometry(red, blue)
    reverse = body_pair_geometry(blue, red)
    live = np.asarray([s.alive for s in red])[:, None] & np.asarray([s.alive for s in blue])[None, :]
    w3, w4 = distance_weights(geometry['distance'], reward, weapon)
    _, reverse_w4 = distance_weights(reverse['distance'], reward, weapon)
    guide_pairs = np.where(
        live & angle_within(np.abs(geometry['body_azimuth']), np.pi / 6)
        & angle_within(np.abs(geometry['body_elevation']), np.pi / 6), reward['guide_reward'] * w3, 0.)
    offense_pairs = np.where(
        live & (geometry['distance'] <= reward['tactical_distance_max'])
        & angle_within(np.abs(geometry['aa']), reward['tactical_aspect_max']),
        body_levels(geometry, reward, 'offense_reward') * w4, 0.)
    defense_pairs = np.where(
        live.T & (reverse['distance'] <= reward['tactical_distance_max'])
        & angle_within(np.abs(reverse['aa']), reward['tactical_aspect_max']),
        body_levels(reverse, reward, 'threat_penalty') * reverse_w4, 0.)
    guide = guide_pairs.max(axis=1)
    offense = offense_pairs.max(axis=1)
    defense = defense_pairs.min(axis=0)
    return dict(guide=guide, tactical_offense=offense, tactical_defense=defense,
                tactical=guide + offense + defense)
