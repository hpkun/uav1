"""Unchanged paper angles and aggregation, with pair-level R4 distance weights."""
import numpy as np
from .v313_geometry import combat_geometry_v313 as combat_geometry
from .v36_geometry import angle_within
from .v36_reward import levels


def distance_weight(distance):
    return np.clip((3000. - np.asarray(distance, dtype=float)) / 2000., 0., 1.)


def tactical_rewards(red, blue, reward):
    geometry = combat_geometry(red, blue)
    reverse = combat_geometry(blue, red)
    live = np.asarray([s.alive for s in red])[:, None] & np.asarray([s.alive for s in blue])[None, :]
    guide = np.where((live & (geometry['distance'] >= reward['guide_distance_min'])
                      & angle_within(geometry['ata'], np.pi / 6)
                      & angle_within(geometry['ha'], np.pi / 6)).any(axis=1), reward['guide_reward'], 0.)
    offense_pairs = np.where(
        live & (geometry['distance'] <= reward['tactical_distance_max'])
        & angle_within(geometry['aa'], reward['tactical_aspect_max']),
        levels(geometry, reward, 'offense_reward') * distance_weight(geometry['distance']), 0.)
    defense_pairs = np.where(
        live.T & (reverse['distance'] <= reward['tactical_distance_max'])
        & angle_within(reverse['aa'], reward['tactical_aspect_max']),
        levels(reverse, reward, 'threat_penalty') * distance_weight(reverse['distance']), 0.)
    offense = offense_pairs.max(axis=1)
    defense = defense_pairs.min(axis=0)
    return dict(guide=guide, tactical_offense=offense, tactical_defense=defense,
                tactical=guide + offense + defense)
