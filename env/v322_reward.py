"""Continuous guide handoff; frozen R4 angles, coefficients and aggregation."""
import numpy as np
from .v313_geometry import combat_geometry_v313 as combat_geometry
from .v36_geometry import angle_within
from .v36_reward import levels


def distance_weights(distance, reward, weapon):
    d_weapon = float(weapon['range_max'])
    d_tactical = float(reward['tactical_distance_max'])
    if not d_tactical > d_weapon:
        raise ValueError('tactical_distance_max must exceed weapon.range_max')
    span = d_tactical - d_weapon
    distance = np.asarray(distance, dtype=float)
    w3 = np.clip((distance - d_weapon) / span, 0., 1.)
    w4 = np.clip((d_tactical - distance) / span, 0., 1.)
    return w3, w4


def tactical_rewards_v322(red, blue, reward, weapon):
    geometry = combat_geometry(red, blue)
    reverse = combat_geometry(blue, red)
    live = np.asarray([s.alive for s in red])[:, None] & np.asarray([s.alive for s in blue])[None, :]
    w3, w4 = distance_weights(geometry['distance'], reward, weapon)
    _, reverse_w4 = distance_weights(reverse['distance'], reward, weapon)
    guide_pairs = np.where(
        live & angle_within(geometry['ata'], np.pi / 6)
        & angle_within(geometry['ha'], np.pi / 6), reward['guide_reward'] * w3, 0.)
    guide = guide_pairs.max(axis=1)
    offense_pairs = np.where(
        live & (geometry['distance'] <= reward['tactical_distance_max'])
        & angle_within(geometry['aa'], reward['tactical_aspect_max']),
        levels(geometry, reward, 'offense_reward') * w4, 0.)
    defense_pairs = np.where(
        live.T & (reverse['distance'] <= reward['tactical_distance_max'])
        & angle_within(reverse['aa'], reward['tactical_aspect_max']),
        levels(reverse, reward, 'threat_penalty') * reverse_w4, 0.)
    offense = offense_pairs.max(axis=1)
    defense = defense_pairs.min(axis=0)
    return dict(guide=guide, tactical_offense=offense, tactical_defense=defense,
                tactical=guide + offense + defense)
