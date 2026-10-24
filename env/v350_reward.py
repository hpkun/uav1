"""Li et al. (2023) Eq. (25), with one deterministic nearest living target.

The paper does not specify 4v4 target aggregation. Selection is a project
completion, shared by R3 and R4, independent of reward values and weapon range.
"""
import numpy as np

from .geometry import engagement_geometry
from .v36_geometry import angle_within
from .v346_reward import paper_branch


def offense_gate(geometry, reward):
    return (geometry.distance <= reward['tactical_distance_max']
            and angle_within(abs(geometry.aa), reward['tactical_aspect_max']))


def paper_pair_rewards(red_geometry, blue_geometry, reward):
    """R3 plus exclusive R41/else-R42, using paper ATA/HA (not 3D cone).

    The first outer condition in Eq. (25) selects R41 even when its inner
    ATA/HA bins yield zero. It must not fall through to a defensive penalty.
    paper_branch reuses the verified inclusive, strictest-bin-first formula.
    """
    guide = (reward['guide_reward']
             if red_geometry.distance >= reward['guide_distance_min']
             and angle_within(abs(red_geometry.ata), np.pi / 6)
             and angle_within(abs(red_geometry.ha), np.pi / 6) else 0.)
    if offense_gate(red_geometry, reward):
        offense = paper_branch(red_geometry, reward, 'offense_reward')
        defense = 0.
    else:
        offense = 0.
        defense = paper_branch(blue_geometry, reward, 'threat_penalty')
    return guide, offense, defense


def nearest_target_rewards(red, blue, reward):
    guide, offense, defense = (np.zeros(len(red), dtype=float) for _ in range(3))
    indices = np.full(len(red), -1, dtype=np.int64)
    for i, own in enumerate(red):
        if not own.alive:
            continue
        candidates = [(engagement_geometry(own, target), j)
                      for j, target in enumerate(blue) if target.alive]
        if not candidates:
            continue
        geometry, j = min(candidates, key=lambda pair: (pair[0].distance, pair[1]))
        indices[i] = j
        guide[i], offense[i], defense[i] = paper_pair_rewards(
            geometry, engagement_geometry(blue[j], own), reward)
    return dict(guide=guide, tactical_offense=offense,
                tactical_defense=defense, tactical=guide + offense + defense), indices
