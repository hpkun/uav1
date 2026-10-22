"""Li et al. (2023), Eq. (25) R4 only; DOI 10.3390/aerospace10070574.

Paper rules: AA/distance gate, independent ATA/HA bins and six constants.
Engineering completion: max offense, min threat, offense precedence, else zero.
No continuous distance/tail weight, guide reward, RNG or target-selector state.
"""
import numpy as np
from .geometry import engagement_geometry
from .v36_geometry import angle_within


PAPER_R4 = dict(offense_reward_30deg=.01, offense_reward_15deg=.02,
    offense_reward_5deg=.1, threat_penalty_30deg=-.015,
    threat_penalty_15deg=-.025, threat_penalty_5deg=-.15,
    tactical_aspect_max=np.pi/6, tactical_distance_max=4000.)


def paper_branch(geometry, reward, prefix):
    """Inclusive paper gates; first matching (strictest) bin, never a sum.

    Angle comparison retains the project's 1e-12 rad roundoff tolerance;
    distance comparison is literal <=, without a range tolerance or weight.
    """
    if (geometry.distance > reward['tactical_distance_max']
            or not angle_within(abs(geometry.aa), reward['tactical_aspect_max'])):
        return 0.
    for degrees in (5, 15, 30):
        if (angle_within(abs(geometry.ata), np.deg2rad(degrees))
                and angle_within(abs(geometry.ha), np.deg2rad(degrees))):
            return reward[f'{prefix}_{degrees}deg']
    return 0.


def tactical_pair_rewards(red, blue, reward):
    """Both matrices indexed [red, blue]; threat swaps attacker/target roles."""
    offense = np.zeros((len(red), len(blue)), dtype=float)
    defense = np.zeros_like(offense)
    for i, own in enumerate(red):
        for j, target in enumerate(blue):
            if own.alive and target.alive:
                offense[i, j] = paper_branch(engagement_geometry(own, target), reward, 'offense_reward')
                defense[i, j] = paper_branch(engagement_geometry(target, own), reward, 'threat_penalty')
    return offense, defense


def tactical_rewards(red, blue, reward):
    """Project convention for ambiguities NOT resolved by paper Eq. (25).

    Diagnostics contain applied contributions, so the inherited R4 ledger stays
    exact: a positive offense suppresses defense, even from another Blue.
    Death filtering uses the inherited post-motion/pre-settlement living set.
    """
    offense_pairs, defense_pairs = tactical_pair_rewards(red, blue, reward)
    offense = offense_pairs.max(axis=1) if len(blue) else np.zeros(len(red))
    threat = defense_pairs.min(axis=1) if len(blue) else np.zeros(len(red))
    defense = np.where(offense > 0., 0., threat)
    return dict(guide=np.zeros(len(red)), tactical_offense=offense,
                tactical_defense=defense, tactical=offense+defense)
