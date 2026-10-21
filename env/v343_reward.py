"""Continuous candidate research rewards, not a claim of paper replication."""
import numpy as np
from .geometry import engagement_geometry


RESEARCH_REWARD = dict(gamma=.99, potential_scale=.1, potential_distance=8000.,
    potential_distance_weight=.5, potential_angle_weight=.5,
    tactical_outer_distance=4000., tactical_span=2000., alignment_max=np.pi/3,
    offense_base=.005, offense_tail=.015, defense_scale=.01)


def state_potential(red, blue, config):
    """Nearest living target, ties by slot index; no cached/sticky target state.

    The target selector is a function of each full state. Changing target or
    losing a target therefore still telescopes, including the same-step kill.
    """
    phi = np.zeros(len(red), dtype=float)
    distance_component = phi.copy(); angle_component = phi.copy()
    indices = np.full(len(red), -1, dtype=int)
    targets = [j for j,t in enumerate(blue) if t.alive]
    for i, own in enumerate(red):
        if not own.alive or not targets:
            continue
        pairs = [(j,engagement_geometry(own,blue[j])) for j in targets]
        j,g = min(pairs, key=lambda item:(item[1].distance,item[0]))
        indices[i] = j
        distance_component[i] = 1.-np.clip(g.distance/config['potential_distance'],0.,1.)
        angle_component[i] = 1.-g.off_boresight/np.pi
        phi[i] = config['potential_scale']*(config['potential_distance_weight']*distance_component[i]
                   + config['potential_angle_weight']*angle_component[i])
    return phi, dict(distance=distance_component, angle=angle_component, target_indices=indices)


def continuous_tactical_rewards(red, blue, config):
    offense_pairs = np.zeros((len(red),len(blue)),dtype=float)
    defense_pairs = offense_pairs.copy()
    for i,a in enumerate(red):
        for j,b in enumerate(blue):
            if not a.alive or not b.alive:
                continue
            g = engagement_geometry(a,b); reverse = engagement_geometry(b,a)
            w = np.clip((config['tactical_outer_distance']-g.distance)/config['tactical_span'],0.,1.)
            align = np.clip(1.-g.off_boresight/config['alignment_max'],0.,1.)
            threat = np.clip(1.-reverse.off_boresight/config['alignment_max'],0.,1.)
            tail = (1.+np.cos(g.aa))/2.
            offense_pairs[i,j] = w*align*(config['offense_base']+config['offense_tail']*tail)
            defense_pairs[i,j] = -config['defense_scale']*w*threat
    offense = offense_pairs.max(axis=1) if len(blue) else np.zeros(len(red))
    defense = defense_pairs.min(axis=1) if len(blue) else np.zeros(len(red))
    return dict(guide=np.zeros(len(red)), tactical_offense=offense,
                tactical_defense=defense, tactical=offense+defense)
