"""v3.13 Equation (25) geometry only: no sensor dependency or visibility gate."""
import numpy as np
from .v313_geometry import combat_geometry_v313 as combat_geometry
from .v36_geometry import angle_within
from .v36_reward import levels


def tactical_rewards(red, blue, reward):
    g = combat_geometry(red, blue)
    reverse = combat_geometry(blue, red)
    live = np.asarray([s.alive for s in red])[:, None] & np.asarray([s.alive for s in blue])[None, :]
    guide = np.where((live & (g['distance'] >= reward['guide_distance_min'])
        & angle_within(g['ata'], np.pi / 6)
        & angle_within(g['ha'], np.pi / 6)).any(axis=1), reward['guide_reward'], 0.)
    offense = np.where(live & (g['distance'] <= reward['tactical_distance_max'])
        & angle_within(g['aa'], reward['tactical_aspect_max']),
        levels(g, reward, 'offense_reward'), 0.).max(axis=1)
    defense = np.where(live.T & (reverse['distance'] <= reward['tactical_distance_max'])
        & angle_within(reverse['aa'], reward['tactical_aspect_max']),
        levels(reverse, reward, 'threat_penalty'), 0.).min(axis=0)
    return dict(guide=guide, tactical_offense=offense, tactical_defense=defense,
                tactical=guide + offense + defense)
