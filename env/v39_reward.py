"""Equation (25) distances, independent of the calibrated weapon envelope."""
import numpy as np
from .sensor import sensor_geometry
from .v36_geometry import combat_geometry, angle_within
from .v36_reward import levels


def tactical_rewards(red, blue, sensor, reward):
    g = combat_geometry(red, blue)
    reverse = combat_geometry(blue, red)
    visible = sensor_geometry(red, blue, sensor)[0]
    guide = np.where((visible & (g['distance'] >= reward['guide_distance_min'])
        & angle_within(g['ata'], np.pi / 6)
        & angle_within(g['ha'], np.pi / 6)).any(axis=1), reward['guide_reward'], 0.)
    offensive = np.where(visible & (g['distance'] <= reward['tactical_distance_max'])
        & angle_within(g['aa'], reward['tactical_aspect_max']),
        levels(g, reward, 'offense_reward'), 0.).max(axis=1)
    # Preserve the existing reverse-Blue threat definition, including threats
    # outside the Red radar FOV. No new Blue-visible gate is introduced.
    live = np.asarray([s.alive for s in blue])[:, None] & np.asarray([s.alive for s in red])[None, :]
    defensive = np.where(live & (reverse['distance'] <= reward['tactical_distance_max'])
        & angle_within(reverse['aa'], reward['tactical_aspect_max']),
        levels(reverse, reward, 'threat_penalty'), 0.).min(axis=0)
    return dict(guide=guide, tactical_offense=offensive, tactical_defense=defensive,
        tactical=guide + offensive + defensive)
