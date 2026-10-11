"""Height LOS angle in NED; horizontal ATA/AA retain historical conventions."""
import numpy as np
from .v36_geometry import angular_difference


def combat_geometry_v313(attackers, targets):
    a = np.asarray([[s.x, s.y, s.z] for s in attackers], dtype=float)
    t = np.asarray([[s.x, s.y, s.z] for s in targets], dtype=float)
    delta = t[None, :, :] - a[:, None, :]
    horizontal = np.hypot(delta[:, :, 0], delta[:, :, 1])
    distance = np.linalg.norm(delta, axis=-1)
    heading = np.arctan2(delta[:, :, 1], delta[:, :, 0])
    own_heading = np.asarray([s.psi for s in attackers])[:, None]
    # Preserve ATA=0 and stable AA when horizontal direction is undefined.
    heading = np.where(horizontal == 0, own_heading, heading)
    elevation = np.arctan2(-delta[:, :, 2], horizontal)
    # Coincidence has no LOS: explicitly choose zero, independent of pitch.
    elevation = np.where(distance == 0, 0., elevation)
    return dict(distance=distance, ata=angular_difference(heading, own_heading),
        ha=np.abs(elevation),
        aa=angular_difference(heading, np.asarray([s.psi for s in targets])[None, :]),
        los_elevation=elevation)
