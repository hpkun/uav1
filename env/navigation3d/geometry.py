"""NED navigation geometry. No engagement or weapon interfaces are used."""
from dataclasses import dataclass
import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def direction(pitch, heading):
    if not np.isfinite([pitch, heading]).all():
        raise ValueError('finite navigation angles required')
    return np.array([np.cos(pitch)*np.cos(heading),
                     np.cos(pitch)*np.sin(heading), -np.sin(pitch)])


def body_basis(pitch, heading):
    """Rows are forward, right, up in the world NED frame (zero roll)."""
    forward = direction(pitch, heading)
    right = np.array([-np.sin(heading), np.cos(heading), 0.])
    return np.stack([forward, right, np.cross(right, forward)])


@dataclass(frozen=True)
class NavigationGeometry:
    distance: float
    horizontal_distance: float
    height_difference: float
    angle: float
    body_azimuth: float
    body_elevation: float
    heading_error: float
    world_elevation: float


def waypoint_geometry(position, pitch, heading, waypoint):
    p, q = np.asarray(position, dtype=float), np.asarray(waypoint, dtype=float)
    if p.shape != (3,) or q.shape != (3,) or not np.isfinite([p, q]).all():
        raise ValueError('finite three-dimensional NED positions required')
    basis = body_basis(pitch, heading)
    with np.errstate(over='ignore'):
        relative = q - p
    if not np.isfinite(relative).all():
        raise ValueError('navigation displacement exceeds floating point range')
    distance = float(np.hypot.reduce(relative))
    horizontal = float(np.hypot(relative[0], relative[1]))
    if not np.isfinite(distance):
        raise ValueError('navigation distance exceeds floating point range')
    if distance == 0:
        # No unique direction at coincidence; all directional errors are zero.
        return NavigationGeometry(0., 0., 0., 0., 0., 0., 0., 0.)
    unit = relative / distance
    forward, right, up = basis @ unit
    cosine = float(np.clip(forward, -1., 1.))
    angle = float(np.arctan2(np.linalg.norm(np.cross(basis[0], unit)), cosine))
    bearing = np.arctan2(relative[1], relative[0]) if horizontal else heading
    return NavigationGeometry(distance, horizontal, float(-relative[2]), angle,
        float(np.arctan2(right, forward)), float(np.arctan2(up, np.hypot(forward, right))),
        float(wrap(bearing-heading)), float(np.arctan2(-relative[2], horizontal)))
