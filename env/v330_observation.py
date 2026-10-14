"""45D observations retaining signed weapon-consistent enemy geometry."""
import numpy as np
from .geometry import engagement_geometry
from .observation import flight_path_frame, _relative_position

OBSERVATION_DIM = 45


def build_team_observations_v330(team, opponents, config):
    if len(team) != 4 or len(opponents) != 4:
        raise ValueError('v3.30 observations require 4v4')
    result = np.zeros((4, OBSERVATION_DIM), dtype=np.float64)
    for i, own in enumerate(team):
        if not own.alive:
            continue
        result[i, :7] = [own.x/config['horizontal_position_scale'],
            own.y/config['horizontal_position_scale'], own.altitude/config['altitude_scale'],
            (own.v-config['speed_center'])/config['speed_scale'], 0., own.psi/np.pi,
            own.theta/(np.pi/3)]
        frame = flight_path_frame(own)
        for slot, j in enumerate(index for index in range(4) if index != i):
            ally = team[j]
            if ally.alive:
                start = 7 + 6*slot
                result[i, start:start+6] = np.concatenate((
                    _relative_position(own, ally, frame, config['relative_position_scale']),
                    [(ally.v-config['speed_center'])/config['speed_scale'],
                     ally.psi/np.pi, ally.theta/(np.pi/3)]))
        for j, enemy in enumerate(opponents):
            if enemy.alive:
                g = engagement_geometry(own, enemy)
                start = 25 + 5*j
                result[i, start:start+5] = [g.distance/config['relative_position_scale'],
                    (enemy.v-config['speed_center'])/config['speed_scale'],
                    g.aa/np.pi, g.boresight_azimuth_error/np.pi,
                    g.boresight_elevation_error/(np.pi/2)]
    if not np.isfinite(result).all():
        raise FloatingPointError('invalid v3.30 observation')
    return result
