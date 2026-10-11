"""Equation (24) five-agent layout using the authoritative combat geometry."""
import numpy as np
from .observation import flight_path_frame, _relative_position
from .v313_geometry import combat_geometry_v313 as combat_geometry

OBSERVATION_DIM = 56


def build_team_observations_v313(team, opponents, config):
    if len(team) != 5 or len(opponents) != 5:
        raise ValueError('v3.13 observations require 5v5')
    geometry = combat_geometry(team, opponents)
    # Keep geometry precision for direct inspection/replay; existing algorithms
    # perform their usual float32 tensor conversion at the network boundary.
    result = np.zeros((5, OBSERVATION_DIM), dtype=np.float64)
    for i, own in enumerate(team):
        if not own.alive:
            continue
        result[i, :7] = [own.x/config['horizontal_position_scale'],
            own.y/config['horizontal_position_scale'], own.altitude/config['altitude_scale'],
            (own.v-config['speed_center'])/config['speed_scale'], 0., own.psi/np.pi,
            own.theta/(np.pi/3)]
        frame = flight_path_frame(own)
        for slot, j in enumerate(index for index in range(5) if index != i):
            ally = team[j]
            if ally.alive:
                start = 7 + 6*slot
                result[i, start:start+6] = np.concatenate((
                    _relative_position(own, ally, frame, config['relative_position_scale']),
                    [(ally.v-config['speed_center'])/config['speed_scale'],
                     ally.psi/np.pi, ally.theta/(np.pi/3)]))
        for j, enemy in enumerate(opponents):
            if enemy.alive:
                start = 31 + 5*j
                result[i, start:start+5] = [geometry['distance'][i,j]/config['relative_position_scale'],
                    (enemy.v-config['speed_center'])/config['speed_scale'],
                    geometry['aa'][i,j]/np.pi, geometry['ata'][i,j]/np.pi,
                    geometry['ha'][i,j]/(np.pi/2)]
    if not np.isfinite(result).all():
        raise FloatingPointError('invalid v3.13 observation')
    return result
