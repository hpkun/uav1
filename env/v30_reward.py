"""Detected-target potential and true-3D soft safety; no legacy reward changes."""
import numpy as np
from .sensor import sensor_geometry


def distance_potential(distance, safe_distance=200., attack_range=1000., detection_range=3000.):
    d=np.asarray(distance,dtype=float)
    return np.where(d<safe_distance,2*d/safe_distance-1,
        np.where(d<=attack_range,1.,np.where(d<detection_range,
            (detection_range-d)/(detection_range-attack_range),0.)))


def potentials(team,targets,sensor,weapon,reward):
    visible,distance,ata_cos,aa_cos=sensor_geometry(team,targets,sensor)
    selection=np.where(visible,distance,np.inf).argmin(axis=1)
    valid=visible.any(axis=1);rows=np.arange(len(team))
    pd=np.where(valid,distance_potential(distance[rows,selection],reward['safe_distance'],
        weapon['range_max'],sensor['range_max']),0.)
    pa=np.where(valid,(ata_cos[rows,selection]+aa_cos[rows,selection])/2,0.)
    phi=reward['distance_weight']*pd+reward['angle_weight']*pa
    return phi,{'distance':pd,'angle':pa,'detected_target_index':np.where(valid,selection,-1)}


def safety_rewards(team,opponents,reward):
    states=team+opponents
    positions=np.asarray([[s.x,s.y,s.z] for s in states])
    distance=np.linalg.norm(positions[:,None,:]-positions[None,:,:],axis=-1)
    np.fill_diagonal(distance,np.inf)
    alive=np.asarray([s.alive for s in states])
    distance[:,~alive]=np.inf
    minimum=distance[:len(team)].min(axis=1)
    return np.where(alive[:len(team)],-reward['safety_penalty_scale']*
                    np.maximum(0.,1-minimum/reward['safe_distance']),0.)


def potential_shaping(current,next_,reward):
    return reward['potential_scale']*(reward['potential_gamma']*next_-current)
