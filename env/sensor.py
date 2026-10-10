"""Shared true-3D detection and tactical geometry for the isolated v3.0 family."""
import numpy as np


def sensor_geometry(observers, targets, config):
    """One geometry definition for observations, fixed policy and potential.

    Rows are observers, columns targets. Coincidence follows the legacy
    boresight convention: aligned LOS when no direction can be defined.
    """
    a=np.asarray([[s.x,s.y,s.z] for s in observers],dtype=float)
    b=np.asarray([[s.x,s.y,s.z] for s in targets],dtype=float)
    delta=b[None,:,:]-a[:,None,:]
    distance=np.linalg.norm(delta,axis=-1)
    los=delta/np.maximum(distance[...,None],1e-12)
    forward=np.asarray([s.velocity_vector()/s.v for s in observers])
    target_forward=np.asarray([s.velocity_vector()/s.v for s in targets])
    ata_cos=np.clip(np.einsum('ijk,ik->ij',los,forward),-1,1)
    aa_cos=np.clip(np.einsum('ijk,jk->ij',los,target_forward),-1,1)
    ata_cos=np.where(distance==0,1.,ata_cos)
    aa_cos=np.where(distance==0,1.,aa_cos)
    visible=(np.asarray([s.alive for s in observers])[:,None]
             & np.asarray([s.alive for s in targets])[None,:]
             & (distance<=float(config['range_max']))
             & (np.arccos(ata_cos)<=float(config['off_boresight_angle_max'])))
    return visible,distance,ata_cos,aa_cos


def is_detected(observer,target,sensor_config):
    return bool(sensor_geometry([observer],[target],sensor_config)[0][0,0])


def tactical_angles_3d(attacker,target):
    _,_,ata,aa=sensor_geometry([attacker],[target],
        {'range_max':float('inf'),'off_boresight_angle_max':np.pi})
    return float(np.arccos(ata[0,0])),float(np.arccos(aa[0,0]))
