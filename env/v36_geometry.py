"""Separate horizontal ATA/AA and vertical HA in NED coordinates."""
import numpy as np

def angular_difference(a,b):
    return np.abs(np.arctan2(np.sin(a-b),np.cos(a-b)))

def combat_geometry(attackers,targets):
    a=np.asarray([[s.x,s.y,s.z] for s in attackers],dtype=float)
    t=np.asarray([[s.x,s.y,s.z] for s in targets],dtype=float)
    delta=t[None,:,:]-a[:,None,:]
    horizontal=np.hypot(delta[:,:,0],delta[:,:,1]);distance=np.linalg.norm(delta,axis=-1)
    heading=np.arctan2(delta[:,:,1],delta[:,:,0])
    own_heading=np.asarray([s.psi for s in attackers])[:,None]
    own_pitch=np.asarray([s.theta for s in attackers])[:,None]
    # No horizontal direction: use attacker heading; exact 3D coincidence
    # additionally uses attacker pitch, defining ATA/HA as zero.
    heading=np.where(horizontal==0,own_heading,heading)
    elevation=np.arctan2(-delta[:,:,2],horizontal)
    elevation=np.where(distance==0,own_pitch,elevation)
    return dict(distance=distance,ata=angular_difference(heading,own_heading),
        ha=angular_difference(elevation,own_pitch),
        aa=angular_difference(heading,np.asarray([s.psi for s in targets])[None,:]),
        los_elevation=elevation)

def angle_within(angle,limit):
    # Roundoff tolerance only, many orders below any modeled angular change.
    return angle<=limit+1e-12
