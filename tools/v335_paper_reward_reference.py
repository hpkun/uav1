"""Offline Eq.(25) pair reference and explicitly project-side comparison variants."""
import numpy as np
from env.geometry import engagement_geometry
from env.v322_reward import distance_weights
from env.v36_geometry import angle_within

PAPER_DISTANCE = 4000.
OFFENSE = {30: .01, 15: .02, 5: .10}
DEFENSE = {30: -.015, 15: -.025, 5: -.15}
AGGREGATION_NOTE = ('This aggregation is a project-side comparison wrapper; '
                    'the paper does not specify the multi-target aggregation rule.')


def pair_geometry(red, blue):
    names=('distance','ata','ha','aa','boresight_azimuth_error','boresight_elevation_error')
    result={k:np.zeros((len(red),len(blue))) for k in
        ('distance','ata_r','ha_r','aa_r','ata_b','ha_b','aa_b','body_az_r','body_el_r','body_az_b','body_el_b')}
    for i,r in enumerate(red):
        for j,b in enumerate(blue):
            forward=engagement_geometry(r,b);reverse=engagement_geometry(b,r)
            result['distance'][i,j]=forward.distance
            for suffix,g in (('r',forward),('b',reverse)):
                for field in ('ata','ha','aa'):result[field+'_'+suffix][i,j]=getattr(g,field)
                result['body_az_'+suffix][i,j]=g.boresight_azimuth_error
                result['body_el_'+suffix][i,j]=g.boresight_elevation_error
    result['live']=np.asarray([r.alive for r in red])[:,None]&np.asarray([b.alive for b in blue])[None,:]
    return result


def aligned(g,side,degrees,geometry):
    if geometry=='paper':
        # Exact literal paper angle conditions, using absolute ATA/HA.
        return (np.abs(g['ata_'+side])<=np.deg2rad(degrees))&(np.abs(g['ha_'+side])<=np.deg2rad(degrees))
    return angle_within(np.abs(g['body_az_'+side]),np.deg2rad(degrees))&angle_within(np.abs(g['body_el_'+side]),np.deg2rad(degrees))


def levels(g,side,geometry,coefficients):
    value=np.zeros_like(g['distance']);tier=np.zeros_like(value,dtype=int)
    for degrees in (30,15,5):
        mask=aligned(g,side,degrees,geometry)
        value=np.where(mask,coefficients[degrees],value);tier=np.where(mask,degrees,tier)
    return value,tier


def pair_variant(g,reward,weapon,geometry,gate):
    d=g['distance'];live=g['live']
    offense_nominal,tier=levels(g,'r',geometry,OFFENSE)
    defense_nominal,_=levels(g,'b',geometry,DEFENSE)
    if geometry=='paper':
        aa_r=np.abs(g['aa_r'])<=np.pi/6;aa_b=np.abs(g['aa_b'])<=np.pi/6
    else:
        # Preserve the existing current numerical angle tolerance in body variants.
        aa_r=angle_within(np.abs(g['aa_r']),reward['tactical_aspect_max'])
        aa_b=angle_within(np.abs(g['aa_b']),reward['tactical_aspect_max'])
    if gate=='hard':
        w3=(d>=PAPER_DISTANCE).astype(float);w4=(d<=PAPER_DISTANCE).astype(float)
    else:
        w3,w4=distance_weights(d,reward,weapon)
        w4=w4*(d<=reward['tactical_distance_max'])
    guide=np.where(live&aligned(g,'r',30,geometry),.001*w3,0.)
    offense=np.where(live&aa_r,offense_nominal*w4,0.)
    defense=np.where(live&aa_b,defense_nominal*w4,0.)
    return dict(r3=guide,offense=offense,defense=defense,r4=offense+defense,
                dense=guide+offense+defense,nominal_offense=offense_nominal,tier=tier)


def paper_pair_rewards(g):
    # Paper A has no Weapon, current reward configuration or normalized-distance dependency.
    return pair_variant(g,{}, {},'paper','hard')


def project_aggregation(pair):
    # The defensive matrix already indexes (Red victim, Blue threat).
    guide=pair['r3'].max(axis=1);offense=pair['offense'].max(axis=1);defense=pair['defense'].min(axis=1)
    return dict(r3=guide,offense=offense,defense=defense,r4=offense+defense,dense=guide+offense+defense)


def comparison(red,blue,reward,weapon):
    g=pair_geometry(red,blue)
    variants=dict(A=paper_pair_rewards(g),B=pair_variant(g,reward,weapon,'body','hard'),
                  C=pair_variant(g,reward,weapon,'paper','continuous'),
                  D=pair_variant(g,reward,weapon,'body','continuous'))
    w3,w4=distance_weights(g['distance'],reward,weapon)
    return g,variants,{k:project_aggregation(v) for k,v in variants.items()},w3,w4


def event_comparison(red_exits,red_ground,red_kills,blue_kills,n):
    # Literal +/-10 coefficients with inherited project event attribution/causes.
    # These attribution and ground/ceiling mappings are PAPER UNSPECIFIED.
    r1=np.zeros(n,dtype=np.float32);r2=np.zeros(n,dtype=np.float32)
    for attackers in red_kills.values():r1[attackers]+=10./len(attackers)
    for i in set(red_ground)|set(blue_kills):r1[i]-=10.
    for i in red_exits:r2[i]-=10.
    return r1,r2
