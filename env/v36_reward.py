"""Post-dynamics/pre-hit geometry rewards with bounded target aggregation."""
import numpy as np
from .sensor import sensor_geometry
from .v36_geometry import combat_geometry,angle_within

def levels(g,config,prefix):
    result=np.zeros_like(g['distance'])
    for degrees in (30,15,5):
        valid=angle_within(g['ata'],np.deg2rad(degrees))&angle_within(g['ha'],np.deg2rad(degrees))
        result=np.where(valid,config[f'{prefix}_{degrees}deg'],result)
    return result

def tactical_rewards(red,blue,sensor,weapon,reward):
    g=combat_geometry(red,blue);reverse=combat_geometry(blue,red)
    visible=sensor_geometry(red,blue,sensor)[0]
    guide=np.where((visible & (g['distance']>weapon['range_max'])
        & (g['distance']<=sensor['range_max']) & angle_within(g['ata'],weapon['ata_max'])
        & angle_within(g['ha'],weapon['ha_max'])).any(axis=1),reward['guide_reward'],0.)
    offensive=levels(g,reward,'offense_reward')
    offensive=np.where(visible & (g['distance']<=weapon['range_max'])
        & angle_within(g['aa'],reward['tactical_aspect_max']),offensive,0.).max(axis=1)
    live=np.asarray([s.alive for s in blue])[:,None]&np.asarray([s.alive for s in red])[None,:]
    defensive=levels(reverse,reward,'threat_penalty')
    defensive=np.where(live & (reverse['distance']<=weapon['range_max'])
        & angle_within(reverse['aa'],reward['tactical_aspect_max']),defensive,0.).min(axis=0)
    return dict(guide=guide,tactical_offense=offensive,tactical_defense=defensive,
        tactical=guide+offensive+defensive)
