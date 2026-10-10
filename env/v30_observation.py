"""Self8/ally7/enemy6 layout with strict zeros for undetected enemies."""
import numpy as np
from .observation import build_team_observations
from .sensor import sensor_geometry


def observation_dim_for_v30(team_size):
    from .observation import observation_dim_for_team_size
    return observation_dim_for_team_size(team_size)+1


def build_v30_observations(team,opponents,config,sensor,ammo,max_ammo,phis):
    # The legacy builder is reused for visible features, then all hidden slots
    # are zeroed before return. It has no persistent state or diagnostics.
    old=build_team_observations(team,opponents,config,phis)
    alive=np.asarray([s.alive for s in team],dtype=float)
    obs=np.concatenate((old[:,:7],(ammo/max_ammo*alive)[:,None],old[:,7:]),axis=1).astype(np.float32)
    visible,_,_,_=sensor_geometry(team,opponents,sensor)
    start=8+7*(len(team)-1)
    enemies=obs[:,start:].reshape(len(team),len(opponents),6)
    enemies[~visible]=0
    return obs
