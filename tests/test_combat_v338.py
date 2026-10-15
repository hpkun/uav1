"""Observation-only v3.38: exact signed slots and frozen v3.37 transitions."""
from copy import deepcopy
from pathlib import Path
import inspect
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v337 import CombatEnvironmentV337
from env.combat_v338 import CombatEnvironmentV338
from env.config import environment_dimensions,validate_config
from env.factory import make_combat_environment
from env.geometry import engagement_geometry
from env.models import AircraftState
from env.v330_observation import build_team_observations_v330
from env.v338_observation import build_team_observations_v338
from env.v336_weapon import PaperATAHAWeapon
from env.fixed_policy import NearestTargetPursuitPolicy
from tools.audit_combat_v336_weapon_geometry import policy_actions


def config(version='338',algorithm='combat_environment'):
    stem=algorithm if algorithm=='combat_environment' else algorithm+'_4v4'
    return yaml.safe_load(Path(f'configs/{stem}_v{version}.yaml').read_text(encoding='utf-8'))


def crafted_states():
    red=[AircraftState(0,0,-3000,225,.55,.4),
         AircraftState(200,-100,-3500,180,-.3,-.7),
         AircraftState(-100,400,-2600,280,.15,1.2),
         AircraftState(-300,-200,-3100,200,-.6,-2.)]
    blue=[AircraftState(1300,-900,-4200,240,-.2,-1.1),
          AircraftState(900,1200,-1800,160,.4,2.8),
          AircraftState(-700,-1200,-4700,260,-.1,1.6),
          AircraftState(-1300,300,-2300,220,.2,-2.6)]
    return red,blue


def exact(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:
            if k!='environment_version':exact(a[k],b[k])
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def allowed_observation_difference(old,new):
    np.testing.assert_array_equal(old[:,:25],new[:,:25])
    for j in range(4):np.testing.assert_array_equal(old[:,25+5*j:28+5*j],new[:,25+5*j:28+5*j])


def test_dispatch_inheritance_45d_and_only_observation_override():
    e=make_combat_environment(config())
    assert type(e) is CombatEnvironmentV338
    assert CombatEnvironmentV338.__bases__==(CombatEnvironmentV337,)
    assert set(CombatEnvironmentV338.__dict__)-{'__module__','__doc__'}=={'environment_version','_observations'}
    assert environment_dimensions(config())==(45,3,4)
    for name in ('__init__','reset','step','_advance','initial_states','dense_combat_reward','build_weapon',
                 '_outcome','_resolve_combat','attempt_hit_for_pair','_entry_attempts'):
        assert getattr(CombatEnvironmentV338,name) is getattr(CombatEnvironmentV337,name)
    obs,_=e.reset(37);assert obs.shape==(4,45)
    assert e.step(np.zeros((4,3)))[0].shape==(4,45)
    assert type(e.weapon) is PaperATAHAWeapon


def test_environment_config_only_version_changed():
    old,new=config('337'),config()
    new['environment_version']='3.37'
    assert new==old
    assert old['arena']['altitude_min']==0 and old['arena']['altitude_max']==6000


def test_every_enemy_slot_exact_signed_authoritative_geometry():
    red,blue=crafted_states();c=config()['observation']
    obs=build_team_observations_v338(red,blue,c)
    for i,own in enumerate(red):
        for j,enemy in enumerate(blue):
            g=engagement_geometry(own,enemy)
            expected=np.array([g.distance/c['relative_position_scale'],(enemy.v-c['speed_center'])/c['speed_scale'],
                               g.aa/np.pi,g.ata/np.pi,g.ha/(np.pi/2)])
            np.testing.assert_allclose(obs[i,25+5*j:30+5*j],expected,rtol=0,atol=1e-14)
    angles=obs[:,25:].reshape(4,4,5)[:,:,2:]
    assert (np.abs(angles)<=1).all()
    for k in range(3):assert (angles[:,:,k]<0).any() and (angles[:,:,k]>0).any()


def test_nonzero_pitch_altitude_difference_discriminates_old_body_fields():
    red,blue=crafted_states();c=config()['observation']
    g=engagement_geometry(red[0],blue[0])
    assert red[0].theta!=0 and red[0].altitude!=blue[0].altitude
    assert not np.isclose(g.ata,g.boresight_azimuth_error,rtol=0,atol=1e-3)
    assert not np.isclose(g.ha,g.boresight_elevation_error,rtol=0,atol=1e-3)
    old=build_team_observations_v330(red,blue,c);new=build_team_observations_v338(red,blue,c)
    np.testing.assert_allclose(old[0,28:30],[g.boresight_azimuth_error/np.pi,g.boresight_elevation_error/(np.pi/2)],rtol=0,atol=1e-14)
    np.testing.assert_allclose(new[0,28:30],[g.ata/np.pi,g.ha/(np.pi/2)],rtol=0,atol=1e-14)
    assert not np.allclose(old[0,28:30],new[0,28:30],rtol=0,atol=1e-3)
    import env.v338_observation as module
    assert module.engagement_geometry is engagement_geometry
    source=inspect.getsource(module.build_team_observations_v338)
    assert not any(name in source for name in ('boresight_azimuth_error','boresight_elevation_error','off_boresight'))


@pytest.mark.parametrize('dead',[False,True])
def test_self_allies_and_enemy_distance_speed_aa_bit_identical(dead):
    red,blue=crafted_states()
    if dead:red[1].alive=False;blue[2].alive=False
    c=config()['observation']
    old=build_team_observations_v330(red,blue,c);new=build_team_observations_v338(red,blue,c)
    allowed_observation_difference(old,new)
    if dead:
        assert not new[1].any() and not new[0,7:13].any() and not new[:,35:40].any()


def test_historical_v337_still_uses_original_builder():
    a,b=MultiUAVCombatEnv(config('337')),MultiUAVCombatEnv(config())
    red,blue=crafted_states()
    for e in (a,b):e.red=deepcopy(red);e.blue=deepcopy(blue)
    np.testing.assert_array_equal(a._observations(),build_team_observations_v330(red,blue,a.config['observation']))
    assert not np.array_equal(a._observations(),b._observations())


def test_200_identical_resets_and_only_two_enemy_columns_change():
    a,b=MultiUAVCombatEnv(config('337')),MultiUAVCombatEnv(config())
    for seed in range(200):
        x,info1=a.reset(seed);y,info2=b.reset(seed)
        assert a.red==b.red and a.blue==b.blue
        exact(info1,info2);exact(a.rng.bit_generator.state,b.rng.bit_generator.state)
        allowed_observation_difference(x,y)


@pytest.mark.parametrize('seed',[0,1,2])
@pytest.mark.parametrize('mode',['fixed','pitch_probe'])
def test_full_episode_all_non_observation_outputs_identical(seed,mode):
    a,b=MultiUAVCombatEnv(config('337')),MultiUAVCombatEnv(config())
    a.reset(seed);b.reset(seed)
    policy=NearestTargetPursuitPolicy(a.config['blue_policy'],a.config['action'])
    while True:
        actions=policy_actions(a,policy,mode)
        x,y=a.step(actions),b.step(actions.copy())
        assert a.red==b.red and a.blue==b.blue and x[2:4]==y[2:4]
        exact(x[1],y[1]);exact(x[4],y[4])
        exact(a.rng.bit_generator.state,b.rng.bit_generator.state)
        exact(a.combat_counts,b.combat_counts)
        exact(a.dense_combat_reward(),b.dense_combat_reward())
        allowed_observation_difference(x[0],y[0])
        for side in ('red','blue'):
            for suffix in ('_alive_mask','_ammo','_fire_cooldown_steps'):
                exact(getattr(a,side+suffix),getattr(b,side+suffix))
        if x[2] or x[3]:break


def test_weapon_eligibility_and_100_identical_hit_draws():
    a,b=MultiUAVCombatEnv(config('337')),MultiUAVCombatEnv(config())
    red,blue=crafted_states()
    red[0]=AircraftState(0,0,-3000,225,.55,0)
    blue[0]=AircraftState(450,100,-3100,240,-.2,math.pi)
    for e in (a,b):e.reset(38);e.red=deepcopy(red);e.blue=deepcopy(blue)
    assert a.weapon==b.weapon
    exact(a.weapon.eligibility(a.red,a.blue,a.red_ammo),b.weapon.eligibility(b.red,b.blue,b.red_ammo))
    assert a.weapon.eligibility(a.red,a.blue,a.red_ammo)[0][0,0]
    hits=0
    for _ in range(100):
        x,y=a.attempt_hit_for_pair(a.red[0],a.blue[0]),b.attempt_hit_for_pair(b.red[0],b.blue[0])
        assert x==y
        hits+=int(x)
        exact(a.rng.bit_generator.state,b.rng.bit_generator.state)
    assert 0<hits<100


@pytest.mark.parametrize('section,key,value',[('weapon','range_max',4000),('reward','tactical_distance_max',4000),
    ('arena','altitude_max',7000),('scenario','center_radius',4800),('observation','speed_scale',50),
    ('blue_policy','desired_speed',250),('simulation','dt',.2)])
def test_strict_validator_rejects_unrelated_drift(section,key,value):
    c=config();c[section][key]=value
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
def test_algorithm_configs_only_environment_identity_changes(name):
    old,new=config('337',name),config('338',name)
    assert new['environment']=={'environment_version':'3.38','team_size':4}
    if 'environment' in old:new['environment']['environment_version']='3.37'
    else:new.pop('environment')  # MAPPO v337 has no explicit identity section.
    assert new==old
    assert new['network']['observation_dim']==45
    if name=='madsac':
        assert new['implementation_version']==5
        assert [new['training'][k] for k in ('steps_per_update','num_updates','policy_delay')]==[100,4,2]
