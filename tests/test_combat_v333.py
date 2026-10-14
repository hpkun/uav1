"""Canonical paper geometry, frozen baseline, protocol and calibration diagnostics."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import inspect
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v332 import CombatEnvironmentV332
from env.combat_v333 import CombatEnvironmentV333
from env.config import validate_config, environment_dimensions
from env.geometry import engagement_geometry
from env.models import AircraftState
from env.v333_config import CHANGES
from env.v333_reward import tactical_rewards_v333


def env():
    e = MultiUAVCombatEnv('configs/combat_environment_v333.yaml')
    e.reset(0)
    return e


def pair(distance=3500, body=0, aa=0, elevation=0):
    # LOS along +x; target heading independently sets AA.
    return AircraftState(0, 0, -3000, 225, math.radians(elevation), -math.radians(body)), AircraftState(distance, 0, -3000, 225, 0, math.radians(aa))


def rewards(a, b):
    e = env()
    return tactical_rewards_v333([a], [b], e.config['reward'], e.config['weapon'])


@pytest.mark.parametrize('distance,body,aa,guide,offense', [
    (5000,0,180,.001,0), (3500,0,180,0,0), (3500,0,0,0,.1),
    (3500,10,0,0,.02), (3500,20,0,0,.01), (3500,0,31,0,0),
    (4100,0,0,.001,0), (4000,0,0,.001,.1),
    (4000,0,180,.001,0), (3999,0,0,0,.1), (4001,0,0,.001,0),
    (5000,31,0,0,0), (3500,31,0,0,0), (3500,30,30,0,.01),
    (3500,15,0,0,.02), (3500,5,0,0,.1)])
def test_canonical_offense(distance,body,aa,guide,offense):
    r = rewards(*pair(distance,body,aa))
    assert r['guide'][0] == pytest.approx(guide)
    assert r['tactical_offense'][0] == pytest.approx(offense)
    np.testing.assert_array_equal(r['tactical'], r['guide']+r['tactical_offense']+r['tactical_defense'])


@pytest.mark.parametrize('angle,penalty', [(30,-.015),(15,-.025),(5,-.15)])
@pytest.mark.parametrize('axis', ['azimuth','elevation'])
def test_reverse_defense(angle,penalty,axis):
    blue, red = pair(body=angle if axis=='azimuth' else 0, elevation=angle if axis=='elevation' else 0)
    assert rewards(red,blue)['tactical_defense'][0] == pytest.approx(penalty)


def test_reverse_aspect_and_distance_gates():
    for d,aa in ((3500,31),(4100,0)):
        blue,red = pair(d,0,aa)
        assert rewards(red,blue)['tactical_defense'][0] == 0


@pytest.mark.parametrize('distance,az,el,aa,eligible', [
    (3999,0,0,0,True),(4000,0,0,0,True),(4001,0,0,0,False),
    (3500,31,0,0,False),(3500,0,31,0,False),
    (3500,0,0,180,True),(3500,30,0,0,True),(3500,0,30,0,True)])
def test_weapon(distance,az,el,aa,eligible):
    a,b=pair(distance,az,aa,el)
    assert env().weapon.qualifies(a,b,6) is eligible
    if aa==180:
        assert abs(engagement_geometry(a,b).aa)==pytest.approx(math.pi)
        r=rewards(a,b)
        assert r['guide'][0]==r['tactical_offense'][0]==0


@pytest.mark.parametrize('distance',[1000,2000,3000,4000])
def test_expected_hit_diagnostic(distance):
    w=env().weapon
    expected=math.erf(math.pi*math.exp(-distance/w.effective_hit_distance)/w.azimuth_noise_scale/math.sqrt(2))
    p=w.expected_hit_probability(distance,0,0)
    assert p==pytest.approx(expected)
    print(f'perfect alignment {distance} m: {p:.12f}')


def test_inheritance_and_frozen_config():
    e=env();old=MultiUAVCombatEnv('configs/combat_environment_v332.yaml')
    assert CombatEnvironmentV333.__bases__==(CombatEnvironmentV332,)
    assert type(e) is CombatEnvironmentV333 and environment_dimensions(e.config)==(45,3,4)
    baseline=deepcopy(e.config);baseline['environment_version']='3.32'
    for section,key,_,prior in CHANGES:baseline[section][key]=prior
    assert baseline==old.config
    for name in ('_advance','step','reset','initial_states','_observations','_outcome','build_weapon','_entry_attempts','_resolve_combat'):
        assert getattr(CombatEnvironmentV333,name) is getattr(CombatEnvironmentV332,name)
    assert 'distance_weights' not in inspect.getsource(tactical_rewards_v333)
    with patch('env.v323_reward.distance_weights',side_effect=AssertionError('continuous weights called')):
        e.dense_combat_reward()
    old.reset(0);old.red,old.blue=map(list,zip(*[pair(2000)]*4))
    assert 0 < old.dense_combat_reward()[1]['guide'][0] < .001
    assert old.weapon.range_max==1000


@pytest.mark.parametrize('section,key,value', [
    ('scenario','center_radius',4000),('weapon','range_max',1000),
    ('reward','guide_distance_min',3000),('reward','tactical_distance_max',3000),
    ('reward','tactical_aspect_max',math.pi/2),('reward','offense_reward_5deg',.2),
    ('weapon','ammo_per_aircraft',7),('fire_control','cooldown_seconds',2),
    ('sensor','range_max',9000),('blue_policy','desired_speed',250),
    ('action','command',{}),('scenario','formation_offsets',[-400,-100,100,400])])
def test_reject_drift(section,key,value):
    c=deepcopy(env().config);c[section][key]=value
    with pytest.raises(ValueError):validate_config(c)


def test_aggregation_and_dead_pairs():
    e=env();red,b0=pair(3500,0,0);_,b1=pair(5000,0,0)
    r=tactical_rewards_v333([red],[b0,b1],e.config['reward'],e.config['weapon'])
    assert r['guide'][0]==.001 and r['tactical_offense'][0]==.1
    b0.alive=b1.alive=False
    assert tactical_rewards_v333([red],[b0,b1],e.config['reward'],e.config['weapon'])['tactical'][0]==0
    b0,r0=pair(3500,0,0);b1,_=pair(3500,20,0)
    assert tactical_rewards_v333([r0],[b0,b1],e.config['reward'],e.config['weapon'])['tactical_defense'][0]==-.15


def test_initialization_200_seeds_and_heading_semantics():
    from tools.audit_combat_v333_engagement import initialization
    e=env();old=MultiUAVCombatEnv('configs/combat_environment_v332.yaml')
    result=initialization(e,range(200))
    assert result['all_checks_passed']
    assert result['center_separation']['mean']==pytest.approx(9600)
    for seed in range(200):
        e.reset(seed);old.reset(seed)
        for side in ('red','blue'):
            a=np.array([s.as_array() for s in getattr(e,side)])
            b=np.array([s.as_array() for s in getattr(old,side)])
            np.testing.assert_array_equal(a[:,2:],b[:,2:])
        assert e._observations().shape==(4,45)


def test_identical_motion_and_outcome_semantics():
    e=env();old=MultiUAVCombatEnv('configs/combat_environment_v332.yaml');old.reset(0)
    for side in ('red','blue'):setattr(old,side,deepcopy(getattr(e,side)))
    actions=np.array([[1,.5,1],[-1,-1,-1],[.2,.1,.2],[0,0,0]])
    for side in ('red','blue'):
        e._advance(getattr(e,side),actions);old._advance(getattr(old,side),actions)
        assert getattr(e,side)==getattr(old,side)
    for s in e.red+e.blue:s.alive=False
    assert e._outcome(False)==(False,False,True,'draw_mutual_destruction')


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
def test_algorithm_config_frozen(name):
    a=yaml.safe_load(Path(f'configs/{name}_4v4_v332.yaml').read_text())
    b=yaml.safe_load(Path(f'configs/{name}_4v4_v333.yaml').read_text())
    if 'environment' in b:b['environment']['environment_version']='3.32'
    assert a==b


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
@pytest.mark.parametrize('previous',['3.30','3.31','3.32'])
def test_historical_checkpoint_rejected(name,previous):
    ec=env().config;cfg=yaml.safe_load(Path(f'configs/{name}_4v4_v333.yaml').read_text())
    s=dict(algorithm={'mappo':'MAPPO','madsac':'madsac','maac':'MAAC'}[name],implementation_version=3,
           extra=dict(environment_version=previous,training_smoke=True))
    if name=='mappo':
        from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
        checks=(validate_checkpoint_for_evaluation,validate_checkpoint_for_resume)
    elif name=='madsac':
        from algorithm.madsac.protocol import validate_madsac_checkpoint
        checks=(validate_madsac_checkpoint,)
    else:
        from algorithm.maac.protocol import validate_checkpoint
        checks=(validate_checkpoint,)
    for check in checks:
        with pytest.raises(RuntimeError,match='environment_version'):check(s,ec,cfg)
