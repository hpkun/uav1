"""Single-variable R4 scaling and frozen v3.30 mechanics regression."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v330 import CombatEnvironmentV330
from env.combat_v331 import CombatEnvironmentV331
from env.config import environment_dimensions,validate_config
from env.models import AircraftState
from env.v331_config import R4_BASELINE
from tools.audit_combat_v331_reward import pair_states,static_table
from tools.audit_combat_v317_v318 import FixedNormal


def env(version='331'):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml');e.reset(31);return e


def dead():return AircraftState(0,0,-3000,225,0,0,False)


def test_dimensions_config_diff_and_frozen_methods():
    a,b=env('330'),env()
    assert CombatEnvironmentV331.__bases__==(CombatEnvironmentV330,)
    assert environment_dimensions(a.config)==environment_dimensions(b.config)==(45,3,4)
    baseline=deepcopy(b.config);baseline['environment_version']='3.30';baseline['reward'].update(R4_BASELINE)
    assert baseline==a.config
    for key,val in R4_BASELINE.items():assert b.config['reward'][key]==2*val==2*a.config['reward'][key]
    for name in ('__init__','reset','step','_advance','_observations','_outcome','dense_combat_reward',
                 '_entry_attempts','_resolve_combat','_resolve_noncombat_losses','build_weapon','initial_states'):
        assert getattr(type(a),name) is getattr(type(b),name)
    for seed in (0,31,199):
        x,ix=a.reset(seed);y,iy=b.reset(seed);np.testing.assert_array_equal(x,y)
        assert a.red==b.red and a.blue==b.blue and a.rng.bit_generator.state==b.rng.bit_generator.state
        assert type(a.fixed_policy) is type(b.fixed_policy)
        for key in ('red_nominal_anchor','blue_nominal_anchor','nominal_anchor_separation'):
            np.testing.assert_equal(ix[key],iy[key])


@pytest.mark.parametrize('key',list(R4_BASELINE))
def test_validator_rejects_wrong_r4_scale(key):
    c=deepcopy(env().config);c['reward'][key]=R4_BASELINE[key]
    with pytest.raises(ValueError,match=key):validate_config(c)


@pytest.mark.parametrize('section,key,value',[
    ('reward','tactical_aspect_max',.6),('reward','tactical_distance_max',4000.),
    ('reward','guide_reward',.002),('reward','guide_distance_min',4000.),
    ('weapon','range_max',800.),('action','command',{}),('scenario','team_size',5)])
def test_validator_rejects_non_r4_drift(section,key,value):
    c=deepcopy(env().config);c[section][key]=value
    with pytest.raises((ValueError,KeyError)):validate_config(c)


@pytest.mark.parametrize('side,level,azimuth',[
    (side,level,angle) for side in ('offense','threat') for level,angle in ((30,20.),(15,10.),(5,2.))])
def test_nonzero_offense_threat_levels_exact_double(side,level,azimuth):
    old,new=env('330'),env();red,blue=pair_states(azimuth=azimuth,threat=side=='threat')
    for e in (old,new):e.red=deepcopy(red)+[dead() for _ in range(3)];e.blue=deepcopy(blue)+[dead() for _ in range(3)]
    a,da=old.dense_combat_reward();b,db=new.dense_combat_reward()
    term='tactical_offense' if side=='offense' else 'tactical_defense'
    key=f'offense_reward_{level}deg' if side=='offense' else f'threat_penalty_{level}deg'
    assert da[term][0]!=0
    assert da[term][0]==pytest.approx(old.config['reward'][key]*.5,abs=1e-15)
    for t in ('tactical_offense','tactical_defense'):np.testing.assert_array_equal(db[t],2*da[t])
    np.testing.assert_array_equal(db['guide'],da['guide'])
    np.testing.assert_allclose(b,da['guide']+2*(da['tactical_offense']+da['tactical_defense']),rtol=0,atol=1e-16)


@pytest.mark.parametrize('kwargs',[dict(aspect=31.),dict(azimuth=31.),dict(elevation=31.),dict(distance=3000.),dict(distance=4000.)])
@pytest.mark.parametrize('threat',[False,True])
def test_frozen_geometry_gates(kwargs,threat):
    for version in ('330','331'):
        e=env(version);e.red,e.blue=pair_states(**kwargs,threat=threat)
        _,d=e.dense_combat_reward()
        assert not d['tactical_offense'].any() and not d['tactical_defense'].any()


def test_r3_active_and_max_min_multi_target_aggregation():
    a,b=env('330'),env()
    red,blue=pair_states(distance=2000.,azimuth=20.)
    _,blue15=pair_states(distance=2000.,azimuth=10.)
    _,blue5=pair_states(distance=2000.,azimuth=2.)
    for e in (a,b):e.red=deepcopy(red)+[dead() for _ in range(3)];e.blue=deepcopy(blue+blue15+blue5)+[dead()]
    _,x=a.dense_combat_reward();_,y=b.dense_combat_reward()
    assert x['guide'][0]==y['guide'][0] and x['guide'][0]==pytest.approx(.0005,abs=1e-15)
    assert x['tactical_offense'][0]==pytest.approx(.05,abs=1e-15)
    assert y['tactical_offense'][0]==2*x['tactical_offense'][0]
    # Reverse the teams to exercise the unchanged min threat aggregation.
    for e in (a,b):e.red,e.blue=e.blue,e.red
    _,x=a.dense_combat_reward();_,y=b.dense_combat_reward()
    np.testing.assert_array_equal(y['tactical_defense'],2*x['tactical_defense'])
    assert np.min(x['tactical_defense'])==pytest.approx(-.075,abs=1e-15)


def test_event_rewards_unchanged_and_mutual_draw_inherited():
    a,b=env('330'),env()
    for e in (a,b):
        e.red=[AircraftState(0,0,-3000,225,0,0),AircraftState(5001,0,-3000,225,0,0),AircraftState(0,0,0,225,0,0),dead()]
        e.blue=[AircraftState(500,0,-3000,225,theta=0,psi=np.pi),dead(),dead(),dead()];e.rng=FixedNormal(0)
    with patch.object(a,'_advance',return_value=np.zeros(4)),patch.object(b,'_advance',return_value=np.zeros(4)):
        x=a.step(np.zeros((4,3)));y=b.step(np.zeros((4,3)))
    assert x[2]==y[2]==True
    for k in ('r1','r2','r3'):np.testing.assert_array_equal(x[4][k+'_rewards'],y[4][k+'_rewards'])
    assert x[4]['r1_rewards'][2]==y[4]['r1_rewards'][2]==-10
    assert x[4]['r2_rewards'][1]==y[4]['r2_rewards'][1]==-10
    assert a.red==b.red and a.blue==b.blue
    for k in ('red_win','blue_win','draw','red_success','termination_reason','red_attack_kills','blue_attack_kills'):
        assert x[4][k]==y[4][k]
    assert y[4]['draw'] and y[4]['termination_reason']=='draw_mutual_destruction'


def test_static_audit_table():
    rows=static_table(env('330').config,env().config)
    assert len(rows)==15
    for row in rows:
        for term,scale in (('guide',1),('tactical_offense',2),('tactical_defense',2)):
            assert row[term]['ratio'] in (None,scale)


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
def test_algorithm_config_only_identity_changed(name):
    old=yaml.safe_load(Path(f'configs/{name}_4v4_v330.yaml').read_text())
    new=yaml.safe_load(Path(f'configs/{name}_4v4_v331.yaml').read_text())
    if 'environment' in new:
        assert new['environment']['environment_version']=='3.31'
        new['environment']['environment_version']='3.30'
    assert new==old
