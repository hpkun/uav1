"""Single-parameter distance recalibration, normalized probability and frozen mechanisms."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v334 import CombatEnvironmentV334
from env.combat_v335 import CombatEnvironmentV335
from env.config import validate_config,environment_dimensions
from env.v335_config import EFFECTIVE_HIT_DISTANCE,BASELINE_EFFECTIVE_HIT_DISTANCE
from env.v317_weapon import BodyFrameGaussianWeapon
from tests.test_combat_v332 import state


def env(v='335'):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{v}.yaml');e.reset(31);return e


def test_identity_exact_only_two_config_differences_and_inherited_methods():
    e=env();old=env('334')
    assert CombatEnvironmentV335.__bases__==(CombatEnvironmentV334,)
    assert type(e) is CombatEnvironmentV335 and environment_dimensions(e.config)==(45,3,4)
    assert EFFECTIVE_HIT_DISTANCE==558.1106265512473==1000/math.log(6)
    c=deepcopy(e.config);c['environment_version']='3.34';c['weapon']['effective_hit_distance']=BASELINE_EFFECTIVE_HIT_DISTANCE
    assert c==old.config
    for name in ('__init__','initial_states','reset','step','_advance','dense_combat_reward','build_weapon','_entry_attempts','_resolve_combat','_outcome','_observations'):
        assert getattr(CombatEnvironmentV335,name) is getattr(CombatEnvironmentV334,name)
    assert type(e.weapon) is type(old.weapon) is BodyFrameGaussianWeapon
    assert replace(e.weapon,effective_hit_distance=BASELINE_EFFECTIVE_HIT_DISTANCE)==old.weapon
    assert e.weapon.azimuth_noise_scale==e.weapon.elevation_noise_scale==0.9984711359155588
    assert e.weapon.range_max==1000 and e.config['scenario']['center_radius']==4000
    assert type(e.fixed_policy) is type(old.fixed_policy)


@pytest.mark.parametrize('distance,expected',[(0,.9983470796),(250,.9556082264),(500,.8010377188),(750,.5882005801),(1000,.4)])
def test_perfect_alignment_theoretical_probability(distance,expected):
    w=env().weapon;p=w.expected_hit_probability(distance,0,0)
    assert p==pytest.approx(expected,abs=5e-11,rel=0)
    assert p==pytest.approx(math.erf(math.pi*math.exp(-distance/w.effective_hit_distance)/w.azimuth_noise_scale/math.sqrt(2)),abs=1e-15)
    if distance==1000:assert p==.4


@pytest.mark.parametrize('distance,expected',[(250,.9950928405345261),(500,.9880982205788561),(750,.975461645365576),(1000,.9556082263649425)])
def test_old_v334_probability_retained(distance,expected):
    assert env('334').weapon.expected_hit_probability(distance,0,0)==pytest.approx(expected,abs=1e-15)


@pytest.mark.parametrize('rho',[0,.25,.5,.75,1])
@pytest.mark.parametrize('azimuth,elevation',[(0,0),(.2,.1),(.4,-.4)])
def test_normalized_threshold_and_probability_invariance(rho,azimuth,elevation):
    old=replace(env('334').weapon,range_max=4000,effective_hit_distance=4000/math.log(6))
    new=env().weapon
    assert math.pi*math.exp(-rho*4000/old.effective_hit_distance)==math.pi*math.exp(-rho*1000/new.effective_hit_distance)
    assert old.expected_hit_probability(rho*4000,azimuth,elevation)==new.expected_hit_probability(rho*1000,azimuth,elevation)


@pytest.mark.parametrize('distance,az,el,aa,ammo,expected',[
    (0,0,0,0,6,True),(999,0,0,180,6,True),(1000,0,0,180,6,True),
    (1001,0,0,0,6,False),(500,31,0,0,6,False),(500,0,31,0,6,False),
    (500,30,0,0,6,True),(500,0,30,0,6,True),(500,0,0,0,0,False)])
def test_eligibility_identical_aa_does_not_gate(distance,az,el,aa,ammo,expected):
    a=state(psi=-math.radians(az),theta=math.radians(el));b=state(distance,psi=math.radians(aa))
    assert env().weapon.qualifies(a,b,ammo) is expected
    assert env('334').weapon.qualifies(a,b,ammo) is expected


@pytest.mark.parametrize('epsilon',[0,.25,-.25,1,-1,3,-3])
def test_actual_hit_uses_one_shared_epsilon_and_unchanged_formula(epsilon):
    class Noise:
        calls=0
        def normal(self,mean,std):
            assert mean==0 and std==1;self.calls+=1;return epsilon
    w=env().weapon;az=.2;el=-.3;r=Noise();d=750
    result=w.attempt_hit(r,d,az,el);t=math.pi*math.exp(-d/w.effective_hit_distance)
    assert r.calls==1
    assert result==(abs(az+w.azimuth_noise_scale*epsilon)<=t and abs(el+w.elevation_noise_scale*epsilon)<=t)
    old=env('334').weapon
    assert old.attempt_hit(Noise(),d*4,az,el)==result


@pytest.mark.parametrize('section,key,value',[
    ('weapon','effective_hit_distance',2232.442506204989),('weapon','effective_hit_distance',558.1),
    ('weapon','azimuth_noise_scale',1),('weapon','elevation_noise_scale',1),('weapon','range_max',4000),
    ('weapon','ammo_per_aircraft',7),('fire_control','cooldown_seconds',2),
    ('reward','offense_reward_5deg',.2),('reward','guide_distance_min',4000),
    ('reward','tactical_distance_max',4000),('reward','tactical_aspect_max',math.pi/2),
    ('scenario','center_radius',4800),('scenario','formation_local_offsets',[[300,0],[-100,250],[-100,0],[-100,-250]]),
    ('action','command',{}),('sensor','range_max',9000),('observation','speed_scale',50),
    ('aircraft','v_max',350),('arena','radius',6000),('blue_policy','desired_speed',250)])
def test_strict_validator_rejects_drift(section,key,value):
    c=deepcopy(env().config);c[section][key]=value
    with pytest.raises(ValueError):validate_config(c)


def test_same_seed_200_resets_and_reward_motion_transition_without_hits():
    a,b=env('334'),env()
    for seed in range(200):
        x,_=a.reset(seed);y,_=b.reset(seed)
        assert a.red==b.red and a.blue==b.blue
        np.testing.assert_array_equal(x,y)
    # Remove hit draws in this comparison only: differing hit outcomes are the experimental variable.
    a.reset(31);b.reset(31)
    with patch.object(a,'_entry_attempts',return_value=[]),patch.object(b,'_entry_attempts',return_value=[]):
        for step in range(160):
            actions=np.tile([.02,.01,0],(4,1))
            x=a.step(actions);y=b.step(actions)
            assert a.red==b.red and a.blue==b.blue and x[2:4]==y[2:4]
            np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
            for i in range(1,5):np.testing.assert_array_equal(x[4][f'r{i}_rewards'],y[4][f'r{i}_rewards'])
            for side in ('red','blue'):
                np.testing.assert_array_equal(getattr(a,side+'_ammo'),getattr(b,side+'_ammo'))
            if x[2] or x[3]:break


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
def test_formal_algorithm_config_frozen(name):
    a=yaml.safe_load(Path(f'configs/{name}_4v4_v334.yaml').read_text());b=yaml.safe_load(Path(f'configs/{name}_4v4_v335.yaml').read_text())
    if 'environment' in b:b['environment']['environment_version']='3.34'
    assert a==b


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
@pytest.mark.parametrize('previous',['3.32','3.33','3.34'])
def test_historical_checkpoint_rejected(name,previous):
    ec=env().config;cfg=yaml.safe_load(Path(f'configs/{name}_4v4_v335.yaml').read_text())
    state=dict(algorithm={'mappo':'MAPPO','madsac':'madsac','maac':'MAAC'}[name],implementation_version=5 if name == 'madsac' else 3,
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
        with pytest.raises(RuntimeError,match='environment_version'):check(state,ec,cfg)


@pytest.mark.parametrize('version',['334','335'])
@pytest.mark.parametrize('distance',[500,1000])
def test_expected_angle_probability_never_increases(version,distance):
    w=env(version).weapon
    probabilities=[w.expected_hit_probability(distance,math.radians(angle),math.radians(angle)) for angle in (0,10,20,30)]
    assert all(a>b for a,b in zip(probabilities,probabilities[1:]))
