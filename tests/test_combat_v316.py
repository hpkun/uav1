"""Version-isolated geometry and shared-Gaussian Equation (8) regressions."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.combat_v315 import CombatEnvironmentV315
from env.combat_v316 import CombatEnvironmentV316
from env.config import load_config, validate_config, environment_dimensions
from env.geometry import engagement_geometry
from env.models import AircraftState
from env.v313_geometry import combat_geometry_v313
from tools.audit_combat_v316_weapon import FixedNormal, geometry_cases


def env(version='316'):
    result=MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml')
    result.reset(31)
    return result


def test_minimal_inheritance_and_frozen_configuration():
    new,old=env(),env('315')
    assert type(new) is CombatEnvironmentV316
    assert set(CombatEnvironmentV316.__dict__)-{'__module__','__doc__'}=={'environment_version','build_weapon'}
    for name in ('__init__','step','reset','_advance','_observations','attempt_hit_for_pair','dense_combat_reward','_entry_attempts','_resolve_combat','_resolve_noncombat_losses','_outcome'):
        assert getattr(CombatEnvironmentV316,name) is getattr(CombatEnvironmentV315,name)
    cfg=deepcopy(new.config);cfg['environment_version']='3.15';assert cfg['weapon'].pop('boresight_max')==math.pi/6
    assert cfg==old.config and environment_dimensions(new.config)==(56,3,5)
    assert type(new.dynamics) is type(old.dynamics) and type(new.integrator) is type(old.integrator)
    for name in ('effective_hit_distance','ata_noise_scale','ha_noise_scale','ammo_per_aircraft'):
        assert getattr(new.weapon,name)==getattr(old.weapon,name)


@pytest.mark.parametrize('value',[None,'bad',float('nan'),float('inf'),float('-inf'),0,-1,math.pi+.001,math.pi,.1,True])
def test_strict_boresight_validation(value):
    cfg=load_config('configs/combat_environment_v316.yaml');cfg['weapon']['boresight_max']=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_missing_boresight_and_extra_config_rejected():
    cfg=load_config('configs/combat_environment_v316.yaml');cfg['weapon'].pop('boresight_max')
    with pytest.raises(ValueError):validate_config(cfg)
    cfg=load_config('configs/combat_environment_v316.yaml');cfg['weapon']['new_delay']=1
    with pytest.raises(ValueError):validate_config(cfg)


@pytest.mark.parametrize('name,attacker,target,expected',geometry_cases(),ids=[x[0] for x in geometry_cases()])
def test_geometry_gates(name,attacker,target,expected):
    old,new=env('315'),env()
    assert new.weapon.qualifies(attacker,target,6)==expected
    g=combat_geometry_v313([attacker],[target]);bore=engagement_geometry(attacker,target).off_boresight
    old_eligible=old.weapon.qualifies(attacker,target,6)
    assert expected==(old_eligible and bore<=math.pi/6+1e-12)
    if name=='replay_red_up_target_below':
        assert old_eligible and np.rad2deg(bore)==pytest.approx(82,abs=.05)
        assert g['distance'][0,0]<4000 and np.rad2deg(g['ata'][0,0])<5 and np.rad2deg(g['ha'][0,0])==pytest.approx(22)
    if name=='replay_blue_up_target_above':assert np.rad2deg(bore)==pytest.approx(4,abs=.05)
    if name in ('ata_outside_boresight_inside','ha_outside_boresight_inside','distance_outside'):
        assert bore<=math.pi/6+1e-12 and not old_eligible


@pytest.mark.parametrize('case',['attacker_dead','target_dead','ammo_zero'])
def test_liveness_and_ammo(case):
    _,a,b,_=geometry_cases()[0]
    if case=='attacker_dead':a.alive=False
    if case=='target_dead':b.alive=False
    assert not env().weapon.qualifies(a,b,0 if case=='ammo_zero' else 6)


@pytest.mark.parametrize('epsilon',[0,.5,-.5,2,-2])
@pytest.mark.parametrize('distance,ata,ha',[(0,0,0),(3000,0,0),(3990,.03,.4),(4000,.5,.5),(4000,.4,.03)])
def test_equation8_exact_fixed_epsilon_one_shared_draw(epsilon,distance,ata,ha):
    w=env().weapon;rng=FixedNormal(epsilon)
    threshold=math.pi*math.exp(-distance/w.effective_hit_distance)
    expected=abs(ata+w.ata_noise_scale*epsilon)<=threshold and abs(ha+w.ha_noise_scale*epsilon)<=threshold
    assert w.attempt_hit(rng,distance,ata,ha)==expected and rng.calls==1


def test_large_negative_epsilon_old_single_sided_error_retained_only_in_old_version():
    old,new=env('315').weapon,env().weapon
    assert old.attempt_hit(FixedNormal(-2),4000,.5,.5)
    assert not new.attempt_hit(FixedNormal(-2),4000,.5,.5)


def test_analytic_shared_interval_intersection_and_disjoint_case():
    w=env().weapon
    d=4000.;threshold=math.pi*math.exp(-d/w.effective_hit_distance);c=w.ata_noise_scale
    expected=math.erf(threshold/c/math.sqrt(2))
    assert w.expected_hit_probability(d,0,0)==pytest.approx(expected)
    assert w.expected_hit_probability(d,10,-10)==0
    # Unequal scales ensure this is an intersection, not a product of marginals.
    varied=replace(w,ata_noise_scale=1.,ha_noise_scale=2.)
    lo=max((-threshold-.2),(-threshold-.4)/2);hi=min((threshold-.2),(threshold-.4)/2)
    want=.5*(math.erf(hi/math.sqrt(2))-math.erf(lo/math.sqrt(2)))
    assert varied.expected_hit_probability(d,.2,.4)==pytest.approx(want)
    with patch.object(type(w),'expected_hit_probability',side_effect=AssertionError('analytic must not sample')):
        w.attempt_hit(FixedNormal(0),d,.2,.4)


def test_pair_hit_passes_paper_geometry_not_true_boresight():
    e=env();_,a,b,_=geometry_cases()[7];g=combat_geometry_v313([a],[b])
    with patch.object(type(e.weapon),'attempt_hit',autospec=True,return_value=False) as hit:
        assert not e.attempt_hit_for_pair(a,b)
    assert hit.call_count==1
    assert hit.call_args.args[2:]==tuple(g[k][0,0] for k in ('distance','ata','ha'))
    assert g['ha'][0,0]!=engagement_geometry(a,b).off_boresight


def targeting_env(version):
    e=env(version)
    own=AircraftState(0,0,-3000,225,math.pi/3,0)
    bad=AircraftState(1000*math.cos(math.radians(22)),0,-3000+1000*math.sin(math.radians(22)),225,0,math.pi)
    good=AircraftState(3000*math.cos(math.pi/6),0,-4500,225,0,math.pi)
    dead=replace(own,alive=False)
    e.red=[own,*[replace(dead) for _ in range(4)]];e.blue=[bad,good,*[replace(dead) for _ in range(3)]]
    e.rng=FixedNormal(0)
    return e


def test_nearest_qualified_target_and_repeat_fire_cadence():
    old,new=targeting_env('315'),targeting_env('316')
    assert old._entry_attempts(old.red,old.blue,None,'red')[0][1]==0
    for _ in range(3):
        shots=new._entry_attempts(new.red,new.blue,None,'red')
        assert len(shots)==1 and shots[0][:2]==(0,1)
    assert new.red_ammo[0]==3 and new.rng.calls==3
    for _ in range(3):new._entry_attempts(new.red,new.blue,None,'red')
    assert not new._entry_attempts(new.red,new.blue,None,'red')
    assert new.red_ammo[0]==0 and new.rng.calls==6


def test_both_sides_use_same_new_eligibility():
    e=targeting_env('316')
    a,b=deepcopy(e.red),deepcopy(e.blue)
    red=e._entry_attempts(a,b,None,'red');blue=e._entry_attempts(deepcopy(a),deepcopy(b),None,'blue')
    assert red==blue and red[0][1]==1


@pytest.mark.parametrize('seed',[1,31,10000000])
def test_observations_scenario_and_flight_exact(seed):
    old,new=env('315'),env();oo,_=old.reset(seed);on,_=new.reset(seed)
    np.testing.assert_array_equal(oo,on)
    actions=np.tile([.4,.5,.8],(5,1))
    for side in ('red','blue'):
        np.testing.assert_array_equal([s.as_array() for s in getattr(old,side)],[s.as_array() for s in getattr(new,side)])
        op=old._advance(getattr(old,side),actions);np_=new._advance(getattr(new,side),actions)
        np.testing.assert_array_equal(op,np_)
        np.testing.assert_array_equal([s.as_array() for s in getattr(old,side)],[s.as_array() for s in getattr(new,side)])
    np.testing.assert_array_equal(old._observations(),new._observations())
    assert not new._observations()[:,4].any()


@pytest.mark.parametrize('distance',[100.,2500.,5000.])
def test_static_rewards_full_decomposition_unchanged(distance):
    old,new=env('315'),env()
    old.red=[AircraftState(i*30,0,-3000,225,.1*i,0) for i in range(5)]
    old.blue=[AircraftState(distance+i*30,0,-3000-50*i,225,0,math.pi) for i in range(5)]
    new.red,new.blue=deepcopy(old.red),deepcopy(old.blue)
    for a,b in zip(old.dense_combat_reward(),new.dense_combat_reward()):
        if isinstance(a,dict):
            for key in a:np.testing.assert_array_equal(a[key],b[key])
        else:np.testing.assert_array_equal(a,b)
    # Hold combat outcomes fixed: changed weapons can legitimately change kills,
    # but reward formulas for identical states/outcomes must remain identical.
    with patch.object(old,'_advance',return_value=np.zeros(5)),patch.object(new,'_advance',return_value=np.zeros(5)),patch.object(old,'_entry_attempts',return_value=[(0,0,True)]),patch.object(new,'_entry_attempts',return_value=[(0,0,True)]):
        _,ro,_,_,io=old.step(np.zeros((5,3)),np.zeros((5,3)));_,rn,_,_,inn=new.step(np.zeros((5,3)),np.zeros((5,3)))
    np.testing.assert_array_equal(ro,rn)
    for key in ['r1_rewards','r2_rewards','r3_rewards','r4_rewards','safety_rewards','paper_rewards','extended_rewards']:
        np.testing.assert_array_equal(io[key],inn[key])


@pytest.mark.parametrize('stem',['mappo','rmappo','ea_mappo','stea_mappo'])
def test_algorithm_configs_byte_identical(stem):
    assert Path(f'configs/{stem}_5v5_v316.yaml').read_bytes()==Path(f'configs/{stem}_5v5_v315.yaml').read_bytes()


def test_recording_dispatch():
    from tools.combat_visualization import RecordingCombatEnv
    e=RecordingCombatEnv('configs/combat_environment_v316.yaml')
    assert isinstance(e,CombatEnvironmentV316) and e.reset(1)[0].shape==(5,56)
