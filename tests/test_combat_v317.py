"""Body axes apply to eligibility and the actual step sampler, not paper state."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.combat_v316 import CombatEnvironmentV316
from env.combat_v317 import CombatEnvironmentV317
from env.combat_v318 import CombatEnvironmentV318
from env.config import load_config,validate_config,environment_dimensions
from env.geometry import engagement_geometry
from env.models import AircraftState
from env.v313_geometry import combat_geometry_v313
from tools.audit_combat_v317_v318 import body_pair,geometry_cases,FixedNormal


def env(version='317',seed=31):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml');e.reset(seed);return e


def test_minimal_inheritance_and_no_old_geometry_or_controller_changes():
    old,new=env('316'),env()
    assert type(new) is CombatEnvironmentV317
    assert set(CombatEnvironmentV317.__dict__)-{'__module__','__doc__'}=={'environment_version','build_weapon','attempt_hit_for_pair','reset'}
    for name in ('__init__','step','_advance','_observations','_entry_attempts','dense_combat_reward','_resolve_combat','_resolve_noncombat_losses','_outcome'):
        assert getattr(CombatEnvironmentV317,name) is getattr(CombatEnvironmentV316,name)
    assert environment_dimensions(new.config)==(56,3,5)
    for section in set(old.config)-{'weapon','environment_version','scenario'}:assert old.config[section]==new.config[section]
    scenario=deepcopy(new.config['scenario']);scenario['center_radius']=4800.
    assert scenario==old.config['scenario']
    assert new.weapon.azimuth_noise_scale==old.weapon.ata_noise_scale
    assert new.weapon.elevation_noise_scale==old.weapon.ha_noise_scale
    assert new.weapon.effective_hit_distance==old.weapon.effective_hit_distance
    assert new.weapon.ammo_per_aircraft==6
    assert set(new.config['weapon'])=={'range_min','range_max','ammo_per_aircraft','effective_hit_distance','boresight_azimuth_max','boresight_elevation_max','azimuth_noise_scale','elevation_noise_scale'}


@pytest.mark.parametrize('name,a,b,expected',geometry_cases(),ids=[c[0] for c in geometry_cases()])
def test_required_body_geometry(name,a,b,expected):
    new,old=env(),env('316');g=engagement_geometry(a,b)
    assert new.weapon.qualifies(a,b,6)==expected
    if name=='pitch45_aligned':
        assert not old.weapon.qualifies(a,b,6)
        assert abs(g.boresight_azimuth_error)<1e-12 and abs(g.boresight_elevation_error)<1e-12
        assert g.off_boresight<3e-8
        assert np.rad2deg(combat_geometry_v313([a],[b])['ha'][0,0])==pytest.approx(45)
    if name=='body_az25_el25_rectangle':assert g.off_boresight>math.pi/6
    if name=='pitch60_LOS_minus22':assert np.rad2deg(g.off_boresight)==pytest.approx(82)


@pytest.mark.parametrize('case',['ammo_zero','attacker_dead','target_dead'])
def test_liveness_and_ammo(case):
    a,b=body_pair()
    if case=='attacker_dead':a.alive=False
    if case=='target_dead':b.alive=False
    assert not env().weapon.qualifies(a,b,0 if case=='ammo_zero' else 6)


@pytest.mark.parametrize('axis',['boresight_azimuth_max','boresight_elevation_max'])
@pytest.mark.parametrize('value',[None,'bad',True,0,-1,float('nan'),float('inf'),math.pi+.01,math.pi,.1])
def test_axis_limits_strictly_frozen(axis,value):
    cfg=load_config('configs/combat_environment_v317.yaml');cfg['weapon'][axis]=value
    with pytest.raises(ValueError):validate_config(cfg)


@pytest.mark.parametrize('field',['ata_max','ha_max','boresight_max','ata_noise_scale','ha_noise_scale','cooldown'])
def test_legacy_and_extra_weapon_fields_rejected(field):
    cfg=load_config('configs/combat_environment_v317.yaml');cfg['weapon'][field]=.5
    with pytest.raises(ValueError):validate_config(cfg)


@pytest.mark.parametrize('epsilon',[0,.5,-.5,2,-2])
@pytest.mark.parametrize('angles',[(0,0),(.4,.2),(-.4,.2),(.4,-.2)])
def test_shared_gaussian_signed_body_equation8(epsilon,angles):
    w=env().weapon;rng=FixedNormal(epsilon);d=w.range_max;az,el=angles
    threshold=math.pi*math.exp(-d/w.effective_hit_distance)
    assert w.attempt_hit(rng,d,az,el)==(abs(az+w.azimuth_noise_scale*epsilon)<=threshold and abs(el+w.elevation_noise_scale*epsilon)<=threshold)
    assert rng.calls==1


def test_probability_is_shared_interval_intersection_and_never_samples():
    w=env().weapon;d=w.range_max;az,el=.4,-.4
    threshold=math.pi*math.exp(-d/w.effective_hit_distance)
    lo=max((-threshold-az)/w.azimuth_noise_scale,(-threshold-el)/w.elevation_noise_scale)
    hi=min((threshold-az)/w.azimuth_noise_scale,(threshold-el)/w.elevation_noise_scale)
    expected=.5*(math.erf(hi/math.sqrt(2))-math.erf(lo/math.sqrt(2)))
    assert w.expected_hit_probability(d,az,el)==pytest.approx(expected)
    assert w.expected_hit_probability(d,10,-10)==0
    with patch.object(type(w),'expected_hit_probability',side_effect=AssertionError('diagnostic may not sample')):
        w.attempt_hit(FixedNormal(0),d,az,el)


@pytest.mark.parametrize('version',['317','318'])
def test_real_step_uses_body_inputs_and_one_draw(version):
    e=env(version);a,b=body_pair(pitch=45)
    dead=lambda:AircraftState(0,0,-3000,225,0,0,False)
    e.red=[a,*[dead() for _ in range(4)]];e.blue=[b,*[dead() for _ in range(4)]];e.rng=FixedNormal(4)
    original=type(e.weapon).attempt_hit
    with patch.object(type(e.weapon),'attempt_hit',autospec=True,side_effect=original) as hit:
        obs,_,term,trunc,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
        assert not term and not trunc and info['red_fire_attempts']==1
        assert hit.call_count==1 and e.rng.calls==1
        g=engagement_geometry(e.red[0],e.blue[0]);args=hit.call_args.args[2:]
        assert args==(g.distance,g.boresight_azimuth_error,g.boresight_elevation_error)
        paper=combat_geometry_v313([e.red[0]],[e.blue[0]])
        assert abs(args[2])<1e-12 and paper['ha'][0,0]==pytest.approx(math.pi/4)
        assert obs.shape==(5,56)
        # Enemy B1's HA is slot35: paper state did not change with weapon inputs.
        assert obs[0,35]==pytest.approx(.5) and obs[0,4]==0


@pytest.mark.parametrize('version',['317','318'])
@pytest.mark.parametrize('seed',[1,31,10000000])
def test_scenario_observation_and_control_exact(version,seed):
    old,new=env('316',seed),env(version,seed)
    _,old_info=old.reset(seed);_,new_info=new.reset(seed)
    assert old_info['nominal_anchor_separation']==9600.
    assert new_info['nominal_anchor_separation']==8000.
    assert new.config['scenario']['formation_offsets']==[-400.,-200.,0.,200.,400.]
    alpha=new_info['radial_angle'];direction=np.array([np.cos(alpha),np.sin(alpha)])
    for side,sign in [('red',-1),('blue',1)]:
        np.testing.assert_array_equal(new_info[f'{side}_nominal_anchor'],sign*4000.*direction)
        expected=np.array([s.as_array() for s in getattr(old,side)])
        expected[:,:2]+=sign*(4000.-4800.)*direction
        actual=np.array([s.as_array() for s in getattr(new,side)])
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
        assert np.all(np.linalg.norm(actual[:,:2],axis=1)<new.arena_radius)
    # Compare unchanged observation/control semantics on identical states.
    new.red,new.blue=deepcopy(old.red),deepcopy(old.blue)
    np.testing.assert_array_equal(old._observations(),new._observations())
    actions=np.tile([.4,.5,.8],(5,1))
    for side in ('red','blue'):
        np.testing.assert_array_equal([s.as_array() for s in getattr(old,side)],[s.as_array() for s in getattr(new,side)])
        np.testing.assert_array_equal(old._advance(getattr(old,side),actions),new._advance(getattr(new,side),actions))
        np.testing.assert_array_equal([s.as_array() for s in getattr(old,side)],[s.as_array() for s in getattr(new,side)])
    np.testing.assert_array_equal(old._observations(),new._observations())
    assert type(old.fixed_policy) is type(new.fixed_policy)


@pytest.mark.parametrize('version',['317','318'])
@pytest.mark.parametrize('distance',[100.,2500.,5000.])
def test_static_reward_and_safety_unchanged(version,distance):
    old,new=env('316'),env(version)
    old.red=[AircraftState(30*i,0,-3000,225,.1*i,0) for i in range(5)]
    old.blue=[AircraftState(distance+30*i,0,-3000-50*i,225,0,math.pi) for i in range(5)]
    new.red,new.blue=deepcopy(old.red),deepcopy(old.blue)
    for a,b in zip(old.dense_combat_reward(),new.dense_combat_reward()):
        if isinstance(a,dict):
            for key in a:np.testing.assert_array_equal(a[key],b[key])
        else:np.testing.assert_array_equal(a,b)
    with patch.object(old,'_advance',return_value=np.zeros(5)),patch.object(new,'_advance',return_value=np.zeros(5)),patch.object(old,'_entry_attempts',return_value=[(0,0,True)]),patch.object(new,'_entry_attempts',return_value=[(0,0,True)]):
        _,ro,to,xo,io=old.step(np.zeros((5,3)),np.zeros((5,3)));_,rn,tn,xn,inn=new.step(np.zeros((5,3)),np.zeros((5,3)))
    np.testing.assert_array_equal(ro,rn);assert (to,xo)==(tn,xn)
    for key in ['r1_rewards','r2_rewards','r3_rewards','r4_rewards','safety_rewards','paper_rewards','extended_rewards']:
        np.testing.assert_array_equal(io[key],inn[key])


@pytest.mark.parametrize('version',['317','318'])
@pytest.mark.parametrize('stem',['mappo','rmappo','ea_mappo','stea_mappo'])
def test_algorithm_configs_byte_identical(version,stem):
    assert Path(f'configs/{stem}_5v5_v{version}.yaml').read_bytes()==Path(f'configs/{stem}_5v5_v316.yaml').read_bytes()


@pytest.mark.parametrize('version',['317','318'])
def test_recording_dispatch(version):
    from tools.combat_visualization import RecordingCombatEnv
    e=RecordingCombatEnv(f'configs/combat_environment_v{version}.yaml')
    assert isinstance(e,CombatEnvironmentV317 if version=='317' else CombatEnvironmentV318)
    assert e.reset(1)[0].shape==(5,56)


@pytest.mark.parametrize('version,expected',[('316',4000.),('317',800.),('318',800.)])
def test_formal_range_version_isolation(version,expected):
    e=env(version)
    assert e.weapon.range_max==expected and e.weapon.range_min==0.
    cfg=deepcopy(e.config);cfg['weapon']['range_max']=800. if version=='316' else 4000.
    with pytest.raises(ValueError):validate_config(cfg)


@pytest.mark.parametrize('version,expected',[('316',4800.),('317',4000.),('318',4000.)])
def test_formal_center_radius_version_isolation(version,expected):
    e=env(version)
    assert e.config['scenario']['center_radius']==expected
    cfg=deepcopy(e.config);cfg['scenario']['center_radius']=4000. if version=='316' else 4800.
    with pytest.raises(ValueError):validate_config(cfg)
