"""Paper-scale anchors, chord geometry, RNG isolation and inherited protocols."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.combat_v37 import CombatEnvironmentV37
from env.combat_v38 import CombatEnvironmentV38
from env.config import load_config,validate_config
from env.v38_scenario import SPAWN_EPSILON,random_combat_states_v38
from env.v36_geometry import combat_geometry
from tools._prepare_v38 import capture37

ROOT=Path(__file__).resolve().parents[1]
def cfg():return load_config(ROOT/'configs/combat_environment_v38.yaml')

@pytest.mark.parametrize('seed',[0,1,2,3,31,71,101,373,10000000,40000000])
def test_anchor_chord_states_visibility_heading_and_exact_draw_order(seed):
    e=MultiUAVCombatEnv(cfg());obs,info=e.reset(seed)
    old=MultiUAVCombatEnv(ROOT/'configs/combat_environment_v37.yaml');old.reset(seed)
    assert type(e) is CombatEnvironmentV38 and obs.shape==(5,66)
    assert e.arena_radius==5000 and 0<SPAWN_EPSILON<=1e-6
    alpha=info['radial_angle'];assert -np.pi<=alpha<=np.pi
    u=np.array([np.cos(alpha),np.sin(alpha)]);transverse=np.array([-np.sin(alpha),np.cos(alpha)])
    np.testing.assert_array_equal(info['red_nominal_anchor'],-5000*u)
    np.testing.assert_array_equal(info['blue_nominal_anchor'],5000*u)
    assert info['nominal_anchor_separation']==10000
    assert np.linalg.norm(info['blue_nominal_anchor']-info['red_nominal_anchor'])==pytest.approx(10000,abs=5e-12)
    for side,sign,nominal in (('red',-1,alpha),('blue',1,alpha+np.pi)):
        states=getattr(e,side);previous=getattr(old,side)
        positions=np.array([[s.x,s.y] for s in states]);laterals=positions@transverse
        assert (np.linalg.norm(positions,axis=1)<=5000).all()
        np.testing.assert_allclose(np.linalg.norm(positions,axis=1),5000-SPAWN_EPSILON,rtol=0,atol=2e-12)
        assert (laterals>=-2500).all() and (laterals<=2500).all()
        assert np.diff(np.sort(laterals)).min()>=400-1e-9
        np.testing.assert_allclose(positions@u,sign*np.sqrt((5000-SPAWN_EPSILON)**2-laterals**2),rtol=0,atol=3e-12)
        np.testing.assert_allclose(laterals,np.array([[s.x,s.y] for s in previous])@transverse,rtol=0,atol=1e-12)
        for s,p in zip(states,previous):
            # Altitude, speed, pitch, heading exactly retain the v3.7 draws.
            np.testing.assert_array_equal(s.as_array()[2:],p.as_array()[2:])
            assert 2900<=s.altitude<=3100 and 215<=s.v<=235 and s.theta==0
            difference=abs(np.arctan2(np.sin(s.psi-nominal),np.cos(s.psi-nominal)))
            assert difference<=cfg()['scenario']['heading_perturbation_max']+1e-12
    assert e.rng.bit_generator.state==old.rng.bit_generator.state
    assert e.fixed_policy.rng.bit_generator.state==old.fixed_policy.rng.bit_generator.state
    for side in ('red','blue'):
        assert info[f'initial_{side}_visible_pairs']==info[f'initial_{side}_fire_eligible_pairs']==0
    d=combat_geometry(e.red,e.blue)['distance']
    assert d.min()>4000 and info['minimum_inter_team_distance']==d.min()
    assert info['mean_inter_team_distance']==d.mean() and info['maximum_inter_team_distance']==d.max()
    assert not np.allclose(d,10000,rtol=0,atol=1e-5)

def test_many_fixed_seeds_no_initial_boundary_loss_or_return_and_independent_offsets():
    e=MultiUAVCombatEnv(cfg());angles=[]
    for seed in range(100):
        _,info=e.reset(seed);angles.append(info['radial_angle'])
        t=np.array([-np.sin(info['radial_angle']),np.cos(info['radial_angle'])])
        red=np.array([[s.x,s.y] for s in e.red])@t;blue=np.array([[s.x,s.y] for s in e.blue])@t
        assert not np.array_equal(red,blue)
        assert all(s.alive for s in e.red+e.blue)
        _,_,_,_,step_info=e.step(np.zeros((5,3)))
        assert step_info['red_boundary_exits']==step_info['blue_boundary_returns']==0
        assert step_info['red_ground_losses']==step_info['blue_ground_losses']==0
    assert len(set(angles))==100

@pytest.mark.parametrize('seed',[1,31,10000000])
def test_same_seed_full_reset_reproducible(seed):
    e=MultiUAVCombatEnv(cfg());a,ia=e.reset(seed)
    before=np.array([s.as_array() for s in e.red+e.blue]);rng=copy.deepcopy(e.rng.bit_generator.state)
    e.step(np.zeros((5,3)));b,ib=e.reset(seed)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(before,[s.as_array() for s in e.red+e.blue])
    assert rng==e.rng.bit_generator.state
    for key in ia:
        if isinstance(ia[key],np.ndarray):np.testing.assert_array_equal(ia[key],ib[key])
        else:assert ia[key]==ib[key]

def test_inherited_parameters_methods_algorithm_bytes_and_old_reference():
    old=load_config(ROOT/'configs/combat_environment_v37.yaml');new=cfg()
    for key in old.keys()-{'environment_version','scenario'}:assert old[key]==new[key]
    assert new['scenario']=={k:v for k,v in old['scenario'].items() if k!='center_radius'}
    for method in ('step','build_weapon','attempt_hit_for_pair','reward_aliases','_entry_attempts',
            '_resolve_combat','_resolve_noncombat_losses','_advance','_outcome','dense_combat_reward'):
        assert getattr(CombatEnvironmentV38,method) is getattr(CombatEnvironmentV37,method)
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        assert (ROOT/f'configs/{name}_5v5_v38.yaml').read_bytes()==(ROOT/f'configs/{name}_5v5_v37.yaml').read_bytes()
    assert capture37()==json.loads((ROOT/'tests/v38_v37_reference.json').read_text())

@pytest.mark.parametrize('section,key,value',[('scenario','center_radius',5000),('scenario','lateral_max',2400),
    ('scenario','heading_perturbation_max',0),('scenario','speed_center',250),('scenario','altitude_center',3200),
    ('scenario','minimum_same_team_lateral_separation',500),('arena','radius',5500),('arena','altitude_max',6500),
    ('sensor','range_max',5000),('weapon','range_max',4000),('weapon','ata_noise_scale',1.),('reward','guide_reward',.002)])
def test_validator_rejects_changes(section,key,value):
    c=cfg();c[section][key]=value
    with pytest.raises(ValueError):validate_config(c)

@pytest.mark.parametrize('seed',[1,31,10000000])
@pytest.mark.parametrize('mode',['ZERO','PURSUIT'])
def test_contact_timing_is_delayed_without_absolute_time_threshold(seed,mode):
    from tools.compare_v38_initialization import trial
    a,b=trial('37',mode,seed),trial('38',mode,seed)
    assert b['initial_minimum_inter_team_distance']>a['initial_minimum_inter_team_distance']
    assert b['first_detection_step']>a['first_detection_step']
    assert b['first_fire_window_step']>a['first_fire_window_step']
    assert b['first_attempt_step']>=b['first_fire_window_step']
