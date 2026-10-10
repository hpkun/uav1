"""Scaled symmetric hardware, exact tactical geometry and legacy hook isolation."""
import copy,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config,validate_config
from env.models import AircraftState
from env.sensor import sensor_geometry
from env.v36_geometry import combat_geometry
from env.v36_reward import tactical_rewards
from env.v36_weapon import MADSACStyleFiniteAmmoWeapon
from env.combat_v35 import CombatEnvironmentV35
from env.combat_v36 import CombatEnvironmentV36
from tools._prepare_v36 import capture35

ROOT=Path(__file__).resolve().parents[1]
def cfg():return load_config(ROOT/'configs/combat_environment_v36.yaml')
def state(x=0,y=0,z=-3000,psi=0,theta=0,alive=True):return AircraftState(x,y,z,225,theta,psi,alive)
def weapon():return MADSACStyleFiniteAmmoWeapon(**cfg()['weapon'])
def pair(d=1500,ata=0,ha=0,aa=0):
    return state(psi=-np.deg2rad(ata),theta=-np.deg2rad(ha)),state(d,psi=-np.deg2rad(aa))
def rewards(red,blue):
    c=cfg();return tactical_rewards(red,blue,c['sensor'],c['weapon'],c['reward'])

@pytest.mark.parametrize('d,ata,ha,yes',[(1999,0,0,True),(2000,0,0,True),(2001,0,0,False),(1500,30,0,True),(1500,30.1,0,False),(1500,0,30,True),(1500,0,30.1,False),(1500,10,31,False),(1500,31,10,False)])
def test_weapon_boundaries(d,ata,ha,yes):
    a,b=pair(d,ata,ha);assert weapon().qualifies(a,b,6)==yes

@pytest.mark.parametrize('ammo,dead',[(0,None),(6,'attacker'),(6,'target')])
def test_dead_or_empty_ammo_rejected(ammo,dead):
    a,b=pair()
    if dead=='attacker':a.alive=False
    if dead=='target':b.alive=False
    assert not weapon().qualifies(a,b,ammo)

@pytest.mark.parametrize('target_psi,aa',[(0,0),(np.pi,np.pi),(np.pi/2,np.pi/2)])
def test_rear_headon_crossing_aspect(target_psi,aa):
    a,b=pair();b.psi=target_psi;g=combat_geometry([a],[b])
    assert g['ata'][0,0]==0 and g['aa'][0,0]==pytest.approx(aa)
    assert weapon().qualifies(a,b,6) # AA never gates fire.

def test_ned_elevation_pitch_alignment_vertical_and_wrap():
    a,b=pair(1000);b.z=-4000;g=combat_geometry([a],[b])
    assert g['los_elevation'][0,0]==pytest.approx(np.pi/4)
    a.theta=np.pi/4;assert combat_geometry([a],[b])['ha'][0,0]==pytest.approx(0)
    b.z=-2000;assert combat_geometry([a],[b])['los_elevation'][0,0]<0
    b.x=0;assert all(np.isfinite(v).all() for v in combat_geometry([a],[b]).values())
    a.psi=np.pi-.01;b=state(-1000,-10);assert combat_geometry([a],[b])['ata'][0,0]<.021
    b=copy.deepcopy(a);g=combat_geometry([a],[b]);assert g['ata'][0,0]==g['ha'][0,0]==0

@pytest.mark.parametrize('ata,ha,expected',[(20,20,.001),(31,20,0),(20,31,0),(120,0,0)])
def test_guide(ata,ha,expected):
    a,b=pair(3000,ata,ha);assert rewards([a],[b])['guide'][0]==expected

def test_guide_visibility_distance_and_no_target_sum():
    a,b=pair(3000,20,20)
    assert rewards([a],[b,copy.deepcopy(b)])['guide'][0]==.001
    b.x=4001;assert rewards([a],[b])['guide'][0]==0
    b.x=2000;assert rewards([a],[b])['guide'][0]==0
    b.alive=False;assert rewards([a],[b])['guide'][0]==0

@pytest.mark.parametrize('ata,ha,aa,value',[(25,25,20,.01),(10,10,20,.02),(4,4,20,.1),(4,4,40,0),(5,5,30,.1),(15,15,30,.02),(30,30,30,.01)])
def test_offensive_levels(ata,ha,aa,value):
    a,b=pair(1500,ata,ha,aa);assert rewards([a],[b])['tactical_offense'][0]==value

@pytest.mark.parametrize('angle,value',[(25,-.015),(10,-.025),(4,-.15)])
def test_defensive_levels_without_observation_visibility(angle,value):
    b,a=pair(1500,angle,angle,20);b.x=-1500;a.x=0
    assert not sensor_geometry([a],[b],cfg()['sensor'])[0].any()
    r=rewards([a],[b]);assert r['tactical_defense'][0]==value and r['guide'][0]==r['tactical_offense'][0]==0

def test_coexisting_offense_defense_and_bounded_multi_target():
    a=state();ahead=state(1500);behind=state(-1500)
    r=rewards([a],[ahead,copy.deepcopy(ahead),behind,copy.deepcopy(behind)])
    assert r['tactical_offense'][0]==.1 and r['tactical_defense'][0]==-.15
    assert r['tactical'][0]==pytest.approx(-.05)
    ahead.alive=behind.alive=False;assert not rewards([a],[ahead,behind])['tactical'].any()
    a.alive=False;assert not rewards([a],[state(1500)])['tactical'].any()

def test_strongest_different_offense_and_threat_selected():
    a=state(theta=-np.deg2rad(4));b=state(1500);c=state(1500,500)
    assert rewards([a],[b,c])['tactical_offense'][0]==.1
    a=state();b=state(-1500,psi=-np.deg2rad(25));c=state(-1500,psi=-np.deg2rad(4))
    assert rewards([a],[b,c])['tactical_defense'][0]==-.15

def test_precombat_cache_survives_target_kill_pbrs_disabled_total_and_metrics():
    e=MultiUAVCombatEnv(cfg());e.reset(31)
    e.red=[state(alive=i==0) for i in range(5)];e.blue=[state(1500,alive=i==0) for i in range(5)]
    with patch.object(e,'_advance',return_value=np.zeros(5)),patch.object(type(e.weapon),'attempt_hit',return_value=True):
        _,r,t,_,info=e.step(np.zeros((5,3)),np.zeros((5,3)))
    assert t and not e.blue[0].alive
    assert info['tactical_offense_rewards'][0]==info['adv_rewards'][0]==.1
    assert info['episode_tactical_total']==info['episode_tactical_offense_total']==.1
    assert info['pbrs_enabled'] is False
    for key in ('phi_current','phi_next','phi_next_for_shaping'):assert not info[key].any()
    np.testing.assert_array_equal(r,sum((info[f'{k}_rewards'] for k in ('event','outcome','adv','safe')),np.zeros(5)).astype(np.float32))
    np.testing.assert_array_equal(info['event_rewards'],info['individual_event_rewards']+info['team_casualty_rewards'])
    np.testing.assert_array_equal(info['tactical_total_rewards'],info['guide_rewards']+info['tactical_offense_rewards']+info['tactical_defense_rewards'])
    from algorithm.common.evaluator import aggregate_combat_records
    record=dict(info,episode_return=float(r.sum()),mean_agent_episode_return=float(r.mean()))
    assert aggregate_combat_records([record])['average_episode_tactical_total']==.1

@pytest.mark.parametrize('side',['red','blue'])
def test_strict_pair_entry_nearest_ammo_and_unselected_entries_consumed(side):
    e=MultiUAVCombatEnv(cfg());e.reset(31)
    own=[state(alive=i==0) for i in range(5)];targets=[state(1500+i*100,alive=i<2) for i in range(5)]
    setattr(e,side,own);setattr(e,'blue' if side=='red' else 'red',targets)
    fire=getattr(e,f'{side}_fire_states');e.steps=1
    assert [j for _,j,_ in e._entry_attempts(own,targets,fire,side)]==[0]
    assert e._entry_attempts(own,targets,fire,side)==[]
    own[0].psi=np.pi;assert e._entry_attempts(own,targets,fire,side)==[]
    own[0].psi=0;assert len(e._entry_attempts(own,targets,fire,side))==1
    assert getattr(e,f'{side}_ammo')[0]==4

@pytest.mark.parametrize('seed',[1,31,101])
def test_red_blue_physical_and_hardware_symmetry(seed):
    e=MultiUAVCombatEnv(cfg());e.reset(seed);rng=np.random.default_rng(seed)
    e.blue=copy.deepcopy(e.red)
    action=rng.uniform(-1,1,(5,3))
    e._advance(e.red,action);e._advance(e.blue,action)
    np.testing.assert_array_equal([s.as_array() for s in e.red],[s.as_array() for s in e.blue])
    np.testing.assert_array_equal(e.red_ammo,e.blue_ammo);assert e.weapon.hit_probability==.7 and e.weapon.ammo_per_aircraft==6
    targets=copy.deepcopy(e.red)
    np.testing.assert_array_equal(e.weapon.eligibility(e.red,targets,e.red_ammo)[0],e.weapon.eligibility(e.blue,targets,e.blue_ammo)[0])
    for a,b in zip(sensor_geometry(e.red,targets,e.config['sensor']),sensor_geometry(e.blue,targets,e.config['sensor'])):np.testing.assert_array_equal(a,b)
    assert e.fixed_policy.sensor_config is e.config['sensor'] and e.spec.v_min==150 and e.spec.v_max==300

def test_mirrored_weapon_geometry_and_radar_observation():
    w=weapon();a,b=pair(1500,20,20)
    am,bm=copy.deepcopy(a),copy.deepcopy(b)
    for s in (am,bm):s.x=-s.x;s.y=-s.y;s.psi+=np.pi
    assert w.qualifies(a,b,6)==w.qualifies(am,bm,6)
    e=MultiUAVCombatEnv(cfg());e.reset(31)
    e.red=[state(alive=i==0) for i in range(5)];e.blue=[state(3500,alive=i==0) for i in range(5)]
    assert e._observations().shape==(5,66) and sensor_geometry(e.red,e.blue,e.config['sensor'])[0][0,0]
    e.blue[0].x=4001;assert not sensor_geometry(e.red,e.blue,e.config['sensor'])[0].any() and not e._observations()[0,36:].any()

def test_configuration_inherited_blue_and_frozen_other_mechanisms():
    old=load_config(ROOT/'configs/combat_environment_v35.yaml');new=cfg()
    for key in old.keys()-{'environment_version','sensor','weapon','reward'}:assert old[key]==new[key]
    assert new['blue_policy']==old['blue_policy'] and new['blue_policy']['close_distance']==1800
    assert not any(key in new['reward'] for key in ('potential_gamma','potential_scale','distance_weight','angle_weight'))
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        assert (ROOT/f'configs/{name}_5v5_v36.yaml').read_bytes()==(ROOT/f'configs/{name}_5v5_v35.yaml').read_bytes()
    for method in ('_entry_attempts','_resolve_combat','_outcome','_advance','event_reward_extension','_resolve_noncombat_losses'):
        assert getattr(CombatEnvironmentV36,method) is getattr(CombatEnvironmentV35,method)

def test_complete_v35_pre_hook_reference():
    assert capture35()==json.loads((ROOT/'tests/v36_v35_reference.json').read_text())

@pytest.mark.parametrize('section,key,value',[('sensor','range_max',3000),('weapon','ata_max',1.),('weapon','hit_probability',.8),('reward','guide_reward',.002),('reward','potential_scale',0)])
def test_strict_config_rejections(section,key,value):
    c=cfg();c[section][key]=value
    with pytest.raises(ValueError):validate_config(c)
