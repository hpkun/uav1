"""Paper decision tree, independent randomness and forced-return isolation."""
import copy,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from env.config import load_config,validate_config,environment_dimensions
from env.models import AircraftState
from env.combat_env import MultiUAVCombatEnv
from env.combat_v33 import CombatEnvironmentV33
from env.combat_v35 import CombatEnvironmentV35
from env.v35_policy import PaperTieredBluePolicy
from env.math_utils import wrap_angle
from tools._capture_v33_reference import canonical,states
from tools._prepare_v35 import capture34

ROOT=Path(__file__).resolve().parents[1]
def config():return load_config(ROOT/'configs/combat_environment_v35.yaml')
def state(x=0,y=0,z=-3000,theta=0,psi=0,alive=True):return AircraftState(x,y,z,225,theta,psi,alive)
def scene(nred=5,nblue=5,d=1000):
    c=config();p=PaperTieredBluePolicy(c['blue_policy'],c['action'],c['sensor']);p.reset(31)
    r=[state(d+i*10,alive=i<nred) for i in range(5)]
    b=[state(y=i*10,alive=i<nblue) for i in range(5)]
    return p,b,r

def test_configuration_and_reward_inheritance():
    a=load_config(ROOT/'configs/combat_environment_v33.yaml');b=config();expected=copy.deepcopy(a)
    expected['environment_version']='3.5';expected['blue_policy']=dict(desired_speed=250.,close_distance=1800.,dispersion_threshold=1500.)
    assert b==expected and environment_dimensions(b)==(66,3,5)
    assert MultiUAVCombatEnv().environment_version=='2.3'
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        assert (ROOT/f'configs/{name}_5v5_v35.yaml').read_bytes()==(ROOT/f'configs/{name}_5v5_v33.yaml').read_bytes()
    for method in ('event_reward_extension','potential','shaping_next_potential','_advance','_outcome','_observations','_entry_attempts','_resolve_combat'):
        assert getattr(CombatEnvironmentV35,method) is getattr(CombatEnvironmentV33,method)

@pytest.mark.parametrize('reds,blues,d,mode',[(5,5,1000,'DETECTED_PURSUIT'),(5,4,1800,'DETECTED_ESCAPE'),(5,4,1801,'DETECTED_PURSUIT'),(3,2,1700,'DETECTED_ESCAPE')])
def test_detected_number_and_distance(reds,blues,d,mode):
    p,b,r=scene(reds,blues,d);p.team_actions(b,r);assert p.last_modes[0]==mode

def test_no_hysteresis():
    p,b,r=scene(5,4,1800);p.team_actions(b,r);assert p.last_modes[0]=='DETECTED_ESCAPE'
    r[0].x=1801;p.team_actions(b,r);assert p.last_modes[0]=='DETECTED_PURSUIT'

def test_pursuit_nearest_visible_not_hidden_closer():
    p,b,r=scene();r[0]=state(-10);r[1]=state(900,400,z=-3100)
    # Place other visible candidates farther for unambiguous LOS.
    for s in r[2:]:s.x=2000
    a=p.team_actions(b,r);assert p.last_targets[0]==1
    np.testing.assert_array_equal(a[0],p.toward_vector(b[0],np.array([900,400,-100.])))

@pytest.mark.parametrize('centroid_z,pitch_sign',[(-3500,-1),(-2500,1)])
def test_escape_global_centroid_ned_pitch(centroid_z,pitch_sign):
    p,b,r=scene(5,4,1000)
    for s in r:s.z=centroid_z
    a=p.team_actions(b,r);center,_=p.cluster(r)
    np.testing.assert_array_equal(a[0],p.toward_vector(b[0],np.array([0,0,-3000])-center))
    assert np.sign(a[0,1])==pitch_sign

def test_hidden_alive_red_changes_escape_direction():
    p,b,r=scene(5,4);r[4]=state(-2000,3000,z=-1000)
    a=p.team_actions(b,r)[0];r[4].y=-3000;r[4].z=-5000
    z=p.team_actions(b,r)[0];assert p.last_modes[0]=='DETECTED_ESCAPE' and not np.array_equal(a,z)

@pytest.mark.parametrize('spread,mode',[(1501,'NO_DETECTION_GUERRILLA'),(1500,'NO_DETECTION_CENTRIPETAL'),(1499,'NO_DETECTION_CENTRIPETAL')])
def test_no_detection_threshold(spread,mode):
    p,b,r=scene(2);r[0]=state(-4000,spread);r[1]=state(-4000,-spread)
    p.team_actions(b,r);assert p.last_modes[0]==mode

def test_dispersion_3d_analytic_and_dead_exclusion():
    r=[state(-4000,0,-1500),state(-4000,0,-4500),state(100000,100000,-100000,alive=False)]
    center,spread=PaperTieredBluePolicy.cluster(r)
    np.testing.assert_array_equal(center,[-4000,0,-3000]);assert spread==1500
    r[1].alive=False;assert PaperTieredBluePolicy.cluster(r)[1]==0

def test_centripetal_3d_action_dead_excluded():
    p,b,r=scene(2);r[0]=state(-4000,200,-3500);r[1]=state(-4000,400,-3700)
    r[4]=state(90000,90000,-90000,alive=False)
    a=p.team_actions(b,r)
    assert p.last_modes[0]=='NO_DETECTION_CENTRIPETAL'
    np.testing.assert_array_equal(a[0],p.toward_vector(b[0],[-4000,300,-600]))

def guerrilla_sequence(seed):
    p,b,r=scene(3,1);r[0]=state(-4000,-3000);r[1]=state(-4000,0);r[2]=state(-4000,3000)
    p.reset(seed);result=[]
    for _ in range(1200):
        p.team_actions(b,r);assert p.last_modes[0]=='NO_DETECTION_GUERRILLA';result.append(p.last_targets[0])
    return result

def test_uniform_alive_sampling_per_step_and_reproducibility():
    a=guerrilla_sequence(31);assert a==guerrilla_sequence(31) and a!=guerrilla_sequence(32)
    assert set(a)=={0,1,2} and all(300<x<500 for x in np.bincount(a))
    assert sum(x!=y for x,y in zip(a,a[1:]))>500

def test_blue_rng_does_not_consume_environment_rng():
    e=MultiUAVCombatEnv(config());e.reset(31);before=copy.deepcopy(e.rng.bit_generator.state)
    e.red=[state(-4000,3000*(i-2)) for i in range(5)];e.blue=[state() for _ in range(5)]
    for _ in range(20):e.fixed_policy.team_actions(e.blue,e.red)
    assert e.rng.bit_generator.state==before
    policy_rng=copy.deepcopy(e.fixed_policy.rng.bit_generator.state)
    e.rng.random(20);assert e.fixed_policy.rng.bit_generator.state==policy_rng
    e.reset(31);a=e.fixed_policy.rng.integers(100,size=20);e.reset(31)
    np.testing.assert_array_equal(a,e.fixed_policy.rng.integers(100,size=20))

@pytest.mark.parametrize('all_coincident',[False,True])
def test_escape_degenerate_fallback(all_coincident):
    p,b,r=scene(2,1);r[0]=state(0 if all_coincident else 100);r[1]=state(0 if all_coincident else -100)
    a=p.team_actions(b,r);assert p.last_modes[0]=='DETECTED_ESCAPE' and np.isfinite(a).all()
    expected=p.action_toward(b[0],np.pi,0.,250.)
    np.testing.assert_array_equal(a[0],expected)

def test_dead_blue_and_empty_targets_safe_reset():
    p,b,r=scene();b[0].alive=False;a=p.team_actions(b,r);np.testing.assert_array_equal(a[0],np.zeros(3))
    assert sum(p.counts.values())==4
    for s in r:s.alive=False
    a=p.team_actions(b,r);assert np.isfinite(a).all() and not a.any()
    p.reset(31);assert not p.last_modes and not p.last_targets and not sum(p.counts.values())

def boundary_env(version='35'):
    e=MultiUAVCombatEnv(ROOT/f'configs/combat_environment_v{version}.yaml');e.reset(31)
    e.red=[state(-2500,i*400) for i in range(5)];e.blue=[state(2500,i*400,psi=np.pi) for i in range(5)]
    return e

@pytest.mark.parametrize('cause',['horizontal','ground','ceiling','both'])
def test_blue_boundary_alive_return_and_legacy_death(cause):
    a,b=boundary_env('33'),boundary_env()
    for e in (a,b):
        s=e.blue[0]
        if cause in ('horizontal','both'):s.x=5001;s.psi=.3
        if cause=='ground':s.z=0;s.theta=-.2
        if cause in ('ceiling','both'):s.z=-6001;s.theta=.2
        e._resolve_noncombat_losses()
    assert not a.blue[0].alive and b.blue[0].alive
    s=b.blue[0];assert np.hypot(s.x,s.y)<=5000 and 0<s.altitude<6000
    assert b.blue_returns['boundary']==1
    assert b.combat_counts['blue']['boundary_exits']==b.combat_counts['blue']['ground_losses']==b.combat_counts['blue']['ceiling_losses']==0
    if cause in ('horizontal','both'):
        assert s.psi==wrap_angle(.3+np.pi) and b.blue_returns['horizontal']==1
    if cause=='ground':assert s.theta==.2 and b.blue_returns['ground']==1
    if cause in ('ceiling','both'):assert s.theta==-.2 and b.blue_returns['ceiling']==1

@pytest.mark.parametrize('cause',['horizontal','ground','ceiling'])
def test_red_boundary_loss_exact(cause):
    a,b=boundary_env('33'),boundary_env()
    for e in (a,b):
        if cause=='horizontal':e.red[0].x=5001
        elif cause=='ground':e.red[0].z=0
        else:e.red[0].z=-6001
    assert a._resolve_noncombat_losses()==b._resolve_noncombat_losses()
    assert canonical(states(a))==canonical(states(b)) and a.combat_counts==b.combat_counts

def test_return_before_weapon_no_event_casualty_ammo_or_resurrection():
    e=boundary_env();e.blue[0]=state(5001,z=-6001,theta=.2)
    e.blue[1]=state(5500,alive=False)
    eligibility=e.weapon.eligibility
    def checked(*args):
        assert e.blue[0].alive and np.hypot(e.blue[0].x,e.blue[0].y)<5000 and e.blue[0].altitude<6000
        return eligibility(*args)
    with patch.object(e,'_advance',return_value=np.zeros(5)),patch.object(type(e.weapon),'eligibility',new=lambda weapon,*args:checked(*args)):
        info=e.step(np.zeros((5,3)),np.zeros((5,3)))[4]
    assert not e.blue[1].alive and e.blue_returns['boundary']==1
    assert not info['event_rewards'].any() and not info['team_casualty_rewards'].any()
    assert not info['outcome_rewards'].any() and info['blue_ammo_used']==0
    assert info['blue_boundary_returns']==1 and info['blue_horizontal_returns']==info['blue_ceiling_returns']==1
    e.reset(31);assert not sum(e.blue_returns.values())

@pytest.mark.parametrize('seed',[1,31,33000000,40000000])
@pytest.mark.parametrize('mode',['ZERO','RANDOM'])
def test_nonboundary_explicit_full_episode_isolation(seed,mode):
    a,b=boundary_env('33'),boundary_env();oa,ia=a.reset(seed);ob,ib=b.reset(seed)
    rng=np.random.default_rng(seed+100000000)
    while True:
        red=np.zeros((5,3)) if mode=='ZERO' else rng.uniform(-1,1,(5,3))
        blue=a.fixed_policy.team_actions(a.blue,a.red)
        oa,ra,ta,tra,ia=a.step(red,blue);ob,rb,tb,trb,ib=b.step(red,blue)
        assert not sum(b.blue_returns.values())
        assert canonical((states(a),oa,ra,ta,tra,a.rng.bit_generator.state))==canonical((states(b),ob,rb,tb,trb,b.rng.bit_generator.state))
        extra={f'blue_{k}_returns' for k in ('boundary','horizontal','ground','ceiling')}|{'blue_boundary_return_episode','environment_version'}
        assert canonical({k:v for k,v in ia.items() if k not in extra})==canonical({k:v for k,v in ib.items() if k not in extra})
        if ta or tra:break

def test_v34_full_prechange_reference():
    assert capture34()==json.loads((ROOT/'tests/v35_v34_reference.json').read_text())

@pytest.mark.parametrize('key',['desired_speed','close_distance','dispersion_threshold'])
def test_frozen_config(key):
    c=config();c['blue_policy'][key]+=1
    with pytest.raises(ValueError):validate_config(c)

def test_old_guard_rejected_and_reward_frozen():
    c=config();c['blue_policy']['guard_radius']=4500
    with pytest.raises(ValueError):validate_config(c)
    c=config();c['reward']['win_reward']=2
    with pytest.raises(ValueError):validate_config(c)
