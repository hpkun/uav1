"""Blue-only protocol, information isolation and historical regressions."""
import copy, json
from pathlib import Path
import numpy as np
import pytest
from env.config import load_config, validate_config, environment_dimensions
from env.models import AircraftState
from env.combat_env import MultiUAVCombatEnv
from env.combat_v33 import CombatEnvironmentV33
from env.combat_v34 import CombatEnvironmentV34
from env.v34_policy import SensorLimitedTieredBluePolicy
from env.v30_policy import SensorLimitedPursuitPolicy
from tools._prepare_v34 import capture33
from tools.audit_combat_v34_blue import paired

ROOT=Path(__file__).resolve().parents[1]
def config():return load_config(ROOT/'configs/combat_environment_v34.yaml')
def state(x=0,y=0,z=-3000,psi=0,alive=True):return AircraftState(x,y,z,225,0,psi,alive)
def scene(enemies=2,friends=1,distance=1100):
    c=config();p=SensorLimitedTieredBluePolicy(c['blue_policy'],c['action'],c['sensor'],c['scenario'])
    blue=[state(alive=False) for _ in range(5)];red=[state(alive=False) for _ in range(5)]
    for i in range(friends):blue[i]=state(y=i*50)
    for i in range(enemies):red[i]=state(distance+i*50)
    return p,blue,red

def test_config_reward_and_mechanisms_frozen():
    a=load_config(ROOT/'configs/combat_environment_v33.yaml');b=config();expected=copy.deepcopy(a)
    expected['environment_version']='3.4';expected['blue_policy'].update(evade_enter_distance=1200.,evade_exit_distance=1600.,threat_radius=1500.,support_radius=1500.)
    assert b==expected and b['reward']==a['reward'] and environment_dimensions(b)==(66,3,5)
    assert MultiUAVCombatEnv().environment_version=='2.3'
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        assert (ROOT/f'configs/{name}_5v5_v34.yaml').read_bytes()==(ROOT/f'configs/{name}_5v5_v33.yaml').read_bytes()
    for name in ('event_reward_extension','_resolve_combat','_advance','potential','shaping_next_potential','_outcome','_observations','_entry_attempts'):
        assert getattr(CombatEnvironmentV34,name) is getattr(CombatEnvironmentV33,name)

@pytest.mark.parametrize('enemies,friends,mode',[(1,1,'PURSUIT'),(2,1,'EVADE'),(2,2,'PURSUIT'),(3,2,'EVADE')])
def test_local_counts(enemies,friends,mode):
    p,b,r=scene(enemies,friends);p.team_actions(b,r);assert p.last_modes[0]==mode

def test_search_action_center_and_invisible_isolation():
    p,b,r=scene();b[0]=state(500,400);r=[state(-4000,i*100) for i in range(5)]
    a=p.team_actions(b,r);assert p.last_modes[0]=='SEARCH'
    expected=p.action_toward(b[0],np.arctan2(-400,-500),0.,250.)
    np.testing.assert_array_equal(a[0],expected)
    r[0]=state(4500,3500,z=-100);np.testing.assert_array_equal(a,p.team_actions(b,r))

def test_visible_pursuit_exact_old_policy():
    p,b,r=scene(1);r[0]=state(700,400,z=-3100)
    old=SensorLimitedPursuitPolicy(p.config,config()['action'],p.sensor_config,{'altitude_center':3000})
    np.testing.assert_array_equal(p.team_actions(b,r),old.team_actions(b,r));assert p.last_modes[0]=='PURSUIT'

def test_enter_threshold():
    p,b,r=scene(distance=1201);p.team_actions(b,r);assert p.last_modes[0]=='PURSUIT'
    r[0].x=1200;p.team_actions(b,r);assert p.last_modes[0]=='EVADE'

def test_hysteresis_ignores_count_change_and_exits_at1600():
    p,b,r=scene();p.team_actions(b,r)
    r[0].x=1400;r[1].alive=False;p.team_actions(b,r);assert p.last_modes[0]=='EVADE'
    r[0].x=1599;p.team_actions(b,r);assert p.last_modes[0]=='EVADE'
    r[0].x=1600;p.team_actions(b,r);assert p.last_modes[0]=='PURSUIT' and not p.evading[0]

def test_lost_target_clears():
    p,b,r=scene();p.team_actions(b,r)
    for s in r:s.x=-2000
    p.team_actions(b,r);assert p.last_modes[0]=='SEARCH' and not p.evading[0]

@pytest.mark.parametrize('x,z',[(4500,-3000),(0,-500),(0,-5500)])
def test_guard_priority_and_exact_old_action(x,z):
    p,b,r=scene();p.team_actions(b,r);b[0].x=x;b[0].z=z
    r[0]=state(x+400,z=z);r[1]=state(x+500,z=z)
    a=p.team_actions(b,r);assert p.last_modes[0]=='BOUNDARY_GUARD' and not p.evading[0]
    np.testing.assert_array_equal(a[0],p._action(b[0],r,np.ones(5,dtype=bool),np.ones(5)))

def test_escape_direction_elevation_and_centroid_scope():
    p,b,r=scene();r[0]=state(600,400);r[1]=state(900,300);r[2]=state(2000,-1000)
    expected=np.array([-750.,-350.]);heading=np.arctan2(expected[1],expected[0])
    a=p.team_actions(b,r);assert p.last_modes[0]=='EVADE'
    np.testing.assert_array_equal(a[0],p.action_toward(b[0],heading,0.,250.))
    assert np.dot([np.cos(heading),np.sin(heading)],expected)/np.linalg.norm(expected)>1-1e-12

def test_invisible_red_cannot_change_evade_counts_or_action():
    p,b,r=scene();r[2]=state(-100);a=p.team_actions(b,r);m=p.last_modes.copy()
    r[2]=state(-2500,900,z=-5000);p.reset()
    np.testing.assert_array_equal(a,p.team_actions(b,r));assert p.last_modes==m

def test_counts_use_three_dimensions_and_living_support():
    p,b,r=scene(2,2);b[1].z=-4600
    p.team_actions(b,r);assert p.last_modes[0]=='EVADE'
    p.reset();b[1].z=-3000;b[1].alive=False;p.team_actions(b,r);assert p.last_modes[0]=='EVADE'
    p.reset();r[1].z=-4600;p.team_actions(b,r);assert p.last_modes[0]=='PURSUIT'

@pytest.mark.parametrize('coincident',[False,True])
def test_degenerate_centroid_finite_deterministic(coincident):
    p,b,r=scene();r[0]=state(0,0 if coincident else 500);r[1]=state(0,0 if coincident else -500)
    a=p.team_actions(b,r);p.reset();np.testing.assert_array_equal(a,p.team_actions(b,r));assert np.isfinite(a).all()

def test_dead_modes_excluded_and_reset():
    p,b,r=scene();a=p.team_actions(b,r)
    np.testing.assert_array_equal(a[1:],np.zeros((4,3)))
    assert sum(p.counts.values())==1 and p.evade_episode
    b[0].alive=False;p.team_actions(b,r);assert not p.evading.any() and sum(p.counts.values())==1
    p.reset();assert not p.evading.any() and p.last_modes==[] and sum(p.counts.values())==0 and not p.evade_episode

def test_environment_episode_reset_and_external_override():
    e=MultiUAVCombatEnv(config());e.reset(31)
    p,b,r=scene();e.blue=b;e.red=r;e.step(np.zeros((5,3)))
    assert e.fixed_policy.evade_episode
    e.reset(31);assert not e.fixed_policy.evading.any() and not e.fixed_policy.evade_episode
    info=e.step(np.zeros((5,3)),np.zeros((5,3)))[4]
    assert 'blue_policy_modes' not in info and sum(e.fixed_policy.counts.values())==0

@pytest.mark.parametrize('seed',[1,31,33000000,39000000])
@pytest.mark.parametrize('mode',['ZERO','RANDOM'])
def test_explicit_blue_full_episode_isolation(seed,mode):assert paired(seed,mode)>0

def test_v33_pre_change_full_reference():
    assert capture33()==json.loads((ROOT/'tests/v34_v33_reference.json').read_text())

@pytest.mark.parametrize('key',['desired_speed','guard_radius','guard_altitude_min','guard_altitude_max','evade_enter_distance','evade_exit_distance','threat_radius','support_radius'])
def test_frozen_blue_config(key):
    c=config();c['blue_policy'][key]+=1
    with pytest.raises(ValueError):validate_config(c)

def test_missing_extra_and_reward_rejected():
    c=config();c['blue_policy'].pop('support_radius')
    with pytest.raises(ValueError):validate_config(c)
    c=config();c['reward']['team_casualty_penalty']=-1
    with pytest.raises(ValueError):validate_config(c)
