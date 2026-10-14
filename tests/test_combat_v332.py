"""True independent rate channels with frozen v3.30 combat mechanisms."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import math
import numpy as np
import pytest
import yaml
from env.combat_env import MultiUAVCombatEnv
from env.combat_v330 import CombatEnvironmentV330
from env.combat_v332 import CombatEnvironmentV332
from env.config import environment_dimensions,validate_config
from env.models import AircraftState
from env.v311_kinematics import advance_aircraft
from env.v332_config import ACTION,BASELINE_ACTION
from tools.audit_combat_v317_v318 import FixedNormal
from tools.audit_combat_v332_kinematics import circle_audit


def env(version='332'):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml');e.reset(31);return e


def state(x=0,y=0,h=3000,v=225,theta=0,psi=0,alive=True):
    return AircraftState(x,y,-h,v,theta,psi,alive)


def test_identity_dimensions_config_and_baseline_frozen():
    e=env();old=env('330');doubled=env('331')
    assert type(e) is CombatEnvironmentV332 and CombatEnvironmentV332.__bases__==(CombatEnvironmentV330,)
    assert environment_dimensions(e.config)==(45,3,4) and e._observations().shape==(4,45)
    baseline=deepcopy(e.config);baseline['environment_version']='3.30';baseline['action']=deepcopy(BASELINE_ACTION)
    assert baseline==old.config and e.config['reward']==old.config['reward']
    assert e.config['reward']['offense_reward_5deg']==.1 and doubled.config['reward']['offense_reward_5deg']==.2
    assert not hasattr(e,'dynamics') and not hasattr(e,'integrator') and 'controller' not in e.config['action']
    for previous in (old,doubled):
        assert previous.config['action']['controller']['normal_load_max']==10
        assert hasattr(previous,'dynamics') and hasattr(previous,'integrator')
    for name in ('step','reset','initial_states','_observations','_outcome','dense_combat_reward','build_weapon',
                 '_entry_attempts','_resolve_combat','_resolve_noncombat_losses','_event_reward_components','reward_aliases'):
        assert getattr(type(e),name) is getattr(type(old),name)
    for seed in (0,31,199):
        a,_=old.reset(seed);b,_=e.reset(seed);np.testing.assert_array_equal(a,b)
        assert old.red==e.red and old.blue==e.blue
        assert e.red_ammo.shape==e.blue_ammo.shape==e.blue_step_target_indices.shape==(4,)
        assert not e._observations()[:,4].any()


@pytest.mark.parametrize('channel',[0,1,2])
@pytest.mark.parametrize('magnitude',[-1.,-.5,-.2,.2,.5,1.])
def test_rate_linearity_and_negative_symmetry(channel,magnitude):
    e=env();s=state();a=np.zeros(3);a[channel]=magnitude
    e.red=[s.copy()]+[state(alive=False) for _ in range(3)]
    actions=np.zeros((4,3));actions[0]=a;e._advance(e.red,actions);n=e.red[0]
    np.testing.assert_allclose([n.psi-s.psi,n.theta-s.theta,n.v-s.v],
                              a*np.array([np.pi/40,np.pi/60,2.5]),rtol=0,atol=1e-12)
    # Position uses the updated orientation and speed exactly as in v3.11.
    np.testing.assert_array_equal(n.as_array(),advance_aircraft(s,a,.1,ACTION['command'],e.spec).as_array())


@pytest.mark.parametrize('speed',[150.,225.,300.])
def test_sustained_level_turn_no_compensation(speed):
    e=env();r=circle_audit(e,speed,16.)
    assert r['action']==[1,0,0] and r['no_ground_loss'] and r['altitude_constant']
    assert r['radius_fitted_m']==pytest.approx(speed*.1/(2*np.sin(np.pi/80)),abs=1e-7)
    assert abs(r['radius_fitted_m']-r['radius_continuous_m'])/r['radius_continuous_m']<.001


def test_combined_channels_monotonic_symmetric_and_equal_for_red_blue():
    e=env();responses=[]
    for a in (.3,.5,1.):
        e.red=[state() for _ in range(4)];e.blue=deepcopy(e.red);actions=np.tile([a,a,0],(4,1))
        e._advance(e.red,actions);e._advance(e.blue,actions)
        assert e.red==e.blue
        responses.append([e.red[0].psi,e.red[0].theta])
        np.testing.assert_allclose(responses[-1],[a*np.pi/40,a*np.pi/60],rtol=0,atol=1e-12)
    assert np.all(np.diff(responses,axis=0)>0)
    e.red=[state() for _ in range(4)];e._advance(e.red,np.tile([1,.05,0],(4,1)))
    assert e.red[0].theta>0 and np.rad2deg(e.red[0].theta)==pytest.approx(.15)


@pytest.mark.parametrize('a',[[1,1,1],[-1,-1,-1],[9,9,9],[-9,-9,-9]])
def test_state_bounds_and_action_clipping(a):
    e=env();s=state(v=300 if a[0]>0 else 150,theta=np.pi/3 if a[0]>0 else -np.pi/3,psi=math.pi-.001)
    e.red=[s.copy() for _ in range(4)];e._advance(e.red,np.tile(a,(4,1)))
    for n in e.red:
        assert 150<=n.v<=300 and -np.pi/3<=n.theta<=np.pi/3 and -np.pi<=n.psi<np.pi
        assert n.v==s.v and n.theta==s.theta


def test_dead_aircraft_unchanged_and_no_legacy_controller_calls():
    e=env();s=state(alive=False);e.red=[s.copy() for _ in range(4)]
    e._advance(e.red,np.ones((4,3)));assert e.red==[s]*4
    e.reset(31)
    with patch('env.combat_v329.action_to_control_v329',side_effect=AssertionError('old controller used')), \
         patch('env.dynamics.PointMassDynamics.derivatives',side_effect=AssertionError('dynamics used')), \
         patch('env.integrator.RK4Integrator.step',side_effect=AssertionError('RK4 used')), \
         patch('env.combat_v332.advance_aircraft',wraps=advance_aircraft) as advance:
        obs,reward,_,_,info=e.step(np.zeros((4,3)))
        assert advance.call_count==8 and obs.shape==(4,45) and reward.shape==(4,)
        assert not info['controller_diagnostics'] and 'executed_red_phi' not in info
        assert not any('mean_nz' in k or 'saturation_nz' in k for k in info)


def test_blue_policy_scaled_errors_and_nearest_switch_coupling():
    e=env();own=state(v=223.75)
    np.testing.assert_allclose(e.fixed_policy.action_toward(own,np.deg2rad(2.25),np.deg2rad(1.5),225),[.5,.5,.5],rtol=0,atol=1e-7)
    assert e.fixed_policy.action_toward(own,np.pi,0,225)[0]==-1 # wrapped exact pi, old policy unchanged
    e.blue=[state()]+[state(alive=False) for _ in range(3)]
    e.red=[state(0,400),state(700)]+[state(alive=False) for _ in range(2)];e.rng=FixedNormal(4)
    with patch.object(e,'_advance',return_value=np.zeros(4)):
        _,_,_,_,i=e.step(np.zeros((4,3)))
        assert i['blue_pursuit_target_indices'].tolist()==[0,-1,-1,-1]
        assert e.weapon.qualifies(e.blue[0],e.red[1],6) and i['blue_step_fire_attempts']==0
        e.red[1].x=200;_,_,_,_,i=e.step(np.zeros((4,3)))
        assert i['blue_pursuit_target_indices'][0]==1 and i['blue_step_fire_attempts']==1
    e.red=[state(100),state(-100),state(alive=False),state(alive=False)]
    assert e.fixed_policy.nearest_target_index(e.blue[0],e.red)==0
    e.red[0].z=-5000;assert e.fixed_policy.nearest_target_index(e.blue[0],e.red)==1


def test_frozen_reward_weapon_outcome_on_identical_states():
    a,b=env('330'),env()
    for e in (a,b):
        e.red=[state(),state(x=5001),state(h=0),state(alive=False)]
        e.blue=[state(x=500,psi=np.pi)]+[state(alive=False) for _ in range(3)];e.rng=FixedNormal(0)
    np.testing.assert_array_equal(a.dense_combat_reward()[0],b.dense_combat_reward()[0])
    with patch.object(a,'_advance',return_value=np.zeros(4)),patch.object(b,'_advance',return_value=np.zeros(4)):
        x=a.step(np.zeros((4,3)));y=b.step(np.zeros((4,3)))
    np.testing.assert_array_equal(x[1],y[1])
    for i in range(1,5):np.testing.assert_array_equal(x[4][f'r{i}_rewards'],y[4][f'r{i}_rewards'])
    assert y[2] and y[4]['draw'] and y[4]['termination_reason']=='draw_mutual_destruction'
    assert y[4]['red_attack_kills']==y[4]['blue_attack_kills']==1
    for side in ('red','blue'):np.testing.assert_array_equal(getattr(a,side+'_ammo'),getattr(b,side+'_ammo'))


@pytest.mark.parametrize('section,key,value',[
    ('action','controller',{'normal_load_max':10}),('action','command',BASELINE_ACTION['command']),
    ('reward','offense_reward_5deg',.2),('weapon','range_max',800.),('scenario','center_radius',4800.),
    ('blue_policy','desired_speed',250.),('scenario','team_size',5)])
def test_validator_rejects_drift(section,key,value):
    c=deepcopy(env().config);c[section][key]=value
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
def test_algorithm_configs_frozen(name):
    a=yaml.safe_load(Path(f'configs/{name}_4v4_v330.yaml').read_text())
    b=yaml.safe_load(Path(f'configs/{name}_4v4_v332.yaml').read_text())
    if 'environment' in b:
        assert b['environment']['environment_version']=='3.32';b['environment']['environment_version']='3.30'
    assert a==b


@pytest.mark.parametrize('name',['mappo','madsac','maac'])
@pytest.mark.parametrize('previous',['3.30','3.31'])
def test_old_action_semantics_checkpoint_rejected(name,previous):
    ec=env().config;cfg=yaml.safe_load(Path(f'configs/{name}_4v4_v332.yaml').read_text())
    state=dict(algorithm='MAPPO' if name=='mappo' else 'MAAC' if name=='maac' else 'madsac',
               implementation_version=3,extra=dict(environment_version=previous,training_smoke=True))
    if name=='mappo':
        from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
        for fn in (validate_checkpoint_for_evaluation,validate_checkpoint_for_resume):
            with pytest.raises(RuntimeError,match='environment_version'):fn(state,ec,cfg)
    elif name=='madsac':
        from algorithm.madsac.protocol import validate_madsac_checkpoint
        with pytest.raises(RuntimeError,match='environment_version'):validate_madsac_checkpoint(state,ec,cfg)
    else:
        from algorithm.maac.protocol import validate_checkpoint
        with pytest.raises(RuntimeError,match='environment_version'):validate_checkpoint(state,ec,cfg)
