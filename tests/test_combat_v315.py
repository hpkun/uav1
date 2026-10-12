"""Only zero phi observation and feasible pitch/speed targets change in v3.15."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import importlib
from types import SimpleNamespace
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.combat_v314 import CombatEnvironmentV314
from env.combat_v315 import CombatEnvironmentV315
from env.config import load_config,validate_config,environment_dimensions
from env.models import AircraftState
from env.control import action_to_target,action_to_control
from env.v315_control import action_to_target_v315,action_to_control_v315
from env.v313_observation import build_team_observations_v313
from env.v314_observation import build_team_observations_v314
from env.v315_observation import build_team_observations_v315
from env.v313_geometry import combat_geometry_v313
from env.dynamics import PointMassDynamics
from env.integrator import RK4Integrator


def env(version='315',seed=1):
    e=MultiUAVCombatEnv(f'configs/combat_environment_v{version}.yaml');e.reset(seed);return e


def state(theta=0.,v=225.,psi=0.,x=0.,altitude=3000.,alive=True):
    return AircraftState(x,0.,-altitude,v,theta,psi,alive)


def test_minimal_factory_inheritance_and_frozen_config():
    new,old=env(),env('314')
    assert type(new) is CombatEnvironmentV315
    assert issubclass(CombatEnvironmentV315,CombatEnvironmentV314)
    assert set(CombatEnvironmentV315.__dict__)-{'__module__','__doc__'}=={'environment_version','_advance','_observations'}
    for name in ('__init__','step','reset','initial_states','build_weapon','dense_combat_reward','_entry_attempts','_resolve_combat','_resolve_noncombat_losses','_outcome'):
        assert getattr(CombatEnvironmentV315,name) is getattr(CombatEnvironmentV314,name)
    cfg=deepcopy(new.config);cfg['environment_version']='3.14';assert cfg==old.config
    assert environment_dimensions(new.config)==(56,3,5)
    assert type(new.dynamics) is type(old.dynamics) is PointMassDynamics
    assert type(new.integrator) is type(old.integrator) is RK4Integrator


@pytest.mark.parametrize('section',['simulation','action','aircraft','arena','scenario','sensor','weapon','reward','observation','blue_policy'])
def test_config_cannot_change_frozen_sections(section):
    cfg=load_config('configs/combat_environment_v315.yaml');cfg[section]['unexpected']=0
    with pytest.raises(ValueError):validate_config(cfg)


@pytest.mark.parametrize('theta_deg',[0,55,60,-55,-60])
@pytest.mark.parametrize('action',[-1,-.5,0,.5,1])
def test_pitch_feasibility_and_historical_unclipped_target(theta_deg,action):
    e=env();s=state(theta=np.deg2rad(theta_deg));a=[0,action,0]
    old=action_to_target(s,a,e.config['action']['command'])
    new=action_to_target_v315(s,a,e.config['action'],e.spec)
    expected=float(np.clip(old.pitch,e.spec.theta_min,e.spec.theta_max))
    assert new.pitch==expected and e.spec.theta_min<=new.pitch<=e.spec.theta_max
    assert old.pitch==s.theta+e.config['action']['command']['pitch_delta_max']*action
    if e.spec.theta_min<=old.pitch<=e.spec.theta_max:assert new==old
    else:assert new.pitch!=old.pitch


@pytest.mark.parametrize('speed',[150,160,225,290,300])
@pytest.mark.parametrize('action',[-1,-.5,0,.5,1])
def test_speed_feasibility_and_historical_unclipped_target(speed,action):
    e=env();s=state(v=speed);a=[0,0,action]
    old=action_to_target(s,a,e.config['action']['command']);new=action_to_target_v315(s,a,e.config['action'],e.spec)
    assert new.speed==float(np.clip(old.speed,150,300)) and 150<=new.speed<=300
    assert old.speed==speed+50*action
    if 150<=old.speed<=300:assert old==new
    else:assert old.speed!=new.speed


@pytest.mark.parametrize('psi',[0,3.,-3.])
@pytest.mark.parametrize('action',[-1,-.5,0,.5,1])
def test_heading_definition_exact(psi,action):
    e=env();s=state(psi=psi);a=[action,.1,.1]
    assert action_to_target_v315(s,a,e.config['action'],e.spec).heading==action_to_target(s,a,e.config['action']['command']).heading


def test_normalized_action_clipping():
    e=env();s=state()
    assert action_to_target_v315(s,[2,-2,2],e.config['action'],e.spec)==action_to_target_v315(s,[1,-1,1],e.config['action'],e.spec)


@pytest.mark.parametrize('theta',[-np.pi/6,0,np.pi/6])
@pytest.mark.parametrize('speed',[175,225,275])
@pytest.mark.parametrize('a',[[-.5,-.2,-.2],[0,0,0],[.5,.2,.2]])
def test_interior_control_and_rk4_exact(theta,speed,a):
    e=env();old=env('314');s=state(theta=theta,v=speed)
    c=action_to_control_v315(s,a,e.config['action'],e.spec)
    assert c==action_to_control(s,a,e.config['action'])
    expected=old.integrator.step(s,c,old.dynamics,old.spec)
    aircraft=[s.copy() for _ in range(5)]
    with patch.object(e.dynamics,'derivatives',wraps=e.dynamics.derivatives) as derivative:
        phis=e._advance(aircraft,np.tile(a,(5,1)))
    assert derivative.call_count==20
    np.testing.assert_array_equal(phis,np.full(5,c.phi))
    for actual in aircraft:np.testing.assert_array_equal(actual.as_array(),expected.as_array())


@pytest.mark.parametrize('theta,action',[(np.pi/3,.6),(-np.pi/3,-.6)])
def test_pitch_boundary_zero_target_rate(theta,action):
    e=env();s=state(theta=theta);a=[.1,action,0]
    new=action_to_target_v315(s,a,e.config['action'],e.spec);old=action_to_target(s,a,e.config['action']['command'])
    assert new.pitch==s.theta and old.pitch!=s.theta
    assert (new.pitch-s.theta)/2==0
    assert action_to_control_v315(s,a,e.config['action'],e.spec)!=action_to_control(s,a,e.config['action'])


@pytest.mark.parametrize('speed,action',[(300,.8),(150,-.8)])
def test_speed_boundary_zero_target_acceleration(speed,action):
    e=env();s=state(v=speed);a=[0,0,action]
    assert action_to_target_v315(s,a,e.config['action'],e.spec).speed==speed
    assert action_to_control_v315(s,a,e.config['action'],e.spec).nx==0
    assert action_to_control(s,a,e.config['action']).nx!=0


def test_zero_phi_slot_nonzero_execution_dead_state_and_55_features():
    e=env();assert not e._observations()[:,4].any()
    e.red[1].alive=False;before=e.red[1].as_array().copy()
    action=np.tile([.5,-.1,0],(5,1));phi=e._advance(e.red,action);e.red_last_executed_phi=phi
    assert np.any(phi!=0) and phi[1]==0
    np.testing.assert_array_equal(before,e.red[1].as_array())
    obs=e._observations();assert not obs[:,4].any() and not obs[1].any()
    np.testing.assert_array_equal(obs,build_team_observations_v313(e.red,e.blue,e.config['observation']))
    old=build_team_observations_v314(e.red,e.blue,e.config['observation'],phi)
    np.testing.assert_array_equal(old[:,np.arange(56)!=4],obs[:,np.arange(56)!=4])
    assert old[:,4].any()
    e.red_last_executed_phi[:]=np.nan
    np.testing.assert_array_equal(e._observations(),obs)  # observation does not read diagnostic phi at all.
    e.reset(1);obs,*_=e.step(action,np.zeros((5,3)))
    assert e.red_last_executed_phi.any() and not obs[:,4].any()


def test_both_sides_use_new_control_and_equal_states():
    e=env();e.red=[state(theta=np.deg2rad(55),v=290,x=100*i) for i in range(5)];e.blue=deepcopy(e.red)
    a=np.tile([.3,1,1],(5,1))
    from env.v315_control import action_to_control_v315 as convert
    with patch('env.combat_v315.action_to_control_v315',wraps=convert) as conversion:
        red=e._advance(e.red,a);blue=e._advance(e.blue,a)
    assert conversion.call_count==10
    np.testing.assert_array_equal(red,blue)
    np.testing.assert_array_equal([s.as_array() for s in e.red],[s.as_array() for s in e.blue])
    e.reset(1)
    with patch('env.combat_v315.action_to_control_v315',wraps=convert) as conversion:e.step(a,a)
    assert conversion.call_count==10


@pytest.mark.parametrize('heading,elevation',[(.7,.2),(-2,-.3),(3.1,.1),(0,1.4)])
def test_blue_policy_target_compatibility(heading,elevation):
    e=env();s=state(psi=2.9,theta=.1)
    a=e.fixed_policy.action_toward(s,heading,elevation,250)
    new=action_to_target_v315(s,a,e.config['action'],e.spec)
    assert abs((new.heading-heading+np.pi)%(2*np.pi)-np.pi)<2e-7
    assert new.pitch==pytest.approx(np.clip(elevation,-np.pi/3,np.pi/3),abs=1e-7)
    assert new.speed==250
    assert type(e.fixed_policy) is type(env('314').fixed_policy)


@pytest.mark.parametrize('seed',[1,31,10000000])
def test_scenario_exact(seed):
    old,new=env('314',seed),env('315',seed)
    for side in ('red','blue'):
        np.testing.assert_array_equal([s.as_array() for s in getattr(old,side)],[s.as_array() for s in getattr(new,side)])


@pytest.mark.parametrize('distance',[100.,2500.,5000.])
def test_static_reward_weapon_sampler_geometry_and_full_decomposition(distance):
    old,new=env('314'),env();old.red=[state(x=30*i,theta=.1*i) for i in range(5)];old.blue=[state(x=distance+30*i,psi=np.pi,altitude=3000+50*i) for i in range(5)]
    new.red,new.blue=deepcopy(old.red),deepcopy(old.blue)
    assert old.config['reward']==new.config['reward'] and old.config['weapon']==new.config['weapon']
    for key,value in combat_geometry_v313(old.red,old.blue).items():np.testing.assert_array_equal(value,combat_geometry_v313(new.red,new.blue)[key])
    for a,b in zip(old.weapon.eligibility(old.red,old.blue,old.red_ammo),new.weapon.eligibility(new.red,new.blue,new.red_ammo)):np.testing.assert_array_equal(a,b)
    with patch.object(old.weapon,'attempt_hit',wraps=old.weapon.attempt_hit) as os,patch.object(new.weapon,'attempt_hit',wraps=new.weapon.attempt_hit) as ns:
        for _ in range(20):assert old.attempt_hit_for_pair(old.red[0],old.blue[0])==new.attempt_hit_for_pair(new.red[0],new.blue[0])
        assert [call.args[1:] for call in os.call_args_list]==[call.args[1:] for call in ns.call_args_list]
    assert old.rng.bit_generator.state==new.rng.bit_generator.state
    # Isolate the inherited full reward path on identical static combat states.
    with patch.object(old,'_advance',return_value=np.zeros(5)),patch.object(new,'_advance',return_value=np.zeros(5)):
        oo,ro,to,xo,io=old.step(np.zeros((5,3)),np.zeros((5,3)));on,rn,tn,xn,inn=new.step(np.zeros((5,3)),np.zeros((5,3)))
    np.testing.assert_array_equal(ro,rn)
    for key in ['r1_rewards','r2_rewards','r3_rewards','r4_rewards','safety_rewards','paper_rewards','extended_rewards','episode_paper_reward_total','episode_safety_total','episode_extended_reward_total']:np.testing.assert_array_equal(io[key],inn[key])
    assert (to,xo)==(tn,xn)


@pytest.mark.parametrize('case',['blue_dead','red_dead','timeout','boundary','ground','ceiling'])
def test_noncombat_and_outcome_regression(case):
    old,new=env('314'),env()
    for e in (old,new):
        if case.endswith('_dead'):
            for s in getattr(e,case.split('_')[0]):s.alive=False
        if case=='boundary':e.red[0].x=5001
        if case=='ground':e.red[0].z=0
        if case=='ceiling':e.red[0].z=-6001
    assert old._resolve_noncombat_losses()==new._resolve_noncombat_losses()
    assert old.combat_counts==new.combat_counts
    assert old._outcome(case=='timeout')==new._outcome(case=='timeout')


@pytest.mark.parametrize('name',['mappo','rmappo','ea_mappo','stea_mappo'])
def test_algorithm_configs_byte_identical(name):
    assert Path(f'configs/{name}_5v5_v315.yaml').read_bytes()==Path(f'configs/{name}_5v5_v314.yaml').read_bytes()


@pytest.mark.parametrize('module_name,class_name',[('mappo','MAPPOTrainingRunner'),('rmappo','RMAPPOTrainingRunner'),('ea_mappo','EAMAPPOTrainingRunner'),('stea_mappo','STEAMAPPOTrainingRunner')])
def test_metrics_version_registration(module_name,class_name):
    module=importlib.import_module(f'algorithm.{module_name}.runner');runner=object.__new__(getattr(module,class_name))
    runner.env_config=load_config('configs/combat_environment_v315.yaml');runner.trainer=SimpleNamespace(sampled_steps=512);runner.evaluation_seeds=[10000000,10000001]
    with patch.object(module,'evaluate',return_value={'evaluation_episodes':2}):row=runner._evaluation_record()
    assert row['environment_version']=='3.15' and row['sampled_steps']==512


def test_recording_factory_dispatch_without_loading_model():
    from tools.combat_visualization import RecordingCombatEnv
    e=RecordingCombatEnv('configs/combat_environment_v315.yaml');assert isinstance(e,CombatEnvironmentV315)
    assert e.reset(1)[0].shape==(5,56)


def test_checkpoint_isolation_and_old_zero_counterfactual_cuda():
    from tools.validate_combat_v315 import check_checkpoint_isolation,check_old_zero_counterfactual
    assert all(row['old_accepted'] and row['cross_version_rejected'] for row in check_checkpoint_isolation().values())
    assert check_old_zero_counterfactual()['matches_prior_zero_counterfactual']


def test_real_vector_auto_reset_zero_phi_observation():
    from tools.validate_combat_v315 import check_vector_reset
    assert check_vector_reset()['every_transition_phi_slot_zero']
