"""Dwell geometry/state, simultaneous resolution, reward ledger and isolation."""
from copy import deepcopy
import numpy as np
import pytest
import yaml
from env.config import load_config,validate_config,environment_dimensions
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.geometry import engagement_geometry
from env.v344_dwell import DwellEnvelope,DwellState,required_dwell_steps


def config(version='344'):
    return load_config(f'configs/combat_environment_v{version}.yaml')


def environment(version='344'):
    e=MultiUAVCombatEnv(config(version));e.reset(7);return e


def state(x=0,y=0,z=-3000,psi=0,theta=0,alive=True):
    return AircraftState(x,y,z,225.,theta,psi,alive)


def frozen(version='344',sharing=False):
    e=environment(version)
    if sharing:
        e.red=[state() for _ in range(4)]
        e.blue=[state(x=1500)]+[state(alive=False) for _ in range(3)]
    else:
        e.red=[state(x=x,psi=np.pi/2) for x in (-2500,-800,800,2500)]
        e.blue=[state(x=x,y=1500,psi=-np.pi/2) for x in (-2500,-800,800,2500)]
    e._advance=lambda states,actions:np.zeros(4)
    e.red_ammo.fill(0);e.blue_ammo.fill(0)
    e.red_fire_cooldown_steps.fill(999);e.blue_fire_cooldown_steps.fill(999)
    return e


@pytest.mark.parametrize('seconds,dt,expected',[(.5,.1,5),(.5,.25,2),(.5,.2,3),(.3,.1,3),(.01,.1,1),(.51,.1,6)])
def test_time_mapping(seconds,dt,expected):
    assert required_dwell_steps(seconds,dt)==expected


@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf'),True])
def test_bad_time(value):
    with pytest.raises(ValueError):required_dwell_steps(value,.1)
    with pytest.raises(ValueError):required_dwell_steps(.5,value)


@pytest.mark.parametrize('version,limit',[('344',30),('345',45)])
@pytest.mark.parametrize('psi',[-179,0,130])
@pytest.mark.parametrize('theta',[-50,0,50])
@pytest.mark.parametrize('offset',[-1e-6,0,1e-6])
def test_3d_boundary(version,limit,psi,theta,offset):
    a=state(psi=np.deg2rad(psi),theta=np.deg2rad(theta))
    f=a.velocity_vector()/a.v;r=np.array([-np.sin(a.psi),np.cos(a.psi),0.])
    angle=np.deg2rad(limit+offset);delta=1500*(np.cos(angle)*f+np.sin(angle)*r)
    b=state(*((a.as_array()[:3]+delta).tolist()))
    e=DwellEnvelope(**config(version)['weapon'])
    assert e.qualifies(a,b,0)==(offset<=0)
    b.psi=2.4;b.theta=-.7
    assert e.qualifies(a,b,0)==(offset<=0)  # target attitude never added as a gate


@pytest.mark.parametrize('distance',[0,1e-10,10,99,100,1999.99,2000,2000.01])
def test_range_coincidence_and_no_safe_distance_gate(distance):
    e=DwellEnvelope(**config()['weapon']);a=state();b=state(x=distance)
    assert e.qualifies(a,b)==(distance<=2000)
    assert np.isfinite(list(vars(engagement_geometry(a,b)).values())).all()


@pytest.mark.parametrize('field',['x','y','z','v','psi','theta'])
def test_nonfinite_state_rejected(field):
    a=state();setattr(a,field,float('nan'))
    with pytest.raises(FloatingPointError):DwellEnvelope(**config()['weapon']).qualifies(a,state())


def test_above_below_pitch_and_no_old_gate():
    a=state(theta=np.deg2rad(50));b=state(x=1500*np.cos(np.deg2rad(20)),z=-3000+1500*np.sin(np.deg2rad(20)))
    w=DwellEnvelope(**config()['weapon']);assert not w.qualifies(a,b)
    a.theta=np.deg2rad(-20);assert w.qualifies(a,b)
    a.theta=np.deg2rad(50);b.z=-3000-1500*np.tan(np.deg2rad(50))
    b.x=1000; b.z=-3000-1000*np.tan(np.deg2rad(50))
    assert w.qualifies(a,b)  # old HA=50 does not restrict the cone


def test_interruption_reentry_switch_and_ties():
    s=DwellState(1,5);valid=np.array([[True,False]]);distance=np.array([[1000.,1500.]])
    for k in (1,2):assert not s.update(valid,distance,k)
    assert s.counts[0]==2
    assert not s.update(np.zeros_like(valid),distance,3)
    assert s.counts[0]==0 and s.metrics['dwell_interrupted']==1
    for k in (4,5,6):s.update(valid,distance,k)
    assert s.counts[0]==3
    s.update(np.array([[False,True]]),distance,7)
    assert s.counts[0]==1 and s.targets[0]==1
    assert s.metrics['dwell_target_switches']==1 and s.metrics['dwell_started']==3
    for k in (8,9,10):assert not s.update(np.array([[False,True]]),distance,k)
    assert s.update(np.array([[False,True]]),distance,11)==[(0,1,True)]
    assert s.metrics['dwell_completed']==1 and s.sampled_counts[0]==5
    assert s.diagnostics()['mean_dwell_duration_steps']==pytest.approx(10/3)
    assert s.metrics['dwell_interrupted']==2 and s.metrics['max_dwell_duration_steps']==5
    tie=DwellState(1,5);tie.update(np.ones((1,2),bool),np.ones((1,2)),1)
    assert tie.targets[0]==0


@pytest.mark.parametrize('version',['344','345'])
def test_four_samples_no_event_fifth_mutual_destruction(version):
    e=frozen(version);before=deepcopy(e.rng.bit_generator.state)
    e.attempt_hit_for_pair=lambda *args:pytest.fail('probability hit called')
    total=np.zeros(4);parts=np.zeros((4,4))
    for k in range(1,6):
        _,r,term,trunc,info=e.step(np.zeros((4,3)),np.zeros((4,3)))
        total+=r;parts+=np.stack([info[f'r{i}_rewards'] for i in range(1,5)])
        np.testing.assert_array_equal(r,sum(info[f'r{i}_rewards'] for i in range(1,5)))
        if k<5:
            assert not term and not trunc and e.red_alive_mask.sum()==4 and e.blue_alive_mask.sum()==4
            assert info['red_dwell_counts'].tolist()==[k]*4
            assert np.all(info['r1_rewards']==0)
        else:
            assert term and not trunc and info['draw']
            assert info['termination_reason']=='draw_mutual_destruction'
            assert not e.red_alive_mask.any() and not e.blue_alive_mask.any()
            assert info['red_dwell_completed']==info['blue_dwell_completed']==4
            assert info['red_attack_kills']==info['blue_attack_kills']==4
            assert np.all(info['red_dwell_sampled_counts']==5)
            assert np.all(info['red_dwell_counts']==0)
            assert np.all(info['phi_next_for_shaping']==0)
            assert np.all(info['r1_rewards']==0)  # +10 and -10 each
        assert info['red_fire_attempts']==info['blue_weapon_hits']==info['red_ammo_used']==0
        assert info['red_first_attempt_step'] is None and info['blue_first_hit_step'] is None
    assert e.rng.bit_generator.state==before
    np.testing.assert_allclose(total,parts.sum(axis=0),atol=1e-13)
    for i in range(1,5):np.testing.assert_array_equal(parts[i-1],info[f'episode_r{i}_rewards'])
    assert e._entry_attempts(e.red,e.blue,None,'red')==[]


@pytest.mark.parametrize('version',['344','345'])
def test_shared_credit_one_victim_no_ammo_cooldown(version):
    e=frozen(version,sharing=True)
    for _ in range(5):result=e.step(np.zeros((4,3)),np.zeros((4,3)))
    _,r,term,trunc,info=result
    assert term and info['red_win'] and not trunc
    np.testing.assert_array_equal(info['r1_rewards'],[2.5]*4)
    assert info['red_dwell_completed']==4 and info['red_dwell_neutralizations']==1
    assert info['red_attack_kills']==1 and info['blue_losses']==4
    assert not info['ammo_cooldown_applicable'] and not info['legacy_fire_metrics_applicable']
    e.reset(7)
    for s in e.dwell.values():
        assert np.all(s.targets==-1) and not s.counts.any()
        assert s.metrics['dwell_started']==s.metrics['dwell_completed']==0


@pytest.mark.parametrize('victim',['attacker','target'])
def test_partial_progress_dies_and_invalidates(victim):
    e=frozen(sharing=True)
    for _ in range(3):e.step(np.zeros((4,3)),np.zeros((4,3)))
    if victim=='attacker':e.red[0].alive=False
    else:e.blue[0].alive=False
    e.step(np.zeros((4,3)),np.zeros((4,3)))
    assert e.dwell['red'].counts[0]==0
    assert e.dwell['red'].metrics['dwell_completed']==0


def test_blue_target_is_step_start_not_post_motion_reselection():
    e=frozen(sharing=True)
    e.red=[state(x=1500,psi=np.pi),state(x=1600,psi=np.pi)]+[state(alive=False) for _ in range(2)]
    e.blue=[state()]+[state(alive=False) for _ in range(3)]
    def advance(states,actions):
        if states is e.red:states[0].x=1900;states[1].x=1000
        return np.zeros(4)
    e._advance=advance
    info=e.step(np.zeros((4,3)),np.zeros((4,3)))[4]
    assert info['blue_pursuit_target_indices'][0]==0
    assert info['blue_dwell_sampled_targets'][0]==0


def test_truncation_and_reward_observation_frozen():
    a=frozen(sharing=True);b=environment('343');b.red=deepcopy(a.red);b.blue=deepcopy(a.blue)
    np.testing.assert_array_equal(a._observations(),b._observations())
    np.testing.assert_array_equal(a.potential()[0],b.potential()[0])
    np.testing.assert_array_equal(a.dense_combat_reward()[0],b.dense_combat_reward()[0])
    a.max_steps=3;total=np.zeros(4)
    phi=a.potential()[0];discounted=np.zeros(4)
    for k in range(3):
        _,r,term,trunc,info=a.step(np.zeros((4,3)),np.zeros((4,3)))
        total+=r;discounted+=.99**k*info['r3_rewards']
    assert trunc and not term and np.all(info['phi_next_for_shaping']==0)
    assert not a.dwell['red'].counts.any() and info['red_dwell_completed']==0
    np.testing.assert_allclose(discounted,-phi,atol=1e-15)


@pytest.mark.parametrize('version',['344','345'])
@pytest.mark.parametrize('algo',['mappo','ippo'])
def test_schema_and_only_required_algorithm_differences(version,algo):
    c=config(version);copy=deepcopy(c);validate_config(c);assert c==copy
    assert environment_dimensions(c)==(49,3,4)
    for key in ('action','aircraft','scenario','simulation','blue_policy','reward','research_reward','observation'):
        assert c[key]==config('343')[key]
    with open(f'configs/{algo}_4v4_v{version}.yaml') as f:new=yaml.safe_load(f)
    with open(f'configs/{algo}_4v4_v343.yaml') as f:old=yaml.safe_load(f)
    new['environment']=old['environment'];assert new==old
    c['weapon']['ammo_per_aircraft']=6
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('version',['344','345'])
def test_repeatable_complete_episode(version):
    a,b=environment(version),environment(version)
    while True:
        x,y=a.step(np.zeros((4,3))),b.step(np.zeros((4,3)))
        np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
        assert a.rng.bit_generator.state==b.rng.bit_generator.state
        if x[2] or x[3]:break
    assert x[4]['red_dwell_neutralizations']==x[4]['red_attack_kills']
