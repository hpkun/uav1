"""Equation (8), reward fidelity, RNG order, and historical isolation."""
import copy,json,math
from pathlib import Path
from unittest.mock import Mock,patch
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.config import load_config,validate_config
from env.v37_weapon import MADSACGaussianFiniteAmmoWeapon,expected_hit_probability
from env.v36_geometry import combat_geometry
from env.v36_reward import tactical_rewards
from tests.test_combat_v36 import state,pair
from tools._prepare_v37 import capture36

ROOT=Path(__file__).resolve().parents[1]
def cfg():return load_config(ROOT/'configs/combat_environment_v37.yaml')
def weapon():return MADSACGaussianFiniteAmmoWeapon(**cfg()['weapon'])
def env():
    e=MultiUAVCombatEnv(cfg());e.reset(31);return e
def setup(e,red,blue):
    e.red=red+[state(alive=False) for _ in range(5-len(red))]
    e.blue=blue+[state(alive=False) for _ in range(5-len(blue))]
def step(e):
    with patch.object(e,'_advance',return_value=np.zeros(5)):
        return e.step(np.zeros((5,3)),np.zeros((5,3)))

@pytest.mark.parametrize('distance,angle,value',[(2000,0,.700000),(2000,5,.668944),(2000,15,.603416),
    (2000,30,.500000),(1500,0,.794100),(1500,30,.616511),(1000,0,.900519),(1000,30,.776407)])
def test_analytic_calibration(distance,angle,value):
    a=math.radians(angle)
    assert expected_hit_probability(distance,a,a)==pytest.approx(value,abs=5.1e-7)

def test_exact_anchor_and_boundary_and_monotonicity():
    p=expected_hit_probability
    assert p(2000,0,0)==pytest.approx(.7,abs=1e-14)
    assert p(2000,math.pi/6,math.pi/6)==pytest.approx(.5,abs=1e-14)
    assert p(1000,0,0)>p(1500,0,0)>p(2000,0,0)
    assert p(2000,0,0)>p(2000,math.pi/12,math.pi/12)>p(2000,math.pi/6,math.pi/6)
    for dim in (0,1):
        values=[p(1500,*(([a,0] if dim==0 else [0,a]))) for a in np.linspace(0,math.pi/6,11)]
        assert np.all(np.diff(values)<=0)

@pytest.mark.parametrize('epsilon,ata,ha,hit',[(0,0,0,True),(0,.6,0,False),(0,0,.6,False),
    (.6,0,0,False),(-.2,.6,.6,True),(.2,.1,.4,False),(.2,.4,.1,False)])
def test_single_shared_normal_no_uniform(epsilon,ata,ha,hit):
    rng=Mock();rng.normal.return_value=epsilon
    assert weapon().attempt_hit(rng,2000,ata,ha)==hit
    rng.normal.assert_called_once_with(0.,1.);rng.random.assert_not_called()

@pytest.mark.parametrize('distance,angle',[(2000,0),(2000,30),(1000,0),(1000,30)])
def test_fixed_seed_monte_carlo(distance,angle):
    w=weapon();a=math.radians(angle);rng=np.random.default_rng(3737)
    observed=sum(w.attempt_hit(rng,distance,a,a) for _ in range(30000))/30000
    assert abs(observed-w.expected_hit_probability(distance,a,a))<.015

@pytest.mark.parametrize('side',['red','blue'])
@pytest.mark.parametrize('reason',['dead_attacker','dead_target','ammo','range','ata','ha'])
def test_ineligible_never_samples(side,reason):
    e=env();a,b=pair();setup(e,[a],[b])
    if reason=='dead_attacker':a.alive=False
    if reason=='dead_target':b.alive=False
    if reason=='ammo':getattr(e,f'{side}_ammo')[0]=0
    if reason=='range':b.x=2001
    if reason=='ata':a.psi=math.pi
    if reason=='ha':a.theta=math.pi/4
    e.rng=Mock()
    attempts=e._entry_attempts(e.red,e.blue,getattr(e,f'{side}_fire_states'),side)
    assert attempts==[];e.rng.normal.assert_not_called();e.rng.random.assert_not_called()

def test_mirror_geometry_probability_and_hit():
    a,b=pair(1500,20,10);am,bm=copy.deepcopy(a),copy.deepcopy(b)
    for s in (am,bm):s.x=-s.x;s.y=-s.y;s.psi+=math.pi
    gs=[combat_geometry([x],[y]) for x,y in ((a,b),(am,bm))]
    args=[tuple(float(g[k][0,0]) for k in ('distance','ata','ha')) for g in gs]
    w=weapon();assert w.expected_hit_probability(*args[0])==pytest.approx(w.expected_hit_probability(*args[1]))
    for epsilon in (-1.,0.,1.):
        assert w.attempt_hit(Mock(normal=Mock(return_value=epsilon)),*args[0])==w.attempt_hit(Mock(normal=Mock(return_value=epsilon)),*args[1])

def test_red_then_blue_sampling_and_simultaneous_resolution():
    e=env();setup(e,[state()],[state(1500,psi=math.pi)])
    rng=Mock();rng.normal.side_effect=[-1.,-1.];e.rng=rng
    _,r,t,_,info=step(e)
    assert t and info['draw'] and info['red_attack_kills']==info['blue_attack_kills']==1
    assert rng.normal.call_count==2;rng.random.assert_not_called()
    assert info['episode_r1_total']==0 and not info['outcome_rewards'].any()
    assert not info['team_casualty_rewards'].any()
    # Asymmetric draws show the retained Red-first order.
    e=env();setup(e,[state()],[state(1500,psi=math.pi)])
    e.rng=Mock();e.rng.normal.side_effect=[2.,-2.]
    _,_,_,_,info=step(e)
    assert info['red_weapon_hits']==0 and info['blue_weapon_hits']==1

@pytest.mark.parametrize('side',['red','blue'])
def test_pair_entry_nearest_consumption_rearm_and_six_ammo(side):
    e=env();own=[state(alive=i==0) for i in range(5)];targets=[state(1500+i*100,alive=i<2) for i in range(5)]
    ammo=getattr(e,f'{side}_ammo');fire=getattr(e,f'{side}_fire_states')
    e.rng=Mock();e.rng.normal.return_value=2.
    for shot in range(6):
        assert [j for _,j,_ in e._entry_attempts(own,targets,fire,side)]==[0]
        assert e._entry_attempts(own,targets,fire,side)==[]
        own[0].psi=math.pi;assert e._entry_attempts(own,targets,fire,side)==[]
        own[0].psi=0
    assert ammo[0]==0 and e.rng.normal.call_count==6
    assert e._entry_attempts(own,targets,fire,side)==[]

@pytest.mark.parametrize('mode',['kill','death','boundary','draw','timeout'])
def test_event_aliases_zero_extras_and_composition(mode):
    e=env();setup(e,[state()],[state(1500)])
    if mode=='death':e.blue[0].x=-1500
    if mode=='boundary':e.red[0].x=5001
    if mode=='draw':e.blue[0].psi=math.pi
    if mode=='timeout':e.blue[0].x=4500;e.steps=e.max_steps-1
    e.rng=Mock();e.rng.normal.return_value=-1.
    _,r,t,tr,info=step(e)
    assert t or tr
    expected={'kill':10,'death':-10,'boundary':0,'draw':0,'timeout':0}[mode]
    assert info['episode_r1_total']==expected
    assert info['episode_r2_total']==(-10 if mode=='boundary' else 0)
    for name in ('outcome','team_casualty'):
        assert not info[f'{name}_rewards'].any() and info[f'episode_{name}_total']==0
    np.testing.assert_array_equal(r,(info['r1_rewards']+info['r2_rewards']+info['r3_rewards']+info['r4_rewards']+info['safety_rewards']).astype(np.float32))
    assert info['pbrs_enabled'] is False and not info['phi_current'].any() and not info['phi_next'].any()
    for i in range(1,5):assert info[f'madsac_r{i}_total']==info[f'episode_r{i}_total']
    if mode=='kill':assert info['red_success'] and info['r4_rewards'][0]==.1 # cached before death
    if mode=='death':assert info['blue_win']
    if mode in ('draw','timeout'):assert info['draw']

def test_casualty_does_not_penalize_other_agents_shared_credit_and_safety():
    e=env();setup(e,[state(5001),state(0,1000)],[state(4000,1000)])
    _,_,_,_,info=step(e);assert not info['team_casualty_rewards'].any() and info['event_rewards'][1]==0
    combat,boundary=e._event_reward_components([],[],{0:[0,1]},{});assert combat[:2].tolist()==[5.,5.]
    e=env();setup(e,[state(),state(0,100)],[state(4500)])
    _,_,_,_,info=step(e);assert (info['safety_rewards'][:2]<0).all()
    assert info['episode_r4_total']==0 and info['episode_safety_total']<0

@pytest.mark.parametrize('angle,offense,defense',[(25,.01,-.015),(10,.02,-.025),(4,.1,-.15)])
def test_inherited_tactical_coefficients(angle,offense,defense):
    c=cfg();a,b=pair(1500,angle,angle,20)
    assert tactical_rewards([a],[b],c['sensor'],c['weapon'],c['reward'])['tactical_offense'][0]==offense
    b,a=pair(1500,angle,angle,20);b.x=-1500;a.x=0
    assert tactical_rewards([a],[b],c['sensor'],c['weapon'],c['reward'])['tactical_defense'][0]==defense
    a,b=pair(3000,20,20)
    assert tactical_rewards([a],[b],c['sensor'],c['weapon'],c['reward'])['guide'][0]==.001

def test_configuration_isolation_and_original_reference():
    old=load_config(ROOT/'configs/combat_environment_v36.yaml');new=cfg()
    for key in old.keys()-{'environment_version','weapon','reward'}:assert old[key]==new[key]
    assert old['weapon']['hit_probability']==.7 and old['reward']['team_casualty_penalty']==-2
    assert old['reward']['win_reward']==20 and old['reward']['lose_penalty']==-20
    for name in ('mappo','rmappo','ea_mappo','stea_mappo'):
        assert (ROOT/f'configs/{name}_5v5_v37.yaml').read_bytes()==(ROOT/f'configs/{name}_5v5_v36.yaml').read_bytes()
    assert capture36()==json.loads((ROOT/'tests/v37_v36_reference.json').read_text())

@pytest.mark.parametrize('d,ata,ha,aa,yes',[(2000,0,0,180,True),(2001,0,0,0,False),
    (1500,30,30,90,True),(1500,30.1,0,0,False),(1500,0,30.1,0,False)])
def test_envelope_unchanged_aa_not_gate(d,ata,ha,aa,yes):
    a,b=pair(d,ata,ha,aa);assert weapon().qualifies(a,b,6)==yes

def test_guide_is_r3_not_r4_and_evaluator_uses_real_aliases():
    e=env();setup(e,[state()],[state(3000)])
    _,r,_,_,info=step(e)
    assert info['r3_rewards'][0]==.001 and info['r4_rewards'][0]==0
    assert info['episode_guide_total']==info['episode_r3_total']==.001
    assert info['episode_tactical_total']==info['episode_r4_total']==0
    from algorithm.common.evaluator import aggregate_combat_records
    record=dict(info,episode_return=float(r.sum()),mean_agent_episode_return=float(r.mean()))
    totals=aggregate_combat_records([record])
    assert totals['average_madsac_r3_total']==.001 and totals['average_episode_tactical_total']==0

def test_same_hardware_dynamics_and_independent_blue_rng():
    e=env();e.blue=copy.deepcopy(e.red);e.blue_ammo[:]=e.red_ammo
    action=np.random.default_rng(37).uniform(-1,1,(5,3))
    e._advance(e.red,action);e._advance(e.blue,action)
    np.testing.assert_array_equal([s.as_array() for s in e.red],[s.as_array() for s in e.blue])
    assert e.fixed_policy.sensor_config is e.config['sensor'] and e.weapon.ammo_per_aircraft==6
    before=copy.deepcopy(e.fixed_policy.rng.bit_generator.state)
    e.weapon.attempt_hit(e.rng,1500,0,0)
    assert before==e.fixed_policy.rng.bit_generator.state

@pytest.mark.parametrize('section,key,value',[('weapon','hit_probability',.7),('weapon','effective_hit_distance',0),
    ('weapon','ata_noise_scale',-1),('weapon','ha_noise_scale',1),('weapon','range_max',4000),
    ('sensor','range_max',3000),('reward','team_casualty_penalty',-2),('reward','win_reward',20),
    ('reward','lose_penalty',-20),('reward','draw_reward',1),('reward','potential_scale',0),('reward','guide_reward',.01)])
def test_frozen_schema_rejects_changes(section,key,value):
    c=cfg();c[section][key]=value
    with pytest.raises(ValueError):validate_config(c)
