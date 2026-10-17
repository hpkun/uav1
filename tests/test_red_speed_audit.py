"""Alignment and diagnostic semantics for the read-only speed analysis."""
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from env.geometry import engagement_geometry
from env.models import AircraftState
from tools.analyze_red_speed import episode_samples,event_indices,describe,grouped_statistics,overshoot_proxies

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    T=4
    red=np.tile([0.,0.,-3000.,200.,0.,0.],(T+1,1,1))
    blue=np.tile([500.,0.,-3000.,225.,0.,0.],(T+1,1,1))
    trace=dict(red_kinematics=red,blue_kinematics=blue,
        red_alive=np.ones((T+1,1),bool),blue_alive=np.ones((T+1,1),bool),
        red_actions=np.zeros((T,1,3)),blue_pursuit_target_indices=np.zeros((T,1),int),
        red_ammo=np.full((T,1),6),blue_ammo=np.full((T,1),6),dt=np.array(.1))
    config=yaml.safe_load((ROOT/'configs/combat_environment_v338.yaml').read_text())
    metadata=dict(weapon_config=config['weapon'],blue_target_available=True,
        initial_ammo={'red':[6],'blue':[6]},events=[],episode_seed=99,dt=.1,
        red_win=False,blue_win=True,termination_reason='blue_win_elimination',episode_return=-10)
    return trace,metadata


def test_exclude_frozen_states_include_last_live_transition():
    t,m=fixture();t['red_alive'][2:]=False
    t['red_kinematics'][2:,0,3]=150
    f,_,_=episode_samples(t,m)
    assert len(f)==2 and f.step.tolist()==[0,1]
    assert f.dv.tolist()==[0,-50] and f.alive_after.tolist()==[True,False]


def test_blue_pursuit_is_not_inferred_from_nearest_distance():
    t,m=fixture();t['blue_pursuit_target_indices'][:]=-1
    f,_,_=episode_samples(t,m)
    assert (f.nearest_blue_diagnostic==0).all()
    assert not f.pursued_by_blue.any()
    m['blue_target_available']=False
    with pytest.raises(AssertionError,match='authoritative'):episode_samples(t,m)


def test_authoritative_geometry_pitch_and_closing_sign():
    t,m=fixture();t['red_kinematics'][:,:,4]=np.deg2rad(20)
    t['blue_kinematics'][:,:,2]=-3000-500*np.tan(np.deg2rad(35))
    f,_,_=episode_samples(t,m)
    g=engagement_geometry(AircraftState(*t['red_kinematics'][0,0]),AircraftState(*t['blue_kinematics'][0,0]))
    assert np.isclose(f.ha.iloc[0],np.deg2rad(35)) and np.isclose(f.elevation_error.iloc[0],np.deg2rad(15))
    assert f.ata.iloc[0]==g.ata and f.aa.iloc[0]==g.aa and f.body_el.iloc[0]==g.boresight_elevation_error
    t,m=fixture();f,_,_=episode_samples(t,m)
    assert np.allclose(f.closing_speed,-25)


def test_actual_event_frame_and_causal_transition_are_distinct():
    t,m=fixture();m['events']=[dict(type='fire_attempt',side='red',attacker=0,step=2,hit=True),
        dict(type='attack_kill',side='red',step=2,attackers=[0])]
    f,v,flags=episode_samples(t,m)
    assert f.loc[f.fire,'step'].tolist()==[1]
    assert f.loc[f.hit,'step'].tolist()==[1] and f.loc[f.kill,'step'].tolist()==[1]
    events=event_indices(t,m,v,flags)
    actual=[e for e in events if e['event'] in ['fire','hit','kill']]
    assert len(actual)==3 and all(e['frame']==2 and e['time_s']==.2 for e in actual)


def test_geometry_window_not_same_as_ammo_eligibility():
    t,m=fixture();m['initial_ammo']['red']=[0];t['red_ammo'][:]=0
    f,_,_=episode_samples(t,m)
    assert f.fire_window.all() and not f.eligible_ammo.any()


def test_delta_speed_not_inferred_from_saturated_action():
    t,m=fixture();t['red_kinematics'][:,:,3]=150;t['red_actions'][:,:,2]=-1
    f,_,_=episode_samples(t,m);s=describe(f)
    assert s['hold']==1 and s['decelerate']==0 and s['av_lt_neg09']==1
    assert s['v_near_min']==1


def test_speed_histogram_and_threshold_boundaries():
    t,m=fixture();t['red_kinematics'][:-1,0,3]=[150,200,240,300]
    f,_,_=episode_samples(t,m);s=describe(f)
    assert s['v_lt200']==.25 and s['v_210_240']==.25 and s['v_gt240']==.25
    assert np.isclose(sum(v for k,v in s.items() if k.startswith('bin_')),1)
    assert s['bin_275-300']==.25


@pytest.mark.parametrize('distance,angle,cell',[(999,29.9,'near_small'),(1000,30,'medium_medium'),(3000,60,'medium_medium'),(3001,60.1,'far_large')])
def test_three_by_three_boundaries(distance,angle,cell):
    t,m=fixture();f,_,_=episode_samples(t,m)
    f['distance']=distance;f['heading_deg']=angle
    rows=grouped_statistics(f)
    assert next(r['count'] for r in rows if r['group']=='distance_heading' and r['condition']==cell)==4


def test_analysis_does_not_mutate_trace():
    t,m=fixture();before=copy.deepcopy(t)
    episode_samples(t,m)
    for name in t:assert np.array_equal(t[name],before[name])


def test_pooled_quartiles_keep_policy_identity_with_shared_episode_seeds():
    t,m=fixture();a,_,_=episode_samples(t,m);b=a.copy()
    a['policy_run']='a';b['policy_run']='b'
    a['episode_return']=-40;a['v']=150
    b['episode_return']=40;b['v']=300
    rows=grouped_statistics(pd.concat([a,b],ignore_index=True))
    top=next(r for r in rows if r['group']=='return_quartile' and r['condition']=='top25')
    bottom=next(r for r in rows if r['group']=='return_quartile' and r['condition']=='bottom25')
    assert top['count']==4 and top['mean']==300
    assert bottom['count']==4 and bottom['mean']==150


def test_checkpoint_inventory_refuses_cpu_fallback(monkeypatch,tmp_path):
    import sys,torch
    from tools.audit_red_speed import main
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(sys,'argv',['audit_red_speed.py','--inventory-only','--output-dir',str(tmp_path/'audit')])
    with pytest.raises(RuntimeError,match='CUDA mandatory'):main()
    assert not (tmp_path/'audit').exists()


def test_shot_establishes_first_window_even_if_simultaneous_kill_removes_target():
    t,m=fixture();t['blue_kinematics'][:,:,0]=4000
    m['events']=[dict(type='fire_attempt',side='red',attacker=0,step=2,hit=True)]
    _,v,flags=episode_samples(t,m)
    events=event_indices(t,m,v,flags)
    assert next(e['frame'] for e in events if e['event']=='first_1km')==2
    assert next(e['frame'] for e in events if e['event']=='first_fire_window')==2


def test_overshoot_proxy_requires_continuous_authoritative_pursuit():
    t,m=fixture();T=20
    for key in ['red_kinematics','blue_kinematics','red_alive','blue_alive']:
        t[key]=np.repeat(t[key][:1],T+1,axis=0)
    t['blue_pursuit_target_indices']=np.zeros((T,1),int)
    t['red_kinematics'][:,0,0]=np.arange(T+1)*20
    t['blue_kinematics'][:,0,0]=-30+np.arange(T+1)*22.5
    events=overshoot_proxies(t)
    assert len(events)==1 and events[0]['frame']==12
    t['blue_pursuit_target_indices'][5:]=-1
    assert overshoot_proxies(t)==[]


def test_v338_canonical_observer_full_episode_exact_semantic_invariance():
    from env.combat_env import MultiUAVCombatEnv
    from tools.combat_visualization import RecordingCombatEnv,states_array
    def exact(a,b):
        if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:exact(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b):exact(x,y)
        else:assert a==b
    ec=yaml.safe_load((ROOT/'configs/combat_environment_v338.yaml').read_text())
    base,observed=MultiUAVCombatEnv(ec),RecordingCombatEnv(ec)
    exact(base.reset(20001000),observed.reset(20001000));rng=np.random.default_rng(41)
    while True:
        action=rng.uniform(-1,1,(4,3)).astype(np.float32)
        a,b=base.step(action),observed.step(action.copy());exact(a,b)
        exact(states_array(base.red+base.blue),states_array(observed.red+observed.blue))
        exact(base.rng.bit_generator.state,observed.rng.bit_generator.state)
        for side in ['red','blue']:
            exact(getattr(base,side+'_ammo'),getattr(observed,side+'_ammo'))
            exact(getattr(base,side+'_fire_cooldown_steps'),getattr(observed,side+'_fire_cooldown_steps'))
        if a[2] or a[3]:break
