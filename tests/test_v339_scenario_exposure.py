"""Acceptance tests for authorized instance-only scenario interventions."""
import math
from copy import deepcopy
import numpy as np
import pytest
from tools.audit_v339_scenario_exposure import (NonlethalSettlement,setup,analyze,pair_conditions,paired_until_fork,LETHAL,NONLETHAL)
from tools.audit_v339_attack_geometry import states,PreFireObserver,red_actions,vector_geometry
from env.models import AircraftState
from env.v338_observation import build_team_observations_v338

def one(e):
    actions,targets=red_actions(e,'pursuit')
    with PreFireObserver(e) as obs:
        before=states(e);result=e.step(actions);obs.finish(before,actions,targets,result)
        return obs.arrays(),result

def mutual():
    e,_,_=setup(0);dead=lambda:AircraftState(0,0,-3000,225,0,0,False)
    e.red=[AircraftState(0,0,-3000,225,0,0)]+[dead() for _ in range(3)]
    e.blue=[AircraftState(500,0,-3000,225,0,math.pi)]+[dead() for _ in range(3)]
    return e

@pytest.mark.parametrize('offset',[0,45,-45,90,-90])
def test_heading_changes_only_authorized_initial_field_and_rebuilds_obs(offset):
    e,obs,before=setup(11,offset);after=states(e)
    np.testing.assert_array_equal(before[:, :, [0,1,2,3,4,6]],after[:,:, [0,1,2,3,4,6]])
    np.testing.assert_array_equal(before[1],after[1])
    error=np.arctan2(np.sin(after[0,:,5]-before[0,:,5]-math.radians(offset)),np.cos(after[0,:,5]-before[0,:,5]-math.radians(offset)))
    assert abs(error).max()<1e-14
    np.testing.assert_array_equal(obs,build_team_observations_v338(e.red,e.blue,e.config['observation']))
    e0,_,_=setup(11);assert e.config==e0.config and e.rng.bit_generator.state==e0.rng.bit_generator.state

def test_nonlethal_suppresses_kill_not_hits_rng_ammo_cooldown_or_motion():
    a,b=mutual(),mutual();original=b._resolve_combat
    x,rx=one(a)
    with NonlethalSettlement(b) as intervention:y,ry=one(b)
    assert b._resolve_combat==original and '_resolve_combat' not in b.__dict__
    for name in ('pre','actions','ammo_post','cooldown_post','attempt_targets','hit','dense'):
        np.testing.assert_array_equal(x[name],y[name])
    np.testing.assert_array_equal(x['post'][...,:6],y['post'][...,:6]);assert a.rng.bit_generator.state==b.rng.bit_generator.state
    assert x['hit'].sum()==2 and y['hit'].sum()==2 and x['credited_kills'].sum()==2 and y['credited_kills'].sum()==0
    assert not a.red[0].alive and not a.blue[0].alive and b.red[0].alive and b.blue[0].alive
    assert rx[2] and not ry[2] and not ry[3] and np.asarray(intervention.suppressed).sum()==2
    assert b.combat_counts['red']['attack_kills']==b.combat_counts['blue']['attack_kills']==0

@pytest.mark.parametrize('z,x,cause',[(1,0,1),(-3000,4999,2),(-6001,0,3)])
def test_nonlethal_keeps_noncombat_deaths(z,x,cause):
    e=mutual();e.red[0].x=x;e.red[0].z=z;e.red[0].psi=0
    with NonlethalSettlement(e):
        with PreFireObserver(e) as obs:
            before=states(e);actions=np.zeros((4,3),np.float32);r=e.step(actions);obs.finish(before,actions,np.full(4,-1),r);d=obs.arrays()
    assert not e.red[0].alive and d['death_cause'][0,0,0]==cause and r[2]

@pytest.mark.parametrize('policy',['zero','pursuit','lateral','vertical'])
def test_paired_prefix_through_first_real_kill(policy):
    result=paired_until_fork(0,policy)
    assert result['rng_equal_at_fork'] and result['fork_time_s'] is not None

@pytest.mark.parametrize('scenario',['tail_1v1_1500','tail_1v1_2500','tail_4v4_1500','tail_4v4_2500'])
def test_positive_control_initial_geometry_and_actual_reward(scenario):
    e,obs,_=setup(0,scenario=scenario);g=vector_geometry(states(e)[None],0);m=pair_conditions(g)
    assert m['r4_nonzero_pair'].any() and m['tail_aspect'].any()
    assert (np.hypot(states(e)[...,:1],states(e)[...,1:2])<5000).all()
    with NonlethalSettlement(e):d,r=one(e)
    assert (d['dense'][:,1]>0).any() and r[4]['tactical_offense_rewards'].sum()>0
    np.testing.assert_array_equal(obs,build_team_observations_v338([AircraftState(*s[:6],alive=bool(s[6])) for s in d['start'][0,0]],
        [AircraftState(*s[:6],alive=bool(s[6])) for s in d['start'][0,1]],e.config['observation']))

def test_aa_alone_is_not_nonzero_r4_and_weight_zero_boundary():
    e,_,_=setup(0,scenario='tail_1v1_2500');e.blue[0].x=e.red[0].x+3000
    c=pair_conditions(vector_geometry(states(e)[None],0))
    assert c['tail_aspect'][0,0,0] and c['r4_geometry'][0,0,0] and not c['r4_nonzero_pair'][0,0,0]
    e.red[0].psi=math.pi
    c=pair_conditions(vector_geometry(states(e)[None],0))
    assert c['tail_aspect'][0,0,0] and not c['r4_geometry'][0,0,0]

def test_same_seed_nonlethal_reproducibility_and_instance_restore():
    a,_,_=setup(2);b,_,_=setup(2);before_config=deepcopy(a.config)
    with NonlethalSettlement(a),NonlethalSettlement(b):
        for _ in range(220):
            x,_=red_actions(a,'vertical');y,_=red_actions(b,'vertical');np.testing.assert_array_equal(x,y)
            rx,ry=a.step(x),b.step(y);np.testing.assert_array_equal(states(a),states(b));np.testing.assert_array_equal(rx[1],ry[1]);assert a.rng.bit_generator.state==b.rng.bit_generator.state
            if rx[2] or rx[3]:break
    assert a.config==before_config and '_resolve_combat' not in a.__dict__

def test_counterfactual_obeys_formal_timeout():
    e,_,_=setup(0);e.steps=999
    with NonlethalSettlement(e):r=e.step(np.zeros((4,3),np.float32))
    assert r[3] and e.steps==1000
