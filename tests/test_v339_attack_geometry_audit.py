"""Independent geometry, accounting and observer-transparency acceptance tests."""
from copy import deepcopy
import math
import numpy as np
import pytest
from tools.audit_v339_attack_geometry import (pair,environment,independent_angles,independent_probability,
    PreFireObserver,states,red_actions,vector_geometry,masks,runs,windows,analyze_episode)
from env.geometry import engagement_geometry
from env.models import AircraftState
from env.v337_reward import tactical_pair_rewards_v337

@pytest.mark.parametrize('pitch',[-60,-45,-30,0,30,45,60])
@pytest.mark.parametrize('ata,ha',[(-30,-30),(0,0),(15,30),(30,30),(45,-45)])
def test_angles_independent_and_pitch_semantics(pitch,ata,ha):
    a,b=pair(500,ata,ha,pitch,aa=20);g=engagement_geometry(a,b);d,x,y,z,alpha=independent_angles(a,b)
    assert g.distance==pytest.approx(d,abs=1e-10)
    assert np.degrees([g.ata,g.ha,g.aa])==pytest.approx([ata,ha,20],abs=1e-10)
    assert g.off_boresight==pytest.approx(alpha,abs=3e-8)
    # Spherical identity is independent of implementation's body-vector code.
    expected=math.acos(np.clip(math.sin(math.radians(pitch))*math.sin(math.radians(ha))+
        math.cos(math.radians(pitch))*math.cos(math.radians(ha))*math.cos(math.radians(ata)),-1,1))
    assert alpha==pytest.approx(expected,abs=3e-8)

@pytest.mark.parametrize('pitch',[-60,-45,0,45,60])
def test_same_position_pitch_does_not_change_weapon_or_paper_rewards(pitch):
    e=environment();a,b=pair(500,pitch=pitch);g=engagement_geometry(a,b)
    assert g.ata==g.ha==0 and math.degrees(g.off_boresight)==pytest.approx(abs(pitch))
    assert e.weapon.qualifies(a,b,6)
    p=tactical_pair_rewards_v337([a],[b],e.config['reward'],e.config['weapon'])
    assert p['tactical_offense'][0,0]==pytest.approx(.1)

@pytest.mark.parametrize('d,yes',[(1000-1e-7,True),(1000,True),(1000+1e-7,False)])
@pytest.mark.parametrize('angle,allowed',[(30-1e-8,True),(30,True),(30+1e-8,False)])
def test_formal_range_and_angle_boundaries(d,yes,angle,allowed):
    e=environment();a,b=pair(d,ata=angle)
    assert e.weapon.qualifies(a,b,6)==(yes and allowed)
    assert not e.weapon.qualifies(a,b,0)
    b.alive=False;assert not e.weapon.qualifies(a,b,6)

@pytest.mark.parametrize('delta',[(0,0,0),(1e-12,0,0),(0,0,500),(0,0,-500),(1e-12,0,-500)])
def test_degenerate_los_finite(delta):
    a,b=pair(pitch=60);b.x,b.y,b.z=a.x+delta[0],a.y+delta[1],a.z+delta[2];g=engagement_geometry(a,b)
    assert np.isfinite([g.ata,g.ha,g.aa,g.off_boresight,g.distance]).all()
    if delta==(0,0,0):assert g.off_boresight==g.ha==g.ata==0

@pytest.mark.parametrize('d',[500,1000,1500,2000,4000])
@pytest.mark.parametrize('ata,ha',[(0,0),(30,30),(30,-30),(15,0)])
def test_shared_gaussian_probability_by_independent_quadrature(d,ata,ha):
    e=environment();x,y=np.deg2rad([ata,ha])
    assert e.weapon.expected_hit_probability(d,x,y)==pytest.approx(independent_probability(d,x,y,e.weapon),abs=1e-14)

def small_environment(kind):
    e=environment(0);dead=lambda:AircraftState(0,0,-3000,225,0,0,False)
    e.red=[dead() for _ in range(4)];e.blue=[dead() for _ in range(4)]
    if kind=='binding':
        e.blue[0]=AircraftState(0,0,-3000,225,0,0)
        e.red[0]=AircraftState(-300,0,-3000,225,0,math.pi)
        e.red[1]=AircraftState(500,0,-3000,225,0,0)
    elif kind=='multi':
        e.red[0]=AircraftState(0,0,-3000,225,0,0)
        e.blue[0]=AircraftState(500,0,-3000,225,0,0)
        e.blue[1]=AircraftState(700,0,-3000,225,0,0)
    else:
        e.red[0]=AircraftState(0,0,-3000,225,0,0)
        e.blue[0]=AircraftState(500,0,-3000,225,0,math.pi)
    return e

def observe_one(e,actions=None):
    actions=np.zeros((4,3),np.float32) if actions is None else actions
    with PreFireObserver(e) as obs:
        before=states(e);result=e.step(actions);obs.finish(before,actions,np.full(4,-1),result)
        data=obs.arrays()
    return data,result

def test_blue_bound_to_start_target_even_when_other_target_attackable():
    e=small_environment('binding');data,result=observe_one(e)
    m=masks(vector_geometry(data['pre'],1),1000)
    assert data['blue_targets'][0,0]==0 and not m['full'][0,0,0] and m['full'][0,0,1]
    assert data['attempt_targets'][0,1,0]==-1 and e.blue_ammo[0]==6
    assert result[4]['blue_pursuit_target_indices'][0]==0

def test_red_multi_target_selects_nearest_and_fires_once():
    e=small_environment('multi');data,_=observe_one(e)
    assert masks(vector_geometry(data['pre'],0),1000)['full'][0,0].sum()==2
    assert data['attempt_targets'][0,0,0]==0 and e.red_ammo[0]==5

def test_prefire_alive_is_distinct_from_same_step_weapon_death():
    e=small_environment('mutual');data,result=observe_one(e)
    assert data['pre'][0,:,0,6].tolist()==[1,1]
    assert data['post'][0,:,0,6].tolist()==[0,0]
    assert data['hit'][0,:,0].tolist()==[True,True]
    assert data['death_cause'][0,:,0].tolist()==[4,4]
    assert result[2] and result[4]['draw']
    rows,_,_=analyze_episode(data,0,'fixture',result[4],e.combat_counts)
    assert all(r['live_pair_steps']==1 and r['pair_full']==1 for r in rows)

def test_offline_nested_ranges_do_not_modify_states_rng_or_denominators():
    a,b=pair(1500);e=small_environment('multi');e.red[0]=a;e.blue[0]=b;e.blue[1].alive=False
    before=deepcopy((e.red,e.blue,e.rng.bit_generator.state))
    g=vector_geometry(states(e)[None],0);m=[masks(g,r) for r in (1000,2000,4000)]
    assert [int(x['live'].sum()) for x in m]==[1,1,1]
    assert [int(x['full'].sum()) for x in m]==[0,1,1]
    assert (e.red,e.blue,e.rng.bit_generator.state)==before

def test_agent_window_survives_target_switch_but_pair_windows_are_separate():
    mask=np.zeros((6,1,2),bool);mask[0:2,0,0]=True;mask[2:5,0,1]=True
    g={key:np.ones_like(mask,dtype=float) for key in ('distance','ata','ha','alpha','pitch','relative_speed','closing_speed')}
    agents=windows(mask,g,{},.1,'agent');pairs=windows(mask,g,{},.1,'pair')
    assert len(agents)==1 and agents[0]['duration_s']==.5 and agents[0]['targets_seen']=='0,1'
    assert [x['duration_s'] for x in pairs]==pytest.approx([.2,.3])
    assert runs([False,True,False,True,True])==[(1,2),(3,5)]
    assert windows(mask[:1],{k:v[:1] for k,v in g.items()},{},.1,'agent')[0]['duration_s']==.1

@pytest.mark.parametrize('mode',['zero','pursuit','lateral','vertical'])
def test_observer_full_episode_preserves_transition_rng_actions_and_calls(mode):
    a,b=environment(7),environment(7)
    with PreFireObserver(b) as obs:
        while True:
            actions,targets=red_actions(a,mode);start=states(b)
            x,y=a.step(actions),b.step(actions.copy());obs.finish(start,actions,targets,y)
            np.testing.assert_array_equal(states(a),states(b))
            assert a.rng.bit_generator.state==b.rng.bit_generator.state
            assert a.combat_counts==b.combat_counts and a.fixed_policy.diagnostics()==b.fixed_policy.diagnostics()
            for index in (0,1):np.testing.assert_array_equal(x[index],y[index])
            assert x[2:4]==y[2:4] and x[4]['termination_reason']==y[4]['termination_reason']
            for side in ('red','blue'):
                for suffix in ('_ammo','_fire_cooldown_steps'):np.testing.assert_array_equal(getattr(a,side+suffix),getattr(b,side+suffix))
            for field in ('executed_red_actions','executed_blue_actions','guide_rewards','tactical_offense_rewards','tactical_defense_rewards'):
                np.testing.assert_array_equal(x[4][field],y[4][field])
            if x[2] or x[3]:break
        assert obs.calls==dict(dense=a.steps,resolve=a.steps)
    assert 'dense_combat_reward' not in b.__dict__ and '_resolve_combat' not in b.__dict__

@pytest.mark.parametrize('cause,z,x',[(1,1,0),(2,-3000,4995),(3,-6001,0)])
def test_noncombat_deaths_excluded_from_prefire_pairs_and_counted(cause,z,x):
    e=small_environment('multi');e.red[0].z=z;e.red[0].x=x
    data,result=observe_one(e)
    assert data['death_cause'][0,0,0]==cause
    assert masks(vector_geometry(data['pre'],0),4000)['live'].sum()==0
    assert data['attempt_targets'][0,0,0]==-1 and result[2]

def test_cooldown_and_ammo_are_separate_from_geometry():
    e=small_environment('multi');e.red_fire_cooldown_steps[0]=2
    data,_=observe_one(e);assert masks(vector_geometry(data['pre'],0),1000)['full'].sum()==2
    assert data['attempt_targets'][0,0,0]==-1 and e.red_fire_cooldown_steps[0]==1
    e=small_environment('multi');e.red_ammo[0]=0
    data,_=observe_one(e);assert masks(vector_geometry(data['pre'],0),1000)['full'].sum()==2
    assert data['attempt_targets'][0,0,0]==-1
