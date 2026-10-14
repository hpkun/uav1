"""Only offline audit helpers and read-only instrumentation are tested."""
import csv
from io import StringIO
import math
import numpy as np
import pytest
from env.combat_env import MultiUAVCombatEnv
from env.models import AircraftState
from env.v323_reward import tactical_rewards_v323
from tools.v335_paper_reward_reference import paper_pair_rewards,pair_geometry,comparison,project_aggregation,event_comparison
from tools.audit_combat_v335_paper_reward import Observer,AuditStats,bin_masks


def geometry(d=3500,ata=0,ha=0,aa=0,reverse_ata=180,reverse_ha=0,reverse_aa=180):
    return {**{k:np.array([[v]],float) for k,v in dict(distance=d,ata_r=math.radians(ata),ha_r=math.radians(ha),aa_r=math.radians(aa),
        ata_b=math.radians(reverse_ata),ha_b=math.radians(reverse_ha),aa_b=math.radians(reverse_aa),
        body_az_r=math.radians(ata),body_el_r=math.radians(ha),body_az_b=math.radians(reverse_ata),body_el_b=math.radians(reverse_ha)).items()},'live':np.array([[True]])}


@pytest.mark.parametrize('d,ata,ha,aa,r3,offense',[(5000,0,0,180,.001,0),(3500,0,0,180,0,0),
    (3500,20,20,20,0,.01),(3500,10,10,20,0,.02),(3500,4,4,20,0,.1),
    (4000,0,0,0,.001,.1),(4001,0,0,0,.001,0),(3999,0,0,0,0,.1),
    (3500,31,0,0,0,0),(3500,0,31,0,0,0),(3500,0,0,31,0,0),
    (5000,31,0,0,0,0),(3500,-20,-20,-20,0,.01),
    (3500,30,30,30,0,.01),(3500,15,15,30,0,.02),(3500,5,5,30,0,.1)])
def test_paper_canonical(d,ata,ha,aa,r3,offense):
    p=paper_pair_rewards(geometry(d,ata,ha,aa))
    assert p['r3'][0,0]==r3 and p['offense'][0,0]==offense


@pytest.mark.parametrize('angle,penalty',[(30,-.015),(15,-.025),(5,-.15),(-10,-.025),(31,0)])
def test_reverse_defense(angle,penalty):
    p=paper_pair_rewards(geometry(reverse_ata=angle,reverse_ha=angle,reverse_aa=20))
    assert p['defense'][0,0]==penalty


def test_reverse_distance_aspect_and_dead_gates():
    for d,aa in ((4001,20),(3500,31)):
        assert paper_pair_rewards(geometry(d=d,reverse_ata=0,reverse_aa=aa))['defense'][0,0]==0
    g=geometry();g['live'][:]=False
    assert all(np.all(paper_pair_rewards(g)[k]==0) for k in ('r3','offense','defense'))


def test_paper_never_reads_body_fields():
    g=geometry(5000);g['body_az_r'][:]=math.pi;g['body_el_r'][:]=math.pi
    assert paper_pair_rewards(g)['r3'][0,0]==.001
    g=geometry(3500,20,20,20);g['body_az_r'][:]=math.pi;g['body_el_r'][:]=math.pi
    assert paper_pair_rewards(g)['offense'][0,0]==.01


def test_geometry_source_ha_is_los_not_pitch_relative():
    r=AircraftState(0,0,-3000,225,math.pi/4,0);b=AircraftState(5000,0,-3000,225,0,0)
    g=pair_geometry([r],[b]);assert g['ata_r'][0,0]==g['ha_r'][0,0]==g['aa_r'][0,0]==0
    assert abs(g['body_el_r'][0,0])==pytest.approx(math.pi/4)
    e=MultiUAVCombatEnv('configs/combat_environment_v335.yaml')
    _,variants,_,_,_=comparison([r],[b],e.config['reward'],e.config['weapon'])
    assert variants['A']['r3'][0,0]==.001 and variants['B']['r3'][0,0]==0


@pytest.mark.parametrize('d,w4',[(1500,.75),(2500,.25),(3500,0)])
@pytest.mark.parametrize('angle,coefficient',[(20,.01),(10,.02),(4,.1)])
def test_weight_only_and_tactical_gate(d,w4,angle,coefficient):
    from tools.v335_paper_reward_reference import pair_variant
    e=MultiUAVCombatEnv('configs/combat_environment_v335.yaml');g=geometry(d,angle,angle,20)
    p=paper_pair_rewards(g);c=pair_variant(g,e.config['reward'],e.config['weapon'],'body','continuous')
    assert p['offense'][0,0]==coefficient
    assert c['offense'][0,0]==pytest.approx(coefficient*w4)


def test_project_aggregation_wrapper_reductions():
    pair=dict(r3=np.array([[.001,0]]),offense=np.array([[0,.1]]),defense=np.array([[-.15,0]]))
    a=project_aggregation(pair)
    assert a['r3'][0]==.001 and a['offense'][0]==.1 and a['defense'][0]==-.15


@pytest.mark.parametrize('seed',[1,31,199])
def test_current_D_exact_against_frozen_runtime_function(seed):
    rng=np.random.default_rng(seed);e=MultiUAVCombatEnv('configs/combat_environment_v335.yaml')
    red=[AircraftState(*rng.uniform([-2000,-2000,-5000,150,-.8,-math.pi],[2000,2000,-1000,300,.8,math.pi]),i!=3) for i in range(4)]
    blue=[AircraftState(*rng.uniform([-2000,-2000,-5000,150,-.8,-math.pi],[2000,2000,-1000,300,.8,math.pi]),i!=2) for i in range(4)]
    _,_,a,_,_=comparison(red,blue,e.config['reward'],e.config['weapon'])
    actual=tactical_rewards_v323(red,blue,e.config['reward'],e.config['weapon'])
    for key,name in (('r3','guide'),('offense','tactical_offense'),('defense','tactical_defense'),('dense','tactical')):
        np.testing.assert_array_equal(a['D'][key],actual[name])


def test_event_mapping_explicit_project_assumptions():
    e=MultiUAVCombatEnv('configs/combat_environment_v335.yaml')
    inputs=([2],[3],{0:[0,1,2]},{0:[0]})
    actual=e._event_reward_components(*inputs);reference=event_comparison(*inputs,4)
    for a,b in zip(actual,reference):np.testing.assert_array_equal(a,b)
    assert reference[1][2]==-10 and reference[0][3]==-10


def test_half_open_distance_bins():
    d=np.array([0,999,1000,1999,2000,2999,3000,3999,4000,4001])
    masks=bin_masks(d)
    assert np.all(np.sum(masks,axis=0)==1) and [m.sum() for m in masks]==[2,2,2,2,2]


def test_observer_does_not_change_trajectory_reward_rng_or_persistent_methods():
    from env.fixed_policy import NearestTargetPursuitPolicy
    a=MultiUAVCombatEnv('configs/combat_environment_v335.yaml');b=MultiUAVCombatEnv('configs/combat_environment_v335.yaml')
    a.reset(31);b.reset(31);policy=NearestTargetPursuitPolicy(a.config['blue_policy'],a.config['action'])
    stats=AuditStats();p=StringIO();q=StringIO();events=StringIO()
    with Observer(a,stats,csv.writer(p),csv.writer(q),events,0,31) as observer:
        for _ in range(180):
            actions=policy.team_actions(a.red,a.blue);x=a.step(actions);y=b.step(actions);observer.consume(x[1],x[4])
            assert a.red==b.red and a.blue==b.blue and x[2:4]==y[2:4]
            np.testing.assert_array_equal(x[0],y[0]);np.testing.assert_array_equal(x[1],y[1])
            assert a.rng.bit_generator.state==b.rng.bit_generator.state
            if x[2] or x[3]:break
    assert 'dense_combat_reward' not in a.__dict__ and '_event_reward_components' not in a.__dict__
    assert stats.pair.n>0 and stats.agent.n>0


def test_statistics_r3_comparisons_and_bins():
    from tools.v335_paper_reward_reference import pair_variant
    e=MultiUAVCombatEnv('configs/combat_environment_v335.yaml');stats=AuditStats()
    for d in (500,1500,3500,5000):
        g=geometry(d,aa=180)
        v={k:pair_variant(g,e.config['reward'],e.config['weapon'],geo,gate) for k,geo,gate in (('A','paper','hard'),('B','body','hard'),('C','paper','continuous'),('D','body','continuous'))}
        a={k:project_aggregation(value) for k,value in v.items()}
        w3=np.clip((g['distance']-1000)/2000,0,1);stats.update(g,v,a,w3,1-w3)
    result=stats.result();c=result['pair']['counts']
    assert result['pair']['sample_count']==result['agent']['sample_count']==4
    assert c['r3_current_active_paper_zero']==c['r3_current_greater']==2
    assert c['r3_both_active']==c['r3_both_zero']==1
    assert result['pair_bins']['3-4km']['means']['D_r3']==.001
    assert result['pair_bins']['3-4km']['means']['A_r3']==0


def test_statistics_weight_losses_zero_ratio_and_gate_exclusion():
    from tools.v335_paper_reward_reference import pair_variant
    e=MultiUAVCombatEnv('configs/combat_environment_v335.yaml');stats=AuditStats()
    for d in (2500,3000,3500):
        g=geometry(d,4,4,20)
        v={k:pair_variant(g,e.config['reward'],e.config['weapon'],geo,gate) for k,geo,gate in (('A','paper','hard'),('B','body','hard'),('C','paper','continuous'),('D','body','continuous'))}
        a={k:project_aggregation(value) for k,value in v.items()}
        w3=np.clip((g['distance']-1000)/2000,0,1);stats.update(g,v,a,w3,1-w3)
    r=stats.result();w=r['w4_suppression']['5']
    assert w['sample_count']==2 and w['zero_effective_count']==1
    assert w['finite_nominal_effective_ratio_mean']==4
    assert w['lost_due_w4']==pytest.approx(.175)
    assert r['pair']['counts']['paper_offense_greater_count']==3
    assert r['pair']['paper_offense_greater_cumulative_difference']==pytest.approx(.275)
