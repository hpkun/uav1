from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path

import numpy as np

from algorithm.modular_mappo.evaluation import (
    aggregate_per_wave_diagnostics, per_wave_episode_diagnostics,
)
from env.config import load_config
from env.factory import make_combat_environment
from env.fixed_policy import GroundAwareNearestTargetPursuitPolicy
from env.models import AircraftState
from tools.plain_transition_diagnostic_common import (
    EVALUATION_SEEDS,FUTURE_FINAL_RANGE,boundary_diagnostics,circular_heading_dispersion,
    canonical_spawn_seed,canonicalize_spawn,classify_death,continuation_should_stop,diagnostic_ground_risk,future_rng_seed,
    pairwise_metrics,time_to_ground,transition_snapshot,
)
import tools.analyze_plain_transition_mechanism as analyzer
import tools.run_plain_transition_mechanism_diagnostic as runner

ROOT=Path(__file__).resolve().parents[1]
ENV=ROOT/"configs/persistent_wave_v2_environment.yaml"


def record(wave,cleared,survivors,duration=10):
    return {"wave_index":wave,"wave_cleared":cleared,"red_survivors_end":survivors,
            "blue_survivors_end":0 if cleared else 2,"red_attack_kills":4 if cleared else 2,
            "red_boundary_exits":0,"red_ground_losses":0,"start_step":wave*10-10,
            "duration_steps":duration}


def flattened(rows):return per_wave_episode_diagnostics({"per_wave_metrics":rows},3)


def test_record_and_clear_conditioning_are_distinct_for_wave1_and_wave2():
    episodes=[flattened([record(1,True,4),record(2,True,3)]),
              flattened([record(1,False,1)]),
              flattened([record(1,True,2),record(2,False,1)])]
    value=aggregate_per_wave_diagnostics(episodes,3)
    assert value["average_red_survivors_after_wave_1_conditional_on_record"]==7/3
    assert value["average_red_survivors_after_wave_1_conditional_on_clear"]==3
    assert value["average_red_survivors_after_wave_2_conditional_on_record"]==2
    assert value["average_red_survivors_after_wave_2_conditional_on_clear"]==3
    assert value["red_survivors_after_wave_1"]==7/3


def test_recorded_and_cleared_flags_are_from_real_records():
    value=flattened([record(1,True,4),record(2,False,1)])
    assert value["wave_1_recorded"] is True and value["wave_1_cleared"] is True
    assert value["wave_2_recorded"] is True and value["wave_2_cleared"] is False
    assert value["wave_3_recorded"] is False and value["wave_3_cleared"] is False


def test_geometry_feature_edge_cases_and_circular_heading():
    one=[AircraftState(0,0,-1000,200,0,0)]
    assert pairwise_metrics(one)==(None,None,None)
    two=one+[AircraftState(3,4,-1000,200,0,0)]
    assert pairwise_metrics(two)==(5,5,5)
    assert circular_heading_dispersion([np.pi-.01,-np.pi+.01])<.02
    assert time_to_ground(100,-20)==5 and time_to_ground(100,0) is None


def test_boundary_margin_radial_velocity_and_time():
    state=AircraftState(4000,0,-1000,200,0,0)
    margin,radial,ttb=boundary_diagnostics(state,5000)
    assert margin==1000 and radial==200 and ttb==5
    state.psi=np.pi
    assert boundary_diagnostics(state,5000)[2] is None


def test_diagnostic_ground_risk_exactly_matches_blue_guard():
    cfg=load_config(ENV);policy=GroundAwareNearestTargetPursuitPolicy(cfg["blue_policy"],cfg["action"],cfg["aircraft"])
    for state,pitch in [(AircraftState(0,0,-100,250,-.2,0),-.4),(AircraftState(0,0,-3000,225,.1,0),.2)]:
        assert diagnostic_ground_risk(state,pitch,cfg)==policy.ground_risk(state,pitch)


def test_death_classification_all_three_causes():
    assert classify_death(AircraftState(0,0,1,200,0,0,False),5000)=="ground"
    assert classify_death(AircraftState(6000,0,-1000,200,0,0,False),5000)=="boundary"
    assert classify_death(AircraftState(0,0,-1000,200,0,0,False),5000)=="combat"
    assert classify_death(AircraftState(6000,0,0,200,0,0,False),5000)=="boundary"


def test_canonical_spawn_seed_is_policy_invariant_and_reproducible():
    cfg=load_config(ENV);env=make_combat_environment(cfg);env.reset(88_200_004)
    for blue in env.blue:blue.alive=False
    _,_,_,_,info=env.step(np.zeros((4,3),np.float32));assert info["spawned_next_wave"]
    first=deepcopy(env);second=deepcopy(env)
    seed1,angle1=canonicalize_spawn(first,44_000_000,2);seed2,angle2=canonicalize_spawn(second,44_000_000,2)
    assert seed1==seed2==canonical_spawn_seed(44_000_000,2) and angle1==angle2
    assert all(np.array_equal(a.as_array(),b.as_array()) for a,b in zip(first.blue,second.blue))
    assert [(a.x,a.y,a.z,a.v,a.theta,a.psi,a.alive) for a in env.red] == [(a.x,a.y,a.z,a.v,a.theta,a.psi,a.alive) for a in first.red]
    assert (env.steps,env.wave_index,env.waves_cleared,env.max_steps)==(first.steps,first.wave_index,first.waves_cleared,first.max_steps)
    assert all(state.armed for state in first.red_fire_states+first.blue_fire_states)


def test_paired_controller_effect_counts_each_entry_once():
    rows=[]
    for case in (1,2):
        for source in (5301,5302,5303):
            for controller in (5301,5302,5303):
                rows.append({"evaluation_seed":case,"source_policy_seed":source,
                             "continuation_policy_seed":controller,"next_wave":2,
                             "next_wave_clear":int(controller==5303)})
    effects=analyzer.paired_controller_effect(rows,2)
    assert all(row["N"]==6 and row["wins"]==6 and row["mean_paired_outcome_delta"]==1 for row in effects)


def test_paired_entry_effect_requires_all_three_sources_and_controllers():
    rows=[]
    for case in (1,2):
        for source in (5301,5302,5303):
            for controller in (5301,5302,5303):
                if not (case==2 and source==5302 and controller==5301):
                    rows.append({"evaluation_seed":case,"source_policy_seed":source,
                                 "continuation_policy_seed":controller,"next_wave":2,
                                 "next_wave_clear":int(source==5303)})
    effects=analyzer.paired_entry_effect(rows,2)
    assert all(row["N"]==1 and row["wins"]==1 for row in effects)


def test_fixed_effect_and_overall_classification_rules():
    assert analyzer.effect_label([1,1,1,1,1,1])=="ENTRY_SUPPORTED_STRONG"
    assert analyzer.effect_label([1,1,1,1,-1,0])=="ENTRY_SUPPORTED_DIRECTIONALLY"
    assert analyzer.effect_label([1,1,1,-1,-1,0])=="ENTRY_MIXED_POSITIVE"
    assert analyzer.controller_label([1,1])=="CONTROLLER_SUPPORTED_DIRECTIONALLY"
    assert analyzer.controller_label([1,0])=="CONTROLLER_MIXED_POSITIVE"
    assert analyzer.overall_entry_label("ENTRY_SUPPORTED_STRONG","ENTRY_SUPPORTED_DIRECTIONALLY")=="ENTRY_OVERALL_SUPPORTED"
    assert analyzer.overall_controller_label("CONTROLLER_SUPPORTED_DIRECTIONALLY","CONTROLLER_MIXED_POSITIVE")=="CONTROLLER_OVERALL_MIXED"
    assert analyzer.classify_overall_mechanism("ENTRY_OVERALL_MIXED","CONTROLLER_OVERALL_SUPPORTED")=="CONTROLLER_QUALITY_DOMINANT"
    assert analyzer.classify_overall_mechanism("ENTRY_OVERALL_MIXED","CONTROLLER_OVERALL_MIXED")=="NEITHER_RESOLVED"


def test_deepcopy_and_transition_deepcopy_fidelity_and_same_observation():
    cfg=load_config(ENV);env=make_combat_environment(cfg);env.reset(88_200_001)
    copy=deepcopy(env);actions=np.zeros((4,3),np.float32)
    for _ in range(20):
        a=env.step(actions);b=copy.step(actions)
        assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1]) and a[2:4]==b[2:4]
    env=make_combat_environment(cfg);env.reset(88_200_002)
    for blue in env.blue:blue.alive=False
    result=env.step(actions);assert result[4]["spawned_next_wave"]
    copy=deepcopy(env);assert np.array_equal(env._observations(),copy._observations())
    for _ in range(20):
        a=env.step(actions);b=copy.step(actions);assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])


def test_transition_features_single_survivor_and_weapon_reset():
    cfg=load_config(ENV);env=make_combat_environment(cfg);env.reset(88_200_003)
    for i,row in enumerate(env.red):row.alive=i==0
    for row in env.blue:row.alive=False
    _,_,_,_,info=env.step(np.zeros((4,3),np.float32));assert info["spawned_next_wave"]
    summary,agents=transition_snapshot(env,5301,44_000_000,1)
    assert summary["red_survivor_count"]==1 and summary["red_pairwise_distance_mean"] is None and summary["red_formation_spread"]==0
    assert all((not x["alive"]) or x["fire_armed"] for x in agents)


def test_matched_rng_44m_and_45m_guards():
    assert EVALUATION_SEEDS==(tuple(range(44_000_000,44_000_050)))
    assert FUTURE_FINAL_RANGE==(45_000_000,45_000_199)
    assert future_rng_seed(44_000_007,2)==440_000_072
    for source in (5301,5302,5303):
        for controller in (5301,5302,5303):assert future_rng_seed(44_000_007,3)==440_000_073


def test_counterfactual_stops_at_exact_next_wave_outcome():
    stop,clear=continuation_should_stop(2,2,{"wave_cleared_this_step":True},False,False);assert stop and clear
    stop,clear=continuation_should_stop(3,2,{"wave_cleared_this_step":True},False,False);assert not stop and not clear
    assert continuation_should_stop(2,2,{"wave_cleared_this_step":False},True,False)==(True,False)


def test_runner_has_no_training_and_analyzer_is_strictly_offline():
    source=inspect.getsource(runner);offline=inspect.getsource(analyzer)
    assert "trainer.update(" not in source and "optimizer.step(" not in source and ".backward(" not in source
    assert "make_combat_environment" not in offline and "evaluate_modular" not in offline and "torch" not in offline
