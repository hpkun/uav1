from __future__ import annotations
import copy, math
from pathlib import Path
import numpy as np
import pytest
import torch
from algorithm.mappo.trainer import MAPPOTrainer
from tools.analyze_single_wave_stability_root_cause import empirical_gt, groups, stats
from tools.single_wave_stability_common import (BEST_STEPS,CROSS_HORIZON_BASE,DIAGNOSTIC_BASE,FORMAL_EVAL_RANGE,action_saturation,active_pursuit_geometry,agent_attempt_transitions,boundary_descriptor,checkpoint_specs,counterfactual_score,cross_horizon_config,deterministic_policy_data,heading_relative_to_outward,optimization_phase_predicates,precursor_indices,radial_velocity,refuse_existing_output,strict_checkpoint_identity,symmetric_cone_components,validate_seed_bank,weapon_hit_probability)

def state_and_spec(index=0):
 spec=checkpoint_specs()[index];return torch.load(spec["path"],map_location="cpu",weights_only=False),spec

def test_01_clean_checkpoint_identity():assert strict_checkpoint_identity(checkpoint_specs()[0])["checkpoint_step"]==1_105_920
def test_02_all_best_final_identities():assert [(x["training_seed"],x["checkpoint_role"],x["expected_step"]) for x in checkpoint_specs()]==[(5401,"best",BEST_STEPS[5401]),(5401,"final",1_500_000),(5402,"best",BEST_STEPS[5402]),(5402,"final",1_500_000),(5403,"best",BEST_STEPS[5403]),(5403,"final",1_500_000)]
def test_03_modular_checkpoint_rejected():
 state,spec=state_and_spec();state["algorithm"]="modular_mappo"
 with pytest.raises(RuntimeError):strict_checkpoint_identity(spec,state)
def test_04_critic_must_be_mlp():
 state,spec=state_and_spec();state["critic_type"]="attention"
 with pytest.raises(RuntimeError):strict_checkpoint_identity(spec,state)
def test_05_wrong_seed_rejected():
 state,spec=state_and_spec();state["extra"]["training_seed"]=999
 with pytest.raises(RuntimeError):strict_checkpoint_identity(spec,state)
def test_06_wrong_step_rejected():
 state,spec=state_and_spec();state["sampled_steps"]+=1
 with pytest.raises(RuntimeError):strict_checkpoint_identity(spec,state)
def test_07_wrong_env_sha_rejected():
 state,spec=state_and_spec();state["extra"]["environment_config_sha256"]="0"*64
 with pytest.raises(RuntimeError):strict_checkpoint_identity(spec,state)
def test_08_wrong_algo_sha_rejected():
 state,spec=state_and_spec();state["extra"]["algorithm_config_sha256"]="0"*64
 with pytest.raises(RuntimeError):strict_checkpoint_identity(spec,state)
def test_09_same_diagnostic_seeds():assert validate_seed_bank(range(DIAGNOSTIC_BASE,DIAGNOSTIC_BASE+50))==list(range(DIAGNOSTIC_BASE,DIAGNOSTIC_BASE+50))
def test_10_formal_seed_overlap_rejected():
 with pytest.raises(RuntimeError):validate_seed_bank(range(48_000_000,48_000_002))
def test_11_deterministic_action_is_tanh_mean():
 trainer=MAPPOTrainer(hidden_dim=16,critic_type="mlp");obs=np.ones((4,52),np.float32);alive=np.ones(4,np.float32);action,raw,_=deterministic_policy_data(trainer,obs,alive)
 assert np.allclose(action,np.tanh(raw))
def test_12_boundary_event_recognition():
 before=np.array([1,1,0,1],bool);after=np.array([1,0,0,1],bool);assert np.flatnonzero(before&~after).tolist()==[1]
def test_13_precursor_lag_indices():assert precursor_indices(100,range(51,100))=={20:80,10:90,5:95,1:99}
def test_14_precursor_keeps_available_early_lags():assert precursor_indices(12,range(12))=={10:2,5:7,1:11}
def test_15_radial_velocity_sign():assert radial_velocity(10,0,5,0)>0 and radial_velocity(10,0,-5,0)<0
def test_16_heading_relative_outward():assert heading_relative_to_outward(1,0,0)==pytest.approx(0) and abs(heading_relative_to_outward(1,0,math.pi))==pytest.approx(math.pi)
def test_17_action_saturation():
 x=np.array([[.91,.2,-1.],[.5,-.995,.1]]);assert action_saturation(x,.9)=={"heading":.5,"pitch":.5,"speed":.5}
def base_row(**kw):return {"lag_steps":10,"radial_velocity":10.,"heading_relative_to_outward":0.,"nearest_blue_distance":5000.,"nearest_blue_off_boresight":0.,"closing_velocity":10.,"weapon_range_max":4000.,"fire_window":False,"steps_since_own_red_attempt":30,**kw}
def test_18_tactical_rule():assert boundary_descriptor([base_row(nearest_blue_distance=1000.)],100,None)=="TACTICAL_OVERSHOOT_CANDIDATE"
def test_19_escape_like_rule():assert boundary_descriptor([base_row(lag_steps=20),base_row(lag_steps=5)],100,None)=="ESCAPE_LIKE_TRAJECTORY_DESCRIPTOR"
def test_20_recent_kill_rule():assert boundary_descriptor([base_row(closing_velocity=-1)],100,50)=="RECENT_KILL_OVERSHOOT_CANDIDATE"
def test_21_no_intent_label():assert "INTENTIONAL" not in boundary_descriptor([base_row(lag_steps=20),base_row(lag_steps=5)],100,None)
def test_22_episode_reward_groups():
 rows=[{"red_success":"True","red_boundary_exits":"0","red_ground_losses":"0","termination_reason":"red_win"},{"red_success":"False","red_boundary_exits":"1","red_ground_losses":"0","termination_reason":"blue_win"}];assert len(groups(rows)["WIN"])==1 and len(groups(rows)["BOUNDARY_FAILURE"])==1
def test_23_empirical_ranking():assert empirical_gt([3,4],[1,2])==1.
def test_24_counterfactual_boundary_penalty():
 row={"episode_return":-5,"R2":-10,"red_boundary_exits":1,"termination_reason":"blue_win","red_success":False};assert counterfactual_score(row,-20,0,0,0)==-15
def test_25_counterfactual_terminal_bonus():
 row={"episode_return":5,"R2":0,"red_boundary_exits":0,"termination_reason":"red_win","red_success":True};assert counterfactual_score(row,-10,-10,20,-20)==25
def test_26_episode_component_stats():assert stats([1,2,3])["median"]==2
def test_27_corrected_weapon_edge():
 p_axis=weapon_hit_probability(2000,math.radians(30),0);p_illegal=weapon_hit_probability(2000,math.radians(30),math.radians(30));assert p_axis>p_illegal
def test_28_symmetric_true_cone():
 a,e=symmetric_cone_components(math.radians(30));assert math.acos(math.cos(a)*math.cos(e))==pytest.approx(math.radians(30))
def test_29_optimization_best_to_final_window():
 f=optimization_phase_predicates(1_105_920)["BEST_TO_FINAL"];assert not f(1_105_920) and f(1_200_000) and f(1_500_000)
def test_30_h1000_only_changes_max_steps():
 cfg={"simulation":{"dt":.1,"max_steps":3000},"x":{"y":1}};out=cross_horizon_config(cfg);assert out["simulation"]["max_steps"]==1000 and out["x"]==cfg["x"] and cfg["simulation"]["max_steps"]==3000
def test_31_cross_horizon_seed_disjoint():assert not set(range(DIAGNOSTIC_BASE,DIAGNOSTIC_BASE+50))&set(range(CROSS_HORIZON_BASE,CROSS_HORIZON_BASE+50))
def test_32_output_refuses_overwrite(tmp_path):
 with pytest.raises(FileExistsError):refuse_existing_output(tmp_path)
def test_33_blue_guard_counter_fields():
 required={"blue_ground_guard_decision_steps","blue_ground_guard_override_steps","blue_ground_guard_activations","blue_ground_guard_activation_ratio","blue_ground_guard_max_duration_steps"};source=Path("tools/audit_single_wave_checkpoint_trajectories.py").read_text();assert all(x in source for x in required)
def test_34_cross_horizon_label_present():assert "CROSS_HORIZON_POLICY_EVALUATION" in Path("tools/audit_single_wave_checkpoint_trajectories.py").read_text()
def test_35_formal_48m_not_in_tool():
 source=Path("tools/audit_single_wave_checkpoint_trajectories.py").read_text();assert "48000000" not in source and "48_000_000" not in source
def test_36_true_to_false_is_attempt():assert agent_attempt_transitions([1,0,0,0],[0,0,0,0],[1,1,1,1]).tolist()==[True,False,False,False]
def test_37_false_to_false_no_attempt():assert not agent_attempt_transitions([0]*4,[0]*4,[1]*4).any()
def test_38_false_to_true_is_rearm():assert not agent_attempt_transitions([0]*4,[1,0,0,0],[1]*4).any()
def test_39_dead_agent_cannot_fake_attempt():assert not agent_attempt_transitions([1,0,0,0],[0,0,0,0],[0,1,1,1]).any()
def test_40_teammate_attempt_does_not_reset_focal_history():
 attempted=agent_attempt_transitions([1,1,0,0],[0,1,0,0],[1]*4);last=[None]*4
 for i in np.flatnonzero(attempted):last[int(i)]=7
 assert last==[7,None,None,None]
def test_41_rear_receding_enemy_not_tactical():
 row=base_row(nearest_blue_distance=3500,nearest_blue_off_boresight=2.,closing_velocity=-1.)
 assert not active_pursuit_geometry(row) and boundary_descriptor([row],100,None)=="UNCLASSIFIED_BOUNDARY_EXIT"
def test_42_front_closing_enemy_is_tactical():
 row=base_row(nearest_blue_distance=3500,nearest_blue_off_boresight=.4,closing_velocity=5.)
 assert active_pursuit_geometry(row) and boundary_descriptor([row],100,None)=="TACTICAL_OVERSHOOT_CANDIDATE"
def test_43_fire_window_is_tactical():assert boundary_descriptor([base_row(fire_window=True,closing_velocity=-1)],100,None)=="TACTICAL_OVERSHOOT_CANDIDATE"
def test_44_escape_requires_own_attempt_history():assert boundary_descriptor([base_row(lag_steps=20,closing_velocity=-1),base_row(lag_steps=5,closing_velocity=-1)],100,None)=="ESCAPE_LIKE_TRAJECTORY_DESCRIPTOR"
def test_45_boundary_failure_excludes_boundary_win():
 rows=[{"red_success":"True","red_boundary_exits":"1","red_ground_losses":"0","termination_reason":"red_win"}];g=groups(rows);assert len(g["BOUNDARY_ANY"])==1 and len(g["BOUNDARY_FAILURE"])==0
def test_46_pure_combat_defeat_excludes_draw():
 rows=[{"red_success":"False","red_boundary_exits":"0","red_ground_losses":"0","termination_reason":"draw_mutual_destruction"}];g=groups(rows);assert not g["PURE_COMBAT_DEFEAT"] and len(g["MUTUAL_DESTRUCTION_DRAW"])==1
def test_47_post_combat_name_removed():
 source=(Path("tools/single_wave_stability_common.py").read_text()+Path("tools/audit_single_wave_checkpoint_trajectories.py").read_text()+Path("tools/analyze_single_wave_stability_root_cause.py").read_text());assert "POST_COMBAT_OVERSHOOT" not in source and "post_combat_overshoot" not in source
def test_48_snapshot_uses_own_attempt_field():
 source=Path("tools/audit_single_wave_checkpoint_trajectories.py").read_text();assert "steps_since_own_red_attempt" in source and "steps_since_any_red_attempt" in source
