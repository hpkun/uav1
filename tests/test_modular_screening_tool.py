import json
from pathlib import Path

import pandas as pd
import pytest

from tools.modular_1p5m_screening import (
    discover_runs,matched_episode_delta,nearest_checkpoint_step,
    requested_m6_mapping,require_cross_variant_permission,
    select_near_budget_checkpoints,stage_label,summarize_episodes,
    sweep_mapping_rows,validate_diagnostic_seeds,wave_weight_stats,
)


def test_near_1p5m_checkpoint_selection_brackets_budget():
    result=select_near_budget_checkpoints([(1_001_472,"low"),(1_505_280,"high"),(2_002_944,"later")],1_500_000)
    assert result["at_or_below"]==(1_001_472,"low")
    assert result["nearest_overall"]==(1_505_280,"high")


def test_exact_and_nearest_checkpoint_step_selection():
    available=[503_808,1_001_472,1_500_000]
    assert nearest_checkpoint_step(503_808,available)==503_808
    assert nearest_checkpoint_step(804_864,available)==1_001_472


def test_screening_seeds_reject_formal_holdout_and_duplicates():
    assert validate_diagnostic_seeds(range(34_000_000,34_000_050))[0]==34_000_000
    with pytest.raises(ValueError,match="formal holdout"):validate_diagnostic_seeds([20_000_010])
    with pytest.raises(ValueError,match="unique"):validate_diagnostic_seeds([34_000_000,34_000_000])


def test_paired_episode_delta_aligns_by_seed_not_row_order():
    baseline=pd.DataFrame({"seed":[2,1],"value":[20.,10.]})
    candidate=pd.DataFrame({"seed":[1,2],"value":[13.,25.]})
    result=matched_episode_delta(candidate,baseline,{"value":"value"})
    assert result["delta_value"]==pytest.approx(4.)


def test_paired_episode_delta_rejects_duplicate_seed():
    duplicate=pd.DataFrame({"seed":[1,1],"value":[1.,2.]})
    baseline=pd.DataFrame({"seed":[1,2],"value":[1.,2.]})
    with pytest.raises(ValueError,match="duplicate seed"):matched_episode_delta(duplicate,baseline,{"value":"value"})


def test_stage_boundaries_are_exact():
    assert stage_label(500_000)=="0-0.5M"
    assert stage_label(500_001)=="0.5-1.0M"
    assert stage_label(1_000_000)=="0.5-1.0M"
    assert stage_label(1_000_001)=="1.0-1.5M"


def test_absent_m5_wave_samples_do_not_create_fake_weight():
    frame=pd.DataFrame({"alive_agent_samples_wave_3":[0,0],"weight_wave_3":[3.,3.]})
    result=wave_weight_stats(frame,3)
    assert result["W3_absent"] is True and result["W3_present_updates"]==0
    assert result["weight_W3_mean"] is None and result["weight_W3_max"] is None


def test_m6_sweep_rows_preserve_requested_and_actual_steps():
    mapping=requested_m6_mapping();rows=sweep_mapping_rows(mapping)
    row=next(item for item in rows if item["requested_step"]==202_752)
    assert row=={"requested_step":202_752,"actual_step":503_808}
    assert next(item for item in rows if item["requested_step"]==104_448)["actual_step"]==104_448


def test_baseline_and_modular_episode_records_share_summary_schema():
    record={"episode_return":1.,"mean_agent_episode_return":.25,"red_success":True,"blue_win":False,"draw":False,
        "timeout":0,"red_losses":1,"blue_losses":4,"red_attack_kills":4,"blue_attack_kills":1,
        "red_boundary_exits":0,"blue_boundary_exits":0,"red_ground_losses":0,"blue_ground_losses":0,
        "episode_length":10,"waves_cleared":1,"clear_wave_1":1,"clear_wave_2":0,"clear_wave_3":0,
        "red_survivors_after_wave_1":3,"red_survivors_after_wave_2":None,"red_survivors_after_wave_3":None}
    assert summarize_episodes([dict(record)]).keys()==summarize_episodes([dict(record)]).keys()
    assert {"average_return","clear_wave_3_probability","kill_loss_ratio"}<=summarize_episodes([record]).keys()


def test_cross_variant_requires_explicit_protocol_permission():
    assert require_cross_variant_permission("direct_v2_3","persistent_wave_v2",True) is True
    assert require_cross_variant_permission("persistent_wave_v2","persistent_wave_v2",False) is False
    with pytest.raises(RuntimeError,match="explicit permission"):require_cross_variant_permission("direct_v2_3","persistent_wave_v2",False)


def test_run_discovery_uses_metadata_not_directory_names(tmp_path: Path):
    specs=[("random_a","MAPPO","persistent_wave_v2",[],3_000_000),("random_b","MAPPO","direct_v2_3",[],3_000_000),
        ("random_c","modular_mappo","persistent_wave_v2",["wave_balancing"],1_500_000),("random_d","modular_mappo","persistent_wave_v2",["warm_start"],1_500_000),
        ("random_e","modular_mappo","persistent_wave_v2",["wave_context"],1_500_000),("random_f","modular_mappo","persistent_wave_v2",["popart"],1_500_000)]
    for name,algorithm,variant,modules,steps in specs:
        directory=tmp_path/name;directory.mkdir();
        (directory/"run_config.json").write_text(json.dumps({"algorithm":algorithm,"seed":2023,"environment_variant":variant,"enabled_modules":modules}),encoding="utf-8")
        (directory/"run_summary.json").write_text(json.dumps({"sampled_steps":steps}),encoding="utf-8")
    found=discover_runs(tmp_path)
    assert found["PW baseline"].name=="random_a" and found["M6"].name=="random_d"
