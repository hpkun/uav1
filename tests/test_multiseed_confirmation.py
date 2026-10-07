import json
from pathlib import Path

import numpy as np
import pytest

from algorithm.mappo.trainer import MAPPOTrainer
from algorithm.train_modular_mappo import load_config
from tools.analyze_multiseed_confirmation import (
    M5_SEED_BASE,M8_PW_SEED_BASE,M8_DIRECT_SEED_BASE,conditional_timeout_metrics,
    _json_default,classify_m8_confirmation,is_m8_primary_checkpoint,resolved_configs_matched,seed_level_summary,
    validate_confirmation_seeds,validate_same_source,
)

ROOT=Path(__file__).resolve().parents[1]


def _checkpoint(path:Path,seed:int,bias:float=0.)->None:
    trainer=MAPPOTrainer(hidden_dim=16,seed=seed)
    with __import__("torch").no_grad():trainer.actor.mean.bias.add_(bias)
    trainer.sampled_steps=100
    trainer.save(path,{"training_seed":seed,"training_gamma":.999,"environment_variant":"direct_v2_3"})


@pytest.mark.parametrize("seed",[2024,2025])
def test_seed_override_and_m5_pair_remain_strictly_matched(seed):
    assert resolved_configs_matched(seed)
    alloff=load_config(ROOT/"configs/pw_alloff_matched_1p5m.yaml");m5=load_config(ROOT/"configs/pw_m5_wave_balance.yaml")
    alloff["training"]["seed"]=seed;m5["training"]["seed"]=seed
    assert alloff["training"]["seed"]==m5["training"]["seed"]==seed


def test_source_seed_mismatch_is_rejected(tmp_path):
    source=tmp_path/"source.pt";_checkpoint(source,2024)
    with pytest.raises(RuntimeError,match="source training seed"):validate_same_source(source,source,2025)


def test_warm_reference_checkpoint_mismatch_is_rejected(tmp_path):
    warm,reference=tmp_path/"warm.pt",tmp_path/"reference.pt";_checkpoint(warm,2024,0);_checkpoint(reference,2024,.1)
    with pytest.raises(RuntimeError,match="warm/reference"):validate_same_source(warm,reference,2024)


def test_confirmation_seed_ranges_are_new_and_holdout_safe():
    for base in (M5_SEED_BASE,M8_PW_SEED_BASE,M8_DIRECT_SEED_BASE):assert len(validate_confirmation_seeds(range(base,base+100)))==100
    with pytest.raises(ValueError,match="formal holdout"):validate_confirmation_seeds([20_000_000])
    with pytest.raises(ValueError,match="development"):validate_confirmation_seeds([35_100_000])


def test_training_seed_is_the_aggregate_unit():
    result=seed_level_summary([1.,2.,3.]);assert result["n_training_seeds"]==3
    assert result["mean"]==2 and result["std"]==1 and "n=3" in result["ci_note"]


def test_conditional_timeout_missing_wave_is_none_not_zero():
    records=[{"reached_wave_2":0,"reached_wave_3":0,"timeout":0,"waves_cleared":0,"episode_length":10,
              "time_to_clear_wave_1":None,"time_to_clear_wave_2":None,"time_spent_in_wave_3":None}]
    result=conditional_timeout_metrics(records)
    assert result["timeout_conditioned_reached_W2"] is None
    assert result["timeout_conditioned_reached_W3"] is None
    assert result["mean_time_to_clear_W1"] is None and result["mean_time_spent_in_W3"] is None
    assert result["episode_length_conditioned_waves_cleared_1"] is None


def test_m8_primary_is_latest_not_best():
    assert is_m8_primary_checkpoint("latest")
    assert not is_m8_primary_checkpoint("best")


def test_summary_json_supports_numpy_scalars():
    payload={"seed":np.int64(2023),"supported":np.bool_(True),"score":np.float64(.5)}
    assert json.loads(json.dumps(payload,default=_json_default))=={
        "seed":2023,"supported":True,"score":.5,
    }


def test_m8_rating_requires_consistent_three_seed_preservation():
    assert classify_m8_confirmation([True,True,False],[False,False,False])=="MIXED"
    assert classify_m8_confirmation([True,True,True],[False,False,False])=="PRESERVATION_ONLY"
    assert classify_m8_confirmation([True,True,True],[True,True,False])=="MULTISEED_SUPPORTED"


def test_serial_script_contains_all_eight_runs_in_protocol_order():
    text=(ROOT/"tools/run_multiseed_confirmation.sh").read_text(encoding="utf-8")
    names=["pw_alloff_matched_1p5m_seed2024","pw_m5_wave_balance_1p5m_seed2024",
        "pw_alloff_matched_1p5m_seed2025","pw_m5_wave_balance_1p5m_seed2025",
        "pw_m6_screen_control_300k_seed2024","pw_m6_m8_anchor_c003_300k_seed2024",
        "pw_m6_screen_control_300k_seed2025","pw_m6_m8_anchor_c003_300k_seed2025"]
    positions=[text.rindex(f"run_one {name}") for name in names]
    assert positions==sorted(positions)
    assert "set -euo pipefail" in text and "--reference-checkpoint \"$DIRECT_2024\"" in text
    assert "--reference-checkpoint \"$DIRECT_2025\"" in text
    for artifact in ("latest.pt","best_eval.pt","run_summary.json","run_config.json"):assert artifact in text
