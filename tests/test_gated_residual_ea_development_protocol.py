from pathlib import Path
import json

from algorithm.train_modular_mappo import load_config
from tools.preflight_gated_residual_ea_development import ROOT, validate


def test_development_preflight_without_output_mutation():
    result=validate(check_outputs=False)
    assert result["status"]=="READY_FOR_GATED_RESIDUAL_EA_DEVELOPMENT"
    assert result["planned_runs"]==12 and result["training_seeds"]==[4101,4102,4103]
    assert result["validation_seed_range"]==[32000000,32000019]
    assert result["reserved_untouched_future_final_test"]==[33000000,33000199]


def test_four_configs_differ_only_in_entity_block():
    names=("mappo","full_ea","residual_ea","gated_ea");configs=[]
    for name in names:
        config=load_config(ROOT/f"configs/dev_grea_{name}_400k.yaml")
        config["modules"].pop("entity_attention");configs.append(config)
    assert all(config==configs[0] for config in configs[1:])


def test_launcher_order_and_never_mentions_reserved_evaluation_seeds():
    text=(ROOT/"tools/run_gated_residual_ea_development.sh").read_text(encoding="utf-8")
    expected=[f"{method}_seed{seed}" for method in ("mappo","full_ea","residual_ea","gated_ea") for seed in (4101,4102,4103)]
    positions=[text.index(value) for value in expected]
    assert positions==sorted(positions)
    assert "33000000" not in text and "32000000" not in text

