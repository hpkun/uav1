from copy import deepcopy
import pytest

import tools.preflight_eawb_formal_multiseed as protocol


@pytest.fixture(scope="module")
def manifest():
    return protocol.load_manifest()


def test_manifest_has_exact_15_run_design_and_fresh_seeds(manifest):
    runs = manifest["runs"]
    assert len(runs) == 15
    assert {run["training_seed"] for run in runs} == {3101, 3102, 3103}
    assert 2023 not in {run["training_seed"] for run in runs}
    main = [run for run in runs if run["group"] == "main_matrix"]
    controls = [run for run in runs if run["group"] == "schedule_control"]
    assert {(run["method"], run["training_seed"]) for run in main} == {
        (method, seed) for method in protocol.MAIN_ENABLEMENT for seed in protocol.TRAINING_SEEDS
    }
    assert {(run["method"], run["training_seed"]) for run in controls} == {
        ("EA-WB Fixed LR", seed) for seed in protocol.TRAINING_SEEDS
    }


def test_method_enablement_common_schedule_and_unrelated_modules_off(manifest):
    result = protocol.validate_plan(manifest, check_outputs=False)
    assert result["planned_runs"] == 15
    assert result["main_matrix_equivalent_except"] == ["entity_attention.enabled", "wave_balancing.enabled"]
    assert result["schedule_control_equivalent_except"] == ["actor_lr_decay.enabled"]
    for run in manifest["runs"]:
        config = protocol.load_config(protocol.ROOT / run["config_path"])
        assert not any(config["modules"][name]["enabled"] for name in protocol.UNRELATED_MODULES)
        assert config["training"]["total_sampled_steps"] == 900_000
        assert config["training"]["device"] == "cuda"


def test_holdout_monitoring_and_primary_checkpoint_are_frozen(manifest):
    assert manifest["formal_holdout"] == {
        "executed": False, "seed_start": 30_000_000, "seed_end": 30_000_199,
        "episodes_per_policy": 200, "common_scenario_seeds": True,
    }
    monitoring = set(range(29_000_000, 29_000_020))
    holdout = set(range(30_000_000, 30_000_200))
    assert not monitoring & holdout
    assert not set(protocol.TRAINING_SEEDS) & (monitoring | holdout)
    assert manifest["checkpoint_selection"]["primary"] == "latest_at_budget"
    assert manifest["checkpoint_selection"]["budget"] == 900_000
    assert manifest["checkpoint_selection"]["best_eval_role"] == "secondary_diagnostic_only"


def test_environment_semantic_and_source_freeze():
    assert protocol.validate_environment_freeze() == {
        "raw_sha256": protocol.FROZEN_ENV_RAW_SHA256,
        "semantic_sha256": protocol.FROZEN_ENV_SEMANTIC_SHA256,
        "source_tree_sha256": protocol.FROZEN_ENV_SOURCE_TREE_SHA256,
    }


def test_preflight_rejects_stale_formal_output(manifest, tmp_path):
    changed = deepcopy(manifest)
    stale = tmp_path / "planned" / "stale"
    stale.mkdir(parents=True)
    (stale / "latest.pt").write_bytes(b"stale")
    changed["runs"][0]["output_dir"] = str(stale)
    with pytest.raises(FileExistsError, match="stale results"):
        protocol.validate_plan(changed)


def test_holdout_preflight_rejects_before_primary_checkpoints_exist(manifest, tmp_path):
    changed = deepcopy(manifest)
    for index, run in enumerate(changed["runs"]):
        run["output_dir"] = str(tmp_path / f"run_{index}")
    with pytest.raises(RuntimeError, match="formal holdout locked.*missing"):
        protocol.checkpoint_inventory(changed)
