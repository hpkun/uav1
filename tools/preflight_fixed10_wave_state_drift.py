"""Read-only checkpoint inventory for the Fixed10 wave-state drift audit."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

try:
    from tools.fixed10_wave_state_drift_common import (
        BANK_PATH, HISTORICAL_SNAPSHOT_PATH, INVENTORY_PATH, ROOT, checkpoint_step,
        inventory_specifications, load_checkpoint, run_dir, sha256, state_dict_sha256,
        json_dump,
    )
except ModuleNotFoundError:  # direct ``python tools/...py`` execution
    from fixed10_wave_state_drift_common import (
        BANK_PATH, HISTORICAL_SNAPSHOT_PATH, INVENTORY_PATH, ROOT, checkpoint_step,
        inventory_specifications, load_checkpoint, run_dir, sha256, state_dict_sha256,
        json_dump,
    )


def build_inventory() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for Fixed10 checkpoint inventory")
    checkpoints = []
    topologies = []
    for spec in inventory_specifications():
        path = spec["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = load_checkpoint(path)
        step = checkpoint_step(checkpoint)
        if step != spec["expected_step"]:
            raise RuntimeError(f"{spec['id']} expected step {spec['expected_step']}, got {step}")
        run = (run_dir(spec["arm"], spec["seed"]) if spec["arm"] is not None else
               ROOT / "outputs" / "diag_mappo_learnability" / f"l3_seed{spec['seed']}")
        run_config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
        network = run_config["network_architecture"]
        actual_contract = {
            "environment_variant": run_config.get("environment_variant"),
            "actor_input_dim": run_config.get("actor_input_dim", network.get("actor_input_dim")),
            "mission_context_dim": run_config.get("mission_context_dim", network.get("actor_context_dim", 0)),
        }
        for key, expected in {"environment_variant": "persistent_wave_v2", "actor_input_dim": 52,
                              "mission_context_dim": 0}.items():
            if actual_contract[key] != expected:
                raise RuntimeError(f"{run}: {key} expected {expected!r}, got {actual_contract[key]!r}")
        runtime_env = yaml.safe_load((run / "runtime_env_config.yaml").read_text(encoding="utf-8"))
        effective_waves = run_config.get("effective_training_total_waves", runtime_env.get("persistent_waves", {}).get("total_waves"))
        if effective_waves != 3:
            raise RuntimeError(f"{run}: expected three waves")
        if network.get("actor_class") != "ModularMAPPOActor" or network.get("actor_gru_hidden_dim") != 0:
            raise RuntimeError(f"{run}: Actor is not the required Plain feed-forward topology")
        if network.get("entity_attention_enabled") or network.get("actor_context_dim") != 0:
            raise RuntimeError(f"{run}: Actor topology includes prohibited context/attention")
        expected_modules = (["actor_gradient_clipping", "actor_lr_decay"] if spec["arm"] is not None else ["actor_lr_decay"])
        if run_config.get("enabled_modules") != expected_modules:
            raise RuntimeError(f"{run}: unexpected module identity {run_config.get('enabled_modules')}")
        topology = {key: network.get(key) for key in (
            "actor_class", "actor_input_dim", "actor_context_dim", "actor_parameter_count",
            "hidden_dim", "actor_gru_hidden_dim", "entity_attention_enabled", "mission_context_dim",
        )}
        topologies.append(topology)
        checkpoints.append({
            **{key: value for key, value in spec.items() if key != "path"},
            "checkpoint_path": str(path.relative_to(ROOT)),
            "checkpoint_sha256": sha256(path),
            "actor_checkpoint_sha256": state_dict_sha256(checkpoint["actor"]),
            "sampled_steps": step,
            "training_seed": run_config["seed"],
            "development_method": run_config.get("development_method", "plain_mappo_source"),
            "network_architecture": network,
            "module_identity": run_config["enabled_modules"],
            "runtime_trainer_sha256": next((
                item["sha256"] for item in run_config.get("runtime_source_manifest_files", [])
                if item["path"] == "algorithm/modular_mappo/trainer.py"
            ), None),
            "runtime_source_manifest_sha256": run_config.get("runtime_source_manifest_sha256"),
            "runtime_source_provenance_available": bool(run_config.get("runtime_source_manifest_files")),
            "environment_config_sha256": run_config.get("effective_training_environment_config_sha256", run_config.get("environment_config_sha256")),
            "branch_parent_checkpoint_sha256": run_config.get("branch_provenance", {}).get("parent_checkpoint_sha256"),
        })
    if any(item != topologies[0] for item in topologies[1:]):
        raise RuntimeError("cross-checkpoint inference topology mismatch")
    source_sha = {item["seed"]: item["checkpoint_sha256"] for item in checkpoints if item["role"] == "branch_source"}
    for item in checkpoints:
        if item["role"] == "branch_source":
            item["branch_parent_source_sha256_match"] = None
            continue
        item["branch_parent_source_sha256_match"] = item["branch_parent_checkpoint_sha256"] == source_sha[item["seed"]]
        if not item["branch_parent_source_sha256_match"]:
            raise RuntimeError(f"{item['id']}: branch parent SHA does not match the fixed source checkpoint")
    result = {
        "status": "PASS",
        "analysis_only": True,
        "training_performed": False,
        "environment_steps_performed": 0,
        "forbidden_45m_used": False,
        "historical_observation_bank": {"path": str(BANK_PATH.relative_to(ROOT)), "exists": BANK_PATH.is_file(),
                                         "sha256": sha256(BANK_PATH) if BANK_PATH.is_file() else None,
                                         "allowed_use": "COMMON_FIXED_STATE_ACTOR_COMPARISON_ONLY"},
        "historical_entry_snapshot_bank": {"path": str(HISTORICAL_SNAPSHOT_PATH.relative_to(ROOT)),
                                            "exists": HISTORICAL_SNAPSHOT_PATH.is_file(),
                                            "sha256": sha256(HISTORICAL_SNAPSHOT_PATH) if HISTORICAL_SNAPSHOT_PATH.is_file() else None,
                                            "allowed_use": "INTERFACE_REFERENCE_ONLY_NOT_FIXED10_NATURAL_ENTRY"},
        "common_inference_topology": topologies[0],
        "checkpoints": checkpoints,
    }
    json_dump(INVENTORY_PATH, result)
    return result


def main() -> None:
    result = build_inventory()
    print(json.dumps({"status": result["status"], "checkpoints": len(result["checkpoints"]),
                      "output": str(INVENTORY_PATH.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
