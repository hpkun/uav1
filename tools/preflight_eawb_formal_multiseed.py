"""Validate the frozen EA-WB formal plan without running training or evaluation."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.train_modular_mappo import load_config

MANIFEST_PATH = ROOT / "experiments" / "ea_wb_formal_multiseed_manifest.json"
ENV_CONFIG = ROOT / "configs" / "persistent_wave_v2_environment.yaml"
OUTPUT_ROOT = ROOT / "outputs" / "formal_eawb"
INVENTORY_PATH = OUTPUT_ROOT / "checkpoint_inventory.json"
TRAINING_SEEDS = (3101, 3102, 3103)
HOLDOUT_START, HOLDOUT_END = 30_000_000, 30_000_199
MONITOR_START, MONITOR_END = 29_000_000, 29_000_019
FROZEN_ENV_RAW_SHA256 = "ad16c516b31c6fd6eeed825da114e53e6092356daed18b2723371750e5dd92b2"
FROZEN_ENV_SEMANTIC_SHA256 = "ca2108c449065f17a3ad8ea287c94e8aa94dadac8b1e20a7b063afbfd22333ee"
# Snapshot/restore is an opt-in training-infrastructure interface; reset/step,
# reward, weapon, Blue policy and configuration semantics remain unchanged.
FROZEN_ENV_SOURCE_TREE_SHA256 = "0aacc84dda1d7d43221f2255f577c2c3810a44d751c680d0236c301d1c9376b6"
MAIN_ENABLEMENT = {
    "MAPPO": (False, False),
    "WB-MAPPO": (False, True),
    "EA-MAPPO": (True, False),
    "EA-WB-MAPPO": (True, True),
}
UNRELATED_MODULES = {
    "advantage_priority", "ppo_stabilization", "recurrent_memory", "popart",
    "wave_context", "multi_wave_reward", "warm_start", "curriculum", "policy_anchor",
}
FROZEN_TRAINING = {
    "actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4,
    "gamma": .999, "gae_lambda": .95, "clip_ratio": .2,
    "entropy_coefficient": .01, "value_loss_coefficient": .5,
    "ppo_epochs": 10, "minibatch_size": 512, "rollout_steps": 256,
    "num_train_envs": 24, "total_sampled_steps": 900_000,
    "evaluation_episodes": 20, "device": "cuda",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_source_tree_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "env").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def enabled_modules(config: dict) -> set[str]:
    return {name for name, value in config["modules"].items() if value.get("enabled", False)}


def normalize_main(config: dict) -> dict:
    value = deepcopy(config)
    value["modules"]["entity_attention"]["enabled"] = False
    value["modules"]["wave_balancing"]["enabled"] = False
    return value


def normalize_schedule(config: dict) -> dict:
    value = deepcopy(config)
    value["modules"]["actor_lr_decay"]["enabled"] = True
    return value


def validate_environment_freeze() -> dict:
    environment = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    actual = {
        "raw_sha256": file_sha256(ENV_CONFIG),
        "semantic_sha256": config_sha256(environment),
        "source_tree_sha256": env_source_tree_sha256(),
    }
    expected = {
        "raw_sha256": FROZEN_ENV_RAW_SHA256,
        "semantic_sha256": FROZEN_ENV_SEMANTIC_SHA256,
        "source_tree_sha256": FROZEN_ENV_SOURCE_TREE_SHA256,
    }
    if actual != expected or environment.get("environment_variant") != "persistent_wave_v2":
        raise RuntimeError(f"frozen environment changed: expected={expected}, actual={actual}")
    return actual


def validate_plan(manifest: dict, check_outputs: bool = True) -> dict:
    runs = manifest.get("runs", [])
    if len(runs) != 15:
        raise RuntimeError("formal manifest must contain exactly 15 planned runs")
    if 2023 in {run["training_seed"] for run in runs}:
        raise RuntimeError("development seed 2023 is forbidden")
    main = [run for run in runs if run["group"] == "main_matrix"]
    controls = [run for run in runs if run["group"] == "schedule_control"]
    expected_main = {(method, seed) for method in MAIN_ENABLEMENT for seed in TRAINING_SEEDS}
    actual_main = {(run["method"], run["training_seed"]) for run in main}
    if actual_main != expected_main or len(main) != 12:
        raise RuntimeError("main matrix must be exactly four methods by three seeds")
    if {(run["method"], run["training_seed"]) for run in controls} != {
            ("EA-WB Fixed LR", seed) for seed in TRAINING_SEEDS} or len(controls) != 3:
        raise RuntimeError("schedule controls must be exactly EA-WB Fixed LR by three seeds")
    if {run["training_seed"] for run in runs} != set(TRAINING_SEEDS):
        raise RuntimeError("formal training seeds must be exactly 3101, 3102, 3103")
    if len({run["output_dir"] for run in runs}) != 15:
        raise RuntimeError("formal output paths must be unique")

    configs: dict[str, dict] = {}
    for run in runs:
        config = load_config(ROOT / run["config_path"])
        configs[run["method"]] = config
        failed = [key for key, expected in FROZEN_TRAINING.items() if config["training"].get(key) != expected]
        if failed:
            raise RuntimeError(f"{run['method']} frozen training fields changed: {failed}")
        if any(config["modules"].get(name, {}).get("enabled", False) for name in UNRELATED_MODULES):
            raise RuntimeError(f"{run['method']} enables an unrelated module")
        expected_ea, expected_wb = (MAIN_ENABLEMENT.get(run["method"], (True, True)))
        decay = run["group"] == "main_matrix"
        actual = (
            bool(config["modules"]["entity_attention"]["enabled"]),
            bool(config["modules"]["wave_balancing"]["enabled"]),
            bool(config["modules"]["actor_lr_decay"]["enabled"]),
        )
        if actual != (expected_ea, expected_wb, decay):
            raise RuntimeError(f"{run['method']} method enablement mismatch: {actual}")
        if actual != (run["entity_attention_enabled"], run["wave_balance_enabled"], run["actor_lr_decay_enabled"]):
            raise RuntimeError(f"manifest/config enablement mismatch for {run['output_dir']}")
        decay_config = config["modules"]["actor_lr_decay"]
        expected_decay = {"enabled":decay, "schedule":"delayed_linear", "start_step":600000,
                          "end_step":900000, "start_lr":3e-4, "end_lr":1e-4}
        expected_schedule = "0..600000:0.0003;600000..900000:linear_to_0.0001" if decay else "fixed_0.0003"
        if decay_config != expected_decay or run["actor_lr_schedule"] != expected_schedule:
            raise RuntimeError(f"{run['method']} actor LR schedule mismatch")
        protocol = config["formal_protocol"]
        if protocol["checkpoint_selection"] != {"primary":"latest_at_budget", "budget":900000, "best_eval_role":"secondary_diagnostic_only"}:
            raise RuntimeError("primary checkpoint protocol changed")
        if config["implementation"]["evaluation_seed_base"] != MONITOR_START:
            raise RuntimeError("monitoring seed base changed")
        record_checks = (
            run["total_sampled_steps"] == 900_000,
            run["environment_variant"] == "persistent_wave_v2",
            run["primary_checkpoint_rule"] == "latest_at_900000",
            (run["formal_holdout_start"], run["formal_holdout_end"], run["formal_holdout_episode_count"])
            == (HOLDOUT_START, HOLDOUT_END, 200),
            run["critic_lr"] == 3e-4,
        )
        if not all(record_checks):
            raise RuntimeError(f"manifest protocol mismatch for {run['output_dir']}")
        output = ROOT / run["output_dir"]
        if check_outputs and output.exists() and any(output.iterdir()):
            raise FileExistsError(f"formal output contains stale results: {output}")

    normalized = [normalize_main(configs[name]) for name in MAIN_ENABLEMENT]
    if any(value != normalized[0] for value in normalized[1:]):
        raise RuntimeError("main matrix differs outside Entity Attention/Wave Balance enablement")
    if normalize_schedule(configs["EA-WB Fixed LR"]) != configs["EA-WB-MAPPO"]:
        raise RuntimeError("fixed-LR control differs outside actor_lr_decay.enabled")
    monitoring = set(range(MONITOR_START, MONITOR_END + 1))
    holdout = set(range(HOLDOUT_START, HOLDOUT_END + 1))
    if len(holdout) != 200 or monitoring & holdout or set(TRAINING_SEEDS) & (monitoring | holdout):
        raise RuntimeError("training/monitoring/formal holdout seeds overlap")
    environment_hashes = validate_environment_freeze()
    return {
        "status": "READY_FOR_EA_WB_FORMAL_MULTISEED_TRAINING",
        "planned_runs": len(runs), "main_runs": len(main), "schedule_controls": len(controls),
        "training_seeds": list(TRAINING_SEEDS),
        "monitoring_seed_range": [MONITOR_START, MONITOR_END],
        "formal_holdout_seed_range": [HOLDOUT_START, HOLDOUT_END],
        "primary_checkpoint": "latest.pt@900000",
        "main_matrix_equivalent_except": ["entity_attention.enabled", "wave_balancing.enabled"],
        "schedule_control_equivalent_except": ["actor_lr_decay.enabled"],
        "environment_freeze": environment_hashes,
    }


def checkpoint_inventory(manifest: dict) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal checkpoint audit")

    missing = []
    records = []
    for run in manifest["runs"]:
        directory = ROOT / run["output_dir"]
        paths = {name: directory / name for name in ("latest.pt", "run_config.json", "run_summary.json")}
        absent = [name for name, path in paths.items() if not path.is_file()]
        if absent:
            missing.append({"output_dir": run["output_dir"], "missing": absent})
            continue
        run_config = json.loads(paths["run_config.json"].read_text(encoding="utf-8"))
        summary = json.loads(paths["run_summary.json"].read_text(encoding="utf-8"))
        state = torch.load(paths["latest.pt"], map_location="cuda", weights_only=False)
        extra = state.get("extra", {})
        expected_config = load_config(ROOT / run["config_path"])
        if (int(run_config.get("seed", -1)) != run["training_seed"] or
                int(run_config.get("total_sampled_steps", -1)) != 900_000 or
                int(summary.get("latest_step", -1)) != 900_000 or
                int(state.get("sampled_steps", -1)) != 900_000 or
                int(extra.get("training_seed", -1)) != run["training_seed"] or
                extra.get("environment_variant") != "persistent_wave_v2" or
                extra.get("algorithm_config_sha256") != config_sha256(expected_config) or
                run_config.get("algorithm_config_sha256") != config_sha256(expected_config)):
            raise RuntimeError(f"completed-run metadata mismatch: {run['output_dir']}")
        records.append({
            "method": run["method"], "training_seed": run["training_seed"],
            "checkpoint": f"{run['output_dir']}/latest.pt",
            "checkpoint_sha256": file_sha256(paths["latest.pt"]), "sampled_steps": 900_000,
        })
    if missing:
        raise RuntimeError(f"formal holdout locked: required primary checkpoints are missing: {missing}")
    return {"protocol_name": manifest["protocol_name"], "primary_rule": "latest_at_900000", "runs": records}


def freeze_inventory(manifest: dict) -> dict:
    inventory = checkpoint_inventory(manifest)
    if INVENTORY_PATH.exists():
        existing = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        if existing != inventory:
            raise RuntimeError("frozen checkpoint inventory differs from current primary checkpoints")
        return existing
    INVENTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INVENTORY_PATH.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    return inventory


def validate_holdout_ready(manifest: dict) -> dict:
    current = checkpoint_inventory(manifest)
    if not INVENTORY_PATH.is_file():
        raise RuntimeError("formal holdout locked: checkpoint inventory has not been frozen")
    frozen = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    if frozen != current:
        raise RuntimeError("formal holdout locked: frozen checkpoint inventory mismatch")
    return {"status":"READY_FOR_FORMAL_HOLDOUT", "primary_checkpoints":15, "inventory":str(INVENTORY_PATH.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--freeze-checkpoint-inventory", action="store_true")
    group.add_argument("--check-holdout-ready", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    if args.freeze_checkpoint_inventory:
        result = freeze_inventory(manifest)
    elif args.check_holdout_ready:
        result = validate_holdout_ready(manifest)
    else:
        result = validate_plan(manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
