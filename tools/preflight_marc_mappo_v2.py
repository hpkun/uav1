#!/usr/bin/env python3
"""Fail-closed CUDA preflight for the MARC-MAPPO V2 433 1M screen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256, runtime_source_manifest
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.train_modular_mappo import load_config

ENV = ROOT / "configs/persistent_wave_v2_blue433_environment.yaml"
BASE = ROOT / "configs/diag_mappo_learnability_common_3m.yaml"
CONFIG = ROOT / "configs/dev_marc_mappo_v2_1m.yaml"
FORMAL_OUTPUT = ROOT / "outputs/dev_marc_mappo_v2_1m_seed5301"
DEFAULT_REPORT = ROOT / "outputs/marc_mappo_v2_preflight.json"


def enabled_modules(config: dict) -> list[str]:
    return sorted(name for name, value in config["modules"].items()
                  if isinstance(value, dict) and value.get("enabled", False))


def validate(*, require_output_absent: bool = True) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; MARC V2 preflight has no CPU fallback")
    env, base, config = load_config(ENV), load_config(BASE), load_config(CONFIG)
    identity = {
        "variant": env.get("environment_variant"),
        "blue_units_per_wave": env["persistent_waves"].get("blue_units_per_wave"),
        "total_waves": env["persistent_waves"].get("total_waves"),
        "max_steps": env["simulation"].get("max_steps"),
        "team_size": env["scenario"].get("team_size"),
        "observation_dim": config["network"].get("observation_dim"),
        "action_dim": config["network"].get("action_dim"),
    }
    expected_identity = {"variant": "persistent_wave_v2", "blue_units_per_wave": [4, 3, 3],
                         "total_waves": 3, "max_steps": 3000, "team_size": 4,
                         "observation_dim": 52, "action_dim": 3}
    if identity != expected_identity:
        raise RuntimeError(f"MARC V2 433 environment identity mismatch: {identity}")
    frozen_hashes = {
        "reward": "ca32c924d8e85f7946fb785212691a460aebc175cea05a8edf19c7e3312a2ac8",
        "blue_policy": "b70d8df8ef5ad3a9f049983d06c8d45e2ec3f3ddcd4e40991bd41e7340c248c9",
        "weapon": "e4926808f99ff7a0f00a008d4c300ce8d3b335e4123055b2e7ec6f8cdf4fcafc",
    }
    actual_hashes = {name: config_sha256(env[name]) for name in frozen_hashes}
    if actual_hashes != frozen_hashes:
        raise RuntimeError(f"frozen reward/Blue/weapon identity changed: {actual_hashes}")
    if config.get("development_method") != "marc_mappo_v2":
        raise RuntimeError("MARC V2 development method mismatch")
    if enabled_modules(config) != ["actor_lr_decay", "milestone_aware_retention_credit"]:
        raise RuntimeError(f"MARC V2 enabled modules mismatch: {enabled_modules(config)}")
    expected_module = {
        "enabled": True, "version": 2, "max_waves": 3,
        "continuation_alpha": 1.0, "wave_balance_temperature": 0.5,
        "wave_weight_min": 0.5, "wave_weight_max": 2.0,
        "deployment_distill_coefficient": 0.05, "elite_segments_per_wave": 8,
        "elite_rows_per_segment": 64, "retention_samples_per_wave": 32,
        "retention_min_rows_per_wave": 32, "retention_stride": 8,
    }
    if config["modules"]["milestone_aware_retention_credit"] != expected_module:
        raise RuntimeError("MARC V2 module constants mismatch")
    for forbidden in ("retention_coefficient", "retention_bank_size_per_wave",
                      "retention_min_samples_per_wave"):
        if forbidden in config["modules"]["milestone_aware_retention_credit"]:
            raise RuntimeError(f"MARC V2 contains forbidden V1 setting: {forbidden}")
    for section in ("network", "runtime_logging"):
        if config[section] != base[section]:
            raise RuntimeError(f"MARC V2 changed baseline {section}")
    for key, value in base["training"].items():
        expected = 1_000_000 if key == "total_sampled_steps" else value
        if config["training"].get(key) != expected:
            raise RuntimeError(f"MARC V2 changed baseline training.{key}")
    if config["implementation"] != base["implementation"]:
        raise RuntimeError("MARC V2 changed baseline implementation")
    validation = config["development_protocol"]["validation"]
    if validation != {"seed_start": 44_000_000, "seed_end": 44_000_049,
                       "episodes": 50, "deterministic": True,
                       "common_scenarios": True, "is_holdout": False}:
        raise RuntimeError("MARC V2 development validation mismatch")
    if config["development_protocol"]["reserved_future_final_test"] != {
            "seed_start": 45_000_000, "seed_end": 45_000_199, "executed": False}:
        raise RuntimeError("MARC V2 reserved final bank mismatch")
    if require_output_absent and FORMAL_OUTPUT.exists():
        raise RuntimeError(f"formal output already exists: {FORMAL_OUTPUT}")
    trainer = build_modular_mappo_trainer(config, "cuda", total_sampled_steps=1_000_000)
    architecture = checkpoint_architecture(trainer)
    expected_architecture = {"actor_input_dim": 52, "actor_context_dim": 0,
                             "actor_gru_hidden_dim": 0, "critic_context_dim": 0,
                             "critic_gru_hidden_dim": 0, "entity_attention_enabled": False,
                             "milestone_aware_retention_credit_enabled": True,
                             "marc_version": 2, "actor_wave_input": False,
                             "log_std_retention": False}
    mismatch = {key: (architecture.get(key), value)
                for key, value in expected_architecture.items()
                if architecture.get(key) != value}
    if mismatch:
        raise RuntimeError(f"MARC V2 architecture mismatch: {mismatch}")
    manifest = runtime_source_manifest(ROOT)
    return {
        "status": "READY_FOR_MARC_MAPPO_V2_433_SCREEN",
        "cuda": torch.cuda.get_device_name(0),
        "environment": identity,
        "enabled_modules": enabled_modules(config),
        "development_method": config["development_method"],
        "marc_config": expected_module,
        "training_seed": config["training"]["seed"],
        "total_sampled_steps": config["training"]["total_sampled_steps"],
        "evaluation_seed_range": [44_000_000, 44_000_049],
        "reserved_45m_executed": False,
        "formal_output_absent": not FORMAL_OUTPUT.exists(),
        "frozen_component_hashes": actual_hashes,
        "environment_config_sha256": config_sha256(env),
        "algorithm_config_sha256": config_sha256(config),
        "runtime_source_manifest_sha256": manifest["runtime_source_manifest_sha256"],
        "network_architecture": architecture,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    report = validate()
    if not args.print_only:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        if output.exists():
            raise FileExistsError(f"refusing to overwrite preflight report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(output)
    print(json.dumps(report, indent=2))
    print(report["status"])


if __name__ == "__main__":
    main()
