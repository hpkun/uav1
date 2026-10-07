#!/usr/bin/env python3
"""Fail-closed CUDA preflight for the MARC-MAPPO V1 1M development screen."""
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

ENV = ROOT / "configs/persistent_wave_v2_environment.yaml"
BASE = ROOT / "configs/diag_mappo_learnability_common_3m.yaml"
MARC = ROOT / "configs/dev_marc_mappo_v1_1m.yaml"
FORMAL_OUTPUT = ROOT / "outputs/dev_marc_mappo_v1_1m_seed5301"
DEFAULT_REPORT = ROOT / "outputs/marc_mappo_v1_preflight.json"


def _enabled(config: dict) -> list[str]:
    return sorted(name for name, value in config["modules"].items()
                  if isinstance(value, dict) and value.get("enabled", False))


def validate() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; MARC preflight does not fall back to CPU")
    env, base, marc = load_config(ENV), load_config(BASE), load_config(MARC)
    identity = (env.get("environment_variant"), env["persistent_waves"]["total_waves"],
                env["simulation"]["max_steps"], env["scenario"]["team_size"])
    if identity != ("persistent_wave_v2", 3, 3000, 4):
        raise RuntimeError(f"environment identity mismatch: {identity}")
    frozen_hashes = {
        "reward": "ca32c924d8e85f7946fb785212691a460aebc175cea05a8edf19c7e3312a2ac8",
        "blue_policy": "b70d8df8ef5ad3a9f049983d06c8d45e2ec3f3ddcd4e40991bd41e7340c248c9",
        "weapon": "e4926808f99ff7a0f00a008d4c300ce8d3b335e4123055b2e7ec6f8cdf4fcafc",
    }
    actual_hashes = {name: config_sha256(env[name]) for name in frozen_hashes}
    if actual_hashes != frozen_hashes:
        raise RuntimeError(f"frozen reward/Blue/weapon identity changed: {actual_hashes}")
    if _enabled(marc) != ["actor_lr_decay", "milestone_aware_retention_credit"]:
        raise RuntimeError(f"MARC enabled-module set mismatch: {_enabled(marc)}")
    expected = {
        "enabled": True, "version": 1, "max_waves": 3,
        "continuation_alpha": 1.0, "wave_balance_temperature": 0.5,
        "wave_weight_min": 0.5, "wave_weight_max": 2.0,
        "retention_coefficient": 0.01, "retention_bank_size_per_wave": 512,
        "retention_samples_per_wave": 32, "retention_min_samples_per_wave": 32,
        "retention_stride": 8,
    }
    if marc["modules"].get("milestone_aware_retention_credit") != expected:
        raise RuntimeError("MARC V1 module constants mismatch")
    for section in ("network", "runtime_logging"):
        if marc[section] != base[section]:
            raise RuntimeError(f"MARC changed baseline {section}")
    for key, value in base["training"].items():
        expected_value = 1_000_000 if key == "total_sampled_steps" else value
        if marc["training"].get(key) != expected_value:
            raise RuntimeError(f"MARC changed baseline training.{key}")
    for key, value in base["implementation"].items():
        if marc["implementation"].get(key) != value:
            raise RuntimeError(f"MARC changed baseline implementation.{key}")
    validation = marc["development_protocol"]["validation"]
    if validation != {"seed_start": 44_000_000, "seed_end": 44_000_049,
                       "episodes": 50, "deterministic": True,
                       "common_scenarios": True, "is_holdout": False}:
        raise RuntimeError("MARC development evaluation protocol mismatch")
    if marc["development_protocol"]["reserved_future_final_test"] != {
            "seed_start": 45_000_000, "seed_end": 45_000_199, "executed": False}:
        raise RuntimeError("MARC future holdout declaration mismatch")
    if FORMAL_OUTPUT.exists():
        raise RuntimeError(f"formal output already exists: {FORMAL_OUTPUT}")
    trainer = build_modular_mappo_trainer(marc, "cuda", total_sampled_steps=1_000_000)
    architecture = checkpoint_architecture(trainer)
    strict_arch = {"actor_input_dim": 52, "actor_context_dim": 0,
                   "actor_gru_hidden_dim": 0, "critic_context_dim": 0,
                   "critic_gru_hidden_dim": 0, "entity_attention_enabled": False,
                   "milestone_aware_retention_credit_enabled": True,
                   "actor_wave_input": False}
    mismatch = {key: (architecture.get(key), value) for key, value in strict_arch.items()
                if architecture.get(key) != value}
    if mismatch:
        raise RuntimeError(f"MARC architecture mismatch: {mismatch}")
    manifest = runtime_source_manifest(ROOT)
    return {
        "status": "READY_FOR_MARC_MAPPO_V1_SCREEN",
        "cuda": torch.cuda.get_device_name(0),
        "environment": {"variant": identity[0], "total_waves": identity[1],
                        "max_steps": identity[2], "observation_dim": 52,
                        "action_dim": 3, "num_agents": 4},
        "enabled_modules": _enabled(marc),
        "marc_config": expected,
        "baseline_training_protocol_unchanged_except_budget": True,
        "training_seed": marc["training"]["seed"],
        "total_sampled_steps": marc["training"]["total_sampled_steps"],
        "evaluation_seed_range": [44_000_000, 44_000_049],
        "reserved_45m_executed": False,
        "formal_output_absent": True,
        "environment_config_sha256": config_sha256(env),
        "frozen_component_hashes": actual_hashes,
        "algorithm_config_sha256": config_sha256(marc),
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
