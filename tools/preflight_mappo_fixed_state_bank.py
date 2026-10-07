#!/usr/bin/env python3
"""Static/CUDA preflight for the MAPPO fixed-state-bank diagnostic."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.diagnose_mappo_fixed_state_bank import (
    BANK_GENERATOR,
    BANK_SEED_BASE,
    CHECKPOINTS,
    DEPLOYMENT_SEED_BASE,
    ENV_CONFIG,
    checkpoint_identity,
    validate_seed_protocol,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate(output_dir: Path) -> dict:
    require(torch.cuda.is_available(), "CUDA is mandatory for checkpoint audit")
    env = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    require(env.get("environment_variant") == "persistent_wave_v2", "wrong environment variant")
    require(int(env["scenario"]["team_size"]) == 4, "Red team size must be 4")
    require(int(env["persistent_waves"]["total_waves"]) == 3, "total_waves must be 3")
    require(int(env["simulation"]["max_steps"]) == 3000, "max_steps must be 3000")
    require(BANK_GENERATOR == "FreshStrong", "bank generator is not frozen FreshStrong")
    require(len(CHECKPOINTS) == 5, "expected exactly five diagnostic checkpoints")
    require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    ranges = validate_seed_protocol(BANK_SEED_BASE, 12, 6)
    require(ranges["deployment"][0] == DEPLOYMENT_SEED_BASE, "deployment range is not 88.71M")
    module = importlib.util.find_spec("tools.diagnose_mappo_fixed_state_bank")
    require(module is not None, "diagnostic script cannot be imported")
    identities = {
        name: checkpoint_identity(name, path, torch.device("cuda:0"))
        for name, path in CHECKPOINTS.items()
    }
    require(all(row["observation_dim"] == 52 for row in identities.values()), "observation dim mismatch")
    require(all(row["action_dim"] == 3 for row in identities.values()), "action dim mismatch")
    require(all(row["num_agents"] == 4 for row in identities.values()), "agent count mismatch")
    requested_final = CHECKPOINTS["FreshFinal"].parent / "final.pt"
    return {
        "status": "READY_FOR_MAPPO_FIXED_STATE_BANK_DIAGNOSTIC",
        "cuda": torch.cuda.get_device_name(0),
        "environment": {
            "variant": "persistent_wave_v2", "observation_dim": 52,
            "action_dim": 3, "red_agents": 4, "total_waves": 3,
            "max_steps": 3000,
        },
        "bank_generator": BANK_GENERATOR,
        "bank_generation_mode": "deterministic_tanh_mean",
        "seed_ranges": ranges,
        "protected_44m_47m_accessed": False,
        "checkpoints": identities,
        "fresh_final_note": {
            "requested_final_pt_exists": requested_final.exists(),
            "exact_budget_fallback": str(CHECKPOINTS["FreshFinal"].resolve()),
            "fallback_sampled_steps": identities["FreshFinal"]["sampled_steps"],
        },
        "output_dir_absent": True,
        "full_diagnostic_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/mappo_fixed_state_bank_diagnostic",
    )
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    print(json.dumps(validate(output), indent=2))


if __name__ == "__main__":
    main()
