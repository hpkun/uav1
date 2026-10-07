"""Static preflight and manifest generation for Jiao 2025 seed-2023 screening."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
from algorithm.modular_mappo.trainer import MODULAR_MAPPO_IMPL_VERSION
from algorithm.train_modular_mappo import load_config
from tools.analyze_jiao2025_reproduction import (
    EXPECTED_MODULES, FRESH_EPISODES, FRESH_SEED_BASE, FORMAL_HOLDOUT,
    PRIMARY_METHODS, SUPPLEMENTARY_METHODS, VALIDATION_EPISODES,
    VALIDATION_SEED_BASE, validate_jiao_config,
    validate_validation_fresh_disjoint,
)

PROTOCOL_NAME = "JIAO2025_SEED2023_1P5M_SCREENING_V2"
DOI = "10.1049/cth2.12781"
ENV_CONFIG = ROOT / "configs" / "persistent_wave_v2_environment.yaml"
CORE_CONFIG = ROOT / "configs" / "jiao2025_core_1p5m.yaml"
FULL_CONFIG = ROOT / "configs" / "jiao2025_full_1p5m.yaml"
CORE_OUTPUT = ROOT / "outputs" / "jiao2025_core_1p5m_seed2023"
FULL_OUTPUT = ROOT / "outputs" / "jiao2025_full_1p5m_seed2023"
MANIFEST = ROOT / "outputs" / "jiao2025_screening_protocol.json"
ANALYSIS_CACHE = ROOT / "outputs" / "jiao2025_reproduction_analysis" / "evaluation_cache"
FROZEN_RAW_SHA256 = {
    ROOT / "configs" / "pw_alloff_matched_1p5m.yaml": "7d74e67462ba5dbe7f463d9ec0db28392a87f2d9c415bbe82923e9b8cee39565",
    ROOT / "configs" / "pw_m5_wave_balance.yaml": "8a3d990ef97b95c8780519546b03a296b6f8e3d0a8980550513449df6625a274",
    ENV_CONFIG: "ad16c516b31c6fd6eeed825da114e53e6092356daed18b2723371750e5dd92b2",
}
FROZEN_ENV_SEMANTIC_SHA256 = "ca2108c449065f17a3ad8ea287c94e8aa94dadac8b1e20a7b063afbfd22333ee"
# Snapshot/restore is an opt-in training-infrastructure interface; reset/step,
# reward, weapon, Blue policy and configuration semantics remain unchanged.
FROZEN_ENV_SOURCE_TREE_SHA256 = "0aacc84dda1d7d43221f2255f577c2c3810a44d751c680d0236c301d1c9376b6"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_source_tree_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "env").rglob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode());digest.update(b"\0")
        digest.update(path.read_bytes());digest.update(b"\0")
    return digest.hexdigest()


def _enabled(config: dict) -> set[str]:
    return {name for name, value in config["modules"].items() if value.get("enabled", False)}


def preflight(write_manifest: bool = True) -> dict:
    for path, expected in FROZEN_RAW_SHA256.items():
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen file changed: {path} expected={expected} actual={actual}")
    environment = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    if environment.get("environment_variant") != "persistent_wave_v2":
        raise RuntimeError("Jiao screening requires persistent_wave_v2")
    if config_sha256(environment) != FROZEN_ENV_SEMANTIC_SHA256:
        raise RuntimeError("frozen environment semantic hash changed")
    if env_source_tree_sha256() != FROZEN_ENV_SOURCE_TREE_SHA256:
        raise RuntimeError("frozen env source tree changed")
    if CORE_OUTPUT.exists() or FULL_OUTPUT.exists():
        raise FileExistsError("Jiao formal output directory already exists; fresh-output protection engaged")
    cache_files = sorted(path.name for path in ANALYSIS_CACHE.glob("*.json")) if ANALYSIS_CACHE.is_dir() else []
    if cache_files:
        raise RuntimeError(f"fresh-comparison cache exists before Jiao screening completion: {cache_files}")
    core = load_config(CORE_CONFIG);full = load_config(FULL_CONFIG)
    validate_jiao_config("Jiao-Core", core);validate_jiao_config("Jiao-Full", full)
    seed_protocol = validate_validation_fresh_disjoint(core)
    if validate_validation_fresh_disjoint(full) != seed_protocol:
        raise RuntimeError("Core/Full validation protocols differ")
    alloff = load_config(ROOT / "configs" / "pw_alloff_matched_1p5m.yaml")
    m5 = load_config(ROOT / "configs" / "pw_m5_wave_balance.yaml")
    if _enabled(alloff) != EXPECTED_MODULES["All-Off"] or _enabled(m5) != EXPECTED_MODULES["WB-MAPPO"]:
        raise RuntimeError("frozen All-Off/M5 module semantics changed")
    for name, config in (("All-Off", alloff), ("WB-MAPPO", m5)):
        if int(config["training"]["total_sampled_steps"]) != 1_500_000 or int(config["training"]["seed"]) != 2023:
            raise RuntimeError(f"{name} budget/seed mismatch")
        if (int(config["implementation"]["evaluation_seed_base"]) != VALIDATION_SEED_BASE or
                int(config["training"]["evaluation_episodes"]) != VALIDATION_EPISODES or
                int(config["training"]["evaluation_interval_sampled_steps"]) != 100_000):
            raise RuntimeError(f"{name} checkpoint-selection protocol mismatch")
    validation = set(range(VALIDATION_SEED_BASE, VALIDATION_SEED_BASE + VALIDATION_EPISODES))
    fresh = set(range(FRESH_SEED_BASE, FRESH_SEED_BASE + FRESH_EPISODES))
    if validation & fresh or validation & FORMAL_HOLDOUT or fresh & FORMAL_HOLDOUT:
        raise RuntimeError("seed protocol overlap")
    manifest = {
        "protocol_name": PROTOCOL_NAME,
        "paper": {"title": "Collaborative decision-making for UAV swarm confrontation based on reinforcement learning", "doi": DOI},
        "repository_implementation": {"baseline_mappo_impl_version": MAPPO_IMPL_VERSION, "modular_mappo_impl_version": MODULAR_MAPPO_IMPL_VERSION},
        "environment": {"variant": "persistent_wave_v2", "config": str(ENV_CONFIG.relative_to(ROOT)),
                        "semantic_sha256": FROZEN_ENV_SEMANTIC_SHA256, "source_tree_sha256": FROZEN_ENV_SOURCE_TREE_SHA256},
        "configs": {"Jiao-Core": {"path": str(CORE_CONFIG.relative_to(ROOT)), "resolved_sha256": config_sha256(core)},
                    "Jiao-Full": {"path": str(FULL_CONFIG.relative_to(ROOT)), "resolved_sha256": config_sha256(full)}},
        "training_seed": 2023, "sampled_step_budget": 1_500_000,
        "checkpoint_selection": {"evaluation_seed_start": VALIDATION_SEED_BASE,
                                 "evaluation_seed_end": VALIDATION_SEED_BASE + VALIDATION_EPISODES - 1,
                                 "evaluation_episodes": VALIDATION_EPISODES,
                                 "evaluation_interval_sampled_steps": 100_000,
                                 "selection_key": ["clear_wave_3_probability", "average_waves_cleared", "raw_environment_return", "negative_average_red_loss"]},
        "fresh_comparison": {"evaluation_seed_start": FRESH_SEED_BASE,
                             "evaluation_seed_end": FRESH_SEED_BASE + FRESH_EPISODES - 1,
                             "evaluation_episodes": FRESH_EPISODES, "executed": False,
                             "rotation_reason": "37M was consumed by a prior development smoke; 38M is the untouched replacement"},
        "contaminated_seed_ranges": [{"start": 37_000_000, "end": 37_000_099,
                                       "reason": "two seeds were used by pre-screening Core/Full smoke"}],
        "seed_overlap": {"validation_vs_fresh": 0, "validation_vs_20m_formal": 0, "fresh_vs_20m_formal": 0},
        "expected_modules": {name: sorted(modules) for name, modules in EXPECTED_MODULES.items()},
        "primary_fair_comparison_methods": list(PRIMARY_METHODS),
        "supplementary_comparison_methods": list(SUPPLEMENTARY_METHODS),
        "planned_outputs": {"Jiao-Core": str(CORE_OUTPUT.relative_to(ROOT)), "Jiao-Full": str(FULL_OUTPUT.relative_to(ROOT))},
        "frozen_controls": {"All-Off_config_sha256": FROZEN_RAW_SHA256[ROOT / "configs" / "pw_alloff_matched_1p5m.yaml"],
                            "M5_config_sha256": FROZEN_RAW_SHA256[ROOT / "configs" / "pw_m5_wave_balance.yaml"]},
        "status": "READY_FOR_1P5M_SCREENING",
    }
    if write_manifest:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = preflight(write_manifest=True)
    print(json.dumps({"manifest": str(MANIFEST), "status": manifest["status"]}, indent=2))


if __name__ == "__main__":
    main()
