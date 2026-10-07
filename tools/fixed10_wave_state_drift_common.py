"""Shared, analysis-only helpers for the Fixed10 wave-state drift audit."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5301, 5302, 5303)
DIAGNOSTIC_SEEDS = tuple(range(88_330_000, 88_330_016))
FORBIDDEN_SEED_MIN = 45_000_000
FORBIDDEN_SEED_MAX = 45_000_199
AUDIT_DIR = ROOT / "outputs" / "fixed10_wave_state_drift_audit"
INVENTORY_PATH = ROOT / "outputs" / "fixed10_wave_state_drift_inventory.json"
BANK_PATH = ROOT / "outputs" / "comprehensive_persistent_wave_audit" / "supplemental_causal_audit" / "observation_bank.npz"
HISTORICAL_SNAPSHOT_PATH = BANK_PATH.parent / "entry_snapshot_bank.pt"


def run_dir(arm: str, seed: int) -> Path:
    return ROOT / "outputs" / f"dev_actor_clip{arm}_fixed10_seed{seed}_300k"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(value.dtype).encode("ascii"))
        h.update(str(tuple(value.shape)).encode("ascii"))
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def checkpoint_step(checkpoint: dict[str, Any]) -> int:
    value = checkpoint.get("sampled_steps")
    if value is None:
        value = checkpoint.get("trainer_state", {}).get("sampled_steps")
    if value is None:
        raise KeyError("checkpoint has no sampled_steps metadata")
    return int(value)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write a fake empty result: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def distribution_stats(values: np.ndarray, include_p95: bool = True) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None if include_p95 else None}
    result = {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
    }
    if include_p95:
        result["p95"] = float(np.quantile(values, 0.95))
    return result


def symmetric_diagonal_gaussian_kl(
    mu_a: np.ndarray, log_std_a: np.ndarray, mu_b: np.ndarray, log_std_b: np.ndarray
) -> np.ndarray:
    """Per-sample symmetric KL, summed over action dimensions."""
    mu_a = np.asarray(mu_a, dtype=np.float64)
    mu_b = np.asarray(mu_b, dtype=np.float64)
    log_std_a = np.asarray(log_std_a, dtype=np.float64)
    log_std_b = np.asarray(log_std_b, dtype=np.float64)
    var_a = np.exp(2.0 * log_std_a)
    var_b = np.exp(2.0 * log_std_b)
    delta2 = (mu_a - mu_b) ** 2
    kl_ab = 0.5 * (2.0 * (log_std_b - log_std_a) + (var_a + delta2) / var_b - 1.0)
    kl_ba = 0.5 * (2.0 * (log_std_a - log_std_b) + (var_b + delta2) / var_a - 1.0)
    result = 0.5 * (kl_ab + kl_ba).sum(axis=-1)
    return np.maximum(result, 0.0)


def validate_bank(observations: np.ndarray, alive: np.ndarray, wave: np.ndarray) -> None:
    if observations.ndim != 3 or observations.shape[1:] != (4, 52):
        raise ValueError(f"observation bank shape mismatch: {observations.shape}")
    if alive.shape != observations.shape[:2]:
        raise ValueError(f"alive bank shape mismatch: {alive.shape}")
    if wave.shape != (observations.shape[0],):
        raise ValueError(f"wave bank shape mismatch: {wave.shape}")
    if not set(np.unique(wave)).issubset({1, 2, 3}):
        raise ValueError("wave bank contains labels outside W1/W2/W3")
    if np.any((alive != 0) & (alive != 1)):
        raise ValueError("alive bank must be binary")


def policy_specifications() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm, label in (("05", "Control05"), ("10", "Treatment10")):
            specs.append({"id": f"{label}_seed{seed}_final", "arm": arm, "label": label,
                          "seed": seed, "role": "final", "path": run_dir(arm, seed) / "final.pt",
                          "expected_step": 1_805_280})
    specs.extend([
        {"id": "Treatment10_seed5303_peak", "arm": "10", "label": "Treatment10", "seed": 5303,
         "role": "peak", "path": run_dir("10", 5303) / "best_eval.pt", "expected_step": 1_701_888},
        {"id": "Treatment10_seed5303_1800192", "arm": "10", "label": "Treatment10", "seed": 5303,
         "role": "checkpoint_1800192", "path": run_dir("10", 5303) / "checkpoint_1800192.pt", "expected_step": 1_800_192},
        {"id": "Control05_seed5303_1800192", "arm": "05", "label": "Control05", "seed": 5303,
         "role": "checkpoint_1800192", "path": run_dir("05", 5303) / "checkpoint_1800192.pt", "expected_step": 1_800_192},
    ])
    return specs


def inventory_specifications() -> list[dict[str, Any]]:
    specs = policy_specifications()
    for seed in SEEDS:
        specs.append({
            "id": f"Plain_source_seed{seed}_1505280", "arm": None, "label": "PlainSource",
            "seed": seed, "role": "branch_source", "path": ROOT / "outputs" /
            "diag_mappo_learnability" / f"l3_seed{seed}" / "checkpoint_1505280.pt",
            "expected_step": 1_505_280,
        })
    return specs


def comparison_specifications() -> list[dict[str, str]]:
    comparisons = [
        {"id": f"seed{seed}_final_control_vs_treatment", "policy_a": f"Control05_seed{seed}_final",
         "policy_b": f"Treatment10_seed{seed}_final", "class": "primary_final"}
        for seed in SEEDS
    ]
    comparisons.extend([
        {"id": "seed5303_treatment_peak_vs_final", "policy_a": "Treatment10_seed5303_peak",
         "policy_b": "Treatment10_seed5303_final", "class": "selection_biased_peak_final"},
        {"id": "seed5303_treatment_last5088", "policy_a": "Treatment10_seed5303_1800192",
         "policy_b": "Treatment10_seed5303_final", "class": "strict_last_5088"},
        {"id": "seed5303_control_last5088", "policy_a": "Control05_seed5303_1800192",
         "policy_b": "Control05_seed5303_final", "class": "strict_last_5088"},
    ])
    return comparisons


def stable_snapshot_identity(snapshot: dict[str, Any], restored: dict[str, Any], env: Any) -> bool:
    metadata = snapshot["metadata"]
    return (
        int(metadata["wave_index"]) == int(restored["wave_index"]) == int(env.wave_index)
        and int(metadata["steps"]) == int(env.steps) == int(env._wave_start_step)
        and np.array_equal(np.asarray(restored["red_alive_mask"]), env.red_alive_mask)
        and np.array_equal(np.asarray(restored["observation"]), env._observations())
    )


def assert_safe_diagnostic_seeds(seeds: Iterable[int]) -> None:
    seeds = [int(seed) for seed in seeds]
    if any(FORBIDDEN_SEED_MIN <= seed <= FORBIDDEN_SEED_MAX for seed in seeds):
        raise ValueError("45M future-final seed access is forbidden")
    if any(not (88_000_000 <= seed < 89_000_000) for seed in seeds):
        raise ValueError("diagnostic environment seeds must remain in the independent 88M range")


def checkpoint_mutation_guard(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def load_algorithm_and_environment(run: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    algorithm = yaml.safe_load((run / "algorithm_config.yaml").read_text(encoding="utf-8"))
    environment = yaml.safe_load((run / "runtime_env_config.yaml").read_text(encoding="utf-8"))
    return algorithm, environment


def clone_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(snapshot)
