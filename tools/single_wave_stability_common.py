"""Shared fail-closed utilities for the clean single-wave stability diagnostic."""
from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
from algorithm.common.protocol import config_sha256
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION

ROOT = Path(__file__).resolve().parents[1]
TRAINING_SEEDS = (5401, 5402, 5403)
BEST_STEPS = {5401: 1_105_920, 5402: 1_105_920, 5403: 1_400_832}
FORMAL_EVAL_RANGE = range(48_000_000, 48_000_050)
DIAGNOSTIC_BASE = 89_200_000
CROSS_HORIZON_BASE = 89_220_000
SMOKE_SEED = 89_290_000
RUN_TEMPLATE = "outputs/mappo_mlp_single_wave_seed{seed}_1p5m"


def run_dir(seed: int) -> Path:
    return ROOT / RUN_TEMPLATE.format(seed=int(seed))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected mapping in {path}")
    return value


def checkpoint_specs() -> list[dict[str, Any]]:
    specs = []
    for seed in TRAINING_SEEDS:
        run = run_dir(seed)
        specs.extend((
            {"training_seed": seed, "checkpoint_role": "best", "expected_step": BEST_STEPS[seed],
             "path": run / "best_eval.pt", "run_dir": run},
            {"training_seed": seed, "checkpoint_role": "final", "expected_step": 1_500_000,
             "path": run / "checkpoint_1500000.pt", "run_dir": run},
        ))
    return specs


def validate_seed_bank(seeds: Iterable[int], forbidden: Iterable[int] = FORMAL_EVAL_RANGE) -> list[int]:
    values = [int(x) for x in seeds]
    if not values or any(b != a + 1 for a, b in zip(values, values[1:])):
        raise RuntimeError("diagnostic seeds must be non-empty, unique, contiguous and ordered")
    overlap = sorted(set(values).intersection(map(int, forbidden)))
    if overlap:
        raise RuntimeError(f"diagnostic seeds overlap 48M formal evaluation bank: {overlap}")
    return values


def strict_checkpoint_identity(spec: dict[str, Any], state: dict[str, Any] | None = None) -> dict[str, Any]:
    path, run = Path(spec["path"]), Path(spec["run_dir"])
    if not path.is_file():
        raise FileNotFoundError(path)
    env = load_yaml(run / "env_config.yaml")
    algo = load_yaml(run / "algorithm_config.yaml")
    state = torch.load(path, map_location="cpu", weights_only=False) if state is None else state
    validate_checkpoint_for_evaluation(state, env, algo, allow_cross_variant=False)
    extra = state.get("extra")
    if not isinstance(extra, dict):
        raise RuntimeError("checkpoint extra metadata missing")
    required = {
        "algorithm": (state.get("algorithm"), "MAPPO"),
        "mappo_impl_version": (state.get("mappo_impl_version"), MAPPO_IMPL_VERSION),
        "critic_type": (state.get("critic_type", extra.get("critic_type")), "mlp"),
        "observation_dim": (extra.get("observation_dim"), 52),
        "action_dim": (extra.get("action_dim"), 3),
        "num_agents": (extra.get("num_agents"), 4),
        "training_seed": (extra.get("training_seed"), int(spec["training_seed"])),
        "environment_variant": (extra.get("environment_variant"), "persistent_wave_v2"),
        "sampled_steps": (state.get("sampled_steps"), int(spec["expected_step"])),
        "environment_config_sha256": (extra.get("environment_config_sha256"), config_sha256(env)),
        "algorithm_config_sha256": (extra.get("algorithm_config_sha256"), config_sha256(algo)),
        "total_waves": (env.get("persistent_waves", {}).get("total_waves"), 1),
        "max_steps": (env.get("simulation", {}).get("max_steps"), 3000),
    }
    bad = {k: {"actual": a, "expected": e} for k, (a, e) in required.items() if a != e}
    if bad:
        raise RuntimeError(f"checkpoint identity mismatch for {path}: {bad}")
    return {"training_seed": int(spec["training_seed"]), "checkpoint_role": spec["checkpoint_role"],
            "checkpoint_step": int(spec["expected_step"]), "checkpoint_sha256": sha256(path),
            "checkpoint_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "environment_config_sha256": config_sha256(env), "algorithm_config_sha256": config_sha256(algo),
            "environment_config": env, "algorithm_config": algo}


def load_clean_mappo(spec: dict[str, Any], device: str = "cuda"):
    if device != "cuda":
        raise RuntimeError("checkpoint diagnostic requires CUDA; CPU fallback is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    state = torch.load(spec["path"], map_location="cpu", weights_only=False)
    identity = strict_checkpoint_identity(spec, state)
    trainer = build_mappo_trainer(identity["algorithm_config"], device)
    trainer.load(spec["path"], allow_legacy_diagnostic=False)
    trainer.actor.eval(); trainer.critic.eval()
    return trainer, identity


def deterministic_policy_data(trainer, observation: np.ndarray, alive: np.ndarray):
    tensor = torch.as_tensor(observation, dtype=torch.float32, device=trainer.device)
    with torch.no_grad():
        distribution = trainer.actor.distribution(tensor)
        raw_mean = distribution.mean
        action = torch.tanh(raw_mean)
        log_std = torch.log(distribution.scale)
    mask = torch.as_tensor(alive, dtype=torch.float32, device=trainer.device)[:, None]
    return tuple(x.mul(mask).cpu().numpy() for x in (action, raw_mean, log_std))


def radial_velocity(x: float, y: float, vx: float, vy: float) -> float:
    radius = math.hypot(x, y)
    return 0.0 if radius == 0 else (x * vx + y * vy) / radius


def heading_relative_to_outward(x: float, y: float, heading: float) -> float:
    return float((heading - math.atan2(y, x) + math.pi) % (2 * math.pi) - math.pi)


def precursor_indices(boundary_step: int, available_steps: Iterable[int], lags=(50, 20, 10, 5, 1)) -> dict[int, int]:
    available = set(map(int, available_steps))
    return {int(lag): int(boundary_step - lag) for lag in lags if boundary_step - lag in available}


def agent_attempt_transitions(armed_before: Iterable[bool], armed_after: Iterable[bool],
                              alive_before: Iterable[bool]) -> np.ndarray:
    """Detect entry-trigger attempts without changing or depending on env info."""
    before = np.asarray(list(armed_before), dtype=bool)
    after = np.asarray(list(armed_after), dtype=bool)
    alive = np.asarray(list(alive_before), dtype=bool)
    if before.shape != (4,) or after.shape != (4,) or alive.shape != (4,):
        raise ValueError("FireState transition vectors must all have shape (4,)")
    return alive & before & ~after


def active_pursuit_geometry(row: dict[str, Any]) -> bool:
    if bool(row.get("fire_window")):
        return True
    distance = row.get("nearest_blue_distance")
    off_boresight = row.get("nearest_blue_off_boresight")
    closing = row.get("closing_velocity")
    range_max = row.get("weapon_range_max", 4000.0)
    return bool(distance is not None and off_boresight is not None and closing is not None
                and float(distance) <= float(range_max)
                and abs(float(off_boresight)) < math.pi / 2 and float(closing) > 0)


def boundary_descriptor(rows: list[dict[str, Any]], boundary_step: int, last_red_kill_step: int | None) -> str:
    """Transparent trajectory descriptor; never an inference of policy intent."""
    if not rows:
        return "UNCLASSIFIED_BOUNDARY_EXIT"
    recent = [r for r in rows if int(r.get("lag_steps", 999)) <= 20]
    recent_kill = last_red_kill_step is not None and 0 <= boundary_step - last_red_kill_step <= 100
    tactical = any(active_pursuit_geometry(r) for r in recent)
    if recent_kill and recent and not tactical:
        return "RECENT_KILL_OVERSHOOT_CANDIDATE"
    if tactical:
        return "TACTICAL_OVERSHOOT_CANDIDATE"
    escape_like = len(recent) >= 2 and all(
        r["radial_velocity"] > 0 and abs(r["heading_relative_to_outward"]) < math.pi / 2
        and not bool(r.get("fire_window")) and not active_pursuit_geometry(r)
        and int(r.get("steps_since_own_red_attempt", 0)) > 20
        for r in recent)
    return "ESCAPE_LIKE_TRAJECTORY_DESCRIPTOR" if escape_like else "UNCLASSIFIED_BOUNDARY_EXIT"


def action_saturation(actions: np.ndarray, threshold: float) -> dict[str, float]:
    values = np.abs(np.asarray(actions, dtype=float).reshape(-1, 3))
    names = ("heading", "pitch", "speed")
    return {name: float(np.mean(values[:, i] > threshold)) if len(values) else 0.0 for i, name in enumerate(names)}


def counterfactual_score(row: dict[str, Any], boundary_penalty: float, timeout_penalty: float,
                         win_bonus: float, mission_loss_penalty: float) -> float:
    value = float(row["episode_return"]) - float(row["R2"]) + boundary_penalty * int(row["red_boundary_exits"])
    if row["termination_reason"] == "red_failure_timeout":
        value += timeout_penalty
    value += win_bonus if bool(row["red_success"]) else mission_loss_penalty
    return value


def cross_horizon_config(environment_config: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(environment_config)
    result["simulation"]["max_steps"] = 1000
    expected = copy.deepcopy(environment_config)
    expected["simulation"]["max_steps"] = 1000
    if result != expected:
        raise RuntimeError("cross-horizon protocol changed more than max_steps")
    return result


def optimization_phase_predicates(best_step: int) -> dict[str, Any]:
    return {"TRAINING_FULL": lambda s: True,
            "BEST_PRECEDING_200K": lambda s: best_step-200_000 < s <= best_step,
            "BEST_TO_FINAL": lambda s: best_step < s <= 1_500_000,
            "LAST_100K": lambda s: s > 1_400_000,
            "LAST_200K": lambda s: s > 1_300_000}


def refuse_existing_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")


def normal_interval_probability(error: float, threshold: float, scale: float = 1.0) -> float:
    if scale == 0:
        return float(abs(error) <= threshold)
    cdf = lambda x: .5 * (1 + math.erf(x / math.sqrt(2)))
    return cdf((threshold - error) / scale) - cdf((-threshold - error) / scale)


def weapon_hit_probability(distance: float, azimuth_error: float, elevation_error: float,
                           effective_distance: float = 2232.442506204989,
                           azimuth_scale: float = 1.0, elevation_scale: float = 1.0) -> float:
    threshold = math.pi * math.exp(-distance / effective_distance)
    return normal_interval_probability(azimuth_error, threshold, azimuth_scale) * normal_interval_probability(
        elevation_error, threshold, elevation_scale)


def symmetric_cone_components(off_boresight: float) -> tuple[float, float]:
    # With flight-frame azimuth/elevation, cos(off)=cos(az)*cos(el).
    beta = math.acos(math.sqrt(max(math.cos(off_boresight), 0.0)))
    return beta, beta


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("status\nNO_ROWS\n", encoding="utf-8"); return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


__all__ = [name for name in globals() if not name.startswith("_")]
