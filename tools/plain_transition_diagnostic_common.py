"""Pure diagnostic helpers for the Plain MAPPO transition-mechanism study."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY_SEEDS = (5301, 5302, 5303)
EVALUATION_SEEDS = tuple(range(44_000_000, 44_000_050))
FUTURE_FINAL_RANGE = (45_000_000, 45_000_199)
OUTPUT_DIR = ROOT / "outputs/plain_transition_mechanism_44m"
CHECKPOINTS = {
    seed: ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/final.pt"
    for seed in POLICY_SEEDS
}


def future_rng_seed(evaluation_seed: int, next_wave: int) -> int:
    """Diagnostic-only RNG, independent of source and continuation policy."""
    if evaluation_seed not in EVALUATION_SEEDS or next_wave not in (2, 3):
        raise ValueError("future RNG requires a 44M case and next wave 2 or 3")
    return int(evaluation_seed * 10 + next_wave)


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def checkpoint_contract(path: Path, require_cuda: bool = True) -> dict[str, Any]:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for checkpoint diagnostic audit")
    device = "cuda:0" if require_cuda else "cpu"
    state = torch.load(path, map_location=device, weights_only=False)
    extra = state.get("extra", {})
    arch = extra.get("network_architecture", {})
    config = extra.get("algorithm_config")
    if int(state.get("sampled_steps", -1)) != 3_000_000:
        raise RuntimeError(f"{path}: checkpoint is not sampled_steps=3M")
    if not isinstance(config, dict):
        raise RuntimeError(f"{path}: missing algorithm_config metadata")
    modules = config.get("modules", {})
    forbidden = [
        name for name, block in modules.items()
        if name != "actor_lr_decay" and isinstance(block, dict) and block.get("enabled", False)
    ]
    if forbidden:
        raise RuntimeError(f"{path}: not Plain MAPPO; enabled={forbidden}")
    if int(arch.get("actor_context_dim", 0)) or int(arch.get("critic_context_dim", 0)):
        raise RuntimeError(f"{path}: context dimensions must both be zero")
    if int(arch.get("actor_gru_hidden_dim", 0)) or int(arch.get("critic_gru_hidden_dim", 0)):
        raise RuntimeError(f"{path}: recurrent memory must be disabled")
    env = extra.get("runtime_environment_config") or extra.get("environment_config")
    if not isinstance(env, dict):
        raise RuntimeError(f"{path}: missing environment metadata")
    if env.get("environment_variant") != "persistent_wave_v2":
        raise RuntimeError(f"{path}: wrong environment variant")
    if int(env["persistent_waves"]["total_waves"]) != 3 or int(env["simulation"]["max_steps"]) != 3000:
        raise RuntimeError(f"{path}: expected three waves and max_steps=3000")
    result = {
        "sampled_steps": int(state["sampled_steps"]),
        "actor_context_dim": int(arch.get("actor_context_dim", 0)),
        "critic_context_dim": int(arch.get("critic_context_dim", 0)),
        "actor_gru_hidden_dim": int(arch.get("actor_gru_hidden_dim", 0)),
        "critic_gru_hidden_dim": int(arch.get("critic_gru_hidden_dim", 0)),
        "training_seed": int(extra.get("training_seed")),
        "environment_variant": env["environment_variant"],
        "total_waves": int(env["persistent_waves"]["total_waves"]),
        "max_steps": int(env["simulation"]["max_steps"]),
        "enabled_modules": list(state.get("enabled_modules", [])),
        "algorithm_config": config,
        "environment_config": env,
        "hidden_dim": int(arch.get("hidden_dim", 256)),
    }
    del state
    if require_cuda:
        torch.cuda.empty_cache()
    return result


def load_trainer(path: Path, device: str = "cuda"):
    from algorithm.modular_mappo.factory import build_modular_mappo_trainer

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("diagnostic checkpoint evaluation requires CUDA")
    contract = checkpoint_contract(path, require_cuda=True)
    trainer = build_modular_mappo_trainer(
        contract["algorithm_config"], device, contract["hidden_dim"], 3_000_000
    )
    trainer.load(path)
    trainer.actor.eval()
    trainer.critic.eval()
    return trainer, contract


def circular_heading_dispersion(values: list[float]) -> float | None:
    if not values:
        return None
    resultant = math.hypot(
        float(np.mean(np.cos(values))), float(np.mean(np.sin(values)))
    )
    return float(math.sqrt(max(0.0, -2.0 * math.log(max(resultant, 1e-12)))))


def time_to_ground(altitude: float, vertical_speed: float) -> float | None:
    return float(altitude / -vertical_speed) if vertical_speed < 0.0 else None


def boundary_diagnostics(state, arena_radius: float) -> tuple[float, float, float | None]:
    radius = float(math.hypot(state.x, state.y))
    margin = float(arena_radius - radius)
    vx, vy = state.velocity_vector()[:2]
    radial_speed = float((state.x * vx + state.y * vy) / max(radius, 1e-12))
    ttb = float(margin / radial_speed) if radial_speed > 0.0 else None
    return margin, radial_speed, ttb


def diagnostic_ground_risk(state, commanded_pitch: float, config: dict[str, Any]) -> tuple[bool, float | None]:
    guard = config["blue_policy"]["ground_avoidance"]
    downward_current = max(-state.v * math.sin(state.theta), 0.0)
    downward_commanded = max(-state.v * math.sin(commanded_pitch), 0.0)
    downward_speed = max(downward_current, downward_commanded)
    if downward_speed <= float(guard.get("downward_speed_epsilon", 1e-6)):
        return False, None
    value = float(state.altitude / downward_speed)
    threshold = float(guard["guard_time_constants"]) * float(
        config["action"]["controller"]["pitch_time_constant"]
    )
    return bool(value <= threshold), value


def classify_death(state, arena_radius: float) -> str:
    # This ordering follows the requested diagnostic classification. Normal
    # environment trajectories virtually never cross both surfaces together.
    if math.hypot(state.x, state.y) > arena_radius:
        return "boundary"
    if state.altitude <= 0.0:
        return "ground"
    return "combat"


def continuation_should_stop(pre_step_wave: int, target_wave: int,
                             info: dict[str, Any], terminated: bool,
                             truncated: bool) -> tuple[bool, bool]:
    cleared = bool(
        pre_step_wave == target_wave and info.get("wave_cleared_this_step", False)
    )
    return bool(cleared or terminated or truncated), cleared


def pairwise_metrics(states: list[Any]) -> tuple[float | None, float | None, float | None]:
    if len(states) < 2:
        return None, None, None
    distances = [
        float(np.linalg.norm(a.as_array()[:3] - b.as_array()[:3]))
        for i, a in enumerate(states) for b in states[i + 1:]
    ]
    return float(np.mean(distances)), float(np.max(distances)), float(np.min(distances))


def _nearest_blue(red, blue):
    from env.geometry import engagement_geometry

    candidates = [(engagement_geometry(red, row).distance, index, row) for index, row in enumerate(blue) if row.alive]
    if not candidates:
        return None
    _, index, target = min(candidates)
    return index, target, engagement_geometry(red, target), engagement_geometry(target, red)


def transition_snapshot(env, source_policy_seed: int, evaluation_seed: int, cleared_wave: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    living = [row for row in env.red if row.alive]
    positions = np.asarray([[row.x, row.y, row.z] for row in living], dtype=float)
    centroid = positions.mean(axis=0)
    spread = float(np.sqrt(np.mean(np.sum((positions - centroid) ** 2, axis=1))))
    pair_mean, pair_max, pair_min = pairwise_metrics(living)
    arena = float(env.arena_radius)
    margins = [arena - math.hypot(row.x, row.y) for row in living]
    radii = [math.hypot(row.x, row.y) for row in living]
    alts = [row.altitude for row in living]
    speeds = [row.v for row in living]
    pitches = [row.theta for row in living]
    vertical = [row.v * math.sin(row.theta) for row in living]
    ttg = [time_to_ground(row.altitude, row.v * math.sin(row.theta)) for row in living]
    rb = [
        (float(np.linalg.norm(np.asarray([b.x-r.x,b.y-r.y,b.z-r.z]))), abs(b.altitude-r.altitude))
        for r in living for b in env.blue if b.alive
    ]
    nearest = [min(float(np.linalg.norm(np.asarray([b.x-r.x,b.y-r.y,b.z-r.z]))) for b in env.blue if b.alive) for r in living]
    from env.geometry import engagement_geometry
    red_ata = [abs(engagement_geometry(r,b).ata) for r in living for b in env.blue if b.alive]
    blue_ata = [abs(engagement_geometry(b,r).ata) for r in living for b in env.blue if b.alive]
    summary = {
        "source_policy_seed": source_policy_seed, "evaluation_seed": evaluation_seed,
        "cleared_wave": cleared_wave, "next_wave": cleared_wave + 1,
        "global_step": int(env.steps), "remaining_horizon": int(env.max_steps-env.steps),
        "red_survivor_count": len(living),
        "red_altitude_mean": float(np.mean(alts)), "red_altitude_min": float(np.min(alts)), "red_altitude_max": float(np.max(alts)),
        "red_speed_mean": float(np.mean(speeds)), "red_speed_std": float(np.std(speeds)),
        "red_pitch_mean": float(np.mean(pitches)), "red_pitch_min": float(np.min(pitches)), "red_pitch_max": float(np.max(pitches)),
        "red_boundary_margin_mean": float(np.mean(margins)), "red_boundary_margin_min": float(np.min(margins)),
        "red_horizontal_radius_mean": float(np.mean(radii)), "red_centroid_radius": float(math.hypot(centroid[0],centroid[1])),
        "red_pairwise_distance_mean": pair_mean, "red_pairwise_distance_max": pair_max, "red_pairwise_distance_min": pair_min,
        "red_formation_spread": spread, "red_heading_dispersion": circular_heading_dispersion([row.psi for row in living]),
        "red_vertical_speed_mean": float(np.mean(vertical)), "red_vertical_speed_min": float(np.min(vertical)),
        "red_current_time_to_ground_min": min((x for x in ttg if x is not None), default=None),
        "spawn_radial_angle": None,
        "spawn_candidate_index": env.last_spawn_candidate_index,
        "minimum_spawn_distance": env.last_minimum_spawn_distance,
        "blue_altitude_mean": float(np.mean([b.altitude for b in env.blue])),
        "blue_altitude_min": float(np.min([b.altitude for b in env.blue])),
        "blue_altitude_max": float(np.max([b.altitude for b in env.blue])),
        "minimum_red_blue_distance": min(x[0] for x in rb),
        "mean_nearest_blue_distance": float(np.mean(nearest)),
        "minimum_vertical_separation": min(x[1] for x in rb),
        "mean_vertical_separation": float(np.mean([x[1] for x in rb])),
        "minimum_red_to_blue_ATA": min(red_ata), "minimum_blue_to_red_ATA": min(blue_ata),
    }
    # The exact angle is returned by step info, and is patched by the runner.
    agents = []
    for index, red in enumerate(env.red):
        margin, _, _ = boundary_diagnostics(red, arena)
        vs = float(red.v * math.sin(red.theta))
        nearest_blue = _nearest_blue(red, env.blue)
        row = {
            "source_policy_seed": source_policy_seed, "evaluation_seed": evaluation_seed,
            "cleared_wave": cleared_wave, "next_wave": cleared_wave+1,
            "red_agent_id": index, "alive": bool(red.alive), "x": red.x, "y": red.y,
            "altitude": red.altitude, "speed": red.v, "heading": red.psi, "pitch": red.theta,
            "last_executed_phi": float(env.red_last_executed_phi[index]),
            "boundary_margin": margin, "horizontal_radius": float(math.hypot(red.x,red.y)),
            "vertical_speed": vs, "current_time_to_ground": time_to_ground(red.altitude,vs),
            "fire_armed": bool(env.red_fire_states[index].armed),
        }
        if nearest_blue is None:
            row.update({k:None for k in ("nearest_blue_id","nearest_blue_distance","nearest_blue_altitude_delta","nearest_blue_AA","nearest_blue_ATA","nearest_blue_HA")})
        else:
            blue_id, blue, geom, _ = nearest_blue
            row.update({"nearest_blue_id":blue_id,"nearest_blue_distance":geom.distance,
                        "nearest_blue_altitude_delta":blue.altitude-red.altitude,
                        "nearest_blue_AA":geom.aa,"nearest_blue_ATA":geom.ata,"nearest_blue_HA":geom.ha})
        agents.append(row)
    return summary, agents


def step_trace_row(env, agent_id: int, action: np.ndarray, wave: int) -> dict[str, Any]:
    from env.control import action_to_target

    state = env.red[agent_id]
    target = action_to_target(state, action, env.config["action"]["command"])
    risk, ttg_command = diagnostic_ground_risk(state, target.pitch, env.config)
    margin, radial_speed, ttb = boundary_diagnostics(state, env.arena_radius)
    nearest = _nearest_blue(state, env.blue)
    row = {
        "wave": wave, "altitude": state.altitude, "speed": state.v,
        "heading": state.psi, "pitch": state.theta,
        "vertical_speed": float(state.v*math.sin(state.theta)), "boundary_margin": margin,
        "radial_velocity": radial_speed, "time_to_boundary": ttb,
        "action_heading": float(action[0]), "action_pitch": float(action[1]), "action_speed": float(action[2]),
        "commanded_pitch": target.pitch, "would_trigger_blue_ground_guard": risk,
        "time_to_ground": ttg_command, "alive_before_step": bool(state.alive),
    }
    if nearest is None:
        row.update({k:None for k in ("nearest_blue_distance","nearest_blue_altitude_delta","nearest_blue_AA","nearest_blue_ATA","nearest_blue_HA")})
    else:
        _, blue, geom, _ = nearest
        row.update({"nearest_blue_distance":geom.distance,"nearest_blue_altitude_delta":blue.altitude-state.altitude,
                    "nearest_blue_AA":geom.aa,"nearest_blue_ATA":geom.ata,"nearest_blue_HA":geom.ha})
    return row


def new_ring_buffers() -> list[deque]:
    return [deque(maxlen=50) for _ in range(4)]


def dump_death_trace(buffer: deque, policy_seed: int, evaluation_seed: int,
                     wave: int, agent_id: int, death_type: str, step: int,
                     post_state: Any | None = None, arena_radius: float | None = None) -> dict[str, Any]:
    rows = list(buffer)
    trace = []
    for index, row in enumerate(reversed(rows)):
        trace.append({"t_minus": index, **row})
    outside = bool(post_state is not None and arena_radius is not None and
                   math.hypot(post_state.x, post_state.y) > arena_radius)
    below = bool(post_state is not None and post_state.altitude <= 0.0)
    return {"policy_seed":policy_seed,"evaluation_seed":evaluation_seed,"wave":wave,
            "agent_id":agent_id,"death_type":death_type,"death_global_step":step,
            "post_outside_boundary":outside,"post_below_ground":below,
            "simultaneous_boundary_ground":bool(outside and below),"trace":trace}


def canonical_spawn_seed(evaluation_seed: int, next_wave: int) -> int:
    """Stable diagnostic-only seed for spawn realization, independent of policy."""
    if evaluation_seed not in EVALUATION_SEEDS or next_wave not in (2, 3):
        raise ValueError("canonical spawn requires a 44M case and next wave 2 or 3")
    return int(1_000_000_000 + (evaluation_seed - 44_000_000) * 10 + next_wave)


def canonicalize_spawn(env, evaluation_seed: int, next_wave: int):
    """Regenerate only Blue using an independent, policy-invariant spawn stream."""
    if int(env.wave_index) != int(next_wave):
        raise ValueError("environment must be a post-spawn entry state")
    seed = canonical_spawn_seed(int(evaluation_seed), int(next_wave))
    red_before = [(s.x, s.y, s.z, s.v, s.theta, s.psi, s.alive) for s in env.red]
    counters_before = (env.steps, env.wave_index, env.waves_cleared, env.max_steps)
    env.rng = np.random.default_rng(seed)
    angle = env._spawn_next_wave()
    red_after = [(s.x, s.y, s.z, s.v, s.theta, s.psi, s.alive) for s in env.red]
    counters_after = (env.steps, env.wave_index, env.waves_cleared, env.max_steps)
    armed = all(state.armed for state in env.red_fire_states + env.blue_fire_states)
    if red_before != red_after or counters_before != counters_after or not armed:
        raise RuntimeError("canonical spawn changed source Red or episode counters")
    return int(seed), float(angle)


def stable_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
