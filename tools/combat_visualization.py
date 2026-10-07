"""Read-only helpers for recording and rendering one combat episode."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np

from env.models import AircraftState
from env.persistent_env import PersistentWaveCombatEnv

TRACE_SCHEMA_VERSION = 1
FEATURE_NAMES = ["x", "y", "z", "v", "theta", "psi"]
RESERVED_SEED_RANGES = ((29_000_000, 29_000_019), (30_000_000, 30_000_199))


def infer_method_display_name(metadata: dict[str, Any]) -> str:
    explicit = metadata.get("method_display_name")
    if explicit:
        return str(explicit)
    modules = set(metadata.get("enabled_modules", []))
    ea = "entity_attention" in modules
    wb = "wave_balancing" in modules
    return "EA-WB-MAPPO" if ea and wb else "EA-MAPPO" if ea else "WB-MAPPO" if wb else "MAPPO"


def state_array(state: AircraftState) -> np.ndarray:
    return np.asarray(state.as_array(), dtype=np.float32)


def states_array(states: list[AircraftState], count: int = 4) -> np.ndarray:
    result = np.full((count, 6), np.nan, dtype=np.float32)
    for index, state in enumerate(states[:count]):
        result[index] = state_array(state)
    return result


class RecordingPersistentWaveCombatEnv(PersistentWaveCombatEnv):
    """Environment-semantics-neutral observer for pre-respawn Blue states."""

    def reset(self, seed: int | None = None):
        self.visualization_spawn_snapshot = None
        return super().reset(seed)

    def _spawn_next_wave(self) -> float:
        self.visualization_spawn_snapshot = {
            "old_wave_index": int(self.wave_index),
            "step": int(self.steps),
            "old_blue": [state.copy() for state in self.blue],
            "red": [state.copy() for state in self.red],
        }
        return super()._spawn_next_wave()


def assert_episode_seed_allowed(seed: int) -> None:
    value = int(seed)
    if any(start <= value <= end for start, end in RESERVED_SEED_RANGES):
        raise ValueError(f"episode seed {value} is reserved for formal protocol")


def ensure_fresh_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is non-empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def checkpoint_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_frame(frames: dict[str, list[Any]], env: RecordingPersistentWaveCombatEnv,
                 active_wave: int, waves_cleared: int, time_s: float,
                 info: dict[str, Any] | None = None) -> None:
    red = states_array(env.red)
    blue = np.full((env.total_waves, 4, 6), np.nan, dtype=np.float32)
    blue_alive = np.zeros((env.total_waves, 4), dtype=bool)
    wave = max(1, min(env.total_waves, int(active_wave)))
    blue[wave - 1] = states_array(env.blue)
    blue_alive[wave - 1] = [state.alive for state in env.blue]
    snapshot = getattr(env, "visualization_spawn_snapshot", None)
    if snapshot is not None and int(snapshot["step"]) == int(env.steps):
        old = int(snapshot["old_wave_index"]) - 1
        blue[old] = states_array(snapshot["old_blue"])
        blue_alive[old] = [state.alive for state in snapshot["old_blue"]]
    frames["red_kinematics"].append(red)
    frames["red_alive"].append(np.asarray([s.alive for s in env.red], dtype=bool))
    frames["blue_kinematics"].append(blue)
    frames["blue_alive"].append(blue_alive)
    frames["steps"].append(int(env.steps))
    frames["time_s"].append(float(time_s))
    frames["active_wave"].append(int(wave))
    frames["waves_cleared"].append(int(waves_cleared))


def frame_arrays(frames: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in frames.items()}


def write_trace(path: Path, frames: dict[str, list[Any]], transitions: dict[str, list[Any]]) -> dict[str, tuple[int, ...]]:
    arrays = frame_arrays(frames)
    arrays.update({key: np.asarray(value) for key, value in transitions.items()})
    np.savez_compressed(path, **arrays)
    return {key: tuple(value.shape) for key, value in arrays.items()}


def read_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        result = {key: source[key].copy() for key in source.files}
    required = {"red_kinematics", "red_alive", "blue_kinematics", "blue_alive", "steps", "time_s", "active_wave", "waves_cleared"}
    missing = required - result.keys()
    if missing or result["red_kinematics"].ndim != 3 or result["blue_kinematics"].ndim != 4:
        raise ValueError(f"invalid trace schema; missing={sorted(missing)}")
    if result["red_kinematics"].shape[0] != result["steps"].shape[0]:
        raise ValueError("trace frame count mismatch")
    return result


def extract_death_frames(trace: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    result = []
    red_alive = trace["red_alive"]
    for agent in range(red_alive.shape[1]):
        for frame in range(1, red_alive.shape[0]):
            if red_alive[frame - 1, agent] and not red_alive[frame, agent]:
                result.append({"side":"red", "agent":agent + 1, "wave":None, "frame":frame, "position":trace["red_kinematics"][frame, agent, [0,1,2]].tolist(), "cause":_death_cause(trace, "red", frame, agent)})
                break
    blue_alive = trace["blue_alive"]
    for wave in range(blue_alive.shape[1]):
        for agent in range(blue_alive.shape[2]):
            for frame in range(1, blue_alive.shape[0]):
                if blue_alive[frame - 1, wave, agent] and not blue_alive[frame, wave, agent]:
                    result.append({"side":"blue", "agent":agent + 1, "wave":wave + 1, "frame":frame, "position":trace["blue_kinematics"][frame, wave, agent, [0,1,2]].tolist(), "cause":_death_cause(trace, "blue", frame, agent, wave)})
                    break
    return result


def attack_totals(trace: dict[str, np.ndarray]) -> dict[str, int]:
    """Return schema-v1 aggregate fire statistics without per-attempt inference."""
    def total(side: str, event: str) -> int:
        return int(np.asarray(trace.get(
            f"{side}_step_{event}", np.zeros(0, dtype=np.int64)
        )).sum())
    attempts = total("red", "fire_attempts") + total("blue", "fire_attempts")
    hits = total("red", "weapon_hits") + total("blue", "weapon_hits")
    kills = total("red", "attack_kills") + total("blue", "attack_kills")
    return {
        "attempts": attempts,
        "hits": hits,
        "misses": attempts - hits,
        "kills": kills,
    }


def annotate_death_attack_sources(
    deaths: list[dict[str, Any]], attack_links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach conservative fire-attempt evidence, never schema-v1 kill credit."""
    result = []
    for original in deaths:
        death = dict(original)
        sources: list[str] = []
        if death.get("cause") == "attack_kill":
            for link in attack_links:
                if int(link.get("frame", -1)) != int(death["frame"]):
                    continue
                if death["side"] == "blue":
                    matches = (
                        link.get("side") == "red"
                        and int(link.get("target", -1)) == int(death["agent"])
                        and int(link.get("wave", -1)) == int(death["wave"])
                    )
                    label = f"R{int(link['attacker'])}" if matches else ""
                else:
                    matches = (
                        link.get("side") == "blue"
                        and int(link.get("target", -1)) == int(death["agent"])
                    )
                    label = (
                        f"Wave {int(link['wave'])} B{int(link['attacker'])}"
                        if matches else ""
                    )
                if matches and label not in sources:
                    sources.append(label)
        death["attack_sources"] = sources
        if len(sources) == 1:
            death["attack_source_evidence"] = (
                f"Single reconstructed attacker: {sources[0]}"
            )
        elif len(sources) > 1:
            death["attack_source_evidence"] = (
                f"Multiple reconstructed attackers: {', '.join(sources)}; "
                "unique killer unavailable"
            )
        else:
            death["attack_source_evidence"] = "Unknown"
        result.append(death)
    return result


def _death_cause(
    trace: dict[str, np.ndarray], side: str, frame: int, agent: int,
    wave: int | None = None,
) -> str:
    """Classify a recorded death without inventing attacker identity."""
    transition = frame - 1
    states = trace[f"{side}_kinematics"]
    state = states[frame, agent] if side == "red" else states[frame, wave, agent]
    finite_position = bool(np.all(np.isfinite(state[:3])))
    if finite_position:
        radius = float(trace.get("arena_radius", np.asarray(np.inf)))
        if float(np.hypot(state[0], state[1])) > radius:
            return "boundary_exit"
        if float(-state[2]) <= 0.0:
            return "ground_impact"
    def count(key: str) -> int:
        values = trace.get(key)
        return int(values[transition]) if values is not None and transition < len(values) else 0
    if not finite_position and count(f"{side}_boundary_exit_delta") > 0:
        return "boundary_exit"
    if not finite_position and count(f"{side}_ground_loss_delta") > 0:
        return "ground_impact"
    attacker = "blue" if side == "red" else "red"
    if count(f"{attacker}_step_attack_kills") > 0:
        return "attack_kill"
    return "unknown"


def extract_events(trace: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    events = []
    mappings = (("red_step_attack_kills", "red_kill"), ("blue_step_attack_kills", "blue_kill"), ("red_step_weapon_hits", "red_hit"), ("blue_step_weapon_hits", "blue_hit"), ("red_ground_loss_delta", "red_ground_loss"), ("blue_ground_loss_delta", "blue_ground_loss"), ("red_boundary_exit_delta", "red_boundary_exit"), ("blue_boundary_exit_delta", "blue_boundary_exit"))
    for key, kind in mappings:
        for index, count in enumerate(trace.get(key, np.zeros(max(0, len(trace["steps"]) - 1), dtype=int))):
            if int(count) > 0:
                events.append({"trace_frame":index + 1, "type":kind, "count":int(count)})
    clears = trace.get("wave_cleared_this_step", np.zeros(max(0, len(trace["steps"]) - 1), dtype=bool))
    spawns = trace.get("spawned_next_wave", np.zeros(max(0, len(trace["steps"]) - 1), dtype=bool))
    for index, flag in enumerate(clears):
        if flag:
            events.append({"trace_frame":index + 1, "type":"wave_cleared", "wave":int(trace["active_wave"][index])})
    for index, flag in enumerate(spawns):
        if flag:
            events.append({"trace_frame":index + 1, "type":"wave_spawned", "wave":int(trace["active_wave"][index + 1])})
    return sorted(events, key=lambda event: (event["trace_frame"], event["type"]))


def blue_losses_at_frame(deaths: list[dict[str, Any]], frame: int) -> int:
    """Count Blue deaths visible at a trace frame (no future leakage)."""
    return sum(1 for death in deaths if death.get("side") == "blue" and int(death.get("frame", 0)) <= int(frame))


def events_up_to_frame(events: list[dict[str, Any]], frame: int) -> list[dict[str, Any]]:
    return [event for event in events if int(event.get("trace_frame", 0)) <= int(frame)]


def recent_events(events: list[dict[str, Any]], frame: int, limit: int = 5) -> list[dict[str, Any]]:
    return events_up_to_frame(events, frame)[-max(0, int(limit)):]


def trace_frame_to_render_index(rendered_frames: list[int], trace_frame: int) -> int:
    """Return first rendered index whose trace frame is at or after the event."""
    for index, frame in enumerate(rendered_frames):
        if int(frame) >= int(trace_frame):
            return index
    return max(0, len(rendered_frames) - 1)


def trajectory_slice(data: np.ndarray, frame: int, trail_length: int = 0) -> np.ndarray:
    start = max(0, int(frame) - int(trail_length)) if trail_length else 0
    return data[start:int(frame) + 1]


def heading_endpoint(state: np.ndarray, length: float = 250.0) -> np.ndarray:
    x, y, z, _v, theta, psi = state
    # Aircraft state uses NED z: positive pitch is a climb, so z decreases.
    return np.asarray([x + length * np.cos(theta) * np.cos(psi), y + length * np.cos(theta) * np.sin(psi), z - length * np.sin(theta)])


def dump_metadata(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


__all__ = ["TRACE_SCHEMA_VERSION", "FEATURE_NAMES", "RecordingPersistentWaveCombatEnv", "assert_episode_seed_allowed", "ensure_fresh_output", "state_array", "states_array", "append_frame", "write_trace", "read_trace", "checkpoint_sha256", "dump_metadata", "infer_method_display_name", "extract_events", "extract_death_frames", "attack_totals", "annotate_death_attack_sources", "blue_losses_at_frame", "events_up_to_frame", "recent_events", "trace_frame_to_render_index", "trajectory_slice", "heading_endpoint"]
