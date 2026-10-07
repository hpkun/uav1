"""Record one deterministic, qualitative persistent-wave combat episode."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_modular_checkpoint
from algorithm.modular_mappo.trainer import MODULAR_MAPPO_IMPL_VERSION
from algorithm.modules.wave_survival_pbrs import mission_context_numpy
from algorithm.train_modular_mappo import load_config
from tools.combat_visualization import (FEATURE_NAMES, TRACE_SCHEMA_VERSION,
    RecordingPersistentWaveCombatEnv, append_frame, assert_episode_seed_allowed,
    checkpoint_sha256, dump_metadata, ensure_fresh_output, write_trace, extract_events)


def resolved(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--episode-seed", required=True, type=int)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wall-timeout-s", type=float, default=0.0)
    args = parser.parse_args()
    if args.device != "cuda":
        raise RuntimeError("CUDA is required for deterministic recording")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for deterministic recording")
    assert_episode_seed_allowed(args.episode_seed)
    checkpoint = resolved(args.checkpoint)
    env_path = resolved(args.env_config)
    output_dir = resolved(args.output_dir)
    if "<" in str(args.checkpoint) or ">" in str(args.checkpoint):
        raise ValueError("--checkpoint still contains a placeholder; replace <chosen_run> with a real run directory")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}; replace <chosen_run> with an existing run directory")
    if not env_path.is_file():
        raise FileNotFoundError(f"environment config not found: {env_path}")
    ensure_fresh_output(output_dir)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    extra = state.get("extra", {})
    algorithm_config = extra.get("algorithm_config")
    if not isinstance(algorithm_config, dict):
        raise RuntimeError("checkpoint lacks self-describing algorithm_config")
    env_config = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    validate_modular_checkpoint(state, env_config, algorithm_config)
    architecture = extra.get("network_architecture", {})
    hidden_dim = int(architecture.get("hidden_dim", algorithm_config["network"]["actor_hidden_layers"][0]))
    trainer = build_modular_mappo_trainer(algorithm_config, "cuda", hidden_dim,
                                          int(extra.get("training_total_sampled_steps", algorithm_config["training"]["total_sampled_steps"])))
    trainer.load(checkpoint)
    env = RecordingPersistentWaveCombatEnv(env_config)
    observation, reset_info = env.reset(args.episode_seed)
    alive = env.red_alive_mask.copy()
    actor_hidden, critic_hidden = trainer.initial_hidden(1)
    actor_phase_reset_flags = np.ones(1, dtype=bool)
    episode_mask = np.zeros(1, dtype=np.float32)
    frames = {key: [] for key in ("red_kinematics", "red_alive", "blue_kinematics", "blue_alive", "steps", "time_s", "active_wave", "waves_cleared")}
    transitions = {key: [] for key in ("red_actions", "local_rewards", "team_reward", "terminated", "truncated", "wave_cleared_this_step", "spawned_next_wave", "red_step_fire_attempts", "blue_step_fire_attempts", "red_step_weapon_hits", "blue_step_weapon_hits", "red_step_attack_kills", "blue_step_attack_kills", "red_boundary_exit_delta", "blue_boundary_exit_delta", "red_ground_loss_delta", "blue_ground_loss_delta")}
    append_frame(frames, env, env.wave_index, env.waves_cleared, 0.0)
    previous_counts = {side: {event: 0 for event in ("boundary_exits", "ground_losses")} for side in ("red", "blue")}
    start = time.time()
    info = reset_info
    while True:
        wave = np.asarray([env.wave_index], dtype=np.int64)
        total = np.asarray([env.total_waves], dtype=np.int64)
        context = mission_context_numpy(
            trainer, wave, total, env.blue_alive_mask[None],
            np.asarray([env.steps]), env.max_steps,
        )
        context = trainer.actor_context_numpy(wave, alive[None], context)
        phase_kwargs = {"wave_indices":np.asarray([env.wave_index]),"actor_phase_reset_flags":actor_phase_reset_flags} if trainer.recurrent.wave_boundary_reset else {}
        actions, _, _, actor_hidden = trainer.act(observation[None], alive[None], deterministic=True, return_policy_data=True, context=context, hidden=actor_hidden, episode_mask=episode_mask, **phase_kwargs)
        _, critic_hidden = trainer.values_step(observation[None], alive[None], context, critic_hidden, episode_mask)
        next_observation, reward, terminated, truncated, info = env.step(actions[0])
        for key in transitions:
            if key == "red_actions": value = actions[0]
            elif key == "local_rewards": value = reward
            elif key == "team_reward": value = float(np.sum(reward))
            elif key == "terminated": value = bool(terminated)
            elif key == "truncated": value = bool(truncated)
            elif key in ("wave_cleared_this_step", "spawned_next_wave"): value = bool(info.get(key, False))
            elif key.endswith("_delta"):
                side = key.split("_")[0]
                event = "boundary_exits" if "boundary" in key else "ground_losses"
                current = int(info.get(f"{side}_{event}", 0)); value = current - previous_counts[side][event]
                previous_counts[side][event] = current
            else: value = int(info.get(key, 0))
            transitions[key].append(value)
        alive = np.asarray(info["red_alive_mask"], dtype=np.float32)
        actor_hidden = trainer.recurrent.apply_alive(actor_hidden, alive[None])
        critic_hidden = trainer.recurrent.apply_alive(critic_hidden, alive[None])
        episode_mask[:] = 1.0
        actor_phase_reset_flags[:] = bool(info.get("spawned_next_wave",False))
        append_frame(frames, env, env.wave_index, env.waves_cleared, env.steps * env.dt, info)
        observation = next_observation
        if terminated or truncated:
            break
        if args.wall_timeout_s > 0 and time.time() - start > args.wall_timeout_s:
            raise RuntimeError(f"recording exceeded configured wall timeout: {args.wall_timeout_s}s")
    arrays = write_trace(output_dir / "episode_trace.npz", frames, transitions)
    metadata = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "checkpoint_sampled_steps": int(state.get("sampled_steps", 0)), "checkpoint_training_seed": extra.get("training_seed"),
        "algorithm": state.get("algorithm"), "modular_mappo_impl_version": state.get("modular_mappo_impl_version"),
        "baseline_mappo_impl_version": state.get("baseline_mappo_impl_version"), "enabled_modules": state.get("enabled_modules", []),
        "module_config_sha256": state.get("module_config_sha256"), "algorithm_config_sha256": extra.get("algorithm_config_sha256"),
        "environment_variant": extra.get("environment_variant"), "environment_version": extra.get("environment_version"),
        "environment_config_sha256": extra.get("environment_config_sha256"),
        "episode_seed": args.episode_seed, "episode_role": "qualitative_visualization_only",
        "not_used_for_quantitative_metrics": True, "dt": env.dt, "max_steps": env.max_steps,
        "arena_radius": env.arena_radius, "total_waves": env.total_waves, "deterministic_policy": True,
        "recorded_frames": int(len(frames["steps"])), "episode_steps": int(env.steps), "simulation_time_s": float(env.steps * env.dt),
        "termination_reason": info.get("termination_reason"), "red_success": bool(info.get("red_success", False)),
        "waves_cleared": int(env.waves_cleared), "final_red_survivors": int(env.red_alive_mask.sum()),
        "final_blue_survivors": int(env.blue_alive_mask.sum()), "total_blue_losses": int(info.get("blue_losses", 0)),
        "episode_return": float(np.sum([sum(x) for x in transitions["local_rewards"]])), "reserved_formal_seed": False,
        "feature_names": FEATURE_NAMES, "trace_shapes": arrays,
    }
    # Keep event metadata optional and derive it from the exact transition arrays.
    metadata["events"] = extract_events({key: np.asarray(value) for key, value in {**frames, **transitions}.items()})
    dump_metadata(output_dir / "metadata.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
