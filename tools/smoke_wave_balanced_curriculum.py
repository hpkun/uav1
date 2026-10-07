"""CUDA-only tiny integration smoke for Wave-Entry Curriculum MAPPO."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.modular_mappo.evaluation import evaluate_modular_episode
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.train_modular_mappo import load_config
from env.config import load_config as load_env_config
from env.persistent_env import PersistentWaveCombatEnv

SMOKE_SEED = 88_000_001


def entry_snapshot(env_config: dict, wave: int, seed: int) -> dict:
    env = PersistentWaveCombatEnv(env_config); env.reset(seed)
    while env.wave_index < wave:
        for aircraft in env.blue: aircraft.alive = False
        _, _, terminated, truncated, info = env.step(np.zeros((4, 3), np.float32))
        if terminated or truncated or not info.get("spawned_next_wave"):
            raise RuntimeError("failed to construct a legal smoke entry snapshot")
    return env.export_curriculum_state()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for Wave-Entry Curriculum smoke")
    output = ROOT / "outputs/dev_wave_balanced_curriculum_combined_tiny_smoke_mixed"
    if output.exists():
        raise FileExistsError(f"smoke output already exists: {output}")
    env_config = load_env_config(ROOT / "configs/persistent_wave_v2_environment.yaml")
    algorithm = load_config(ROOT / "configs/dev_wave_balanced_curriculum_3m.yaml")
    runner = ModularMAPPOTrainingRunner(
        env_config, algorithm, num_envs=3, total_sampled_steps=12,
        device="cuda", seed=SMOKE_SEED, output_dir=output, smoke=True,
    )
    clone = None
    try:
        module = runner.trainer.wave_entry_curriculum
        enabled_modules = runner.trainer.module_protocol()["enabled_modules"]
        expected_modules = ["actor_lr_decay", "wave_balancing", "wave_entry_curriculum"]
        if enabled_modules != expected_modules:
            raise RuntimeError(
                f"combined smoke module mismatch: expected {expected_modules}, got {enabled_modules}"
            )
        snapshots = {wave: entry_snapshot(env_config, wave, SMOKE_SEED + wave) for wave in (2, 3)}
        for wave in (2, 3):
            for index in range(module.min_bank_entries):
                module.add_natural_entry(wave, snapshots[wave], source_sampled_steps=0,
                    source_training_seed=SMOKE_SEED, source_env_id=0,
                    source_episode_reset_seed=SMOKE_SEED + wave, red_survivors=4)
        for _ in range(module.min_natural_episodes): module.record_natural_episode(0)
        restored = {}
        for _ in range(32):
            runner.episode_start_wave[0] = 1
            restored = runner._apply_wave_entry_curriculum_resets(
                np.asarray([True, False, False]),
                [{"waves_cleared": 0}, {}, {}],
            )
            if restored: break
        if not restored:
            raise RuntimeError("tiny smoke did not exercise a curriculum restore")
        metadata = restored[0]
        runner.observations = runner.vector.current_observations.copy()
        runner.alive = runner.vector.current_alive_masks.copy()
        runner.blue_alive[0] = metadata["blue_alive_mask"]
        runner.wave[0] = metadata["wave_index"]
        runner.total[0] = metadata["total_waves"]
        runner.episode_steps[0] = metadata["steps"]
        runner.episode_mask[0] = 0.0
        rollout = runner.collect_rollout(4)
        if not np.all(np.isfinite(rollout.actions)) or not np.all(np.isfinite(rollout.old_log_probs)):
            raise FloatingPointError("current actor did not produce finite on-policy rollout data")
        if rollout.wave_indices.shape != (4, 3) or rollout.alive_masks.shape != (4, 3, 4):
            raise RuntimeError("combined smoke rollout lacks real wave/alive tensors")
        metrics = runner.trainer.update(rollout)
        required = ("actor_loss", "value_loss", "entropy", "approx_kl")
        if not all(math.isfinite(float(metrics[key])) for key in required):
            raise FloatingPointError("non-finite PPO metric in WEC smoke")
        weight_metrics = {key: float(metrics[key]) for key in (
            "weight_wave_1", "weight_wave_2", "weight_wave_3",
            "effective_wave_weight_mean",
        )}
        for wave in (1, 2, 3):
            count = float(metrics[f"alive_agent_samples_wave_{wave}"])
            weight = weight_metrics[f"weight_wave_{wave}"]
            if count > 0 and (not math.isfinite(weight) or not 0.0 < weight <= 3.0):
                raise FloatingPointError(f"invalid weight for present wave {wave}: {weight}")
        if not math.isclose(
            weight_metrics["effective_wave_weight_mean"], 1.0,
            rel_tol=1e-6, abs_tol=1e-6,
        ):
            raise RuntimeError("effective wave weight mean is not one")
        checkpoint = output / "smoke_checkpoint.pt"
        runner.save_checkpoint(checkpoint)
        clone = ModularMAPPOTrainingRunner(
            env_config, algorithm, num_envs=3, total_sampled_steps=24,
            device="cuda", seed=SMOKE_SEED, output_dir=output / "resume_clone",
            smoke=True, resume_mode=True,
        )
        clone.resume(checkpoint)
        restored_sizes = {wave: len(clone.trainer.wave_entry_curriculum.bank[wave]) for wave in (2, 3)}
        evaluation_env = dict(env_config)
        evaluation_env = json.loads(json.dumps(evaluation_env))
        evaluation_env["simulation"]["max_steps"] = 4
        evaluation = evaluate_modular_episode(
            runner.trainer, evaluation_env, SMOKE_SEED + 98, include_trace=True
        )
        if int(evaluation["wave_trace"][0]) != 1:
            raise RuntimeError("combined smoke evaluation did not start from W1")
        if restored_sizes[2] <= 0 or restored_sizes[3] <= 0:
            raise RuntimeError("checkpoint/resume lost a wave-entry bank")
        wec = module.diagnostics()
        rollout_metrics = runner.last_rollout_metrics
        report = {"status": "PASS", "device": torch.cuda.get_device_name(0),
            "seed": SMOKE_SEED, "bank_sizes": {wave: len(module.bank[wave]) for wave in (2,3)},
            "enabled_modules": enabled_modules,
            **{f"transition_fraction_wave_{wave}": float(rollout_metrics[f"transition_fraction_wave_{wave}"]) for wave in (1,2,3)},
            **{f"alive_agent_fraction_wave_{wave}": float(rollout_metrics[f"alive_agent_fraction_wave_{wave}"]) for wave in (1,2,3)},
            **weight_metrics,
            **{key: float(wec[key]) for key in (
                "wec_probability_w1", "wec_probability_w2", "wec_probability_w3",
                "wec_bank_size_w2", "wec_bank_size_w3",
            )},
            "curriculum_restore_wave": int(metadata["wave_index"]),
            "restored_global_steps": int(metadata["steps"]),
            "ppo_metrics": {key: float(metrics[key]) for key in required},
            "checkpoint": str(checkpoint), "resume_bank_sizes": restored_sizes,
            "evaluation_first_wave": int(evaluation["wave_trace"][0]),
            "evaluation_curriculum_used": False, "formal_evaluation": False,
            "future_final_45m_used": False}
        (output / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        runner.vector.close()
        if clone is not None: clone.vector.close()


if __name__ == "__main__":
    main()
