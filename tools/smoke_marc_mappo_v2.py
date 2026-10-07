#!/usr/bin/env python3
"""Tiny CUDA smoke for MARC V2 collection, loss, update and checkpointing."""
from __future__ import annotations

import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.train_modular_mappo import load_config


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for MARC V2 smoke")
    config = deepcopy(load_config(ROOT / "configs/dev_marc_mappo_v2_1m.yaml"))
    config["network"]["actor_hidden_layers"] = [16, 16]
    config["network"]["critic_hidden_layers"] = [16, 16]
    config["training"].update({"ppo_epochs": 1, "minibatch_size": 8})
    module_config = config["modules"]["milestone_aware_retention_credit"]
    module_config.update({"retention_min_rows_per_wave": 1,
                          "retention_samples_per_wave": 1,
                          "retention_stride": 1})
    trainer = build_modular_mappo_trainer(config, "cuda", total_sampled_steps=64)

    # Exercise the same Runner collection/finalization methods used by training.
    runner = ModularMAPPOTrainingRunner.__new__(ModularMAPPOTrainingRunner)
    runner.trainer = trainer
    runner.num_envs = 1
    runner.episode_steps = np.asarray([0])
    runner.marc_wave_start_steps = np.asarray([0])
    runner.marc_pending_episode = [{1: [], 2: [], 3: []}]
    observations = np.arange(4 * 52, dtype=np.float32).reshape(1, 4, 52) / 100
    actions = np.asarray([[[.1, .2, .3], [.4, .5, .6], [.7, .8, .9], [0, 0, 0]]],
                         dtype=np.float32)
    alive = np.asarray([[1, 1, 1, 0]], dtype=np.float32)
    runner._marc_collect_candidates(observations, alive, np.asarray([1]), actions)
    completed = runner._marc_finalize_step(np.asarray([1]), np.asarray([False]), [{
        "spawned_next_wave": True, "red_success": False, "episode_length": 1,
        "red_alive_mask": [1, 1, 1, 0],
    }])
    if not np.array_equal(completed[0]["target_actions"], actions[0, :3]):
        raise RuntimeError("runner did not preserve executed-action alignment")

    rng = np.random.default_rng(4)
    time_steps, envs, agents = 2, 1, 4
    obs = rng.normal(size=(time_steps, envs, agents, 52)).astype(np.float32)
    raw = rng.normal(size=(time_steps, envs, agents, 3)).astype(np.float32)
    rollout_actions = np.tanh(raw)
    rollout_alive = np.ones((time_steps, envs, agents), np.float32)
    waves = np.asarray([[1], [2]], np.int64)
    with torch.no_grad():
        distribution = trainer.actor.distribution(torch.as_tensor(obs, device="cuda"))
        old_log_probs = trainer.actor._squashed_log_prob(
            distribution, torch.as_tensor(raw, device="cuda"),
            torch.as_tensor(rollout_actions, device="cuda")
        ).cpu().numpy()
    rewards = rng.normal(size=(time_steps, envs, agents)).astype(np.float32)
    rollout = ModularRolloutBatch(
        observations=obs, actions=rollout_actions, raw_actions=raw,
        old_log_probs=old_log_probs, rewards=rewards,
        raw_environment_rewards=rewards.copy(),
        dones=np.zeros((time_steps, envs), np.float32), alive_masks=rollout_alive,
        next_observations=obs.copy(), next_alive_masks=rollout_alive.copy(),
        wave_indices=waves, total_waves=np.full((time_steps, envs), 3),
        contexts=np.zeros((time_steps, envs, 0), np.float32),
        next_contexts=np.zeros((time_steps, envs, 0), np.float32),
        episode_masks=np.ones((time_steps, envs), np.float32),
        wave_transition_flags=np.asarray([[1], [0]], np.float32),
        marc_success_segments=completed,
    )
    metrics = trainer.update(rollout)
    required = ("actor_loss", "value_loss", "approx_kl", "entropy",
                "marc_v2_deployment_distill_loss", "marc_v2_action_mse_wave1")
    if any(not np.isfinite(metrics[key]) for key in required):
        raise FloatingPointError(f"non-finite MARC V2 smoke metrics: {metrics}")
    if trainer.milestone_aware_retention_credit.elite_row_count(1) != 3:
        raise RuntimeError("elite segment was not committed")

    with tempfile.TemporaryDirectory(prefix="marc_v2_smoke_") as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        trainer.save(checkpoint, {"training_seed": 5301})
        restored = build_modular_mappo_trainer(config, "cuda", total_sampled_steps=64)
        restored.load(checkpoint, strict_protocol=True, restore_rng=False)
        original = trainer.milestone_aware_retention_credit.state_dict()
        recovered = restored.milestone_aware_retention_credit.state_dict()
        if original["insertion_counter"] != recovered["insertion_counter"]:
            raise RuntimeError("MARC V2 checkpoint counter mismatch")
        if not np.array_equal(original["elite_segments"]["1"][0]["target_actions"],
                              recovered["elite_segments"]["1"][0]["target_actions"]):
            raise RuntimeError("MARC V2 checkpoint elite actions mismatch")
    report = {
        "status": "MARC_MAPPO_V2_CUDA_SMOKE_PASS",
        "cuda": torch.cuda.get_device_name(0),
        "runner_action_alignment": True,
        "elite_segments_wave1": len(trainer.milestone_aware_retention_credit.elite_segments[1]),
        "elite_rows_wave1": trainer.milestone_aware_retention_credit.elite_row_count(1),
        "loss_metrics_finite": True,
        "backward_update_finite": True,
        "checkpoint_roundtrip": True,
        "training_performed": False,
    }
    print(json.dumps(report, indent=2))
    print(report["status"])


if __name__ == "__main__":
    main()
