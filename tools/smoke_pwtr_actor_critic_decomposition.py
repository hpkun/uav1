"""CUDA-only tiny smoke for the four PWTR actor/critic decomposition modes."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_pwtr_branch
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.train_modular_mappo import load_config

SOURCE = ROOT / "outputs/diag_mappo_learnability/l3_seed5302/checkpoint_1505280.pt"
NAMES = ("current_actor_only", "current_critic_only", "recent_actor_only", "recent_critic_only")


def digest(value) -> str:
    result = hashlib.sha256()
    def add(item):
        if torch.is_tensor(item): result.update(item.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(item, np.ndarray): result.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str): result.update(str(key).encode()); add(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item: add(child)
        else: result.update(repr(item).encode())
    add(value); return result.hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def natural_batch(trainer, wave: int):
    length = 128
    obs = np.random.default_rng(wave).normal(size=(length, 1, 4, 52)).astype("f")
    with torch.no_grad():
        tensor = torch.as_tensor(obs[:, 0], device="cuda"); alive = torch.ones((length, 4), device="cuda")
        distribution, _ = trainer.actor.distribution_step(tensor, None, None, None, alive)
        raw = distribution.sample(); log_prob = trainer.actor._squashed_log_prob(distribution, raw, torch.tanh(raw))
    return type("Batch", (), {"observations": obs, "next_observations": obs + .01,
        "raw_actions": raw.cpu().numpy()[:, None], "old_log_probs": log_prob.cpu().numpy()[:, None],
        "rewards": np.full((length, 1, 4), .1, "f"), "dones": np.zeros((length, 1), "f"),
        "alive_masks": np.ones((length, 1, 4), "f"), "next_alive_masks": np.ones((length, 1, 4), "f"),
        "wave_indices": np.full((length, 1), wave, dtype=np.int64),
        "wave_transition_flags": np.zeros((length, 1), "f")})()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="outputs/smoke_pwtr_actor_critic_decomposition")
    args = parser.parse_args(); output = ROOT / args.output_dir
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is mandatory")
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    source_sha = file_sha(SOURCE); state = torch.load(SOURCE, map_location="cpu", weights_only=False)
    environment = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    configs = {name: load_config(ROOT / f"configs/dev_pwtr_{name}_300k.yaml") for name in NAMES}
    provenances = {name: validate_pwtr_branch(state, environment, config,
        {"training_seed": 5302, "training_num_envs": 24, "training_smoke": False}) for name, config in configs.items()}

    # Real vector environment and current actor: all four modes must collect the same first rollout.
    rollouts = {}
    for name in NAMES:
        runner = ModularMAPPOTrainingRunner(environment, configs[name], 2, 1_805_280, "cuda", 5302,
            output / f"rollout_{name}", False, resume_mode=True,
            branch_provenance={**provenances[name], "parent_checkpoint_sha256": source_sha})
        runner.branch_from(SOURCE, provenances[name]["intervention"], source_sha)
        rollouts[name] = runner.collect_rollout(2)
        runner.vector.close()
    fields = ("observations", "actions", "raw_actions", "old_log_probs", "raw_environment_rewards",
              "alive_masks", "next_alive_masks", "wave_indices", "dones")
    reference = rollouts[NAMES[0]]
    rollout_match = {field: all(np.array_equal(getattr(reference, field), getattr(rollouts[name], field)) for name in NAMES[1:]) for field in fields}
    if not all(rollout_match.values()): raise RuntimeError(f"first rollout mismatch: {rollout_match}")

    # Before any replay is eligible, the fresh MAPPO update must be bitwise identical.
    fresh = {}
    for name in NAMES:
        trainer = build_modular_mappo_trainer(configs[name], "cuda", 256, 1_805_280)
        trainer.load(SOURCE, strict_protocol=False, restore_rng=True)
        metrics = trainer.update(deepcopy(rollouts[name]))
        fresh[name] = {"actor": digest(trainer.actor.state_dict()), "critic": digest(trainer.critic.state_dict()),
            "actor_optimizer": digest(trainer.actor_optimizer.state_dict()), "critic_optimizer": digest(trainer.critic_optimizer.state_dict()),
            "actor_steps": metrics["pwtr_fresh_actor_optimizer_steps"], "critic_steps": metrics["pwtr_fresh_critic_optimizer_steps"],
            "replay_batches": metrics["pwtr_replay_batches"]}
    fresh_match = all(fresh[name] == fresh[NAMES[0]] for name in NAMES[1:])
    if not fresh_match or fresh[NAMES[0]]["replay_batches"] != 0: raise RuntimeError(f"fresh update mismatch: {fresh}")

    # Exercise real replay backward/optimizer paths with natural-format W2/W3 segments.
    replay = {}; eligibility = {}
    for name in NAMES:
        trainer = build_modular_mappo_trainer(configs[name], "cuda", 256, 1_805_280)
        trainer.load(SOURCE, strict_protocol=False, restore_rng=True)
        module = trainer.persistent_wave_trajectory_replay
        generation = 1 if name.startswith("current") else 0
        module.ingest_rollout(natural_batch(trainer, 2), generation)
        module.ingest_rollout(natural_batch(trainer, 3), generation)
        module.current_rollout_generation = 1; module.current_later_states = 1024; module.fresh_rollout_count = 2
        eligible = module.eligible(1)
        eligibility[name] = {key: len(value) for key, value in eligible.items()}
        if not eligible: raise RuntimeError(f"{name}: expected eligible replay segments")
        actor_before, critic_before = digest(trainer.actor.state_dict()), digest(trainer.critic.state_dict())
        reward_before = digest([segment["rewards"] for rows in module.partitions.values() for segment in rows])
        metrics = trainer._pwtr_replay_phase()
        row = {"budget": metrics["pwtr_replay_batches_budget"], "batches": metrics["pwtr_replay_batches"],
            "actor_steps": metrics["pwtr_replay_actor_optimizer_steps_this_phase"],
            "critic_steps": metrics["pwtr_replay_critic_optimizer_steps_this_phase"],
            "actor_changed": actor_before != digest(trainer.actor.state_dict()),
            "critic_changed": critic_before != digest(trainer.critic.state_dict()),
            "reward_unchanged": reward_before == digest([segment["rewards"] for rows in module.partitions.values() for segment in rows]),
            "finite": all(np.isfinite(value) for value in metrics.values() if isinstance(value, (int, float)))}
        actor_only = name.endswith("actor_only")
        expected = ((row["actor_steps"] > 0 and row["critic_steps"] == 0 and row["actor_changed"] and not row["critic_changed"])
                    if actor_only else (row["actor_steps"] == 0 and row["critic_steps"] > 0 and not row["actor_changed"] and row["critic_changed"]))
        if not expected or not row["reward_unchanged"] or not row["finite"]: raise RuntimeError(f"{name}: decomposition failure {row}")
        replay[name] = row
    budgets = {(row["budget"], row["batches"]) for row in replay.values()}
    if budgets != {(2.0, 2.0)}: raise RuntimeError(f"budget mismatch: {budgets}")
    report = {"status": "PWTR_ACTOR_CRITIC_DECOMPOSITION_CUDA_SMOKE_PASS", "cuda": torch.cuda.get_device_name(0),
        "first_rollout_bitwise": rollout_match, "fresh_mappo_update_bitwise_before_replay": fresh_match,
        "fresh_update": fresh, "replay": replay, "eligibility": eligibility,
        "matched_replay_budget": True, "environment_reward_unchanged": True,
        "all_metrics_finite": True, "source_checkpoint_unchanged": source_sha == file_sha(SOURCE),
        "formal_training_started": False, "formal_evaluation_started": False, "45m_used": False}
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
