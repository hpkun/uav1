"""CUDA-only tiny PWTR integration smoke. Never starts a formal experiment."""
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
from algorithm.modules import replay_batch_budget, wave_stratified_permutation
from algorithm.train_modular_mappo import load_config

SOURCE = ROOT / "outputs/diag_mappo_learnability/l3_seed5302/checkpoint_1505280.pt"


def digest(value):
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v): h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, np.ndarray): h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            for key in sorted(v, key=str): h.update(str(key).encode()); add(v[key])
        elif isinstance(v, (list, tuple)):
            for item in v: add(item)
        else: h.update(repr(v).encode())
    add(value); return h.hexdigest()


def checkpoint_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="outputs/smoke_pwtr_mappo")
    args = parser.parse_args(); out = ROOT / args.output_dir
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is mandatory for PWTR smoke")
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    source_sha = checkpoint_hash(SOURCE); state = torch.load(SOURCE, map_location="cpu", weights_only=False)
    env = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    plain_cfg = load_config(ROOT / "configs/dev_pwtr_plain_300k.yaml")
    full_cfg = load_config(ROOT / "configs/dev_pwtr_full_300k.yaml")
    runtime = {"training_seed": 5302, "training_num_envs": 24, "training_smoke": False}
    plain_prov = validate_pwtr_branch(state, env, plain_cfg, runtime)
    full_prov = validate_pwtr_branch(state, env, full_cfg, runtime)

    # The real environment path: identical source state, RNG, reset and two vector steps.
    rollouts = {}
    for name, config, provenance in (("plain", plain_cfg, plain_prov), ("full", full_cfg, full_prov)):
        runner = ModularMAPPOTrainingRunner(env, config, 2, 1_805_280, "cuda", 5302, out / name, False,
                                            resume_mode=True, branch_provenance={**provenance, "parent_checkpoint_sha256": source_sha})
        runner.branch_from(SOURCE, provenance["intervention"], source_sha)
        rollouts[name] = runner.collect_rollout(2)
        runner.vector.close()
    fields = ("observations", "actions", "raw_actions", "old_log_probs", "raw_environment_rewards",
              "alive_masks", "next_alive_masks", "wave_indices", "dones")
    first_rollout = {field: np.array_equal(getattr(rollouts["plain"], field), getattr(rollouts["full"], field)) for field in fields}
    if not all(first_rollout.values()): raise RuntimeError(f"first rollout mismatch: {first_rollout}")

    # Disabled capability must remain exact Plain, including optimizer and RNG state.
    disabled_cfg = deepcopy(plain_cfg); disabled_cfg["modules"]["persistent_wave_trajectory_replay"] = {"enabled": False}
    control = build_modular_mappo_trainer(plain_cfg, "cuda", 256, 1_805_280)
    disabled = build_modular_mappo_trainer(disabled_cfg, "cuda", 256, 1_805_280)
    control.load(SOURCE, strict_protocol=False, restore_rng=True); disabled.load(SOURCE, strict_protocol=False, restore_rng=True)
    saved_cpu = torch.get_rng_state(); saved_cuda = torch.cuda.get_rng_state_all()
    mc = control.update(deepcopy(rollouts["plain"])); rc = control.capture_rng_state()
    torch.set_rng_state(saved_cpu); torch.cuda.set_rng_state_all(saved_cuda)
    md = disabled.update(deepcopy(rollouts["plain"])); rd = disabled.capture_rng_state()
    parity = (digest(control.actor.state_dict()) == digest(disabled.actor.state_dict()) and
              digest(control.critic.state_dict()) == digest(disabled.critic.state_dict()) and
              digest(control.actor_optimizer.state_dict()) == digest(disabled.actor_optimizer.state_dict()) and
              digest(control.critic_optimizer.state_dict()) == digest(disabled.critic_optimizer.state_dict()) and
              digest(rc) == digest(rd) and mc == md)
    if not parity: raise RuntimeError("PWTR-disabled update is not bitwise Plain-equivalent")

    # Pure conservation plus a real CUDA replay backward/step on natural-format segments.
    waves = np.asarray([1] * 13 + [2] * 11 + [3] * 7)
    order = wave_stratified_permutation(waves, 8, np.random.default_rng(5))
    conservation = sorted(order.tolist()) == list(range(len(waves)))
    def batch(trainer, wave):
        t = 128; obs = np.random.default_rng(wave).normal(size=(t, 1, 4, 52)).astype("f")
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs[:, 0], device="cuda")
            alive_tensor = torch.ones((t, 4), device="cuda")
            distribution, _ = trainer.actor.distribution_step(obs_tensor, None, None, None, alive_tensor)
            raw_tensor = distribution.sample()
            log_tensor = trainer.actor._squashed_log_prob(distribution, raw_tensor, torch.tanh(raw_tensor))
        raw = raw_tensor.cpu().numpy()[:, None]
        behavior_log_probs = log_tensor.cpu().numpy()[:, None]
        return type("Batch", (), {"observations": obs, "next_observations": obs + .01,
            "raw_actions": raw, "old_log_probs": behavior_log_probs,
            "rewards": np.full((t, 1, 4), .1, "f"), "dones": np.zeros((t, 1), "f"),
            "alive_masks": np.ones((t, 1, 4), "f"), "next_alive_masks": np.ones((t, 1, 4), "f"),
            "wave_indices": np.full((t, 1), wave, dtype=np.int64), "wave_transition_flags": np.zeros((t, 1), "f")})()
    compute_configs = {
        "current_extra": load_config(ROOT / "configs/dev_pwtr_current_extra_300k.yaml"),
        "uniform_recent": load_config(ROOT / "configs/dev_pwtr_uniform_recent_300k.yaml"),
        "priority_recent": load_config(ROOT / "configs/dev_pwtr_priority_recent_300k.yaml"),
        "full": full_cfg,
    }
    compute = {}; full = None
    for name, config in compute_configs.items():
        trainer = build_modular_mappo_trainer(config, "cuda", 256, 1_805_280)
        trainer.load(SOURCE, strict_protocol=False, restore_rng=True)
        module = trainer.persistent_wave_trajectory_replay
        collection_generation = 1 if name == "current_extra" else 0
        module.ingest_rollout(batch(trainer, 2), collection_generation)
        module.ingest_rollout(batch(trainer, 3), collection_generation)
        if name == "full":
            for source_wave in (1, 2):
                source = batch(trainer, source_wave); source.wave_transition_flags[-1, 0] = 1
                module.ingest_rollout(source, collection_generation)
                module.ingest_rollout(batch(trainer, source_wave + 1), collection_generation)
        module.current_rollout_generation = 1
        module.current_later_states = 1024
        module.fresh_rollout_count = 2
        actor_before, critic_before = digest(trainer.actor.state_dict()), digest(trainer.critic.state_dict())
        reward_before = digest([segment["rewards"] for rows in module.partitions.values() for segment in rows])
        metrics = trainer._pwtr_replay_phase()
        compute[name] = {
            "budget": metrics["pwtr_replay_batches_budget"],
            "actual_batches": metrics["pwtr_replay_batches"],
            "actor_steps": metrics["pwtr_replay_actor_optimizer_steps_this_phase"],
            "critic_steps": metrics["pwtr_replay_critic_optimizer_steps_this_phase"],
            "fill_fraction": metrics["pwtr_replay_budget_fill_fraction"],
            "actor_changed": actor_before != digest(trainer.actor.state_dict()),
            "critic_changed": critic_before != digest(trainer.critic.state_dict()),
            "reward_unchanged": reward_before == digest([segment["rewards"] for rows in module.partitions.values() for segment in rows]),
            "finite": all(np.isfinite(value) for value in metrics.values() if isinstance(value, (int, float))),
        }
        if name == "full": full = trainer; replay_metrics = metrics
    matched_values = {(row["budget"], row["actual_batches"], row["actor_steps"], row["critic_steps"]) for row in compute.values()}
    compute_matched = matched_values == {(2.0, 2.0, 2.0, 2.0)}
    bridge_budget_safe = compute["full"]["budget"] == 2.0 and compute["full"]["actual_batches"] == 2.0
    if not compute_matched or not bridge_budget_safe or not all(row["finite"] and row["actor_changed"] and row["critic_changed"] and row["reward_unchanged"] for row in compute.values()):
        raise RuntimeError(f"PWTR real compute matching failed: {compute}")
    module = full.persistent_wave_trajectory_replay
    actor_changed = compute["full"]["actor_changed"]; critic_changed = compute["full"]["critic_changed"]
    reward_unchanged = compute["full"]["reward_unchanged"]; finite = compute["full"]["finite"]

    checkpoint = out / "pwtr_roundtrip.pt"; full.save(checkpoint)
    restored = build_modular_mappo_trainer(full_cfg, "cuda", 256, 1_805_280); restored.load(checkpoint, strict_protocol=True, restore_rng=True)
    roundtrip = digest(module.state_dict()) == digest(restored.persistent_wave_trajectory_replay.state_dict())
    if not roundtrip: raise RuntimeError("PWTR memory/RNG checkpoint roundtrip failed")
    result = {"status": "PWTR_CUDA_TINY_SMOKE_PASS", "cuda": torch.cuda.get_device_name(0),
              "first_rollout_bitwise": first_rollout, "plain_disabled_bitwise_parity": parity,
              "stratified_sample_conservation": conservation, "memory_counts": module.memory_counts(),
              "w2_w3_extraction": len(module.partitions["W2_INTERNAL"]) > 0 and len(module.partitions["W3_INTERNAL"]) > 0,
              "bridge12_bridge23_extraction": len(module.partitions["BRIDGE_12"]) > 0 and len(module.partitions["BRIDGE_23"]) > 0,
              "replay_actor_changed": actor_changed, "replay_critic_changed": critic_changed,
              "replay_metrics_finite": finite, "environment_reward_unchanged": reward_unchanged,
              "current_recent_budget_matched": compute_matched, "bridge_does_not_change_budget": bridge_budget_safe,
              "compute_matching": compute,
              "checkpoint_memory_rng_roundtrip": roundtrip, "source_checkpoint_unchanged": source_sha == checkpoint_hash(SOURCE),
              "formal_training_started": False, "formal_evaluation_started": False, "45m_used": False,
              "replay_metrics": replay_metrics}
    (out / "smoke_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
