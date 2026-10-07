#!/usr/bin/env python3
"""CUDA-only full-10-epoch matched smoke for Actor clipping 0.5 vs 1.0."""
from __future__ import annotations

from copy import deepcopy
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_actor_grad_clip_branch, validate_pwtr_branch
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from tools.smoke_actor_grad_clip_mappo import (
    digest, fsha, global_rng, raw_gradient_identity, restore_rng,
)

SOURCE = ROOT / "outputs/diag_mappo_learnability/l3_seed5302/checkpoint_1505280.pt"


def first_minibatch_actor_geometry(trainer, rollout, rng):
    restore_rng(rng)
    obs = torch.as_tensor(rollout.observations.reshape(-1, 4, 52)[:16], device="cuda")
    act = torch.as_tensor(rollout.actions.reshape(-1, 4, 3)[:16], device="cuda")
    raw = torch.as_tensor(rollout.raw_actions.reshape(-1, 4, 3)[:16], device="cuda")
    old = torch.as_tensor(rollout.old_log_probs.reshape(-1, 4)[:16], device="cuda")
    alive = torch.as_tensor(rollout.alive_masks.reshape(-1, 4)[:16], device="cuda")
    count = obs.shape[0]
    advantage = torch.linspace(-1, 1, count * 4, device="cuda").reshape(count, 4) * alive
    old_value = torch.zeros(count, 4, device="cuda")
    target = torch.ones(count, 4, device="cuda")
    weights = torch.ones(count, device="cuda")
    context = torch.zeros(count, 0, device="cuda")
    loss = trainer._loss_step(obs, act, raw, old, alive, advantage, old_value, target, weights, context)
    trainer.actor_optimizer.zero_grad()
    (loss[0] - trainer.entropy_coefficient * loss[2] + loss[3]).backward()
    parameters = trainer.actor.trainable_policy_parameters()
    pre = trainer._gradient_norm(parameters)
    torch.nn.utils.clip_grad_norm_(parameters, trainer._actor_grad_clip_limit())
    post = trainer._gradient_norm(parameters)
    trainer.actor_optimizer.zero_grad()
    return {"preclip": pre, "postclip": post, "limit": trainer._actor_grad_clip_limit()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/smoke_actor_grad_clip_fixed10")
    args = parser.parse_args()
    out = ROOT / args.output_dir
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA mandatory")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    env = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text())
    state = torch.load(SOURCE, map_location="cpu", weights_only=False)
    source_sha = fsha(SOURCE)
    configs = {
        "Plain": load_config(ROOT / "configs/dev_pwtr_plain_300k.yaml"),
        "Clip05": load_config(ROOT / "configs/dev_actor_grad_clip_05_control_300k.yaml"),
        "Clip10": load_config(ROOT / "configs/dev_actor_grad_clip_10_300k.yaml"),
    }
    if any(config["training"]["ppo_epochs"] != 10 for config in configs.values()):
        raise RuntimeError("smoke requires ppo_epochs=10")
    runtime = {"training_seed": 5302, "training_num_envs": 24, "training_smoke": False}
    provenance = {
        "Plain": validate_pwtr_branch(state, env, configs["Plain"], runtime),
        "Clip05": validate_actor_grad_clip_branch(state, env, configs["Clip05"], runtime),
        "Clip10": validate_actor_grad_clip_branch(state, env, configs["Clip10"], runtime),
    }
    runners, rollouts, initial = {}, {}, {}
    for name in ("Plain", "Clip05", "Clip10"):
        runner = ModularMAPPOTrainingRunner(
            env, configs[name], 2, 1_805_280, "cuda", 5302, out / name,
            False, resume_mode=True,
            branch_provenance={**provenance[name], "parent_checkpoint_sha256": source_sha},
        )
        runner.branch_from(SOURCE, provenance[name]["intervention"], source_sha)
        runners[name] = runner
        initial[name] = {
            "actor": digest(runner.trainer.actor.state_dict()),
            "critic": digest(runner.trainer.critic.state_dict()),
            "actor_optimizer": digest(runner.trainer.actor_optimizer.state_dict()),
            "critic_optimizer": digest(runner.trainer.critic_optimizer.state_dict()),
            "rng": digest(runner.trainer.capture_rng_state()),
        }
        rollouts[name] = runner.collect_rollout(2)
    fields = ("observations", "actions", "raw_actions", "old_log_probs", "raw_environment_rewards", "alive_masks", "wave_indices", "dones")
    physical = {field: all(np.array_equal(getattr(rollouts["Plain"], field), getattr(rollouts[name], field)) for name in ("Clip05", "Clip10")) for field in fields}
    for runner in runners.values():
        runner.vector.close()
    branch_identity = {key: len({initial[name][key] for name in initial}) == 1 for key in initial["Plain"]}
    if not all(branch_identity.values()) or not all(physical.values()):
        raise RuntimeError(f"branch/rollout identity failure: {branch_identity}/{physical}")
    raw_identity = raw_gradient_identity(runners["Clip05"].trainer, runners["Clip10"].trainer, rollouts["Plain"])
    if not all(raw_identity.values()):
        raise RuntimeError(f"raw gradient identity failed: {raw_identity}")
    geometry_rng = global_rng()
    first_geometry = {
        name: first_minibatch_actor_geometry(runners[name].trainer, rollouts["Plain"], geometry_rng)
        for name in ("Clip05", "Clip10")
    }
    if not (first_geometry["Clip05"]["preclip"] == first_geometry["Clip10"]["preclip"] and
            .4999 < first_geometry["Clip05"]["postclip"] <= .50001 and
            .9999 < first_geometry["Clip10"]["postclip"] <= 1.00001):
        raise RuntimeError(f"first-minibatch actor clip geometry failed: {first_geometry}")
    trainers = {name: build_modular_mappo_trainer(configs[name], "cuda", 256, 1_805_280) for name in configs}
    for trainer in trainers.values():
        trainer.load(SOURCE, strict_protocol=False, restore_rng=False)
        if trainer.ppo_epochs != 10:
            raise RuntimeError("trainer did not resolve ten epochs")
    rng = global_rng()
    results, post, deltas = {}, {}, {}
    for name in ("Plain", "Clip05", "Clip10"):
        before = (trainers[name].actor_update_count, trainers[name].critic_update_count, trainers[name].ppo_update_count)
        restore_rng(rng)
        results[name] = trainers[name].update(deepcopy(rollouts[name]))
        deltas[name] = {
            "actor": trainers[name].actor_update_count - before[0],
            "critic": trainers[name].critic_update_count - before[1],
            "ppo": trainers[name].ppo_update_count - before[2],
        }
        post[name] = {
            "actor": digest(trainers[name].actor.state_dict()),
            "critic": digest(trainers[name].critic.state_dict()),
            "actor_optimizer": digest(trainers[name].actor_optimizer.state_dict()),
            "critic_optimizer": digest(trainers[name].critic_optimizer.state_dict()),
            "trainer_rng": digest(trainers[name].capture_rng_state()),
        }
    expected_steps = 10
    if any(value != {"actor": expected_steps, "critic": expected_steps, "ppo": 1} for value in deltas.values()):
        raise RuntimeError(f"ten-epoch execution failed: {deltas}")
    for result in results.values():
        if result["ppo_epochs_executed"] != 10 or result["ppo_minibatches_executed"] != 10:
            raise RuntimeError("runtime epoch diagnostics failed")
    control_noop = {key: post["Plain"][key] == post["Clip05"][key] for key in post["Plain"]}
    control_noop.update({key: results["Plain"][key] == results["Clip05"][key] for key in ("actor_loss", "value_loss", "actor_grad_norm", "critic_grad_norm")})
    critic_identity = {key: post["Clip05"][key] == post["Clip10"][key] for key in ("critic", "critic_optimizer")}
    if not all(control_noop.values()) or not all(critic_identity.values()):
        raise RuntimeError(f"matched identity failure: {control_noop}/{critic_identity}")
    if not (results["Clip05"]["actor_grad_clip_limit"] == .5 and results["Clip10"]["actor_grad_clip_limit"] == 1.0):
        raise RuntimeError("actor clip limit mismatch")
    if results["Clip05"]["actor_grad_clip_critic_limit"] != .5 or results["Clip10"]["actor_grad_clip_critic_limit"] != .5:
        raise RuntimeError("critic clip limit mismatch")
    checkpoints = {}
    for name in ("Clip05", "Clip10"):
        path = out / f"{name}.pt"
        trainers[name].save(path)
        restored = build_modular_mappo_trainer(configs[name], "cuda", 256, 1_805_280)
        restored.load(path, strict_protocol=True, restore_rng=False)
        checkpoints[name] = digest(restored.actor.state_dict()) == digest(trainers[name].actor.state_dict())
    cross_rejected = []
    for checkpoint_name, config_name in (("Clip10", "Clip05"), ("Clip05", "Clip10")):
        try:
            build_modular_mappo_trainer(configs[config_name], "cuda", 256, 1_805_280).load(out / f"{checkpoint_name}.pt", strict_protocol=True, restore_rng=False)
            cross_rejected.append(False)
        except RuntimeError:
            cross_rejected.append(True)
    finite = all(np.isfinite(value) for name in results for value in results[name].values() if isinstance(value, (int, float)))
    report = {
        "status": "ACTOR_GRAD_CLIP_FIXED10_CUDA_SMOKE_PASS",
        "cuda": torch.cuda.get_device_name(0), "branch_identity": branch_identity,
        "two_step_physical_rollout_identity": physical,
        "first_matched_minibatch_raw_gradient_identity": raw_identity,
        "first_matched_minibatch_actor_geometry": first_geometry,
        "ppo_epochs": 10, "optimizer_step_deltas": deltas,
        "control05_plain_full10_noop": control_noop,
        "critic_identity_clip05_clip10": critic_identity,
        "actor_geometry": {
            "clip05_post": results["Clip05"]["actor_grad_clip_postclip_norm"],
            "clip10_post": results["Clip10"]["actor_grad_clip_postclip_norm"],
            "clip05_limit": results["Clip05"]["actor_grad_clip_limit"],
            "clip10_limit": results["Clip10"]["actor_grad_clip_limit"],
        },
        "strict_resume": checkpoints, "cross_config_resume_rejected": cross_rejected,
        "all_finite": finite, "source_unchanged": source_sha == fsha(SOURCE),
        "uses_45m": False, "formal_training_started": False,
    }
    if not all(checkpoints.values()) or not all(cross_rejected) or not finite:
        raise RuntimeError("checkpoint/finite smoke failure")
    (out / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
