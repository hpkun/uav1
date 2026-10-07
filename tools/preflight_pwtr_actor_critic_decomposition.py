"""Strict CUDA/static preflight for the PWTR actor-vs-critic decomposition."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import runtime_source_manifest
from algorithm.modular_mappo.protocol import validate_pwtr_branch
from algorithm.train_modular_mappo import load_config

SEEDS = (5301, 5302, 5303)
SOURCE, TARGET = 1_505_280, 1_805_280
NEW = ("current_actor_only", "current_critic_only", "recent_actor_only", "recent_critic_only")
OLD = ("stratified", "current_extra", "uniform_recent")
CORE = (
    "algorithm/modules/persistent_wave_trajectory_replay.py",
    "algorithm/modular_mappo/trainer.py", "algorithm/modular_mappo/runner.py",
    "algorithm/modular_mappo/networks.py", "env/combat_env.py", "env/persistent_env.py",
)
EXPECTED = {
    "current_actor_only": (True, True, "current", False, False, True, False),
    "current_critic_only": (True, True, "current", False, False, False, True),
    "recent_actor_only": (True, True, "recent_uniform", False, False, True, False),
    "recent_critic_only": (True, True, "recent_uniform", False, False, False, True),
}
OLD_EXPECTED = {
    "stratified": ("pwtr_stratified", True, False, "recent_uniform", False, False, False, False),
    "current_extra": ("pwtr_current_extra", True, True, "current", False, False, True, True),
    "uniform_recent": ("pwtr_uniform_recent", True, True, "recent_uniform", False, False, True, True),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mode(config: dict) -> tuple:
    module = config["modules"]["persistent_wave_trajectory_replay"]
    return (bool(module["fresh_wave_stratification"]), bool(module["replay_enabled"]),
            module["replay_source"], bool(module["priority_enabled"]),
            bool(module["bridge_enabled"]), bool(module["actor_replay"]),
            bool(module["critic_replay"]))


def validate_old_reference(path: Path, branch: str, seed: int, parent_checkpoint_sha256: str) -> dict:
    required = ("run_summary.json", "run_config.json", "evaluation_history.csv",
                "training_metrics.jsonl", "optimization_metrics.jsonl", "algorithm_config.yaml",
                "latest.pt", "final.pt")
    missing = [name for name in required if not (path / name).is_file()]
    if missing: raise RuntimeError(f"{path}: missing {missing}")
    summary = json.loads((path / "run_summary.json").read_text(encoding="utf-8"))
    if int(summary.get("sampled_steps", -1)) != TARGET: raise RuntimeError(f"{path}: incomplete")
    run = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
    if (int(run.get("seed", -1)), run.get("environment_variant")) != (seed, "persistent_wave_v2"):
        raise RuntimeError(f"{path}: identity mismatch")
    provenance = run.get("branch_provenance")
    if not isinstance(provenance, dict) or provenance.get("parent_checkpoint_sha256") != parent_checkpoint_sha256:
        raise RuntimeError(f"{path}: parent checkpoint SHA mismatch or missing")
    algorithm = yaml.safe_load((path / "algorithm_config.yaml").read_text(encoding="utf-8"))
    module = algorithm.get("modules", {}).get("persistent_wave_trajectory_replay", {})
    actual = (algorithm.get("development_method"), bool(module.get("fresh_wave_stratification")),
              bool(module.get("replay_enabled")), module.get("replay_source"),
              bool(module.get("priority_enabled")), bool(module.get("bridge_enabled")),
              bool(module.get("actor_replay")), bool(module.get("critic_replay")))
    if actual != OLD_EXPECTED[branch]: raise RuntimeError(f"{path}: old reference branch identity mismatch: {actual}")
    fixed = tuple(int(module.get(key, -1)) for key in ("sequence_length", "bridge_half_length",
                  "min_segment_length", "partition_capacity", "actor_max_age_updates"))
    if fixed != (128, 64, 32, 32, 2): raise RuntimeError(f"{path}: old reference fixed constants mismatch: {fixed}")
    with (path / "evaluation_history.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    exact = [row for row in rows if int(float(row["sampled_steps"])) == TARGET]
    if len(exact) != 1: raise RuntimeError(f"{path}: exact endpoint missing")
    row = exact[0]
    if (int(float(row["evaluation_episodes"])), int(float(row["evaluation_seed_base"])),
            int(float(row["evaluation_seed_end"]))) != (50, 44_000_000, 44_000_049):
        raise RuntimeError(f"{path}: validation protocol mismatch")
    for filename in ("training_metrics.jsonl", "optimization_metrics.jsonl"):
        lines = [line for line in (path / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines: raise RuntimeError(f"{path}/{filename}: empty")
        for line in lines:
            for value in json.loads(line).values():
                if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                    raise RuntimeError(f"{path}/{filename}: non-finite")
    return {"branch": branch, "seed": seed, "sampled_steps": TARGET,
            "evaluation_episodes": 50, "evaluation_seed_range": [44_000_000, 44_000_049],
            "parent_checkpoint_sha256": parent_checkpoint_sha256,
            "parent_checkpoint_sha256_match": True, "branch_identity_match": True}


def main() -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is mandatory for PWTR decomposition preflight")
    env = yaml.safe_load((ROOT / "configs/persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    if (env["environment_variant"], env["persistent_waves"]["total_waves"],
            env["simulation"]["max_steps"], env["scenario"]["team_size"]) != ("persistent_wave_v2", 3, 3000, 4):
        raise RuntimeError("frozen environment identity mismatch")
    configs = {name: load_config(ROOT / f"configs/dev_pwtr_{name}_300k.yaml") for name in NEW}
    expected_training = {"gamma": .999, "gae_lambda": .95, "clip_ratio": .2,
        "entropy_coefficient": .01, "value_loss_coefficient": .5, "max_grad_norm": .5,
        "rollout_steps": 256, "ppo_epochs": 10, "minibatch_size": 512,
        "num_train_envs": 24, "evaluation_episodes": 50,
        "evaluation_interval_sampled_steps": 100000, "total_sampled_steps": TARGET}
    for name, config in configs.items():
        if any(config["training"][key] != value for key, value in expected_training.items()):
            raise RuntimeError(f"{name}: training mismatch")
        if (config["network"]["observation_dim"], config["network"]["action_dim"], config["network"]["num_agents"]) != (52, 3, 4):
            raise RuntimeError(f"{name}: network mismatch")
        if int(config["implementation"]["evaluation_seed_base"]) != 44_000_000: raise RuntimeError(f"{name}: seed range mismatch")
        branch = config["development_branch"]
        if (branch["source_sampled_steps"], branch["additional_sampled_steps"], branch["target_sampled_steps"],
                branch["actor_optimizer_restore"], branch["critic_optimizer_restore"], branch["rng_restore"]) != (SOURCE, 300_000, TARGET, True, True, True):
            raise RuntimeError(f"{name}: branch/restore mismatch")
        enabled = sorted(key for key, value in config["modules"].items() if isinstance(value, dict) and value.get("enabled", False))
        if enabled != ["actor_lr_decay", "persistent_wave_trajectory_replay"] or exact_mode(config) != EXPECTED[name]:
            raise RuntimeError(f"{name}: exact module mode mismatch")

    source_rows, validations = {}, {}
    for seed in SEEDS:
        checkpoint = ROOT / f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_{SOURCE}.pt"
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(state.get("sampled_steps", -1)) != SOURCE or int(state["extra"].get("training_seed", -1)) != seed:
            raise RuntimeError(f"seed{seed}: source identity mismatch")
        if state.get("enabled_modules") != ["actor_lr_decay"] or float(state["actor_optimizer"]["param_groups"][0]["lr"]) != 1e-4:
            raise RuntimeError(f"seed{seed}: source modules/LR mismatch")
        source_rows[str(seed)] = {"path": str(checkpoint), "sha256": sha(checkpoint)}
        validations[str(seed)] = {name: validate_pwtr_branch(state, env, config,
            {"training_seed": seed, "training_num_envs": 24, "training_smoke": False}) for name, config in configs.items()}

    references = [validate_old_reference(ROOT / f"outputs/dev_pwtr_{branch}_seed{seed}_300k", branch, seed,
                  source_rows[str(seed)]["sha256"])
                  for seed in SEEDS for branch in OLD]
    historical = json.loads((ROOT / "outputs/dev_pwtr_stratified_seed5301_300k/run_config.json").read_text(encoding="utf-8"))
    old_hashes = {row["path"]: row["sha256"] for row in historical["runtime_source_manifest_files"]}
    core = {path: {"historical": old_hashes.get(path), "current": sha(ROOT / path)} for path in CORE}
    if any(row["historical"] != row["current"] for row in core.values()): raise RuntimeError("frozen PWTR core source changed")
    formal = [ROOT / f"outputs/dev_pwtr_{name}_seed{seed}_300k" for seed in SEEDS for name in NEW]
    existing = [str(path) for path in formal if path.exists()]
    if existing: raise RuntimeError(f"new formal outputs already exist: {existing}")
    manifest = runtime_source_manifest(ROOT)
    report = {"status": "READY_FOR_PWTR_ACTOR_CRITIC_DECOMPOSITION_300K",
        "cuda": torch.cuda.get_device_name(0), "source_checkpoints": source_rows,
        "config_validations": validations, "old_reference_runs": references,
        "old_reference_count": len(references), "new_formal_output_count": len(formal),
        "new_formal_outputs_absent": True, "frozen_core_sha256": core,
        "runtime_source_manifest_sha256": manifest["runtime_source_manifest_sha256"],
        "evaluation_seed_range": [44_000_000, 44_000_049], "reserved_45m_used": False}
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
