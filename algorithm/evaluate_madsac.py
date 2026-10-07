"""Read-only actor deployment evaluation for MADSAC checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.madsac.evaluation import evaluate_madsac
from algorithm.madsac.protocol import validate_madsac_checkpoint
from algorithm.madsac.trainer import MADSACTrainer
from algorithm.common.protocol import config_sha256


def resolved(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("stochastic", "deterministic"), default="stochastic")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--env-config", default="configs/combat_environment.yaml")
    parser.add_argument("--algorithm-config", default="configs/madsac.yaml")
    parser.add_argument("--seed-base", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    env_config = yaml.safe_load(resolved(args.env_config).read_text(encoding="utf-8"))
    algorithm_config = yaml.safe_load(resolved(args.algorithm_config).read_text(encoding="utf-8"))
    checkpoint = resolved(args.checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    training_seed = int(state.get("extra", {}).get("training_seed", -1))
    validate_madsac_checkpoint(
        state,
        env_config,
        algorithm_config,
        expected_training_seed=training_seed,
    )
    n, t, i = algorithm_config["network"], algorithm_config["training"], algorithm_config["implementation"]
    architecture = state.get("extra", {}).get("network_architecture", {})
    hidden_dim = int(architecture.get("hidden_dim", n["actor_hidden_layers"][0]))
    attention_heads = int(architecture.get("attention_heads", n["attention_heads"]))
    trainer = MADSACTrainer(
        n["observation_dim"], n["action_dim"], n["num_agents"], hidden_dim,
        attention_heads, t["actor_learning_rate"], t["critic_learning_rate"],
        t["gamma"], t["alpha"], t["tau"], t["policy_delay"], args.device,
        training_seed, i["actor_activation"], i["critic_activation"], i["log_std_min"], i["log_std_max"],
    )
    trainer.load_for_evaluation(checkpoint)
    base = int(i["evaluation_seed_base"] if args.seed_base is None else args.seed_base)
    count = int(t["evaluation_episodes"] if args.episodes is None else args.episodes)
    if count <= 0:
        raise ValueError("episodes must be positive")
    seeds = tuple(range(base, base + count))
    extra = state["extra"]
    result = {"algorithm": "madsac", "checkpoint": str(checkpoint), "mode": args.mode,
              "environment_seed_range": [seeds[0], seeds[-1]],
              "policy_seed": i["evaluation_policy_seed"] if args.mode == "stochastic" else None,
              "implementation_version": state["implementation_version"],
              "protocol_complete": True,
              "checkpoint_sampled_steps": int(state["sampled_steps"]),
              "checkpoint_training_seed": training_seed,
              "checkpoint_training_gamma": extra["gamma"],
              "checkpoint_training_num_envs": extra["training_num_envs"],
              "checkpoint_training_total_sampled_steps": extra["training_total_sampled_steps"],
              "checkpoint_training_smoke": extra["training_smoke"],
              "checkpoint_effective_hidden_dim": hidden_dim,
              "checkpoint_environment_version": extra["environment_version"],
              "evaluation_environment_version": env_config["environment_version"],
              "checkpoint_environment_config_sha256": extra["environment_config_sha256"],
              "checkpoint_algorithm_config_sha256": extra["algorithm_config_sha256"],
              "evaluation_environment_config_sha256": config_sha256(env_config),
              "provided_algorithm_config_sha256": config_sha256(algorithm_config),
              "observation_dim": n["observation_dim"], "action_dim": n["action_dim"],
              "num_agents": n["num_agents"], "device": args.device,
              "holdout_seed_base": seeds[0], "holdout_seed_end": seeds[-1],
              **evaluate_madsac(trainer, env_config, seeds, args.mode, i["evaluation_policy_seed"])}
    text = json.dumps(result, indent=2)
    if args.output:
        output = resolved(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
