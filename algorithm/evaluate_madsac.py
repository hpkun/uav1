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


def resolved(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("stochastic", "deterministic"), default="stochastic")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--env-config", default="configs/persistent_wave_v2_environment.yaml")
    parser.add_argument("--algorithm-config", default="configs/madsac_persistent_wave_v2_3m.yaml")
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
    seeds = tuple(range(44_000_000, 44_000_050))
    result = {"checkpoint": str(checkpoint), "mode": args.mode,
              "environment_seed_range": [seeds[0], seeds[-1]],
              "policy_seed": 770001 if args.mode == "stochastic" else None,
              "future_final_45m_untouched": True,
              **evaluate_madsac(trainer, env_config, seeds, args.mode, 770001)}
    text = json.dumps(result, indent=2)
    if args.output:
        resolved(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
