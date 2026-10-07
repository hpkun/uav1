"""Formal entry point for the independent MADSAC baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.madsac.protocol import validate_madsac_config
from algorithm.madsac.runner import MADSACTrainingRunner


def resolved(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", default="configs/persistent_wave_v2_environment.yaml")
    parser.add_argument("--algorithm-config", default="configs/madsac_persistent_wave_v2_3m.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--total-sampled-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    env_path, algorithm_path = resolved(args.env_config), resolved(args.algorithm_config)
    env_config = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    algorithm_config = yaml.safe_load(algorithm_path.read_text(encoding="utf-8"))
    validate_madsac_config(env_config, algorithm_config)
    output = resolved(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty MADSAC output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("env_config.yaml", env_config),
                        ("runtime_env_config.yaml", env_config),
                        ("algorithm_config.yaml", algorithm_config)):
        (output / name).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    training = algorithm_config["training"]
    run_config = {
        "algorithm": "madsac", "seed": int(training["seed"] if args.seed is None else args.seed),
        "device": str(args.device or training["device"]),
        "num_envs": int(args.num_envs or training["num_train_envs"]),
        "total_sampled_steps": int(args.total_sampled_steps or training["total_sampled_steps"]),
        "smoke": bool(args.smoke), "environment_variant": env_config["environment_variant"],
        "environment_config_path": str(env_path), "algorithm_config_path": str(algorithm_path),
        "environment_config_sha256": config_sha256(env_config),
        "algorithm_config_sha256": config_sha256(algorithm_config),
        "formal_exact_resume_supported": False, "replay_buffer_in_checkpoint": False,
        "sampled_steps_unit": "environment_transitions_not_agent_transitions",
        "primary_evaluation_mode": "stochastic", "evaluation_policy_seed": 770001,
        "actor_policy_gradient": "own_action_only_other_joint_actions_detached",
        "objective_reduction": "per_transition_agent_sum_then_replay_batch_mean",
        "evaluation_policy_rng": "independent_deterministic_stream_per_environment_seed",
        "future_final_45m_untouched": True,
        "paper_reported": algorithm_config["metadata"]["paper_reported"],
        "project_implementation_choice": algorithm_config["metadata"]["project_implementation_choice"],
    }
    (output / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    runner = MADSACTrainingRunner(
        env_config, algorithm_config, args.num_envs, args.total_sampled_steps,
        args.device, args.seed, output, args.smoke,
    )
    summary = runner.run()
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output),
                      "sampled_steps": summary["sampled_steps"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
