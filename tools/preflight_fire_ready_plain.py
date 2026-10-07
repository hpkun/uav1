"""Fail-closed static/protocol preflight for FireReady-Plain."""
from __future__ import annotations

from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.train_modular_mappo import load_config
from env.factory import make_combat_environment
from env.weapon import FireState


LEGACY = ROOT / "configs/persistent_wave_v2_environment.yaml"
READY = ROOT / "configs/persistent_wave_v2_fire_ready_environment.yaml"
ALGORITHM = ROOT / "configs/dev_fire_ready_plain_3m.yaml"
MANIFEST = ROOT / "experiments/fire_ready_plain_development_manifest.json"
REGISTRY = ROOT / "experiments/current_seed_provenance.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(5301, 5302, 5303))
    args = parser.parse_args()
    legacy_cfg = yaml.safe_load(LEGACY.read_text(encoding="utf-8"))
    ready_cfg = yaml.safe_load(READY.read_text(encoding="utf-8"))
    algorithm = load_config(ALGORITHM)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    normalized = deepcopy(ready_cfg)
    require(normalized["observation"].pop("include_own_fire_ready", None) is True,
            "FireReady opt-in flag missing")
    require(normalized == legacy_cfg, "environment differs outside own fire-ready observation flag")
    legacy, ready = make_combat_environment(legacy_cfg), make_combat_environment(ready_cfg)
    legacy_obs, _ = legacy.reset(9_900_001)
    ready_obs, _ = ready.reset(9_900_001)
    require(legacy_obs.shape == (4, 52) and legacy.observation_dim == 52, "legacy is not 52D")
    require(ready_obs.shape == (4, 53) and ready.observation_dim == 53, "FireReady is not 53D")
    require(np.array_equal(legacy_obs, ready_obs[:, :52]), "first 52 features changed")
    require(np.array_equal(ready_obs[:, 52], np.ones(4)), "reset readiness is not one")

    before = ready._observations().copy()
    ready.red_fire_states[0].armed = False
    after = ready._observations()
    require(np.array_equal(before[:, :52], after[:, :52]), "armed alias changed base observation")
    require(before[0, 52] == 1 and after[0, 52] == 0, "53rd bit does not expose armed")
    require(np.array_equal(before[1:], after[1:]), "own readiness leaked across agents")
    ready.red[1].alive = False
    require(np.count_nonzero(ready._observations()[1]) == 0, "dead Red observation is not all zero")

    # FireState lifecycle: consumption, out-of-window re-arm, intermediate spawn reset.
    ready = make_combat_environment(ready_cfg)
    ready.reset(9_900_002)
    for index in range(1, 4):
        ready.red[index].alive = ready.blue[index].alive = False
    ready.red[0].x = ready.red[0].y = 0.0; ready.red[0].z = -3000.0; ready.red[0].psi = 0.0
    ready.blue[0].x = 1000.0; ready.blue[0].y = 0.0; ready.blue[0].z = -3000.0; ready.blue[0].psi = np.pi
    class AlwaysMiss:
        @staticmethod
        def normal():
            return 100.0
    ready.rng = AlwaysMiss()
    observation, _, terminated, truncated, info = ready.step(
        np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32))
    require(not terminated and not truncated and info["red_step_fire_attempts"] == 1,
            "entry trigger did not attempt fire")
    require(observation[0, 52] == 0, "post-attempt next observation readiness is not zero")
    ready.blue[0].x = 4500.0
    observation, _, _, _, info = ready.step(
        np.zeros((4, 3), np.float32), np.zeros((4, 3), np.float32))
    require(info["red_step_fire_attempts"] == 0 and observation[0, 52] == 1,
            "out-of-window next observation readiness did not re-arm")
    ready = make_combat_environment(ready_cfg); ready.reset(9_900_003)
    ready.red_fire_states = [FireState(False) for _ in range(4)]
    for state in ready.blue: state.alive = False
    observation, _, terminated, truncated, info = ready.step(np.zeros((4, 3), np.float32))
    require(not terminated and not truncated and info["spawned_next_wave"], "intermediate wave did not spawn")
    require(np.array_equal(observation[:, 52], ready.red_alive_mask), "post-spawn readiness mismatch")

    training, network = algorithm["training"], algorithm["network"]
    require(network == {"observation_dim": 53, "action_dim": 3, "num_agents": 4,
                        "actor_hidden_layers": [256, 256], "critic_hidden_layers": [256, 256],
                        "attention_heads": 2}, "network protocol mismatch")
    expected_training = {"gamma": .999, "gae_lambda": .95, "rollout_steps": 256,
                         "num_train_envs": 24, "total_sampled_steps": 3_000_000,
                         "ppo_epochs": 10, "minibatch_size": 512}
    for key, expected in expected_training.items():
        require(training[key] == expected, f"training {key} mismatch")
    enabled = sorted(name for name, cfg in algorithm["modules"].items() if cfg.get("enabled"))
    require(enabled == ["actor_lr_decay"], f"enabled modules mismatch: {enabled}")
    trainer = build_modular_mappo_trainer(algorithm, "cpu", 256, 3_000_000)
    require(trainer.actor.backbone[0].in_features == 53, "actor base input is not 53")
    require(trainer.critic.embedding[0].in_features == 53, "critic base input is not 53")
    require(trainer.actor.mean.out_features == 3, "action dimension changed")
    protocol = algorithm["development_protocol"]
    require(protocol["validation"] == {"seed_start": 44_000_000, "seed_end": 44_000_049,
            "episodes": 50, "deterministic": True, "common_scenarios": True, "is_holdout": False},
            "44M development evaluation mismatch")
    require(protocol["reserved_future_final_test"] == {"seed_start": 45_000_000,
            "seed_end": 45_000_199, "executed": False}, "45M protocol mismatch")
    require(registry["evaluation_ranges"]["45000000..45000199"]["executed"] is False,
            "45M registry is no longer untouched")
    require(manifest["observation_schema"] == "fire_ready_v1" and manifest["from_scratch"] is True,
            "manifest identity mismatch")
    checked_seeds = (args.seed,) if args.seed is not None else (5301, 5302, 5303)
    existing = [str(ROOT / f"outputs/dev_fire_ready_plain_3m/seed{seed}")
                for seed in checked_seeds
                if (ROOT / f"outputs/dev_fire_ready_plain_3m/seed{seed}").exists()]
    require(not existing, f"formal output directories already exist: {existing}")
    print("FIRE_READY_PLAIN_STATIC_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
