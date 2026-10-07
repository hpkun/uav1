"""Offline A/B/C/D feasibility ablation on the spawn-stress state set."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from env.config import load_config
from env.persistent_env import PersistentWaveCombatEnv
from tools.validate_persistent_wave_environment import CATEGORIES, red_layout


ROOT = PROJECT_ROOT


def diagnose_batch(config: dict[str, Any], start: int, stop: int) -> dict[str, int]:
    env = PersistentWaveCombatEnv(config)
    solved = {rule: 0 for rule in ("A", "B", "C", "D")}
    angles = np.linspace(-np.pi, np.pi, 72, endpoint=False)
    for case_index in range(start, stop):
        seed = 70_000_000 + case_index
        category = CATEGORIES[case_index % len(CATEGORIES)]
        survivors = (case_index // len(CATEGORIES)) % 4 + 1
        env.reset(seed)
        env.red = red_layout(category, survivors, env.rng)
        feasible = {rule: False for rule in solved}
        alive_red = [state for state in env.red if state.alive]
        for angle in angles:
            candidate = env._candidate_blue_wave(float(angle))
            inside = all(
                np.hypot(blue.x, blue.y) < env.arena_radius
                for blue in candidate
            )
            distance_ok = all(
                np.linalg.norm(np.array([
                    blue.x - red.x, blue.y - red.y, blue.z - red.z,
                ])) >= 2500.0
                for blue in candidate for red in alive_red
            )
            red_safe = not any(
                env._in_fire_window(red, blue)
                for red in alive_red for blue in candidate
            )
            blue_safe = not any(
                env._in_fire_window(blue, red)
                for blue in candidate for red in alive_red
            )
            base = inside and distance_ok
            feasible["A"] |= base and red_safe and blue_safe
            feasible["B"] |= base
            feasible["C"] |= base and red_safe
            feasible["D"] |= base and blue_safe
        for rule, exists in feasible.items():
            solved[rule] += int(exists)
    return solved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--output", default="outputs/persistent_wave_spawn_constraint_ablation.json"
    )
    args = parser.parse_args()
    config = load_config(ROOT / "configs/persistent_wave_environment.yaml")
    workers = max(1, min(args.workers, args.cases))
    boundaries = np.linspace(0, args.cases, workers + 1, dtype=int)
    if workers == 1:
        batches = [diagnose_batch(config, 0, args.cases)]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batches = list(executor.map(
                diagnose_batch, [config] * workers,
                boundaries[:-1].tolist(), boundaries[1:].tolist(),
            ))
    rules = {
        rule: {
            "solvable": sum(batch[rule] for batch in batches),
            "unsolvable": args.cases - sum(batch[rule] for batch in batches),
            "solvable_rate": sum(batch[rule] for batch in batches) / args.cases,
            "unsolvable_rate": 1.0 - sum(batch[rule] for batch in batches) / args.cases,
        }
        for rule in ("A", "B", "C", "D")
    }
    result = {
        "cases": args.cases,
        "candidate_directions": 72,
        "workers": workers,
        "rules": rules,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
