"""Regression tests for the repository's direct-execution layout."""
from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from algorithm.mappo.runner import MAPPOTrainingRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_configs() -> tuple[dict, dict]:
    environment = yaml.safe_load(
        (PROJECT_ROOT / "configs/combat_environment.yaml").read_text(encoding="utf-8")
    )
    algorithm = yaml.safe_load(
        (PROJECT_ROOT / "configs/mappo.yaml").read_text(encoding="utf-8")
    )
    return environment, algorithm


def test_runner_uses_exact_output_directory(tmp_path):
    environment, algorithm = load_configs()
    run_dir = tmp_path / "exact_run"
    runner = MAPPOTrainingRunner(
        environment,
        algorithm,
        num_envs=1,
        total_sampled_steps=1,
        output_dir=run_dir,
        smoke=True,
    )
    try:
        assert runner.output_dir == run_dir
        assert not any(run_dir.glob("run_seed_*"))
    finally:
        runner.vector.close()


def test_direct_entry_runs_outside_project_with_spawn_and_flat_results():
    script = PROJECT_ROOT / "algorithm/train_mappo.py"
    with tempfile.TemporaryDirectory(prefix="uav_entry_cwd_") as cwd_value:
        cwd = Path(cwd_value)
        run_dir = cwd / "direct_entry_run"
        process_env = os.environ.copy()
        process_env.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--smoke",
                "--device",
                "cpu",
                "--num-envs",
                "1",
                "--total-sampled-steps",
                "4",
                "--env-config",
                "configs/combat_environment.yaml",
                "--algorithm-config",
                "configs/mappo.yaml",
                "--output-dir",
                str(run_dir),
            ],
            cwd=cwd,
            env=process_env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert cwd.resolve() != PROJECT_ROOT.resolve()
        expected = {
            "algorithm_config.yaml",
            "env_config.yaml",
            "latest.pt",
            "optimization_metrics.jsonl",
            "run_config.json",
            "run_summary.json",
            "train.log",
            "training_metrics.jsonl",
        }
        assert expected <= {path.name for path in run_dir.iterdir() if path.is_file()}
        assert not list(run_dir.glob("run_seed_*"))
        assert not [path for path in run_dir.iterdir() if path.is_dir()]
        log = (run_dir / "train.log").read_text(encoding="utf-8")
        assert "backend=multiprocess_spawn" in log
        assert "workers=1" in log

        original_run_config = (run_dir / "run_config.json").read_text(
            encoding="utf-8"
        )
        original_runtime = json.loads(original_run_config)
        resumed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--smoke",
                "--device",
                "cpu",
                "--num-envs",
                "1",
                "--total-sampled-steps",
                "8",
                "--env-config",
                "configs/combat_environment.yaml",
                "--algorithm-config",
                "configs/mappo.yaml",
                "--resume",
                str(run_dir / "latest.pt"),
            ],
            cwd=cwd,
            env=process_env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        history = [
            json.loads(line)
            for line in (run_dir / "resume_history.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert history[-1]["checkpoint_sampled_steps"] == 4
        assert history[-1]["total_sampled_steps"] == 8
        assert history[-1]["original_seed"] == original_runtime["seed"]
        assert history[-1]["effective_seed"] == original_runtime["seed"]
        assert history[-1]["original_num_envs"] == 1
        assert history[-1]["effective_num_envs"] == 1
        assert history[-1]["original_total_sampled_steps"] == 4
        assert history[-1]["effective_total_sampled_steps"] == 8
        assert history[-1]["extended_training_target"] is True
        assert history[-1]["rollback_performed"] is True
        assert history[-1]["environment_config_sha256"]
        assert history[-1]["algorithm_config_sha256"]
        assert list(run_dir.glob("run_summary.pre_resume_*.json"))
        assert (run_dir / "run_config.json").read_text(
            encoding="utf-8"
        ) == original_run_config
