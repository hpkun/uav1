"""Focused regression tests for formal experiment infrastructure."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch
import yaml

from algorithm.common.checkpoint import (
    validate_checkpoint_for_evaluation,
    validate_checkpoint_for_resume,
)
from algorithm.common.protocol import canonical_json, config_sha256
from algorithm.mappo.evaluation import evaluate_mappo_checkpoint
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION
from algorithm.train_mappo import (
    prepare_resume_rollback,
    reject_stale_resume_checkpoint,
    resolve_runtime_settings,
    ensure_fresh_output_directory,
    resolve_run_paths,
    validate_resume_config_snapshots,
)
from tools.aggregate_holdout_results import aggregate_holdout_results
from tools.aggregate_training_runs import (
    aggregate_training_histories,
    summarize_values,
)
from tools.evaluate_policy_matrix import evaluate_policy_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict:
    return yaml.safe_load((PROJECT_ROOT / "configs" / name).read_text(encoding="utf-8"))


def checkpoint_state(variant: str = "direct_v2_3") -> dict:
    trainer = build_mappo_trainer(load_yaml("mappo.yaml"), "cpu")
    return trainer.checkpoint_state({
        "environment_version": "2.3",
        "environment_variant": variant,
        "mappo_impl_version": MAPPO_IMPL_VERSION,
        "observation_dim": 52,
        "action_dim": 3,
        "num_agents": 4,
    })


def complete_checkpoint_state(
    environment: dict, algorithm: dict, *, seed: int = 2023,
    sampled_steps: int = 4, total_sampled_steps: int = 8,
) -> dict:
    trainer = build_mappo_trainer(algorithm, "cpu")
    trainer.sampled_steps = sampled_steps
    return trainer.checkpoint_state({
        "environment_version": str(environment["environment_version"]),
        "environment_variant": environment.get(
            "environment_variant", "direct_v2_3"
        ),
        "mappo_impl_version": MAPPO_IMPL_VERSION,
        "observation_dim": 52,
        "action_dim": 3,
        "num_agents": 4,
        "training_seed": seed,
        "training_gamma": float(algorithm["training"]["gamma"]),
        "training_num_envs": 1,
        "training_total_sampled_steps": total_sampled_steps,
        "training_smoke": True,
        "effective_hidden_dim": int(algorithm["network"]["actor_hidden_layers"][0]),
        "environment_config_sha256": config_sha256(environment),
        "algorithm_config_sha256": config_sha256(algorithm),
    })


def test_protocol_fingerprint_is_canonical_and_content_sensitive():
    left = {"b": [2, 3], "a": {"x": 1}}
    right = {"a": {"x": 1}, "b": [2, 3]}
    assert canonical_json(left) == canonical_json(right)
    assert config_sha256(left) == config_sha256(right)
    changed = {"a": {"x": 2}, "b": [2, 3]}
    assert config_sha256(left) != config_sha256(changed)
    algorithm = load_yaml("mappo.yaml")
    changed_algorithm = json.loads(json.dumps(algorithm))
    changed_algorithm["training"]["gamma"] = 0.999
    assert config_sha256(algorithm) != config_sha256(changed_algorithm)
    environment = load_yaml("combat_environment.yaml")
    changed_environment = json.loads(json.dumps(environment))
    changed_environment["simulation"]["max_steps"] += 1
    assert config_sha256(environment) != config_sha256(changed_environment)


def test_resume_runtime_inheritance_mismatch_and_extension():
    algorithm = load_yaml("mappo.yaml")
    run_config = {
        "seed": 2023, "num_envs": 1, "total_sampled_steps": 8,
        "smoke": True, "device": "cpu",
    }
    state = {"sampled_steps": 4, "extra": {"training_total_sampled_steps": 8}}
    inherited = resolve_runtime_settings(
        algorithm, seed=None, num_envs=None, total_sampled_steps=None,
        device=None, smoke=None, run_config=run_config, checkpoint_state=state,
    )
    assert inherited["seed"] == 2023
    assert inherited["num_envs"] == 1
    assert inherited["total_sampled_steps"] == 8
    assert inherited["smoke"] is True
    with pytest.raises(RuntimeError, match="seed mismatch"):
        resolve_runtime_settings(
            algorithm, seed=2024, num_envs=None, total_sampled_steps=None,
            device=None, smoke=None, run_config=run_config, checkpoint_state=state,
        )
    with pytest.raises(RuntimeError, match="num_envs mismatch"):
        resolve_runtime_settings(
            algorithm, seed=None, num_envs=2, total_sampled_steps=None,
            device=None, smoke=None, run_config=run_config, checkpoint_state=state,
        )
    extended = resolve_runtime_settings(
        algorithm, seed=None, num_envs=None, total_sampled_steps=12,
        device="cuda", smoke=None, run_config=run_config, checkpoint_state=state,
    )
    assert extended["extended_training_target"] is True
    assert extended["total_sampled_steps"] == 12
    assert extended["device"] == "cuda"
    with pytest.raises(RuntimeError, match="cannot be smaller"):
        resolve_runtime_settings(
            algorithm, seed=None, num_envs=None, total_sampled_steps=7,
            device=None, smoke=None, run_config=run_config, checkpoint_state=state,
        )


def test_legacy_resume_requires_explicit_critical_runtime_fields():
    with pytest.raises(RuntimeError, match="--seed"):
        resolve_runtime_settings(
            load_yaml("mappo.yaml"), seed=None, num_envs=1,
            total_sampled_steps=8, device="cpu", smoke=True,
            run_config=None, checkpoint_state={"sampled_steps": 4},
        )


def test_checkpoint_contains_formal_training_protocol_metadata(tmp_path):
    environment = load_yaml("combat_environment.yaml")
    algorithm = load_yaml("mappo.yaml")
    from algorithm.mappo.runner import MAPPOTrainingRunner
    runner = MAPPOTrainingRunner(
        environment, algorithm, num_envs=1, total_sampled_steps=8,
        device="cpu", seed=2023, output_dir=tmp_path / "run", smoke=True,
    )
    checkpoint = tmp_path / "run" / "manual.pt"
    try:
        runner.save_checkpoint(checkpoint)
    finally:
        runner.vector.close()
    extra = torch.load(checkpoint, map_location="cpu", weights_only=False)["extra"]
    assert extra["training_seed"] == 2023
    assert extra["training_gamma"] == pytest.approx(0.99)
    assert extra["training_num_envs"] == 1
    assert extra["training_total_sampled_steps"] == 8
    assert extra["training_smoke"] is True
    assert extra["environment_config_sha256"] == config_sha256(environment)
    assert extra["algorithm_config_sha256"] == config_sha256(algorithm)


def test_complete_checkpoint_rejects_wrong_source_algorithm_config(tmp_path):
    environment = load_yaml("combat_environment.yaml")
    algorithm = load_yaml("mappo.yaml")
    checkpoint = tmp_path / "model.pt"
    torch.save(complete_checkpoint_state(environment, algorithm), checkpoint)
    wrong = json.loads(json.dumps(algorithm))
    wrong["training"]["gamma"] = 0.999
    with pytest.raises(RuntimeError, match="algorithm config fingerprint"):
        evaluate_mappo_checkpoint(
            checkpoint, wrong, environment, "cpu", [100, 101]
        )


def test_evaluation_seed_protocol_rejects_gaps_and_duplicates(tmp_path):
    for seeds in ([1, 3], [1, 1], [2, 1]):
        with pytest.raises(ValueError, match="contiguous"):
            evaluate_mappo_checkpoint(
                tmp_path / "missing.pt", {}, {}, "cpu", seeds
            )


def test_checkpoint_resume_and_evaluation_variant_contracts():
    state = checkpoint_state("direct_v2_3")
    direct = load_yaml("combat_environment.yaml")
    persistent = load_yaml("persistent_wave_v2_environment.yaml")
    algorithm = load_yaml("mappo.yaml")
    validate_checkpoint_for_resume(state, direct, algorithm)
    with pytest.raises(RuntimeError, match="environment_variant mismatch"):
        validate_checkpoint_for_resume(state, persistent, algorithm)
    with pytest.raises(RuntimeError, match="environment_variant mismatch"):
        validate_checkpoint_for_evaluation(state, persistent, algorithm)
    validate_checkpoint_for_evaluation(
        state, persistent, algorithm, allow_cross_variant=True
    )


def test_cross_evaluation_never_relaxes_version_impl_or_dimensions():
    persistent = load_yaml("persistent_wave_v2_environment.yaml")
    algorithm = load_yaml("mappo.yaml")
    state = checkpoint_state("direct_v2_3")
    state["extra"]["environment_version"] = "2.2"
    with pytest.raises(RuntimeError, match="environment_version mismatch"):
        validate_checkpoint_for_evaluation(
            state, persistent, algorithm, allow_cross_variant=True
        )
    state = checkpoint_state("direct_v2_3")
    state["mappo_impl_version"] = MAPPO_IMPL_VERSION + 1
    with pytest.raises(RuntimeError, match="implementation mismatch"):
        validate_checkpoint_for_evaluation(
            state, persistent, algorithm, allow_cross_variant=True
        )
    state = checkpoint_state("direct_v2_3")
    state["extra"]["observation_dim"] = 51
    with pytest.raises(RuntimeError, match="observation_dim mismatch"):
        validate_checkpoint_for_evaluation(
            state, persistent, algorithm, allow_cross_variant=True
        )


def test_reusable_evaluation_metadata(tmp_path, monkeypatch):
    checkpoint = tmp_path / "direct.pt"
    torch.save(checkpoint_state("direct_v2_3"), checkpoint)

    class DummyTrainer:
        def load(self, path):
            assert Path(path) == checkpoint

    monkeypatch.setattr(
        "algorithm.mappo.evaluation.build_mappo_trainer",
        lambda config, device: DummyTrainer(),
    )
    monkeypatch.setattr(
        "algorithm.mappo.evaluation.evaluate",
        lambda trainer, config, seeds: {
            "average_return": 3.5,
            "evaluation_episodes": len(seeds),
        },
    )
    with pytest.warns(RuntimeWarning, match="diagnostic only"):
        result = evaluate_mappo_checkpoint(
            checkpoint,
            load_yaml("mappo.yaml"),
            load_yaml("persistent_wave_v2_environment.yaml"),
            "cpu",
            [20_000_000, 20_000_001],
            allow_cross_variant=True,
        )
    assert result["checkpoint_environment_variant"] == "direct_v2_3"
    assert result["evaluation_environment_variant"] == "persistent_wave_v2"
    assert result["cross_variant_evaluation"] is True
    assert result["holdout_seed_base"] == 20_000_000
    assert result["holdout_seed_end"] == 20_000_001
    assert result["evaluation_episodes"] == 2
    assert result["protocol_complete"] is False


def test_policy_matrix_writes_four_cells_with_identical_seeds(tmp_path, monkeypatch):
    calls = []

    def fake_evaluate(checkpoint, algorithm, environment, device, seeds,
                      allow_cross_variant=False):
        seeds = list(seeds)
        calls.append(seeds)
        source = "direct_v2_3" if "direct" in Path(checkpoint).name else "persistent_wave_v2"
        target = environment.get("environment_variant", "direct_v2_3")
        return {
            "algorithm": "MAPPO",
            "checkpoint": str(checkpoint),
            "checkpoint_environment_variant": source,
            "evaluation_environment_variant": target,
            "cross_variant_evaluation": source != target,
            "evaluation_episodes": len(seeds),
            "holdout_seed_base": seeds[0],
            "holdout_seed_end": seeds[-1],
            "average_return": 1.0,
        }

    monkeypatch.setattr(
        "tools.evaluate_policy_matrix.evaluate_mappo_checkpoint", fake_evaluate
    )
    monkeypatch.setattr(
        "tools.evaluate_policy_matrix.validate_matrix_source",
        lambda role, checkpoint, algorithm, environment: {
            "training_gamma": 0.99,
            "algorithm_config_sha256": "a",
            "environment_config_sha256": "e",
            "sampled_steps": 4,
            "protocol_complete": True,
        },
    )
    results = evaluate_policy_matrix(
        tmp_path / "direct.pt",
        tmp_path / "persistent.pt",
        {}, {},
        {"environment_variant": "direct_v2_3"},
        {"environment_variant": "persistent_wave_v2"},
        [101, 102], "cpu", tmp_path / "matrix",
    )
    assert len(results) == 4
    assert calls == [[101, 102]] * 4
    assert [row["cross_variant_evaluation"] for row in results.values()] == [
        False, True, True, False
    ]
    expected = {
        "direct_to_direct.json", "direct_to_persistent.json",
        "persistent_to_direct.json", "persistent_to_persistent.json",
        "matrix_summary.csv", "matrix_summary.json", "evaluation_manifest.json",
    }
    assert expected <= {path.name for path in (tmp_path / "matrix").iterdir()}


def test_policy_matrix_rejects_swapped_checkpoint_roles(tmp_path):
    direct_environment = load_yaml("combat_environment.yaml")
    persistent_environment = load_yaml("persistent_wave_v2_environment.yaml")
    direct_algorithm = load_yaml("mappo.yaml")
    persistent_algorithm = load_yaml("mappo_persistent_wave.yaml")
    direct_checkpoint = tmp_path / "direct.pt"
    persistent_checkpoint = tmp_path / "persistent.pt"
    torch.save(
        complete_checkpoint_state(direct_environment, direct_algorithm),
        direct_checkpoint,
    )
    torch.save(
        complete_checkpoint_state(persistent_environment, persistent_algorithm),
        persistent_checkpoint,
    )
    with pytest.raises(RuntimeError, match="environment_variant mismatch"):
        evaluate_policy_matrix(
            persistent_checkpoint, direct_checkpoint,
            direct_algorithm, persistent_algorithm,
            direct_environment, persistent_environment,
            [101], "cpu", tmp_path / "swapped",
        )


def test_policy_matrix_manifest_preserves_source_gamma_identity(tmp_path, monkeypatch):
    direct_environment = load_yaml("combat_environment.yaml")
    persistent_environment = load_yaml("persistent_wave_v2_environment.yaml")
    direct_algorithm = load_yaml("mappo.yaml")
    persistent_algorithm = load_yaml("mappo_persistent_wave.yaml")
    direct_checkpoint = tmp_path / "direct.pt"
    persistent_checkpoint = tmp_path / "persistent.pt"
    torch.save(complete_checkpoint_state(direct_environment, direct_algorithm), direct_checkpoint)
    torch.save(complete_checkpoint_state(persistent_environment, persistent_algorithm), persistent_checkpoint)

    def fake_evaluate(checkpoint, algorithm, environment, device, seeds,
                      allow_cross_variant=False):
        source = torch.load(checkpoint, map_location="cpu", weights_only=False)["extra"]
        target = environment.get("environment_variant", "direct_v2_3")
        return {
            "algorithm": "MAPPO", "checkpoint": str(checkpoint),
            "checkpoint_environment_variant": source["environment_variant"],
            "evaluation_environment_variant": target,
            "cross_variant_evaluation": source["environment_variant"] != target,
            "evaluation_episodes": len(seeds), "holdout_seed_base": seeds[0],
            "holdout_seed_end": seeds[-1], "average_return": 0.0,
        }

    monkeypatch.setattr(
        "tools.evaluate_policy_matrix.evaluate_mappo_checkpoint", fake_evaluate
    )
    output = tmp_path / "matrix"
    evaluate_policy_matrix(
        direct_checkpoint, persistent_checkpoint,
        direct_algorithm, persistent_algorithm,
        direct_environment, persistent_environment,
        [101], "cpu", output,
    )
    manifest = json.loads((output / "evaluation_manifest.json").read_text())
    assert manifest["direct_training_gamma"] == pytest.approx(0.99)
    assert manifest["persistent_training_gamma"] == pytest.approx(0.999)
    assert manifest["direct_protocol_complete"] is True
    assert manifest["persistent_protocol_complete"] is True


def test_fresh_output_directory_safety(tmp_path):
    missing = tmp_path / "missing"
    ensure_fresh_output_directory(missing)
    assert missing.is_dir()
    ensure_fresh_output_directory(missing)
    (missing / "occupied.txt").write_text("x", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty"):
        ensure_fresh_output_directory(missing)


def test_fresh_rejection_occurs_before_runner_creation(tmp_path, monkeypatch):
    import algorithm.train_mappo as entry

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        entry, "MAPPOTrainingRunner",
        lambda *args, **kwargs: pytest.fail("runner/spawn must not be created"),
    )
    monkeypatch.setattr(sys, "argv", [
        "train_mappo.py", "--smoke", "--output-dir", str(occupied),
    ])
    with pytest.raises(RuntimeError, match="non-empty"):
        entry.main()


def test_resume_output_resolution_and_snapshot_protection(tmp_path):
    checkpoint = tmp_path / "run" / "latest.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    output, resume = resolve_run_paths(None, str(checkpoint), 7)
    assert output == checkpoint.parent.resolve()
    assert resume == checkpoint.resolve()
    output, _ = resolve_run_paths(str(checkpoint.parent), str(checkpoint), 7)
    assert output == checkpoint.parent.resolve()
    with pytest.raises(RuntimeError, match="checkpoint.parent"):
        resolve_run_paths(str(tmp_path / "other"), str(checkpoint), 7)

    env_config = load_yaml("combat_environment.yaml")
    algorithm_config = load_yaml("mappo.yaml")
    (checkpoint.parent / "env_config.yaml").write_text(
        yaml.safe_dump(env_config), encoding="utf-8"
    )
    (checkpoint.parent / "algorithm_config.yaml").write_text(
        yaml.safe_dump(algorithm_config), encoding="utf-8"
    )
    assert validate_resume_config_snapshots(
        checkpoint.parent, env_config, algorithm_config
    ) == []
    changed_env = dict(env_config)
    changed_env["environment_variant"] = "persistent_wave_v2"
    with pytest.raises(RuntimeError, match="env_config.yaml mismatch"):
        validate_resume_config_snapshots(
            checkpoint.parent, changed_env, algorithm_config
        )
    changed_algorithm = json.loads(json.dumps(algorithm_config))
    changed_algorithm["training"]["gamma"] = 0.5
    with pytest.raises(RuntimeError, match="algorithm_config.yaml mismatch"):
        validate_resume_config_snapshots(
            checkpoint.parent, env_config, changed_algorithm
        )


def test_resume_rollback_backs_up_and_truncates_future_records(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "checkpoint_4.pt"
    torch.save({"sampled_steps": 4}, checkpoint)
    for name in ("training_metrics.jsonl", "optimization_metrics.jsonl"):
        (run_dir / name).write_text(
            '\n'.join(json.dumps({"sampled_steps": step, "x": step})
                      for step in (2, 4, 6)) + '\n', encoding="utf-8"
        )
    with (run_dir / "evaluation_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["sampled_steps", "score"])
        writer.writeheader()
        writer.writerows([
            {"sampled_steps": 4, "score": 1},
            {"sampled_steps": 6, "score": 2},
        ])
    torch.save({"sampled_steps": 6}, run_dir / "best_eval.pt")
    (run_dir / "run_summary.json").write_text("{}", encoding="utf-8")
    result = prepare_resume_rollback(run_dir, checkpoint, 4)
    assert result["rollback_performed"] is True
    assert result["rollback_from_max_logged_steps"] == 6
    for name in ("training_metrics.jsonl", "optimization_metrics.jsonl"):
        records = [json.loads(line) for line in (run_dir / name).read_text().splitlines()]
        assert [record["sampled_steps"] for record in records] == [2, 4]
        assert list(run_dir.glob(f"{Path(name).stem}.pre_resume_*.jsonl"))
    assert not (run_dir / "best_eval.pt").exists()
    assert list(run_dir.glob("best_eval.pre_resume_*.pt"))
    assert list(run_dir.glob("run_summary.pre_resume_*.json"))


def test_stale_resume_rejected_when_newer_regular_checkpoint_exists(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    selected = run_dir / "checkpoint_4.pt"
    torch.save({"sampled_steps": 4}, selected)
    torch.save({"sampled_steps": 8}, run_dir / "checkpoint_8.pt")
    with pytest.raises(RuntimeError, match="stale resume checkpoint"):
        reject_stale_resume_checkpoint(run_dir, selected)


def write_history(path: Path, rows: list[dict], seed: int = 2023) -> None:
    path.mkdir()
    environment = load_yaml("combat_environment.yaml")
    algorithm = load_yaml("mappo.yaml")
    (path / "env_config.yaml").write_text(
        yaml.safe_dump(environment), encoding="utf-8"
    )
    (path / "algorithm_config.yaml").write_text(
        yaml.safe_dump(algorithm), encoding="utf-8"
    )
    (path / "run_config.json").write_text(json.dumps({
        "algorithm": "MAPPO", "seed": seed, "device": "cpu",
        "num_envs": 1, "total_sampled_steps": 200, "smoke": True,
        "effective_hidden_dim": 64,
    }), encoding="utf-8")
    with (path / "evaluation_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("run_count", [3, 5, 7])
def test_training_history_aggregation_for_arbitrary_run_count(tmp_path, run_count):
    run_dirs = []
    for index in range(run_count):
        rows = [
            {"sampled_steps": 100, "average_return": index + 1,
             "win_rate": index / 10, "only_first": 9 if index == 0 else ""},
            {"sampled_steps": 200, "average_return": index + 2,
             "win_rate": index / 10, "only_first": 9 if index == 0 else ""},
        ]
        run_dir = tmp_path / f"run_{index}"
        write_history(run_dir, rows, seed=2023 + index)
        run_dirs.append(run_dir)
    _, manifest = aggregate_training_histories(run_dirs, tmp_path / "summary")
    assert manifest["number_of_runs"] == run_count
    assert manifest["aggregated_metrics"] == ["average_return", "win_rate"]
    assert "only_first" in manifest["skipped_columns"]
    assert (tmp_path / "summary/training_curve_summary.csv").is_file()


@pytest.mark.parametrize(
    "mutation,field",
    [
        ("gamma", "training_gamma"),
        ("environment", "environment_config_sha256"),
        ("budget", "total_sampled_steps"),
        ("duplicate_seed", "unique training_seed"),
    ],
)
def test_training_aggregation_rejects_protocol_mismatch(tmp_path, mutation, field):
    first, second = tmp_path / "first", tmp_path / "second"
    rows = [{"sampled_steps": 100, "average_return": 1.0}]
    write_history(first, rows, seed=2023)
    write_history(second, rows, seed=2024)
    if mutation == "gamma":
        config = yaml.safe_load((second / "algorithm_config.yaml").read_text())
        config["training"]["gamma"] = 0.999
        (second / "algorithm_config.yaml").write_text(yaml.safe_dump(config))
    elif mutation == "environment":
        config = yaml.safe_load((second / "env_config.yaml").read_text())
        config["simulation"]["max_steps"] += 1
        (second / "env_config.yaml").write_text(yaml.safe_dump(config))
    else:
        config = json.loads((second / "run_config.json").read_text())
        if mutation == "budget":
            config["total_sampled_steps"] = 201
        else:
            config["seed"] = 2023
        (second / "run_config.json").write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match=field):
        aggregate_training_histories([first, second], tmp_path / "summary")


def test_student_t_ci_and_no_common_training_step(tmp_path):
    stats = summarize_values([1.0, 2.0, 3.0])
    assert stats["mean"] == 2.0
    assert stats["std"] == pytest.approx(1.0)
    assert stats["sem"] == pytest.approx(1 / 3 ** 0.5)
    assert stats["ci95_lower"] == pytest.approx(2 - 4.303 / 3 ** 0.5)
    first, second = tmp_path / "first", tmp_path / "second"
    write_history(first, [{"sampled_steps": 100, "score": 1}], seed=1)
    write_history(second, [{"sampled_steps": 200, "score": 2}], seed=2)
    with pytest.raises(RuntimeError, match="no common sampled_steps"):
        aggregate_training_histories([first, second], tmp_path / "none")


def holdout_result(**overrides) -> dict:
    result = {
        "algorithm": "MAPPO",
        "checkpoint": "model.pt",
        "checkpoint_environment_version": "2.3",
        "checkpoint_environment_variant": "direct_v2_3",
        "evaluation_environment_version": "2.3",
        "evaluation_environment_variant": "persistent_wave_v2",
        "cross_variant_evaluation": True,
        "mappo_impl_version": 2,
        "holdout_seed_base": 20_000_000,
        "holdout_seed_end": 20_000_199,
        "evaluation_episodes": 200,
        "observation_dim": 52,
        "action_dim": 3,
        "num_agents": 4,
        "checkpoint_sampled_steps": 3_000_000,
        "checkpoint_training_seed": None,
        "checkpoint_training_gamma": 0.99,
        "checkpoint_training_num_envs": 24,
        "checkpoint_training_total_sampled_steps": 3_000_000,
        "checkpoint_training_smoke": False,
        "checkpoint_effective_hidden_dim": 256,
        "checkpoint_environment_config_sha256": "env-source",
        "checkpoint_algorithm_config_sha256": "algorithm",
        "evaluation_environment_config_sha256": "env-target",
        "provided_algorithm_config_sha256": "algorithm",
        "protocol_complete": True,
        "average_return": 10.0,
        "average_red_loss": 2.0,
    }
    result.update(overrides)
    return result


def write_holdouts(tmp_path: Path, results: list[dict]) -> list[Path]:
    paths = []
    for index, result in enumerate(results):
        result = dict(result)
        if result.get("checkpoint_training_seed") is None:
            result["checkpoint_training_seed"] = 2023 + index
        path = tmp_path / f"holdout_{index}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        paths.append(path)
    return paths


def test_holdout_aggregation_and_metadata_exclusion(tmp_path):
    paths = write_holdouts(tmp_path, [
        holdout_result(average_return=9.0),
        holdout_result(average_return=10.0),
        holdout_result(average_return=11.0),
    ])
    summary, manifest = aggregate_holdout_results(paths, tmp_path / "summary")
    assert summary["metrics"]["average_return"]["mean"] == 10.0
    assert "evaluation_episodes" not in manifest["aggregated_metrics"]
    assert (tmp_path / "summary/holdout_summary.csv").is_file()
    assert (tmp_path / "summary/holdout_summary.json").is_file()


@pytest.mark.parametrize(
    "field,value",
    [
        ("algorithm", "OTHER"),
        ("checkpoint_environment_variant", "persistent_wave_v2"),
        ("evaluation_environment_variant", "direct_v2_3"),
        ("holdout_seed_base", 30_000_000),
        ("holdout_seed_end", 30_000_199),
        ("evaluation_episodes", 100),
        ("checkpoint_training_gamma", 0.999),
        ("checkpoint_algorithm_config_sha256", "different-algorithm"),
        ("evaluation_environment_config_sha256", "different-environment"),
    ],
)
def test_holdout_protocol_mismatch_rejected(tmp_path, field, value):
    paths = write_holdouts(tmp_path, [holdout_result(), holdout_result(**{field: value})])
    with pytest.raises(RuntimeError, match=field):
        aggregate_holdout_results(paths, tmp_path / "summary")


def test_holdout_rejects_incomplete_and_duplicate_training_seeds(tmp_path):
    incomplete = write_holdouts(tmp_path, [
        holdout_result(protocol_complete=False), holdout_result(),
    ])
    with pytest.raises(RuntimeError, match="incomplete"):
        aggregate_holdout_results(incomplete, tmp_path / "incomplete")
    duplicate_dir = tmp_path / "duplicates"
    duplicate_dir.mkdir()
    duplicate = write_holdouts(duplicate_dir, [
        holdout_result(checkpoint_training_seed=2023),
        holdout_result(checkpoint_training_seed=2023),
    ])
    with pytest.raises(RuntimeError, match="unique checkpoint_training_seed"):
        aggregate_holdout_results(duplicate, tmp_path / "duplicate_summary")


def test_evaluate_cli_runs_outside_project_and_requires_explicit_cross_flag(tmp_path):
    checkpoint = tmp_path / "direct.pt"
    torch.save(checkpoint_state("direct_v2_3"), checkpoint)
    direct_env = load_yaml("combat_environment.yaml")
    direct_env["simulation"]["max_steps"] = 1
    persistent_env = load_yaml("persistent_wave_v2_environment.yaml")
    persistent_env["simulation"]["max_steps"] = 1
    direct_path = tmp_path / "direct_env.yaml"
    persistent_path = tmp_path / "persistent_env.yaml"
    direct_path.write_text(yaml.safe_dump(direct_env), encoding="utf-8")
    persistent_path.write_text(yaml.safe_dump(persistent_env), encoding="utf-8")
    script = PROJECT_ROOT / "algorithm/evaluate_mappo.py"
    process_env = os.environ.copy()
    process_env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="uav_eval_cwd_") as cwd:
        common = [
            sys.executable, str(script), "--checkpoint", str(checkpoint),
            "--algorithm-config", "configs/mappo.yaml", "--seed-base", "20000000",
            "--episodes", "1", "--device", "cpu",
        ]
        same_output = tmp_path / "same.json"
        same = subprocess.run(
            common + ["--env-config", str(direct_path), "--output", str(same_output)],
            cwd=cwd, env=process_env, capture_output=True, text=True, timeout=120,
        )
        assert same.returncode == 0, same.stdout + same.stderr
        assert json.loads(same_output.read_text())["cross_variant_evaluation"] is False
        strict = subprocess.run(
            common + ["--env-config", str(persistent_path), "--output", str(tmp_path / "strict.json")],
            cwd=cwd, env=process_env, capture_output=True, text=True, timeout=120,
        )
        assert strict.returncode != 0
        assert "environment_variant mismatch" in strict.stderr
        cross_output = tmp_path / "cross.json"
        cross = subprocess.run(
            common + ["--env-config", str(persistent_path), "--output", str(cross_output),
                      "--allow-cross-variant"],
            cwd=cwd, env=process_env, capture_output=True, text=True, timeout=120,
        )
        assert cross.returncode == 0, cross.stdout + cross.stderr
        result = json.loads(cross_output.read_text())
        assert result["cross_variant_evaluation"] is True
        assert result["checkpoint_environment_variant"] == "direct_v2_3"
        assert result["evaluation_environment_variant"] == "persistent_wave_v2"
