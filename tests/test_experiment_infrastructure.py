"""Combat checkpoint, resume safety and aggregation regressions."""
import copy
import csv
import json
import sys
from pathlib import Path
import pytest
import torch
import yaml
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation, validate_checkpoint_for_resume
from algorithm.common.protocol import config_sha256
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.train_mappo import (prepare_resume_rollback, reject_stale_resume_checkpoint,
    ensure_fresh_output_directory)
from tools.aggregate_training_runs import aggregate_training_histories, summarize_values
from tools.aggregate_holdout_results import aggregate_holdout_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_yaml(name):
    return yaml.safe_load((PROJECT_ROOT / 'configs' / name).read_text(encoding='utf-8'))


def test_checkpoint_protocol_rejects_version_dimensions_and_fingerprint(tmp_path):
    environment, algorithm = load_yaml('combat_environment.yaml'), load_yaml('mappo.yaml')
    trainer = build_mappo_trainer(algorithm, 'cuda', hidden_dim=64)
    state = trainer.checkpoint_state({'environment_version': '2.3', 'observation_dim': 52,
        'action_dim': 3, 'num_agents': 4, 'environment_config_sha256': config_sha256(environment),
        'algorithm_config_sha256': config_sha256(algorithm)})
    validate_checkpoint_for_resume(state, environment, algorithm)
    validate_checkpoint_for_evaluation(state, environment, algorithm)
    for key, value, message in [('environment_version', '0', 'environment_version'),
                                ('observation_dim', 53, 'observation_dim'),
                                ('algorithm_config_sha256', 'wrong', 'sha256')]:
        changed = copy.deepcopy(state); changed['extra'][key] = value
        with pytest.raises(RuntimeError, match=message):
            validate_checkpoint_for_resume(changed, environment, algorithm)


def test_training_aggregation_requires_same_protocol_and_distinct_seeds(tmp_path):
    environment, algorithm = load_yaml('combat_environment.yaml'), load_yaml('mappo.yaml')
    runs = []
    for seed, value in [(1, 2.), (2, 4.), (3, 6.)]:
        run = tmp_path / str(seed); run.mkdir(); runs.append(run)
        (run / 'env_config.yaml').write_text(yaml.safe_dump(environment))
        (run / 'algorithm_config.yaml').write_text(yaml.safe_dump(algorithm))
        (run / 'run_config.json').write_text(json.dumps({'algorithm': 'MAPPO', 'seed': seed,
            'num_envs': 1, 'total_sampled_steps': 32, 'smoke': True,
            'effective_hidden_dim': 64, 'device': 'cuda'}))
        (run / 'evaluation_history.csv').write_text(f'sampled_steps,red_win_rate,episode_return\n16,0.5,{value}\n')
    rows, manifest = aggregate_training_histories(runs, tmp_path / 'aggregation')
    assert manifest['number_of_runs'] == 3
    assert next(row for row in rows if row['metric'] == 'episode_return')['mean'] == 4
    changed = copy.deepcopy(environment); changed['simulation']['max_steps'] = 99
    (runs[0] / 'env_config.yaml').write_text(yaml.safe_dump(changed))
    with pytest.raises(RuntimeError, match='protocol mismatch'):
        aggregate_training_histories(runs, tmp_path / 'invalid')
    assert summarize_values([2., 4.])['std'] == pytest.approx(2 ** .5)


def test_holdout_aggregation_checks_seed_policy_and_implementation(tmp_path):
    inputs = []
    for seed in (1, 2):
        path = tmp_path / f'{seed}.json'; inputs.append(path)
        path.write_text(json.dumps({'algorithm': 'madsac', 'protocol_complete': True,
            'checkpoint_training_seed': seed, 'implementation_version': 1,
            'mode': 'stochastic', 'policy_seed': 77, 'evaluation_episodes': 2,
            'holdout_seed_base': 100, 'holdout_seed_end': 101,
            'red_win_rate': .5, 'episode_return': float(seed)}))
    summary, manifest = aggregate_holdout_results(inputs, tmp_path / 'aggregate')
    assert summary['metrics']['episode_return']['mean'] == 1.5
    assert 'policy_seed' not in manifest['aggregated_metrics']
    changed = json.loads(inputs[0].read_text()); changed['policy_seed'] = 78
    inputs[0].write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match='policy_seed'):
        aggregate_holdout_results(inputs, tmp_path / 'invalid')

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
