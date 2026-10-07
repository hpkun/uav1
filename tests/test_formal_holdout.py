import json
from pathlib import Path

import pytest

from tools.prepare_formal_holdout import (
    EPISODES_PER_POLICY, FORMAL_SEED_END, FORMAL_SEED_START, METHODS,
    PRIMARY_METRICS, TRAINING_SEEDS, build_manifest, discover_formal_runs,
    formal_episode_seeds, validate_manifest_files, validate_manifest_schema,
    validate_matched_pair,
)
from tools.run_formal_holdout import (
    build_tasks, formal_conclusion, initial_state, mean_std_table,
    prepare_output_directory, progress_path, seed_level_deltas, summarize_policy,
    validate_task_progress,
)


@pytest.fixture(scope="module")
def manifest():
    return build_manifest()


def test_discovery_allows_only_alloff_and_m5(manifest):
    assert {run["method"] for run in manifest["runs"]} == set(METHODS)


def test_discovery_requires_exactly_three_training_seeds():
    runs = discover_formal_runs()
    assert {seed for _, seed in runs} == set(TRAINING_SEEDS)
    assert len(runs) == 6


@pytest.mark.parametrize("seed", TRAINING_SEEDS)
def test_alloff_and_m5_are_strictly_matched(seed):
    runs = discover_formal_runs()
    result = validate_matched_pair(runs[("All-Off", seed)], runs[("M5 Wave Balance", seed)])
    assert result["matched"] is True
    assert result["allowed_difference"] == "modules.wave_balancing.enabled"


def test_primary_protocol_is_fixed_to_best(manifest):
    tasks = build_tasks(manifest)
    assert manifest["primary_checkpoint_protocol"] == "best_eval.pt"
    assert all(task["checkpoint"].endswith("best_eval.pt") for task in tasks if task["checkpoint_role"] == "best")


def test_secondary_protocol_is_fixed_to_latest(manifest):
    tasks = build_tasks(manifest)
    assert manifest["secondary_checkpoint_protocol"] == "latest.pt"
    assert all(task["checkpoint"].endswith("latest.pt") for task in tasks if task["checkpoint_role"] == "latest")


def test_formal_seed_range_is_exactly_locked():
    seeds = formal_episode_seeds()
    assert seeds == list(range(20_000_000, 20_000_200))
    assert seeds[0] == FORMAL_SEED_START and seeds[-1] == FORMAL_SEED_END


def test_episode_count_is_exactly_200_per_policy(manifest):
    tasks = build_tasks(manifest)
    assert len(tasks) == 12
    assert all(len(task["episode_seeds"]) == EPISODES_PER_POLICY for task in tasks)
    assert sum(len(task["episode_seeds"]) for task in tasks) == 2400


def test_method_list_cannot_be_modified(manifest):
    changed = dict(manifest, methods=[*METHODS, "forbidden"])
    with pytest.raises(RuntimeError, match="frozen formal protocol mismatch"):
        validate_manifest_schema(changed)


def test_primary_metric_order_cannot_be_modified(manifest):
    changed = dict(manifest, primary_metric_order=list(reversed(PRIMARY_METRICS)))
    with pytest.raises(RuntimeError, match="frozen formal protocol mismatch"):
        validate_manifest_schema(changed)


def test_checkpoint_sha_change_is_rejected(manifest):
    changed = json.loads(json.dumps(manifest))
    changed["runs"][0]["best_checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="locked file changed"):
        validate_manifest_files(changed)


def test_manifest_change_is_rejected_by_resume_exact(tmp_path, manifest):
    tasks = build_tasks(manifest, smoke=True)
    prepare_output_directory(tmp_path, manifest, "a" * 64, tasks, False, True)
    with pytest.raises(RuntimeError, match="manifest_sha256"):
        prepare_output_directory(tmp_path, manifest, "b" * 64, tasks, True, True)


def test_existing_results_default_to_refusal(tmp_path, manifest):
    (tmp_path / "partial.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already contains data"):
        prepare_output_directory(tmp_path, manifest, "a" * 64, build_tasks(manifest, smoke=True), False, True)


def test_resume_exact_accepts_only_identical_protocol(tmp_path, manifest):
    tasks = build_tasks(manifest, smoke=True)
    state = prepare_output_directory(tmp_path, manifest, "a" * 64, tasks, False, True)
    resumed = prepare_output_directory(tmp_path, manifest, "a" * 64, tasks, True, True)
    assert resumed["task_fingerprints"] == state["task_fingerprints"]


def test_deleted_committed_episode_is_never_selectively_rerun(tmp_path, manifest):
    task = build_tasks(manifest, smoke=True)[0]
    ledger = progress_path(tmp_path, task)
    ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({"episode_seed":99_000_000,"cache_sha256":"0" * 64}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="selective rerun refused"):
        validate_task_progress(tmp_path, task)


def _episode(reached_w2=0, reached_w3=0):
    return {
        "method": "All-Off", "training_seed": 2023, "checkpoint_role": "best",
        "checkpoint_step": 1, "W1_clear": 0, "W2_clear": 0, "W3_clear": 0,
        "waves_cleared": 0, "episode_return": 0, "red_losses": 0,
        "blue_losses": 0, "red_boundary_exits": 0, "red_ground_losses": 0,
        "timeout": 0, "episode_length": 10, "reached_W2": reached_w2,
        "reached_W3": reached_w3, "time_to_clear_W1": None,
        "time_to_clear_W2": None, "time_spent_in_W3": None,
    }


def test_conditional_timeout_unreached_wave_is_none():
    summary = summarize_policy([_episode()])
    assert summary["timeout_conditioned_reached_W2"] is None
    assert summary["timeout_conditioned_reached_W3"] is None
    assert summary["mean_time_to_clear_W1"] is None


def _summaries():
    rows = []
    for role in ("best", "latest"):
        for seed_index, seed in enumerate(TRAINING_SEEDS):
            for method_index, method in enumerate(METHODS):
                base = seed_index + method_index
                rows.append({
                    "method": method, "training_seed": seed, "checkpoint_role": role,
                    "W1": base, "W2": base, "W3": base, "average_waves": base,
                    "return": base, "red_loss": 3 - base, "blue_loss": base,
                    "K_L": base, "boundary": 0, "ground": 0, "timeout": 0,
                })
    return rows


def test_training_seed_is_statistical_unit_n3():
    rows = seed_level_deltas(_summaries(), "best")
    aggregate = [row for row in rows if row["row_type"] == "aggregate"]
    assert aggregate and all(row["n_training_seeds"] == 3 for row in aggregate)
    assert all("n=3" in row["ci_note"] for row in aggregate)


def test_mean_std_uses_three_policy_summaries_not_episodes():
    table = mean_std_table(_summaries())
    assert len(table) == 2
    assert all(row["n_training_seeds"] == 3 for row in table)


def test_smoke_uses_only_nonformal_seed(manifest):
    tasks = build_tasks(manifest, smoke=True)
    assert all(task["episode_seeds"] == [99_000_000] for task in tasks)
    assert not any(FORMAL_SEED_START <= seed <= FORMAL_SEED_END for task in tasks for seed in task["episode_seeds"])


def test_frozen_conclusion_rule_uses_training_seed_directions():
    assert formal_conclusion(seed_level_deltas(_summaries(), "best")) == "M5_FORMAL_HOLDOUT_SUPPORTED"
