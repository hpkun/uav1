"""Focused, non-performance tests for the baseline learnability protocol."""
from __future__ import annotations

import hashlib
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from algorithm.modules.actor_lr_decay import ActorLRDecayModule
from algorithm.modules.curriculum import CurriculumController
from algorithm.common.protocol import config_sha256
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner,validate_runtime_environment_contract
from algorithm.train_modular_mappo import load_config
from env.factory import make_combat_environment
from env.fixed_policy import GroundAwareNearestTargetPursuitPolicy
from env.models import AircraftState
from tools.analyze_mappo_baseline_learnability import (
    audit_actual_environment, completed_training_episodes, diagnostic_labels, enriched, safe_ratio,
    select_endpoint,
)
from tools.preflight_mappo_baseline_learnability import (
    ENV_PATHS, freshness_scan, load_yaml, normalized_env, retired_33m_evidence_ok,
    validate, validate_configs, validate_seed_registry,
)

ROOT = Path(__file__).resolve().parents[1]


def state(x=0.0, alive=True):
    return AircraftState(x=x, y=0.0, z=-3000.0, v=225.0, psi=0.0, theta=0.0, alive=alive)


def test_protocol_action_environment_and_modules_are_frozen():
    result = validate_configs()
    algorithm = result["algorithm"]
    assert algorithm["network"]["observation_dim"] == 52
    assert algorithm["network"]["action_dim"] == 3
    assert [k for k, v in algorithm["modules"].items() if v.get("enabled", False)] == ["actor_lr_decay"]
    assert hashlib.sha256((ROOT / "env/fixed_policy.py").read_bytes()).hexdigest() == result["manifest"]["frozen_contract"]["blue_policy_source_sha256"]


def test_ladder_diff_and_first_wave_are_exact_matched():
    envs = {k: load_yaml(v) for k, v in ENV_PATHS.items()}
    assert [(envs[k]["persistent_waves"]["total_waves"], envs[k]["simulation"]["max_steps"]) for k in ("L1", "L2", "L3")] == [(1, 1000), (2, 2000), (3, 3000)]
    assert normalized_env(envs["L1"]) == normalized_env(envs["L2"]) == normalized_env(envs["L3"])
    instances = [make_combat_environment(envs[k]) for k in ("L1", "L2", "L3")]
    observations = [env.reset(7_650_321) for env in instances]
    assert np.array_equal(observations[0][0], observations[1][0])
    assert np.array_equal(observations[0][0], observations[2][0])
    for attr in ("red", "blue"):
        arrays = [np.stack([s.as_array() for s in getattr(env, attr)]) for env in instances]
        assert np.array_equal(arrays[0], arrays[1]) and np.array_equal(arrays[0], arrays[2])
    assert envs["L1"]["weapon"] == envs["L2"]["weapon"] == envs["L3"]["weapon"]
    assert envs["L1"]["reward"] == envs["L2"]["reward"] == envs["L3"]["reward"]
    assert envs["L1"]["blue_policy"] == envs["L2"]["blue_policy"] == envs["L3"]["blue_policy"]


def test_ladder_effective_runtime_matches_declared_with_disabled_curriculum():
    result = validate_configs(); controller = CurriculumController(result["algorithm"]["modules"]["curriculum"])
    for condition, expected in {"L1": (1,1000), "L2": (2,2000), "L3": (3,3000)}.items():
        declared = result["envs"][condition]
        effective = controller.runtime_config(declared, 1_500_000)
        contract = validate_runtime_environment_contract(declared,effective,declared,curriculum_enabled=False)
        assert (contract["effective_training"]["total_waves"],contract["effective_training"]["max_steps"]) == expected
        assert contract["declared"]["config_sha256"] == contract["effective_training"]["config_sha256"]


def test_runner_disabled_curriculum_uses_effective_waves_and_never_rebuilds(monkeypatch,tmp_path):
    builds=[]
    def fake_make_vector(self,episode_indices=None):
        builds.append(1)
        self.vector=SimpleNamespace(episode_indices=np.zeros(self.num_envs,dtype=np.int64),close=lambda:None)
        self.observations=np.zeros((self.num_envs,4,52),np.float32)
        self.alive=np.ones((self.num_envs,4),np.float32);self.blue_alive=self.alive.copy()
        self.wave=np.ones(self.num_envs,np.int64);self.total=np.full(self.num_envs,self.current_waves,np.int64)
        self.episode_steps=np.zeros(self.num_envs,np.int64);self.episode_mask=np.zeros(self.num_envs,np.float32)
        self.actor_hidden,self.critic_hidden=self.trainer.initial_hidden(self.num_envs)
    monkeypatch.setattr(ModularMAPPOTrainingRunner,"_make_vector",fake_make_vector)
    cfg=load_config(ROOT/"configs/diag_mappo_learnability_common_3m.yaml")
    runner=ModularMAPPOTrainingRunner(load_yaml(ENV_PATHS["L1"]),cfg,1,8,"cpu",88_100_001,tmp_path,True)
    before=copy.deepcopy(runner.runtime_env_config);runner.trainer.sampled_steps=2_000_000
    runner._maybe_curriculum()
    assert runner.current_waves==1 and runner.runtime_env_config==before and len(builds)==1


def test_blue_reselects_nearest_alive_without_stale_target():
    cfg = load_yaml(ENV_PATHS["L3"])
    policy = GroundAwareNearestTargetPursuitPolicy(cfg["blue_policy"], cfg["action"], cfg["aircraft"])
    own, targets = state(), [state(100.0), state(200.0)]
    assert policy.nearest_target_index(own, targets) == 0
    targets[0].alive = False
    assert policy.nearest_target_index(own, targets) == 1
    targets[1].alive = False
    assert policy.nearest_target_index(own, targets) is None


def test_lr_matches_900k_protocol_and_holds_afterward():
    cfg = validate_configs()["algorithm"]
    decay = ActorLRDecayModule(cfg["modules"]["actor_lr_decay"])
    assert decay.learning_rate(600_000, 3e-4) == pytest.approx(3e-4)
    assert decay.learning_rate(750_000, 3e-4) == pytest.approx(2e-4)
    assert decay.learning_rate(900_000, 3e-4) == pytest.approx(1e-4)
    assert decay.learning_rate(3_000_000, 3e-4) == pytest.approx(1e-4)
    assert cfg["training"]["critic_learning_rate"] == pytest.approx(3e-4)


def test_l2_artificial_wave_respawn_uses_same_policy():
    env = make_combat_environment(load_yaml(ENV_PATHS["L2"])); env.reset(7_650_322)
    policy_id = id(env.fixed_policy)
    for aircraft in env.blue: aircraft.alive = False
    _, _, terminated, truncated, info = env.step(np.zeros((4, 3), np.float32))
    assert not terminated and not truncated and info["spawned_next_wave"]
    assert info["wave_index"] == 2 and id(env.fixed_policy) == policy_id
    assert all(aircraft.alive for aircraft in env.blue)


@pytest.mark.parametrize("condition,missing", [("L1", ("w2", "w3")), ("L2", ("w3",))])
def test_analyzer_preserves_structurally_missing_wave_fields(condition, missing):
    run = {"condition": condition, "training_seed": 1}
    row = {"sampled_steps": 900000, "clear_wave_1_probability": .5,
           "average_waves_cleared": .5, "average_return": 0,
           "average_red_loss": 1, "average_blue_loss": 2,
           "average_red_ground_losses": 0, "average_red_boundary_exits": 0,
           "average_episode_length": 100, "timeout_rate": 0}
    if condition == "L2": row["clear_wave_2_probability"] = .2
    result = enriched(run, row)
    assert all(result[key] is None for key in missing)
    assert result["w1"] == .5


def test_candidate_seeds_are_fresh_and_final_range_is_reassigned():
    scan = freshness_scan(checkpoints=False)["hits"]
    assert all(row["path"].replace("\\","/").startswith(("outputs/diag_mappo_learnability/",
                                                             "outputs/mappo_baseline_learnability_audit/",
                                                             "outputs/actor_mission_context_preflight/",
                                                             "outputs/dev_actor_mission_context_3m/",
                                                             "outputs/actor_mission_context_analysis/",
                                                             "outputs/mission_aware_film_preflight/",
                                                             "outputs/dev_mission_aware_film_3m/",
                                                             "outputs/mission_aware_film_analysis/"))
               for key in ("training","evaluation") for row in scan[key])
    assert scan["future_final"] == []
    assert any(row["value"] == 33_000_000 for row in scan["retired_33m"])
    registry = validate_configs()["registry"]["evaluation_ranges"]
    assert registry["33000000..33000199"]["status"] == "CONTAMINATED_RETIRED_FINAL_RANGE"
    assert registry["42000000..42000049"]["status"] == "EXPOSED"
    assert registry["44000000..44000049"]["status"] == "CURRENT_LEARNABILITY_DEVELOPMENT"
    assert registry["45000000..45000199"]["executed"] is False


def test_full_audit_accepts_historical_33m_evidence_with_portable_path():
    registry = copy.deepcopy(validate_configs()["registry"])
    registry["evaluation_ranges"]["33000000..33000199"]["evidence"] = (
        r"outputs\dev_ea_hwb_stable_cuda_smoke\post_resume_eval_2ep.json")
    freshness = {"hits": {"retired_33m": [
        {"path": "outputs/dev_ea_hwb_stable_cuda_smoke/post_resume_eval_2ep.json",
         "value": 33000000, "field": "metadata.evaluation_seed_base"},
        {"path": "outputs/dev_ea_hwb_stable_cuda_smoke/post_resume_eval_2ep.json",
         "value": 33000001, "field": "metadata.evaluation_seed_end"}]}}
    assert retired_33m_evidence_ok(freshness, registry)


def test_launch_check_allows_missing_retired_33m_archive(monkeypatch):
    configs = validate_configs()
    monkeypatch.setattr("tools.preflight_mappo_baseline_learnability.validate_configs", lambda: configs)
    monkeypatch.setattr("tools.preflight_mappo_baseline_learnability.freshness_scan",
                        lambda checkpoints: {"hits": {"training": [], "evaluation": [],
                            "retired_33m": [], "future_final": []}, "checkpoints_scanned": 0})
    monkeypatch.setattr("tools.preflight_mappo_baseline_learnability.write_audit", lambda result: None)
    result = validate(deep_freshness=False, smoke=False, launch_check=True)
    assert result["status"] == "READY_FOR_MAPPO_BASELINE_LEARNABILITY_DIAGNOSTIC"


def test_launch_check_rejects_wrong_retired_registry_status():
    registry = copy.deepcopy(validate_configs()["registry"])
    registry["evaluation_ranges"]["33000000..33000199"]["status"] = "UNTOUCHED"
    with pytest.raises(RuntimeError, match="33M is not retired"):
        validate_seed_registry(registry)


def test_launch_check_rejects_any_current_45m_evidence(monkeypatch):
    configs = validate_configs()
    monkeypatch.setattr("tools.preflight_mappo_baseline_learnability.validate_configs", lambda: configs)
    monkeypatch.setattr("tools.preflight_mappo_baseline_learnability.freshness_scan",
                        lambda checkpoints: {"hits": {"training": [], "evaluation": [],
                            "retired_33m": [], "future_final": [{"path": "outputs/current.json",
                            "value": 45000000, "field": "metadata.evaluation_seed_base"}]},
                            "checkpoints_scanned": 0})
    monkeypatch.setattr("tools.preflight_mappo_baseline_learnability.write_audit", lambda result: None)
    result = validate(deep_freshness=False, smoke=False, launch_check=True)
    assert "future-final 45M block is not fully fresh" in result["blockers"]


def test_analyzer_uses_nearest_real_evaluation_without_interpolation():
    rows = [{"sampled_steps": 811008}, {"sampled_steps": 909312}, {"sampled_steps": 1007616}]
    assert select_endpoint(rows, 900_000)["sampled_steps"] == 909312


def test_conditional_wave_metrics_preserve_undefined_denominators():
    assert safe_ratio(.4, .8) == pytest.approx(.5)
    assert safe_ratio(0, 0) is None
    assert safe_ratio(None, .5) is None
    l1 = enriched({"condition": "L1", "training_seed": 1}, {"clear_wave_1_probability": .8})
    l2 = enriched({"condition": "L2", "training_seed": 1}, {"clear_wave_1_probability": .8, "clear_wave_2_probability": .4})
    assert l1["q2_w2_given_w1"] is None and l1["q3_w3_given_w2"] is None
    assert l2["q2_w2_given_w1"] == pytest.approx(.5) and l2["q3_w3_given_w2"] is None


def test_persistent_difficulty_is_unresolved_when_l1_is_unstable():
    def row(condition, step, w1, low, q2=None, q3=None):
        return {"condition": condition, "sampled_steps": step, "w1_mean": w1,
                "w1_min": low, "q2_w2_given_w1_mean": q2,
                "q3_w3_given_w2_mean": q3}
    rows = [row("L1", 900_000, .2, .1), row("L1", 3_000_000, .3, .2),
            row("L2", 3_000_000, .2, .1, .1), row("L3", 3_000_000, .1, 0, .1, 0)]
    labels = diagnostic_labels(rows)
    assert labels["persistent_wave"] == "PERSISTENT_WAVE_DIFFICULTY_UNRESOLVED"
    assert labels["persistent_wave_reason"] == "SINGLE_WAVE_BASELINE_UNSTABLE"


def test_first_wave_interference_and_later_distribution_candidates():
    def rows(l2_w1, l3_w1, q2, q3):
        base = lambda c, s, w, low, a=None, b=None: {"condition": c, "sampled_steps": s,
            "w1_mean": w, "w1_min": low, "q2_w2_given_w1_mean": a, "q3_w3_given_w2_mean": b}
        return [base("L1", 900_000, .6, .5), base("L1", 3_000_000, .9, .8),
                base("L2", 3_000_000, l2_w1, l2_w1, q2), base("L3", 3_000_000, l3_w1, l3_w1, q2, q3)]
    assert "LONG_HORIZON_OPTIMIZATION_INTERFERENCE" in diagnostic_labels(rows(.7, .65, .9, .9))["diagnostic_candidates"]
    assert "LATER_WAVE_DISTRIBUTION_DIFFICULTY" in diagnostic_labels(rows(.88, .86, .6, .5))["diagnostic_candidates"]


def test_completed_episode_count_uses_recorded_rows_only(tmp_path):
    path = tmp_path / "training_metrics.jsonl"
    path.write_text('\n'.join(['{"sampled_steps": 10}', '{"sampled_steps": 20}', '{"sampled_steps": 30}']), encoding="utf-8")
    assert completed_training_episodes(path, 20) == 2
    assert completed_training_episodes(path, 25) == 2
    assert completed_training_episodes(tmp_path / "missing.jsonl", 20) is None


@pytest.mark.parametrize("condition,declared,runtime,expected",[
    ("L1",(1,1000),(3,1000),"INVALID_FOR_ORIGINAL_LADDER"),
    ("L2",(2,2000),(3,2000),"INVALID_FOR_ORIGINAL_LADDER"),
    ("L3",(3,3000),(3,3000),"VALID_FULL_3_WAVE_BASELINE"),
])
def test_analyzer_classifies_historical_runtime_override(tmp_path,condition,declared,runtime,expected):
    source=load_yaml(ENV_PATHS[condition]);effective=copy.deepcopy(source)
    source["persistent_waves"]["total_waves"],source["simulation"]["max_steps"]=declared
    effective["persistent_waves"]["total_waves"],effective["simulation"]["max_steps"]=runtime
    import yaml,json
    (tmp_path/"env_config.yaml").write_text(yaml.safe_dump(source),encoding="utf-8")
    (tmp_path/"runtime_env_config.yaml").write_text(yaml.safe_dump(effective),encoding="utf-8")
    (tmp_path/"run_config.json").write_text(json.dumps({"environment_config_sha256":config_sha256(source)}),encoding="utf-8")
    (tmp_path/"run_summary.json").write_text('{"sampled_steps":3000000}',encoding="utf-8")
    manifest={"conditions":{condition:{"total_waves":declared[0],"max_steps":declared[1]}}}
    state={"extra":{"current_total_waves":runtime[0]}}
    assert audit_actual_environment(tmp_path,{"condition":condition},manifest,state)["status"]==expected
