"""Strict offline preflight plus optional tiny CUDA smoke for the learnability ladder."""
from __future__ import annotations

import argparse, csv, hashlib, json, re, sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from algorithm.common.protocol import config_sha256
from algorithm.modules.actor_lr_decay import ActorLRDecayModule
from algorithm.modules.curriculum import CurriculumController
from algorithm.modular_mappo.runner import (ModularMAPPOTrainingRunner,
    validate_runtime_environment_contract)
from algorithm.train_modular_mappo import load_config
from env.factory import make_combat_environment

MANIFEST = ROOT / "experiments/mappo_baseline_learnability_manifest.json"
SEED_REGISTRY = ROOT / "experiments/current_seed_provenance.json"
ALGORITHM = ROOT / "configs/diag_mappo_learnability_common_3m.yaml"
ENV_PATHS = {
    "L1": ROOT / "configs/diag_learnability_1wave_environment.yaml",
    "L2": ROOT / "configs/diag_learnability_2wave_environment.yaml",
    "L3": ROOT / "configs/persistent_wave_v2_environment.yaml",
}
AUDIT = ROOT / "outputs/mappo_baseline_learnability_preflight_audit"
EXPECTED_SEEDS = (5301, 5302, 5303)
EXPECTED_EVAL = (44000000, 44000049)
FUTURE_FINAL = (45000000, 45000199)
PROPOSED_OUTPUT_ROOT = ROOT / "outputs/diag_mappo_learnability_corrected"
OFF = ("wave_context", "recurrent_memory", "popart", "multi_wave_reward",
       "wave_survival_pbrs", "wave_balancing", "warm_start", "curriculum",
       "policy_anchor", "entity_attention", "advantage_priority", "ppo_stabilization")
DECLARATIONS = {p.resolve() for p in [MANIFEST, SEED_REGISTRY, ALGORITHM, Path(__file__).resolve(),
    ROOT / "tools/run_mappo_baseline_learnability.sh", ROOT / "tests/test_mappo_baseline_learnability.py",
    ROOT / "configs/dev_actor_mission_context_3m.yaml",
    ROOT / "experiments/actor_mission_context_development_manifest.json",
    ROOT / "tools/preflight_actor_mission_context.py", ROOT / "tools/run_actor_mission_context_3m.sh",
    ROOT / "tools/analyze_actor_mission_context.py", ROOT / "tests/test_actor_mission_context.py",
    ROOT / "configs/dev_mission_aware_film_3m.yaml",
    ROOT / "experiments/mission_aware_film_development_manifest.json",
    ROOT / "tools/preflight_mission_aware_film.py", ROOT / "tools/run_mission_aware_film_3m.sh",
    ROOT / "tools/analyze_mission_aware_film.py", ROOT / "tests/test_mission_aware_film.py"]}
DECLARATIONS.update(p.resolve() for p in [
    ROOT / "configs/dev_team_credit_control_300k.yaml",
    ROOT / "configs/dev_team_credit_teammean_300k.yaml",
    ROOT / "tools/preflight_team_credit_screen.py",
    ROOT / "tools/smoke_team_credit_screen.py",
    ROOT / "tools/analyze_team_credit_screen.py",
    ROOT / "tools/run_team_credit_screen_300k.sh",
    ROOT / "tests/test_team_mean_credit.py",
])


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalized_env(value: dict) -> dict:
    row = deepcopy(value)
    row["simulation"]["max_steps"] = "ALLOWED"
    row["persistent_waves"]["total_waves"] = "ALLOWED"
    return row


def validate_seed_registry(registry: dict) -> None:
    ranges = registry["evaluation_ranges"]
    if ranges["33000000..33000199"]["status"] != "CONTAMINATED_RETIRED_FINAL_RANGE":
        raise RuntimeError("33M is not retired in current seed registry")
    if ranges["44000000..44000049"]["status"] != "CURRENT_LEARNABILITY_DEVELOPMENT":
        raise RuntimeError("44M development status mismatch")
    if ranges["45000000..45000199"] != {
            "status": "CURRENT_FUTURE_FINAL_BLOCK", "executed": False,
            "selection": "first complete fresh 200-seed block scanned from 45M upward",
            "freshness_evidence": {"repository_text_hits": 0, "outputs_structured_or_log_hits": 0,
                "checkpoint_metadata_hits": 0, "outputs_files_scanned": 4088, "checkpoints_scanned": 356}}:
        raise RuntimeError("future-final registry mismatch")


def states(env):
    return np.stack([s.as_array() for s in env.red]), np.stack([s.as_array() for s in env.blue])


def validate_configs() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    registry = json.loads(SEED_REGISTRY.read_text(encoding="utf-8"))
    envs = {name: load_yaml(path) for name, path in ENV_PATHS.items()}
    if any(normalized_env(envs[name]) != normalized_env(envs["L3"]) for name in ("L1", "L2")):
        raise RuntimeError("environment ladder differs outside total_waves/max_steps")
    expected = {"L1": (1, 1000), "L2": (2, 2000), "L3": (3, 3000)}
    for name, (waves, steps) in expected.items():
        if (envs[name]["persistent_waves"]["total_waves"], envs[name]["simulation"]["max_steps"]) != (waves, steps):
            raise RuntimeError(f"{name} ladder identity mismatch")
    frozen = manifest["frozen_contract"]
    if config_sha256(envs["L3"]) != frozen["l3_environment_config_sha256"]:
        raise RuntimeError("L3 semantic hash changed")
    if hashlib.sha256(ENV_PATHS["L3"].read_bytes()).hexdigest() != frozen["l3_environment_file_sha256"]:
        raise RuntimeError("L3 file hash changed")
    if config_sha256(envs["L3"]["action"]) != frozen["action_config_sha256"]:
        raise RuntimeError("action scaling changed")
    if config_sha256(envs["L3"]["blue_policy"]) != frozen["blue_policy_config_sha256"]:
        raise RuntimeError("Blue policy config changed")
    if hashlib.sha256((ROOT / "env/fixed_policy.py").read_bytes()).hexdigest() != frozen["blue_policy_source_sha256"]:
        raise RuntimeError("Blue policy source changed")

    cfg = load_config(ALGORITHM)
    t = cfg["training"]
    expected_training = {"actor_learning_rate": 3e-4, "critic_learning_rate": 3e-4,
        "gamma": .999, "gae_lambda": .95, "clip_ratio": .2,
        "value_loss_coefficient": .5, "entropy_coefficient": .01,
        "rollout_steps": 256, "ppo_epochs": 10, "minibatch_size": 512,
        "num_train_envs": 24, "total_sampled_steps": 3_000_000,
        "evaluation_episodes": 50, "evaluation_interval_sampled_steps": 100_000,
        "device": "cuda"}
    bad = {k: (t.get(k), v) for k, v in expected_training.items() if t.get(k) != v}
    if bad: raise RuntimeError(f"training protocol mismatch: {bad}")
    if cfg["network"] != {"observation_dim": 52, "action_dim": 3, "num_agents": 4,
            "actor_hidden_layers": [256, 256], "critic_hidden_layers": [256, 256], "attention_heads": 2}:
        raise RuntimeError("baseline topology changed")
    enabled = [k for k, v in cfg["modules"].items() if v.get("enabled", False)]
    if enabled != ["actor_lr_decay"] or any(cfg["modules"][k]["enabled"] for k in OFF):
        raise RuntimeError(f"not pure baseline protocol: {enabled}")
    curriculum = CurriculumController(cfg["modules"]["curriculum"])
    effective_runtime = {}
    for name, declared in envs.items():
        effective = curriculum.runtime_config(declared, 0)
        contract = validate_runtime_environment_contract(
            declared, effective, declared, curriculum_enabled=curriculum.enabled)
        d, e = contract["declared"], contract["effective_training"]
        effective_runtime[name] = {
            "declared_waves": d["total_waves"], "effective_waves": e["total_waves"],
            "declared_max_steps": d["max_steps"], "effective_max_steps": e["max_steps"],
            "declared_hash": d["config_sha256"], "effective_hash": e["config_sha256"],
            "match": d["config_sha256"] == e["config_sha256"],
        }
        if not effective_runtime[name]["match"]:
            raise RuntimeError(f"{name} effective runtime environment mismatch")
    decay = ActorLRDecayModule(cfg["modules"]["actor_lr_decay"])
    expected_lr = {0: 3e-4, 300_000: 3e-4, 600_000: 3e-4,
                   750_000: 2e-4, 900_000: 1e-4, 1_500_000: 1e-4, 3_000_000: 1e-4}
    for step, value in expected_lr.items():
        if not np.isclose(decay.learning_rate(step, 3e-4), value, rtol=0, atol=1e-15):
            raise RuntimeError(f"actor LR mismatch at {step}")
    for seed in (7_700_001, 7_700_002):
        instances = [make_combat_environment(envs[name]) for name in ("L1", "L2", "L3")]
        resets = [env.reset(seed) for env in instances]
        ref_red, ref_blue = states(instances[0])
        for env, (obs, info) in zip(instances[1:], resets[1:]):
            red, blue = states(env)
            if not (np.array_equal(red, ref_red) and np.array_equal(blue, ref_blue)
                    and np.array_equal(obs, resets[0][0])
                    and np.array_equal(info["red_alive_mask"], resets[0][1]["red_alive_mask"])
                    and np.array_equal(info["blue_alive_mask"], resets[0][1]["blue_alive_mask"])):
                raise RuntimeError("first-wave reset is not exact matched")
        for env in instances:
            if not all(s.armed for s in env.red_fire_states + env.blue_fire_states):
                raise RuntimeError("weapon initial state mismatch")
    runs = manifest["runs"]
    if (manifest.get("protocol_role") != "development_diagnostic_only"
            or manifest.get("warm_start") is not False or manifest.get("from_scratch") is not True
            or manifest.get("action_space") != "frozen_3d_continuous"):
        raise RuntimeError("manifest diagnostic/from-scratch contract mismatch")
    if len(runs) != 9 or [(r["training_seed"], r["condition"]) for r in runs] != [
            (s, c) for s in EXPECTED_SEEDS for c in ("L1", "L2", "L3")]:
        raise RuntimeError("9-run serial matrix mismatch")
    corrected = manifest.get("corrected_runs", [])
    expected_corrected = [(s,c,f"outputs/diag_mappo_learnability_corrected/{c.lower()}_seed{s}")
                          for s in EXPECTED_SEEDS for c in ("L1","L2","L3")]
    if [(r["training_seed"],r["condition"],r["output_dir"]) for r in corrected] != expected_corrected:
        raise RuntimeError("corrected 9-run output matrix mismatch")
    if manifest["evaluation"] != {"seed_start": 44000000, "seed_end": 44000049,
            "episodes": 50, "deterministic": True, "common_scenarios": True}:
        raise RuntimeError("evaluation protocol mismatch")
    validate_seed_registry(registry)
    if cfg["development_protocol"]["reserved_future_final_test"] != {
            "seed_start": 45000000, "seed_end": 45000199, "executed": False}:
        raise RuntimeError("algorithm future-final range mismatch")
    return {"manifest": manifest, "registry": registry, "envs": envs,
            "algorithm": cfg, "lr": expected_lr, "effective_runtime": effective_runtime}


def classify_seed(value: int):
    if value in EXPECTED_SEEDS: return "training"
    if EXPECTED_EVAL[0] <= value <= EXPECTED_EVAL[1]: return "evaluation"
    if 33_000_000 <= value <= 33_000_199: return "retired_33m"
    if FUTURE_FINAL[0] <= value <= FUTURE_FINAL[1]: return "future_final"
    return None


def repo_relative_posix(path: Path) -> str:
    """Return a repository-relative path stable across Windows and POSIX hosts."""
    return path.relative_to(ROOT).as_posix()


def retired_33m_evidence_ok(freshness: dict, registry: dict) -> bool:
    """Validate retired 33M provenance when a full local archive is available."""
    expected = "outputs/dev_ea_hwb_stable_cuda_smoke/post_resume_eval_2ep.json"
    entry = registry["evaluation_ranges"]["33000000..33000199"]
    evidence = Path(str(entry.get("evidence", "")).replace("\\", "/")).as_posix()
    if evidence != expected:
        return False
    rows = [row for row in freshness["hits"]["retired_33m"] if row["path"] == expected]
    values = {row["field"]: row["value"] for row in rows}
    return (values.get("metadata.evaluation_seed_base") == 33000000
            and values.get("metadata.evaluation_seed_end") == 33000001
            and entry.get("observed_used_subset") == [33000000, 33000001])


def freshness_scan(checkpoints: bool = True) -> dict:
    hits = {"training": [], "evaluation": [], "retired_33m": [], "future_final": []}
    # Scan all repository text except outputs, binary/git data and this protocol's declarations.
    suffixes = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".md", ".txt", ".py", ".sh"}
    pattern = re.compile(r"(?<!\d)(5301|5302|5303|\d{8})(?!\d)")
    for path in ROOT.rglob("*"):
        if (not path.is_file() or path.suffix.lower() not in suffixes or path.resolve() in DECLARATIONS
                or ROOT / ".git" in path.parents or ROOT / "outputs" in path.parents
                or any(parent.name.startswith(".pytest_tmp") for parent in path.parents)): continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in set(pattern.findall(text)):
            category = classify_seed(int(token))
            # Current protocol declarations are excluded above. Historical 33M
            # prose is provenance, while any 45M mention outside the current
            # declarations invalidates the proposed future block.
            if category in ("training", "evaluation", "future_final"):
                hits[category].append({"path": repo_relative_posix(path), "value": int(token), "field": "text"})
    # Outputs: only seed-bearing structured fields/log declarations, avoiding metric-value false positives.
    def walk(value, parts=()):
        found = []
        if isinstance(value, dict):
            for key, item in value.items():
                child = parts + (str(key),)
                declaration = any(part in {"reserved_future_final_test", "seed_provenance", "algorithm_config"} for part in child)
                if "seed" in str(key).lower() and not declaration:
                    for candidate in item if isinstance(item, list) else [item]:
                        if isinstance(candidate, (int, float)) and float(candidate).is_integer():
                            cat = classify_seed(int(candidate))
                            if cat: found.append((cat, int(candidate), ".".join(child)))
                found.extend(walk(item, child))
        elif isinstance(value, list):
            for i, item in enumerate(value): found.extend(walk(item, parts + (str(i),)))
        return found
    output = ROOT / "outputs"
    for path in output.rglob("*") if output.exists() else []:
        if not path.is_file() or AUDIT in path.parents: continue
        found = []
        try:
            if path.suffix == ".json": found = walk(json.loads(path.read_text(encoding="utf-8", errors="ignore")))
            elif path.suffix == ".jsonl":
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    if line.strip(): found.extend(walk(json.loads(line)))
            elif path.suffix == ".csv":
                with path.open(encoding="utf-8-sig", errors="ignore", newline="") as f:
                    for row in csv.DictReader(f):
                        for key, value in row.items():
                            if key and "seed" in key.lower() and value and re.fullmatch(r"\d+(?:\.0+)?", value.strip()):
                                cat = classify_seed(int(float(value)))
                                if cat: found.append((cat, int(float(value)), key))
            elif path.suffix == ".log":
                text = path.read_text(encoding="utf-8", errors="ignore")
                for token in set(pattern.findall(text)):
                    if re.search(rf"(?i)seed(?:_base|_end)?\s*[:=]\s*['\"]?{token}(?!\d)", text):
                        cat = classify_seed(int(token))
                        if cat: found.append((cat, int(token), "log_seed_declaration"))
        except (OSError, ValueError, json.JSONDecodeError, csv.Error): pass
        for cat, value, field in found: hits[cat].append({"path": repo_relative_posix(path), "value": value, "field": field})
    checkpoint_count = 0
    if checkpoints and output.exists():
        if not torch.cuda.is_available(): raise RuntimeError("CUDA required for checkpoint metadata freshness audit")
        for path in output.rglob("*.pt"):
            checkpoint_count += 1
            try:
                state = torch.load(path, map_location="cuda", weights_only=False)
                for cat, value, field in walk(state.get("extra", {}) if isinstance(state, dict) else {}):
                    hits[cat].append({"path": repo_relative_posix(path), "value": value, "field": f"checkpoint.extra.{field}"})
            except Exception as exc:
                raise RuntimeError(f"checkpoint metadata unreadable: {path}: {exc}") from exc
    for key in hits:
        unique = {(r["path"], r["value"], r["field"]): r for r in hits[key]}
        hits[key] = list(unique.values())
    return {"hits": hits, "checkpoints_scanned": checkpoint_count}


def blue_existing_safety() -> dict:
    rows = []
    for path in (ROOT / "outputs").rglob("evaluation_history.csv") if (ROOT / "outputs").exists() else []:
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("average_blue_ground_losses") not in (None, ""):
                        rows.append((float(row["average_blue_ground_losses"]), float(row.get("average_blue_boundary_exits") or 0), str(path.relative_to(ROOT))))
        except (OSError, ValueError, csv.Error): pass
    return {"evaluation_rows": len(rows), "blue_ground_positive_rows": sum(g > 0 for g, _, _ in rows),
            "max_average_blue_ground_losses": max((g for g, _, _ in rows), default=None),
            "blue_boundary_positive_rows": sum(b > 0 for _, b, _ in rows),
            "mean_average_blue_boundary_exits": float(np.mean([b for _, b, _ in rows])) if rows else None,
            "blue_boundary_rows_ge_0p5": sum(b >= .5 for _, b, _ in rows),
            "max_average_blue_boundary_exits": max((b for _, b, _ in rows), default=None)}


def write_audit(result: dict) -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "preflight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    blue = result["blue_policy_audit"]
    (AUDIT / "blue_policy_audit.md").write_text(f"""# Blue rule-policy audit

- Selection: nearest currently alive Red, recomputed on every decision; deterministic index tie-break.
- Dead targets: excluded immediately; there is no cached target index.
- Guidance: LOS heading/elevation and desired speed 250 in the frozen 3D increment-action contract.
- Ground guard: stateless time-to-ground threshold plus diagnostics only; no hidden behavior state machine.
- Boundary: no explicit boundary guard. Existing evaluations: {blue['existing_safety']}.
- Respawn: fresh Blue receives the same policy on the next normal step; transient guard diagnostics are reset.
- Weapon: fire-window target selection is independently recomputed from alive targets; attempts for both teams are formed before kill resolution, preserving simultaneous-kill semantics.
- Chattering: nearest-distance crossings can switch targets without hysteresis; this is the declared nearest-target rule, with no stale-index implementation defect found.
- Wave invariance: W1/W2/W3 use the same code/config; only intended Red persistence and fresh-wave geometry change the state distribution.

BLUE_RULE_POLICY_IMPLEMENTATION_BUG_NOT_FOUND

BLUE_RULE_POLICY_KEEP_FROZEN
""", encoding="utf-8")
    (AUDIT / "literature_design_mapping.md").write_text("""# Literature design mapping

| Design | Problem addressed in literature | Matching symptom in this project | Needed now | Later priority |
|---|---|---|---|---|
| Explicit round information | Removes round/phase aliasing in multi-round decisions | Plausible only if later-wave degradation remains after basic skill is stable | No; isolate baseline first | High if L1 is stable and L2/L3 collapse |
| Recurrent/GRU policy | Handles history dependence and partial observability | Plausible for persistent survivor state, not yet isolated | No | Medium-high after ladder evidence |
| Easy-to-hard curriculum | Improves exploration as task complexity grows | Directly relevant if from-scratch L1 succeeds but L2/L3 fail | No; would confound this diagnostic | High conditional priority |
| Multi-round-specific reward | Aligns optimization with later-round completion | Relevant if W1 is retained but conditional W2/W3 stalls | No | Medium; formulation requires separate study |
| Opponent curriculum/diversity | Reduces overfitting and opponent non-stationarity | Fixed Blue is useful control; audit found no implementation bug | No | Low for current diagnostic |
| Local/global critic representation | Improves CTDE state/value estimation | Prior mission-context evidence did not establish a clear remedy | No | Medium-low |
| Action-space simplification | Reduces control-search complexity | Conflicts with required full 3D control question | REJECTED_BY_PROJECT_DESIGN | Rejected |
| Blue opponent redesign | Changes opponent difficulty/coverage | No implementation blocker found | NOT_NEEDED_UNLESS_BLUE_AUDIT_FINDS_BUG | Low |
""", encoding="utf-8")


def cuda_smoke(configs: dict) -> dict:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; CPU fallback forbidden")
    root = ROOT / "outputs/mappo_baseline_learnability_smoke_v3"
    rows = {}
    for index, condition in enumerate(("L1", "L3")):
        out = root / condition.lower()
        reuse = out.exists()
        if not reuse:
            cfg = deepcopy(configs["algorithm"])
            smoke_seed = 88_000_001 + index
            cfg["training"].update({"rollout_steps": 2, "ppo_epochs": 1, "minibatch_size": 8,
                "num_train_envs": 1, "total_sampled_steps": 8, "evaluation_episodes": 1,
                "evaluation_interval_sampled_steps": 10_000_000, "seed": smoke_seed})
            cfg["implementation"]["evaluation_seed_base"] = 88_001_000 + index
            cfg["development_protocol"]["validation"].update({"seed_start": 88_001_000 + index, "seed_end": 88_001_000 + index, "episodes": 1})
            runner = ModularMAPPOTrainingRunner(configs["envs"][condition], cfg, 1, 8, "cuda", smoke_seed, out, True)
            runner.run()
        if not (out / "latest.pt").is_file(): raise RuntimeError(f"incomplete existing smoke: {out}")
        state = torch.load(out / "latest.pt", map_location="cuda", weights_only=False)
        finite = all(torch.isfinite(v).all().item() for group in (state["actor"], state["critic"]) for v in group.values())
        expected_waves = configs["envs"][condition]["persistent_waves"]["total_waves"]
        actual_waves = int(state.get("extra", {}).get("runtime_total_waves", -1))
        if actual_waves != expected_waves:
            raise RuntimeError(f"{condition} smoke runtime waves mismatch: {actual_waves} != {expected_waves}")
        rows[condition] = {"sampled_steps": int(state["sampled_steps"]), "finite": finite,
            "actual_runtime_waves": actual_waves, "checkpoint": str(out / "latest.pt"),
            "reused_existing": reuse}
        if not finite: raise RuntimeError(f"{condition} nonfinite checkpoint")
    # Focused L2 transition; no performance evaluation.
    env = make_combat_environment(configs["envs"]["L2"]); env.reset(8_800_100)
    for aircraft in env.blue: aircraft.alive = False
    _, _, terminated, truncated, info = env.step(np.zeros((4, 3), np.float32))
    if terminated or truncated or not info["spawned_next_wave"] or info["wave_index"] != 2:
        raise RuntimeError("L2 artificial respawn path failed")
    rows["L2_artificial_respawn"] = True
    (root / "smoke_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def validate(*, deep_freshness: bool, smoke: bool, launch_check: bool = False) -> dict:
    configs = validate_configs()
    fresh = freshness_scan(checkpoints=deep_freshness)
    blockers = []
    historical_prefixes = ("outputs/diag_mappo_learnability/",
                           "outputs/mappo_baseline_learnability_audit/",
                           "outputs/actor_mission_context_preflight/",
                           "outputs/dev_actor_mission_context_3m/",
                           "outputs/actor_mission_context_analysis/",
                           "outputs/mission_aware_film_preflight/",
                           "outputs/dev_mission_aware_film_3m/",
                           "outputs/mission_aware_film_analysis/")
    unexpected_development_hits = [row for key in ("training", "evaluation") for row in fresh["hits"][key]
                                   if not row["path"].replace("\\", "/").startswith(historical_prefixes)]
    if unexpected_development_hits:
        blockers.append("candidate training/evaluation seeds are not fresh")
    if fresh["hits"]["future_final"]:
        blockers.append("future-final 45M block is not fully fresh")
    if not launch_check and not retired_33m_evidence_ok(fresh, configs["registry"]):
        blockers.append("retired 33M contamination evidence is missing")
    existing = [str((PROPOSED_OUTPUT_ROOT / f"{condition.lower()}_seed{seed}").relative_to(ROOT))
                for seed in EXPECTED_SEEDS for condition in ("L1", "L2", "L3")
                if (PROPOSED_OUTPUT_ROOT / f"{condition.lower()}_seed{seed}").exists()]
    if existing: blockers.append(f"non-fresh output directories: {existing}")
    result = {"status": ("NOT_READY_FOR_MAPPO_BASELINE_LEARNABILITY_DIAGNOSTIC" if blockers else "READY_FOR_MAPPO_BASELINE_LEARNABILITY_DIAGNOSTIC"),
        "blockers": blockers,
        "training_seeds": list(EXPECTED_SEEDS), "evaluation_seeds": list(EXPECTED_EVAL),
        "retired_final_range": [33000000, 33000199], "future_final_range": list(FUTURE_FINAL),
        "future_final_executed": False,
        "checkpoints_scanned": fresh["checkpoints_scanned"], "freshness_hits": fresh["hits"],
        "historical_development_seed_hits_allowed": True,
        "action_dim": 3, "action_config_sha256": configs["manifest"]["frozen_contract"]["action_config_sha256"],
        "environment_ladder": configs["effective_runtime"],
        "first_wave_exact_match": True, "enabled_modules": ["actor_lr_decay"],
        "blue_policy_audit": {"implementation_bug_found": False, "keep_frozen": True, "existing_safety": blue_existing_safety()},
        "cuda_smoke": cuda_smoke(configs) if smoke else "NOT_REQUESTED"}
    write_audit(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-check", action="store_true")
    parser.add_argument("--cuda-smoke", action="store_true")
    parser.add_argument("--skip-checkpoint-scan", action="store_true")
    args = parser.parse_args()
    result = validate(deep_freshness=not args.skip_checkpoint_scan, smoke=args.cuda_smoke,
                      launch_check=args.launch_check)
    print(json.dumps(result, indent=2))
    if result["blockers"]: raise SystemExit(2)


if __name__ == "__main__": main()
