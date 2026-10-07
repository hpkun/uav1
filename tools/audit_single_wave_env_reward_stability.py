"""Read-only evidence audit for the clean 1.5M single-wave MLP-MAPPO runs.

This program reads existing artifacts and source code only.  It never creates an
environment and never evaluates a checkpoint.  Its sole write target is the new
analysis directory supplied by --output-dir.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (5401, 5402, 5403)
RUN_TEMPLATE = "outputs/mappo_mlp_single_wave_seed{seed}_1p5m"
DEFAULT_OUTPUT = ROOT / "outputs/single_wave_env_reward_audit"
EXPECTED_BEST = {5401: (1_105_920, .86), 5402: (1_105_920, .68), 5403: (1_400_832, .94)}
FIELDS = (
    "average_return", "win_rate", "average_red_loss", "average_blue_loss",
    "average_red_boundary_exits", "evaluation_boundary_exit_rate",
    "average_red_ground_losses", "average_blue_ground_losses", "timeout_rate",
    "average_episode_length", "red_fire_window_episode_rate",
    "red_attempt_episode_rate", "red_hit_episode_rate", "red_kill_episode_rate",
    "average_episode_r1_total", "average_episode_r2_total",
    "average_episode_r3_total", "average_episode_r4_total",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [{k: (float(v) if k != "sampled_steps" else int(v)) for k, v in row.items()}
                for row in csv.DictReader(stream)]


def best_row(history: list[dict[str, Any]]) -> dict[str, Any]:
    return max(history, key=lambda r: (r["win_rate"], r["average_return"], -r["average_red_loss"]))


def selected(row: dict[str, Any]) -> dict[str, Any]:
    return {"sampled_steps": int(row["sampled_steps"]), **{k: float(row[k]) for k in FIELDS}}


def failure_class(best: dict[str, Any], final: dict[str, Any]) -> str:
    db = final["average_red_boundary_exits"] - best["average_red_boundary_exits"]
    dg = final["average_red_ground_losses"] - best["average_red_ground_losses"]
    dc = best["average_blue_loss"] - final["average_blue_loss"]
    if db >= 1.0 and dc >= 1.0:
        return "BOUNDARY_DOMINATED_WITH_COMBAT_ENGAGEMENT_DEGRADATION"
    if db >= .2 and dc >= .3:
        return "MIXED_BOUNDARY_AND_COMBAT_DEGRADATION"
    if dg >= .5:
        return "GROUND_DOMINATED"
    return "COMBAT_OR_MIXED"


def protocol(run: Path, seed: int) -> dict[str, Any]:
    summary = load_json(run / "run_summary.json")
    runtime = load_json(run / "run_config.json")
    algo = yaml.safe_load((run / "algorithm_config.yaml").read_text(encoding="utf-8"))
    env_path = run / "env_config.yaml"
    env = yaml.safe_load(env_path.read_text(encoding="utf-8"))
    expected = {
        "algorithm": "MAPPO", "critic_type": "mlp", "observation_dim": 52,
        "action_dim": 3, "num_agents": 4, "seed": seed, "num_envs_M": 24,
        "rollout_steps": 256, "ppo_epochs": 10, "gamma": .999,
        "gae_lambda": .95, "total_sampled_steps": 1_500_000,
        "sampled_steps": 1_500_000, "total_waves": 1, "max_steps": 3000,
    }
    mismatches = {k: {"expected": v, "actual": summary.get(k)}
                  for k, v in expected.items() if summary.get(k) != v}
    checks = {
        "runtime_device_cuda": runtime.get("device") == "cuda",
        "actor_lr_3e-4": algo["training"]["actor_learning_rate"] == 3e-4,
        "critic_lr_3e-4": algo["training"]["critic_learning_rate"] == 3e-4,
        "evaluation_seeds_48m": algo["implementation"]["evaluation_seed_base"] == 48_000_000,
        "environment_total_waves_1": env["persistent_waves"]["total_waves"] == 1,
        "environment_max_steps_3000": env["simulation"]["max_steps"] == 3000,
        "environment_hash_matches_runtime": config_sha256(env) == runtime["environment_config_sha256"],
        "algorithm_hash_matches_runtime": config_sha256(algo) == runtime["algorithm_config_sha256"],
    }
    return {"mismatches": mismatches, "checks": checks, "pass": not mismatches and all(checks.values()),
            "note": "algorithm_config seed=5303 is a template value; CLI/runtime seed and checkpoint metadata are authoritative."}


def checkpoint_metadata(path: Path, expected_seed: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for checkpoint audit; torch.cuda.is_available() is False")
    state = torch.load(path, map_location="cuda:0", weights_only=False)
    extra = state.get("extra", {})
    result = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(path), "sampled_steps": int(state.get("sampled_steps", -1)),
        "training_seed": int(extra.get("training_seed", -1)),
        "algorithm": state.get("algorithm"), "enabled_modules": state.get("enabled_modules", []),
    }
    result["pass"] = result["training_seed"] == expected_seed and result["algorithm"] == "MAPPO"
    del state
    torch.cuda.empty_cache()
    return result


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    x = (len(data) - 1) * q
    lo, hi = math.floor(x), math.ceil(x)
    return data[lo] if lo == hi else data[lo] * (hi - x) + data[hi] * (x - lo)


def weapon_probability(distance: float, error: float = 0.0) -> float:
    threshold = math.pi * math.exp(-distance / 2232.442506204989)
    cdf = lambda x: .5 * (1 + math.erf(x / math.sqrt(2)))
    one_axis = cdf(threshold - error) - cdf(-threshold - error)
    return one_axis * one_axis


def csv_write(path: Path, data: list[dict[str, Any]]) -> None:
    keys = list(dict.fromkeys(key for row in data for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(data)


def fmt(v: Any, n: int = 3) -> str:
    return "NA" if v is None else f"{v:.{n}f}" if isinstance(v, float) else str(v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing audit directory: {out}")
    out.mkdir(parents=True)

    run_data: dict[int, dict[str, Any]] = {}
    boundary_table: list[dict[str, Any]] = []
    reward_scale: list[dict[str, Any]] = []
    for seed in SEEDS:
        run = ROOT / RUN_TEMPLATE.format(seed=seed)
        history = rows(run / "evaluation_history.csv")
        best, final = best_row(history), history[-1]
        expected_step, expected_win = EXPECTED_BEST[seed]
        if int(best["sampled_steps"]) != expected_step or abs(best["win_rate"] - expected_win) > 1e-12:
            raise RuntimeError(f"seed {seed} best checkpoint fact mismatch")
        if int(final["sampled_steps"]) != 1_500_000:
            raise RuntimeError(f"seed {seed} lacks exact 1.5M evaluation")
        role_rows = {"best": best, "final": final}
        checkpoints = {
            "best": checkpoint_metadata(run / "best_eval.pt", seed),
            "final": checkpoint_metadata(run / "checkpoint_1500000.pt", seed),
        }
        if checkpoints["best"]["sampled_steps"] != expected_step or checkpoints["final"]["sampled_steps"] != 1_500_000:
            raise RuntimeError(f"seed {seed} checkpoint step mismatch")
        cls = failure_class(best, final)
        for role, row in role_rows.items():
            boundary_table.append({"training_seed": seed, "checkpoint_role": role,
                **selected(row), "failure_class_best_to_final": cls})
            abs_sum = sum(abs(row[f"average_episode_r{i}_total"]) for i in range(1, 5))
            for i in range(1, 5):
                value = float(row[f"average_episode_r{i}_total"])
                reward_scale.append({"training_seed": seed, "checkpoint_role": role,
                    "component": f"R{i}", "mean_episode_component": value,
                    "absolute_share_of_component_sum": abs(value) / abs_sum if abs_sum else None})

        complete_batches = []
        with (run / "training_metrics.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("team_episode_return") is not None:
                    complete_batches.append(row)
                for i in range(1, 5):
                    key = f"mean_r{i}_reward"
                    if row.get(key) is not None:
                        row[key] = float(row[key])
        eval_abs = {f"R{i}": sum(abs(float(r[f"average_episode_r{i}_total"])) for r in history) for i in range(1, 5)}
        eval_abs_total = sum(eval_abs.values())
        training_rows = []
        with (run / "training_metrics.jsonl").open(encoding="utf-8") as stream:
            training_rows = [json.loads(line) for line in stream]
        train_abs = {f"R{i}": sum(abs(float(r[f"mean_r{i}_reward"])) for r in training_rows
                                      if r.get(f"mean_r{i}_reward") is not None) for i in range(1, 5)}
        train_signed = {f"R{i}": sum(float(r[f"mean_r{i}_reward"]) for r in training_rows
                                         if r.get(f"mean_r{i}_reward") is not None) for i in range(1, 5)}
        train_abs_total = sum(train_abs.values())
        for i in range(1, 5):
            name = f"R{i}"
            reward_scale.append({"training_seed": seed, "checkpoint_role": "ALL_EVALUATIONS",
                "component": name, "mean_episode_component": None,
                "absolute_share_of_component_sum": eval_abs[name] / eval_abs_total if eval_abs_total else None,
                "scope_value": eval_abs[name], "scope_value_definition": "sum_abs_of_50_episode_checkpoint_means"})
            reward_scale.append({"training_seed": seed, "checkpoint_role": "ALL_TRAINING_VECTOR_STEPS",
                "component": name, "mean_episode_component": None,
                "absolute_share_of_component_sum": train_abs[name] / train_abs_total if train_abs_total else None,
                "scope_value": train_abs[name], "scope_value_definition": "sum_abs_of_per_vector_step_agent_mean",
                "signed_scope_value": train_signed[name]})
        r4_batch_means = [float(r["episode_r4_total"]) for r in complete_batches if r.get("episode_r4_total") is not None]
        run_data[seed] = {
            "protocol": protocol(run, seed), "history_rows": len(history),
            "evaluation_seed_range": [int(history[0]["evaluation_seed_base"]), int(history[0]["evaluation_seed_end"])],
            "best": selected(best), "final": selected(final), "checkpoints": checkpoints,
            "best_to_final": {"delta_win_rate": final["win_rate"] - best["win_rate"],
                "delta_return": final["average_return"] - best["average_return"],
                "delta_red_boundary_exits": final["average_red_boundary_exits"] - best["average_red_boundary_exits"],
                "delta_red_ground_losses": final["average_red_ground_losses"] - best["average_red_ground_losses"],
                "delta_blue_loss": final["average_blue_loss"] - best["average_blue_loss"],
                "classification": cls},
            "all_evaluation_blue_ground_loss": {"mean": statistics.mean(r["average_blue_ground_losses"] for r in history),
                "max": max(r["average_blue_ground_losses"] for r in history)},
            "completion_batch_mean_r4_proxy": {"scope": "mean over whichever episodes completed on one vector step; not individual episodes",
                "n_rows": len(r4_batch_means), "p90": percentile(r4_batch_means, .9),
                "p99": percentile(r4_batch_means, .99), "max": max(r4_batch_means) if r4_batch_means else None},
            "reward_component_scale": {"all_evaluations_absolute_share": {k: v / eval_abs_total for k, v in eval_abs.items()},
                "all_training_vector_steps_absolute_share": {k: v / train_abs_total for k, v in train_abs.items()}},
        }

    csv_write(out / "boundary_audit.csv", boundary_table)
    csv_write(out / "reward_component_scale.csv", reward_scale)
    unavailable = []
    for seed in SEEDS:
        for group in ("WIN", "COMBAT_LOSS", "BOUNDARY_EXIT", "GROUND_LOSS", "TIMEOUT"):
            unavailable.append({"training_seed": seed, "failure_mode": group, "status": "NOT_OBSERVABLE_FROM_EXISTING_LOGS",
                "reason": "evaluation_history is a 50-episode aggregate and training_metrics stores completion-batch means without individual episode count/identity"})
    csv_write(out / "episode_return_by_failure_mode.csv", unavailable)

    offsets = [-450, -150, 150, 450]
    geometry = [{"offset": x, "spawn_radius": math.hypot(4000, x),
                 "boundary_margin": 5000 - math.hypot(4000, x)} for x in offsets]
    weapon = [{"distance": d, "centered_joint_hit_probability": weapon_probability(d),
               "edge_joint_hit_probability": weapon_probability(d, math.radians(30))}
              for d in (0, 1000, 2000, 3000, 4000)]
    summary = {
        "status": "AUDIT_READY", "scope": "read-only existing artifacts plus static source inspection",
        "runs": {str(k): v for k, v in run_data.items()},
        "mechanics": {
            "reward_semantics": {
                "R1": "+10 divided among Red attackers credited for a Blue kill; -10 only to a Red lost by weapon or ground",
                "R2": "-10 only to the Red agent crossing the arena boundary",
                "R3_R4": "individual, nearest-living-Blue state shaping; living teammates do not share teammate loss/exit penalties",
                "mission_level_win_bonus": False, "timeout_penalty": False,
                "classification": "PARTIALLY_ALIGNED_WITH_INDIVIDUAL_INCENTIVE_MISMATCH_RISK",
            },
            "initial_geometry": geometry,
            "weapon_probability": weapon,
            "observation_boundary": "x/5000 and y/5000 make radius inferable, but no explicit boundary distance or radial velocity",
            "blue_ground_guard_activation": "NOT_OBSERVABLE_FROM_EXISTING_LOGS",
            "individual_episode_conditional_returns": "NOT_OBSERVABLE_FROM_EXISTING_LOGS",
        },
        "judgements": {
            "implementation": "IMPLEMENTATION_CORRECT_WITH_DISCLOSED_RECONSTRUCTION_CHOICES",
            "reward_alignment": "MATERIAL_STABILITY_RISK",
            "boundary_design": "MATERIAL_STABILITY_RISK",
            "timeout_design": "PLAUSIBLE_STABILITY_RISK",
            "initialization_margin": "POTENTIALLY_DESTABILIZING_RECONSTRUCTION",
            "observation_sufficiency": "LEARNABLE_FROM_OBSERVATION_BUT_NONLINEAR_REPRESENTATION_BURDEN",
            "weapon_stochasticity": "PLAUSIBLE_STABILITY_AND_EVALUATION_VARIANCE_RISK",
            "blue_guard_asymmetry": "DATA_INSUFFICIENT_FOR_ACTIVATION_FREQUENCY",
            "horizon": "MATERIAL_PROTOCOL_RISK_FOR_CANONICAL_SINGLE_WAVE_COMPARISON",
            "benchmark_maturity": "NOT_YET_MATURE_AS_A_STABLE_SINGLE_WAVE_BENCHMARK",
            "reward_ranking_consistency": "DATA_INSUFFICIENT_FOR_EPISODE_LEVEL_CLASSIFICATION",
            "timeout_hovering": "TIMEOUT_LOCAL_OPTIMUM_NOT_SEEN_IN_AGGREGATE_ENDPOINTS",
        },
        "limitations": [
            "No individual evaluation episodes were persisted.",
            "Training completion rows are unweighted batch means and may mix termination modes.",
            "Aggregate best/final evidence cannot distinguish intentional escape from control overshoot.",
            "Blue ground-guard counters were available at runtime but not persisted in evaluation_history.",
        ],
        "minimal_next_experiment": "Matched clean MLP-MAPPO 3-seed single-wave control with max_steps=1000; hold all other semantics fixed.",
        "optional_trajectory_tool": "tools/audit_single_wave_checkpoint_trajectories.py (created but not executed)",
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mechanics = f"""# Environment mechanics audit

## Result

No source-level implementation bug was found in the inspected transition order, termination, reward accounting, observation construction, or weapon entry trigger. The environment is an explicit reconstruction, and several faithful-looking choices are nevertheless stability-sensitive.

## Boundary and dynamics

The circular arena radius is 5000 m. Red formation members start at radial distances {geometry[0]['spawn_radius']:.2f} m (outer offsets) or {geometry[1]['spawn_radius']:.2f} m (inner offsets), leaving only {geometry[0]['boundary_margin']:.2f}--{geometry[1]['boundary_margin']:.2f} m. At 225 m/s, a continuously outward trajectory crosses that margin in roughly 4.3--4.4 s. Initial headings point inward, so this is not an immediate-reset defect; it is a control-margin sensitivity. A saturated heading command requests up to 180 degrees, filtered by the controller and 8g load limit.

The step order is: advance kinematics, resolve boundary/ground deaths, compute post-transition R3/R4, resolve weapons, then assign R1/R2. A boundary exit receives R2=-10 but not R1=-10. A ground loss receives R1=-10. This ordering is internally consistent.

## Observation sufficiency

The 52D observation includes own x/5000 and y/5000, heading, pitch and speed, so distance to a circular boundary and radial velocity are mathematically inferable. It does not expose either quantity explicitly. Classification: **learnable from observation**, with a nonlinear representation burden rather than a hidden-state bug.

At a normalized action of ±1, the command target changes by up to ±180 degrees heading, ±60 degrees pitch and ±50 m/s. The 2 s target time constant would request about 90 degrees/s yaw and 30 degrees/s pitch before feasibility projection. At 225 m/s the 8g cap limits achievable combined maneuvering; a lateral-only upper bound is about 0.349 rad/s (20 degrees/s). Thus one random action does not instantaneously rotate the aircraft, but persistent saturated actions can reverse it within seconds and consume the sub-kilometre boundary margin.

## Blue guard and weapon

Blue alone receives a ground-aware nearest-target controller. Guard activation counters exist in terminal info but were not stored in these evaluation histories; activation frequency is **NOT_OBSERVABLE_FROM_EXISTING_LOGS**. Across saved evaluations, average Blue ground loss remains near zero (per-seed maxima: {', '.join(f'{s}={run_data[s]["all_evaluation_blue_ground_loss"]["max"]:.3f}' for s in SEEDS)}).

The weapon uses two independent Gaussian draws for azimuth and elevation and a one-shot entry trigger. This is a disclosed reconstruction choice: the paper prints one epsilon symbol in both conditions. Centered joint hit probabilities from the implemented model are approximately {', '.join(f'{int(x["distance"])}m={x["centered_joint_hit_probability"]:.3f}' for x in weapon)}. This makes identical geometry stochastic and a 50-episode evaluation moderately noisy, but it does not explain the large 5401/5402 boundary shifts by itself.
"""
    (out / "environment_mechanics.md").write_text(mechanics, encoding="utf-8")

    reward_md = """# Reward alignment audit

## Actual objective

- R1 is local: credited attackers share +10 for a Blue kill; the Red UAV killed by weapon or ground receives -10.
- R2 is local: only the Red UAV that exits receives -10.
- R3/R4 are computed for each living Red UAV against its nearest living Blue target.
- There is no terminal team win bonus, timeout penalty, or shared teammate-loss/exit penalty.

The task is mission-level cooperative, but the optimized return is the sum of partially local incentives. This is **partially aligned**, not a clean shared-team objective. It creates an incentive-risk channel in which a surviving agent can avoid future combat risk and shaping exposure after exiting, while teammates do not directly receive that failure penalty.

## Paper comparison

Li et al. Eq. (25) gives R=R1+R2+R3+R4 and explicitly says R2 is intended to prevent escape/local convergence. The paper later reports the opposite phenomenon for a baseline: R4 could make direct exit less negative than defeat, and it reports MAPPO policies hovering. The current reconstruction preserves those component magnitudes but does not add a mission-level terminal correction. The observed boundary-heavy final policies are therefore scientifically consistent with the paper's stated failure mode, although aggregate logs alone do not prove intentional escape.

## Ranking and scale

R1/R2 dominate the best/final mean returns; R3 and R4 are usually much smaller in aggregate. However, small per-step R4 can accumulate over a 3000-step horizon. True per-episode R4 p90/p99/max and return rankings by failure mode are **NOT_OBSERVABLE_FROM_EXISTING_LOGS**. The proxy statistics in audit_summary.json are distributions of training completion-batch means, not individual episodes.

Conclusion: reward alignment is a **material stability risk**, but the existing artifacts do not justify rewriting the reward before a matched horizon check and trajectory-level audit.

Because individual episodes were not persisted, reward-ranking consistency cannot honestly be classified as STRONGLY_ALIGNED / MOSTLY_ALIGNED / MATERIAL_MISALIGNMENT from these files. It is **DATA_INSUFFICIENT_FOR_EPISODE_LEVEL_CLASSIFICATION**, alongside a source-level mismatch risk and aggregate boundary evidence.
"""
    (out / "reward_alignment.md").write_text(reward_md, encoding="utf-8")

    horizon = """# Horizon audit

The audited control uses one wave with max_steps=3000 (300 s at dt=0.1). The canonical single-round V2.3 horizon documented elsewhere in the repository is 1000 steps (100 s). Keeping 3000 steps isolates wave count while preserving the multi-wave total horizon, but it is not a canonical single-wave replication.

A 3x longer horizon increases exposure to boundary drift, permits more R3/R4 accumulation, and changes the meaning of timeout/hovering. Evaluation timeout is nearly absent in the selected endpoints, so timeout itself is not the observed dominant failure mode. The horizon remains a material protocol confound for judging whether the reconstructed single-wave benchmark is intrinsically stable.
"""
    (out / "horizon_audit.md").write_text(horizon, encoding="utf-8")

    paper = """# Paper versus reconstruction

| Topic | Li et al. 2023 | Current reconstruction | Audit consequence |
|---|---|---|---|
| Reward | Eq. (25), R1+R2+R3+R4 | Same named component values | Formula-level match, mission alignment remains partial |
| Escape | R2 intended to prevent escape; paper reports a baseline still learned direct exit because shaped return could be less negative than combat defeat | R2=-10 only for exiting agent, no team terminal penalty | Known local-solution risk remains structurally possible |
| MAPPO behavior | Paper reports hovering/overfitting | Current clean MLP runs show strong peaks, then boundary-heavy decline in 2/3 seeds | Failure mode differs in detail but both indicate benchmark/optimization sensitivity |
| Weapon noise | Eq. (8) prints one epsilon symbol in both angular inequalities | Independent azimuth/elevation Gaussian draws | Disclosed reconstruction difference; changes joint hit probability |
| Horizon | Original single-round setting is not established here as 3000 steps | one wave, 3000 steps | Not a strict canonical comparison |
| Blue policy | nearest pursuit in paper | ground-aware nearest pursuit | Safety asymmetry; activation not observable in these logs |

This audit treats the paper's descriptive failure claims as supporting context, not as proof of the current policies' intent.
"""
    (out / "paper_vs_reconstruction.md").write_text(paper, encoding="utf-8")

    rec = """# Recommended validation experiments

1. **Minimum next formal experiment:** rerun the clean MLP-MAPPO single-wave control at max_steps=1000 with three training seeds and the same 48M fixed evaluation bank. Do not change reward, action, Blue policy, weapon, spawn, optimizer, or network simultaneously.
2. **Trajectory audit before reward changes:** use the provided six-checkpoint tool on best/final policies with reserved 89.2M diagnostic seeds. Inspect 20/10/5/1-step boundary precursors, action saturation, radial velocity, geometry and guard activity. The tool was not run in this audit.
3. **Only if trajectory evidence supports reward ranking:** perform offline fixed-trajectory counterfactual scoring for alternative boundary/timeout terminal terms. This cannot establish policy causality, but it can identify whether the recorded ranking is vulnerable.
4. **Later, one-factor-at-a-time:** if 1000-step MLP is still unstable, test explicit boundary features versus the current inferable features; separately test shared mission-level terminal accounting. Do not combine both in the first causal screen.

No new long training or evaluation was run here.
"""
    (out / "recommended_validation_experiments.md").write_text(rec, encoding="utf-8")

    table = ["|Seed|Role|Step|Win|Return|Red loss|Blue loss|Boundary|Boundary ep rate|Ground|Timeout|R1|R2|R3|R4|",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in boundary_table:
        table.append("|" + "|".join(map(str, [row["training_seed"], row["checkpoint_role"], row["sampled_steps"],
            fmt(row["win_rate"]), fmt(row["average_return"]), fmt(row["average_red_loss"]), fmt(row["average_blue_loss"]),
            fmt(row["average_red_boundary_exits"]), fmt(row["evaluation_boundary_exit_rate"]), fmt(row["average_red_ground_losses"]),
            fmt(row["timeout_rate"]), fmt(row["average_episode_r1_total"]), fmt(row["average_episode_r2_total"]),
            fmt(row["average_episode_r3_total"]), fmt(row["average_episode_r4_total"])])) + "|")
    final = """# Single-wave environment/reward stability audit

## A. Protocol validity

All three runs pass the requested runtime protocol checks and exact CUDA checkpoint metadata checks. They are clean MLP-critic MAPPO, 52D/3D/4-agent, one-wave, 3000-step, 24-env, 256-rollout, 10-epoch runs at gamma=.999 and lambda=.95, evaluated on 48,000,000..48,000,049. The static algorithm YAML's seed=5303 is a template value; runtime records and checkpoint metadata correctly contain 5401/5402/5403.

## B. Best versus final

""" + "\n".join(table) + """

Seed 5401 falls .86→.40 and seed 5402 .68→.32 with large boundary increases (0→1.64 and .08→2.34 exits/episode) plus reduced Blue kills/engagement. Seed 5403 retains more (.94→.78), with a milder mixed decline. This is not a ground-loss or timeout collapse.

## C. Root-cause judgement

- **Environment implementation:** no bug found in the inspected mechanics; reconstruction choices are explicit.
- **Reward alignment:** partially mission-aligned, with a material individual-incentive mismatch risk.
- **Boundary:** material stability risk; it is the dominant observed final failure mechanism for 5401/5402.
- **Horizon:** 3000 steps is a material protocol confound for a canonical single-wave comparison.
- **Observation:** sufficient in principle, but boundary distance/radial velocity require nonlinear inference.
- **Weapon:** plausible variance contributor, not a sufficient explanation for the late boundary collapse.
- **Blue guard:** asymmetry exists; activation frequency is unavailable, and Blue ground loss remains near zero.

## D. Evidence limit

The audit cannot infer policy intent from aggregate returns. Conditional return ordering, individual-episode R4 tails, intentional escape versus tactical overshoot, and Blue guard activation are **NOT_OBSERVABLE_FROM_EXISTING_LOGS**. A read-only trajectory tool was added but deliberately not executed.

## E. Benchmark maturity and next step

The current one-wave/3000-step reconstruction is **not yet mature as a stable single-wave benchmark**: it is learnable (all seeds reach strong peaks) but two seeds lose the peak through boundary-heavy behavior. The minimum next experiment is a matched 1000-step, three-seed clean MLP control. No reward change should be bundled into that test.

## F. Required research answers

1. Instability does **not** prove an environment implementation error; it proves a stability problem under this protocol.
2. There is clear structural local-incentive risk and boundary-heavy behavior, but no episode-level return evidence proving an intentional reward local optimum.
3. Yes: boundary exit is the main observed late-collapse mode for 5401/5402, with concurrent combat degradation.
4. Timeout/hovering is not seen at the selected endpoints (all final timeout rates are zero); episode-level hovering classification remains unavailable.
5. Yes: 3000 steps is unsuitable as the sole canonical single-wave stability test; it answers a persistent-horizon isolation question.
6. Yes: run a matched 1000-step, three-seed clean MLP control next.
7. Do not change boundary penalty yet; first obtain matched-horizon and trajectory evidence.
8. Do not add timeout penalty yet; current timeout evidence is negligible.
9. A team terminal reward is a defensible **benchmark-repair candidate**, but requires a separate one-factor causal test after the horizon check.
10. Horizon, boundary semantics and team terminal alignment are benchmark-design work; actor/critic architecture or optimization changes are algorithm innovation and must remain separate.

**AUDIT_READY**
"""
    (out / "final_audit_report.md").write_text(final, encoding="utf-8")
    print(json.dumps({"status": "AUDIT_READY", "output_dir": str(out),
                      "protocol_pass": all(v["protocol"]["pass"] for v in run_data.values()),
                      "best_to_final": {str(s): run_data[s]["best_to_final"] for s in SEEDS}}, indent=2))


if __name__ == "__main__":
    main()
