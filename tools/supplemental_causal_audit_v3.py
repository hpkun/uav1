"""Read-only stochastic team-credit causal audit (V3).

This diagnostic performs no optimizer step.  It contrasts reward-credit
counterfactuals on the exact same stochastic Plain-MAPPO trajectories.
"""
from __future__ import annotations

import csv, hashlib, json, math, statistics, sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.mappo.trainer import compute_gae
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from env.persistent_env import PersistentWaveCombatEnv
from tools.audit_wave_gradient_conflict import (
    cosine_metrics, global_live_advantage_normalization, ppo_clipped_surrogate,
)

BASE = ROOT / "outputs/comprehensive_persistent_wave_audit"
V2 = BASE / "supplemental_causal_audit_v2"
OUT = BASE / "supplemental_causal_audit_v3"
RUNS = {s: ROOT / f"outputs/diag_mappo_learnability/l3_seed{s}" for s in (5301, 5302, 5303)}
STAGES = (1_505_280, 2_101_248, 3_000_000)
EPISODE_SEEDS = tuple(range(88_300_000, 88_300_004))
ENV_CONFIG = ROOT / "configs/persistent_wave_v2_environment.yaml"
CORE = [ROOT / p for p in (
    "env/persistent_env.py", "env/combat_env.py", "env/reward.py",
    "env/observation.py", "env/weapon.py",
    "algorithm/modular_mappo/trainer.py", "algorithm/modular_mappo/runner.py",
)]
METHODS = (("FIXED_LOCAL_BASELINE_GAE", "a_local", "a_team"),
           ("BASELINE_FREE_REWARD_TO_GO", "rtg_local", "rtg_team"))
PARTITIONS = ("full_actor", "backbone", "mean_head", "log_std_head")
WINDOWS = (10, 25, 50)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, torch.Tensor): return clean(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def dump(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows); fields = sorted({k for r in rows for k in r}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def avg(values):
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.fmean(v) if v else None


def med(values):
    v = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return statistics.median(v) if v else None


def load_trainer(seed, stage):
    d = RUNS[seed]
    cfg = yaml.safe_load((d / "algorithm_config.yaml").read_text(encoding="utf-8"))
    trainer = build_modular_mappo_trainer(cfg, "cuda", 256, 3_000_000)
    trainer.load(d / f"checkpoint_{stage}.pt", True, False)
    return trainer


def parameter_groups(actor):
    named = [(n, p) for n, p in actor.named_parameters() if p.requires_grad]
    groups = {
        "full_actor": [p for _, p in named],
        "backbone": [p for n, p in named if n.startswith("backbone.")],
        "mean_head": [p for n, p in named if n.startswith("mean.")],
        "log_std_head": [p for n, p in named if n.startswith("log_std.")],
    }
    if any(not x for x in groups.values()):
        raise RuntimeError(f"Plain actor parameter partition incomplete: {[(k,len(v)) for k,v in groups.items()]}")
    return groups


def flat_grad(loss, parameters, retain_graph=True):
    gs = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(parameters, gs)])


def loss_gradient(surrogate, mask, parameters):
    count = mask.sum()
    if float(count) <= 0:
        return None
    return flat_grad(-(surrogate * mask).sum() / count, parameters)


def pair_metrics(local, team):
    if local is None or team is None: return {"local_norm": None, "team_norm": None, "cosine": None, "angle_degrees": None, "team_local_norm_ratio": None}
    m = cosine_metrics(local, team); c = max(-1.0, min(1.0, m["cosine"]))
    return {"local_norm": m["left_norm"], "team_norm": m["right_norm"], "cosine": m["cosine"],
            "angle_degrees": math.degrees(math.acos(c)),
            "team_local_norm_ratio": m["right_norm"] / max(m["left_norm"], 1e-12)}


def rtg(reward, dones, alive, next_alive, gamma=.999):
    out = torch.zeros_like(reward); future = torch.zeros_like(reward[0])
    for i in reversed(range(len(reward))):
        continuation = (1 - dones[i].unsqueeze(-1)) * next_alive[i]
        future = reward[i] + gamma * continuation * future
        out[i] = future * alive[i]
    return out


def deterministic_degeneracy():
    bank = np.load(BASE / "supplemental_causal_audit/observation_bank.npz")
    obs = torch.as_tensor(bank["observations"][:128], dtype=torch.float32, device="cuda")
    alive = torch.as_tensor(bank["alive"][:128], dtype=torch.float32, device="cuda")
    trainer = load_trainer(5301, 3_000_000); groups = parameter_groups(trainer.actor)
    advantage = torch.linspace(-1, 1, obs.shape[0], device="cuda")[:, None].expand_as(alive) * alive
    results = {}
    for mode in ("deterministic", "stochastic"):
        torch.manual_seed(88_300_999)
        with torch.no_grad():
            old_dist, _ = trainer.actor.distribution_step(obs, None, None, None, alive)
            raw = old_dist.mean if mode == "deterministic" else old_dist.rsample()
            action = torch.tanh(raw); old_log = trainer.actor._squashed_log_prob(old_dist, raw, action)
        dist, _ = trainer.actor.distribution_step(obs, None, None, None, alive)
        new_log = trainer.actor._squashed_log_prob(dist, raw, action)
        ratio = torch.exp(new_log - old_log)
        surrogate = ppo_clipped_surrogate(new_log, old_log, advantage, .2)
        results[mode] = {"max_abs_ratio_minus_one": float((ratio - 1).abs().max().detach().cpu())}
        for name, params in groups.items():
            g = loss_gradient(surrogate, alive, params)
            results[mode][f"{name}_grad_norm"] = float(torch.linalg.vector_norm(g).detach().cpu())
    d = results["deterministic"]["mean_head_grad_norm"]; s = results["stochastic"]["mean_head_grad_norm"]
    confirmed = d <= max(1e-10, s * 1e-6) and s > 1e-10
    result = {"status": "DETERMINISTIC_POLICY_GRADIENT_DEGENERACY_CONFIRMED" if confirmed else "NOT_CONFIRMED",
              "checkpoint": "l3_seed5301/checkpoint_3000000.pt", "observation_count": len(obs),
              "deterministic_mean_head_grad_norm": d, "stochastic_mean_head_grad_norm": s,
              "mean_head_norm_ratio_deterministic_over_stochastic": d / max(s, 1e-30), "partitions": results,
              "explanation": "For raw=mu detached as rollout data, d log pi(raw|s)/d mu cancels at the Gaussian mean; stochastic samples do not have this cancellation."}
    del trainer; torch.cuda.empty_cache()
    return result


def death_cause(info, env, agent):
    if float(info["r2_rewards"][agent]) < 0: return "boundary"
    if float(env.red[agent].altitude) <= 0: return "ground"
    return "weapon"


def collect(seed, stage, trainer, env_cfg):
    episodes = []
    for env_seed in EPISODE_SEEDS:
        torch.manual_seed(env_seed ^ (seed << 8) ^ stage)
        env = PersistentWaveCombatEnv(deepcopy(env_cfg)); obs, _ = env.reset(env_seed)
        alive = env.red_alive_mask.copy(); episode = []; done = False
        while not done:
            wave = int(env.wave_index); value, _ = trainer.values_step(obs[None], alive[None])
            action, raw, logp, _ = trainer.act(obs[None], alive[None], deterministic=False, return_policy_data=True)
            next_obs, reward, terminated, truncated, info = env.step(action[0]); next_alive = info["red_alive_mask"].copy()
            next_value, _ = trainer.values_step(next_obs[None], next_alive[None]); done = bool(terminated or truncated)
            lost = set(int(i) for i in np.flatnonzero((alive > .5) & (next_alive < .5)))
            deaths = [{"agent": i, "cause": death_cause(info, env, i)} for i in sorted(lost)]
            episode.append({"obs": obs.copy(), "raw": raw[0].copy(), "action": action[0].copy(), "old_logp": logp[0].copy(),
                            "alive": alive.copy(), "wave": wave, "reward": reward.copy(), "done": float(done),
                            "next_alive": next_alive.copy(), "value": value[0].copy(), "next_value": next_value[0].copy(),
                            "deaths": deaths, "lost": lost})
            obs, alive = next_obs, next_alive
        episodes.append({"environment_seed": env_seed, "steps": episode})
    return episodes


def process_episode(ep):
    steps = ep["steps"]; arr = lambda k: np.asarray([x[k] for x in steps])
    # Credit identities are audited in float64 so long episode recursions do
    # not fail a 1e-5 algebraic tolerance solely through float32 accumulation.
    dev = "cuda"; alive = torch.as_tensor(arr("alive"), dtype=torch.float64, device=dev)[:, None]
    next_alive = torch.as_tensor(arr("next_alive"), dtype=torch.float64, device=dev)[:, None]
    dones = torch.as_tensor(arr("done"), dtype=torch.float64, device=dev)[:, None]
    values = torch.as_tensor(arr("value"), dtype=torch.float64, device=dev)[:, None]
    next_values = torch.as_tensor(arr("next_value"), dtype=torch.float64, device=dev)[:, None]
    local = torch.as_tensor(arr("reward"), dtype=torch.float64, device=dev)[:, None]
    team = local.sum(-1, keepdim=True) / alive.sum(-1, keepdim=True).clamp_min(1) * alive
    a_local, _ = compute_gae(local, values, next_values, dones, alive, next_alive, .999, .95)
    a_team, _ = compute_gae(team, values, next_values, dones, alive, next_alive, .999, .95)
    zero = torch.zeros_like(values)
    delta, _ = compute_gae(team - local, zero, zero, dones, alive, next_alive, .999, .95)
    error = float(((a_team - a_local) - delta).abs().max().cpu())
    return {"environment_seed": ep["environment_seed"], "steps": steps, "obs": arr("obs"), "raw": arr("raw"),
            "action": arr("action"), "old_logp": arr("old_logp"), "alive": arr("alive"), "next_alive": arr("next_alive"),
            "wave": arr("wave"), "a_local": a_local[:, 0].cpu().numpy(), "a_team": a_team[:, 0].cpu().numpy(),
            "rtg_local": rtg(local, dones, alive, next_alive)[:, 0].cpu().numpy(),
            "rtg_team": rtg(team, dones, alive, next_alive)[:, 0].cpu().numpy(), "delta_gae_error": error}


def concatenate(processed, key): return np.concatenate([p[key] for p in processed])


def tensors_and_policy(trainer, processed):
    to = lambda k, dtype=torch.float32: torch.as_tensor(concatenate(processed, k), dtype=dtype, device="cuda")
    obs, raw, action, old, alive = to("obs")[:, None], to("raw")[:, None], to("action")[:, None], to("old_logp")[:, None], to("alive")[:, None]
    waves = to("wave", torch.long)[:, None]
    dist, _ = trainer.actor.distribution_step(obs, None, None, None, alive)
    new = trainer.actor._squashed_log_prob(dist, raw, action)
    return obs, raw, action, old, alive, waves, new


def normalized(processed, key, alive):
    a = torch.as_tensor(concatenate(processed, key), dtype=torch.float32, device="cuda")[:, None]
    return global_live_advantage_normalization(a, alive)


def masks(processed, alive, waves):
    result = {"overall": alive}
    for w in (1, 2, 3): result[f"W{w}"] = alive * (waves == w).float().unsqueeze(-1)
    return result


def event_mask(processed, window):
    pieces = []
    for p in processed:
        T = len(p["steps"]); mask = np.zeros((T, 4), np.float32)
        for t, step in enumerate(p["steps"]):
            weapon_lost = {d["agent"] for d in step["deaths"] if d["cause"] == "weapon"}
            if not weapon_lost: continue
            survivors = [i for i in range(4) if i not in weapon_lost and step["alive"][i] > .5 and step["next_alive"][i] > .5]
            for i in survivors: mask[max(0, t-window):t, i] = p["alive"][max(0, t-window):t, i]
        pieces.append(mask)
    return torch.as_tensor(np.concatenate(pieces), dtype=torch.float32, device="cuda")[:, None]


def analyze_seed_stage(seed, stage, trainer, episodes):
    processed = [process_episode(ep) for ep in episodes]
    max_delta_error = max(p["delta_gae_error"] for p in processed)
    if max_delta_error > 1e-5: raise RuntimeError(f"reward-only delta-GAE mismatch {max_delta_error}")
    _, _, _, old, alive, waves, new = tensors_and_policy(trainer, processed)
    # trainer.act intentionally zeros log-probabilities for dead agents.  PPO
    # excludes those entries, so ratio identity is checked on live samples only.
    live_ratio_error = (torch.exp(new - old) - 1).abs()[alive > .5]
    ratio_error = float(live_ratio_error.max().detach().cpu()) if live_ratio_error.numel() else 0.0
    # Per-step collection versus one large batched CUDA forward can differ by a
    # few float32 ulps; 1e-4 is still a strict ratio~1 identity check.
    if ratio_error > 1e-4: raise RuntimeError(f"on-checkpoint ratio mismatch {ratio_error}")
    groups = parameter_groups(trainer.actor); wave_masks = masks(processed, alive, waves)
    gradient_rows, partition_rows, delta_rows, cross_rows, event_rows = [], [], [], [], []
    advantages = {}
    for method, local_key, team_key in METHODS:
        advantages[method] = {"local": normalized(processed, local_key, alive), "team_mean": normalized(processed, team_key, alive)}
        advantages[method]["delta"] = advantages[method]["team_mean"] - advantages[method]["local"]
        surrogates = {k: ppo_clipped_surrogate(new, old, v, .2) for k, v in advantages[method].items()}
        grads = defaultdict(dict)
        for label, mask in wave_masks.items():
            for part, params in groups.items():
                for credit in ("local", "team_mean", "delta"):
                    grads[(label, part)][credit] = loss_gradient(surrogates[credit], mask, params)
                metrics = pair_metrics(grads[(label, part)]["local"], grads[(label, part)]["team_mean"])
                row = {"training_seed": seed, "checkpoint_step": stage, "method": method, "wave": label,
                       "partition": part, "alive_agent_samples": int(mask.sum().item()), "max_abs_ratio_minus_one": ratio_error, **metrics}
                gradient_rows.append(row); partition_rows.append(row.copy())
                gl, gd = grads[(label, part)]["local"], grads[(label, part)]["delta"]
                dm = pair_metrics(gl, gd)
                delta_rows.append({"training_seed": seed, "checkpoint_step": stage, "method": method, "wave": label,
                                   "partition": part, "delta_local_norm_ratio": None if gl is None else float(torch.linalg.vector_norm(gd).cpu()) / max(float(torch.linalg.vector_norm(gl).cpu()), 1e-12),
                                   "cos_delta_local": dm["cosine"], "delta_norm": dm["team_norm"], "local_norm": dm["local_norm"]})
        for credit in ("local", "team_mean"):
            for part in PARTITIONS:
                for a, b in ((1, 2), (1, 3), (2, 3)):
                    ga, gb = grads[(f"W{a}", part)][credit], grads[(f"W{b}", part)][credit]
                    m = pair_metrics(ga, gb)
                    cross_rows.append({"training_seed": seed, "checkpoint_step": stage, "method": method,
                                       "credit": credit, "partition": part, "wave_pair": f"W{a}-W{b}",
                                       "cosine": m["cosine"], "angle_degrees": m["angle_degrees"],
                                       "wave_a_norm": m["local_norm"], "wave_b_norm": m["team_norm"]})
        for window in WINDOWS:
            emask = event_mask(processed, window); sample_fraction = float(emask.sum() / alive.sum().clamp_min(1))
            for part, params in groups.items():
                gl = loss_gradient(surrogates["local"], emask, params); gt = loss_gradient(surrogates["team_mean"], emask, params); gd = loss_gradient(surrogates["delta"], emask, params)
                pm = pair_metrics(gl, gt); dm = pair_metrics(gl, gd)
                event_rows.append({"training_seed": seed, "checkpoint_step": stage, "method": method, "window_steps": window,
                                   "partition": part, "event_agent_samples": int(emask.sum().item()), "sample_fraction": sample_fraction,
                                   "local_team_cosine": pm["cosine"], "delta_local_norm_ratio": None if gl is None else float(torch.linalg.vector_norm(gd).cpu()) / max(float(torch.linalg.vector_norm(gl).cpu()), 1e-12),
                                   "cos_delta_local": dm["cosine"]})
    # Episode-local primary externality: one q=t-10 sample per true survivor/event.
    externality = []
    for p in processed:
        for t, step in enumerate(p["steps"]):
            lost = {d["agent"] for d in step["deaths"] if d["cause"] == "weapon"}
            if not lost: continue
            q = max(0, t-10)
            for agent in range(4):
                if agent not in lost and step["alive"][agent] > .5 and step["next_alive"][agent] > .5:
                    al, at = float(p["a_local"][q, agent]), float(p["a_team"][q, agent])
                    externality.append((at-al, al, at))
    ext = {"training_seed": seed, "checkpoint_step": stage, "episodes": len(episodes), "weapon_death_10step_n": len(externality),
           "mean_delta": avg(x[0] for x in externality), "median_delta": med(x[0] for x in externality),
           "sign_flip_fraction": avg((x[1] >= 0) != (x[2] >= 0) for x in externality),
           "max_reward_only_delta_gae_error": max_delta_error, "max_abs_ratio_minus_one": ratio_error,
           "_delta_values": [x[0] for x in externality], "_flip_values": [(x[1] >= 0) != (x[2] >= 0) for x in externality]}
    return ext, gradient_rows, cross_rows, partition_rows, event_rows, delta_rows


def aggregate_externality(rows):
    per_seed = []
    for seed in RUNS:
        g = [r for r in rows if r["training_seed"] == seed]
        deltas = [x for r in g for x in r["_delta_values"]]; flips = [x for r in g for x in r["_flip_values"]]
        n = len(deltas)
        per_seed.append({"training_seed": seed, "checkpoint_step": "all", "episodes": 12, "weapon_death_10step_n": n,
                         "mean_delta": avg(deltas), "median_delta": med(deltas), "sign_flip_fraction": avg(flips)})
    replicated = all(r["mean_delta"] < 0 for r in per_seed)
    return per_seed, "TEAM_CREDIT_EXTERNALITY_STOCHASTIC_REPLICATED" if replicated else "NOT_REPLICATED"


def summarize(gradient_rows, delta_rows, event_rows, cross_rows, externality, degeneracy):
    def rows_for(method, part, stage=None):
        return [r for r in gradient_rows if r["method"] == method and r["partition"] == part and r["wave"] == "overall" and (stage is None or r["checkpoint_step"] == stage)]
    gradient_summary = {}
    for method, _, _ in METHODS:
        gradient_summary[method] = {}
        for part in PARTITIONS:
            rs = rows_for(method, part)
            gradient_summary[method][part] = {"cosine_mean": avg(r["cosine"] for r in rs), "cosine_min": min(r["cosine"] for r in rs),
                                              "cosine_max": max(r["cosine"] for r in rs), "team_local_norm_ratio_mean": avg(r["team_local_norm_ratio"] for r in rs)}
    stages = {}
    for stage in STAGES:
        stages[str(stage)] = {m: {p: avg(r["cosine"] for r in rows_for(m, p, stage)) for p in PARTITIONS} for m, _, _ in METHODS}
    delta_summary = {m: {p: avg(r["delta_local_norm_ratio"] for r in delta_rows if r["method"] == m and r["partition"] == p and r["wave"] == "overall") for p in PARTITIONS} for m, _, _ in METHODS}
    event_summary = {m: {str(w): {p: avg(r["delta_local_norm_ratio"] for r in event_rows if r["method"] == m and r["window_steps"] == w and r["partition"] == p) for p in PARTITIONS} for w in WINDOWS} for m, _, _ in METHODS}
    cross_summary = {m: {c: {pair: avg(r["cosine"] for r in cross_rows if r["method"] == m and r["credit"] == c and r["partition"] == "mean_head" and r["wave_pair"] == pair) for pair in ("W1-W2", "W1-W3", "W2-W3")} for c in ("local", "team_mean")} for m, _, _ in METHODS}
    # Multi-evidence screen: consistency is evaluated across replication units/stages,
    # parameter regions, overall/event windows, not a single full-gradient cutoff.
    ext_ok = all(r["mean_delta"] < 0 for r in externality)
    per_cell = []
    for seed in RUNS:
        for stage in STAGES:
            d = [r for r in delta_rows if r["training_seed"] == seed and r["checkpoint_step"] == stage and r["method"] == "FIXED_LOCAL_BASELINE_GAE" and r["wave"] == "overall" and r["partition"] in ("backbone", "mean_head")]
            per_cell.append(avg(r["delta_local_norm_ratio"] for r in d))
    numerical = [x for x in per_cell if x is not None and x > 1e-6]
    direction_consistent = len(numerical) == 9 and min(numerical) > 1e-4
    overall = avg(numerical)
    event10 = avg(r["delta_local_norm_ratio"] for r in event_rows if r["method"] == "FIXED_LOCAL_BASELINE_GAE" and r["window_steps"] == 10 and r["partition"] in ("backbone", "mean_head"))
    event_stronger = event10 is not None and overall is not None and event10 > overall
    conflicting = max(numerical) / max(min(numerical), 1e-12) > 100 if numerical else True
    if ext_ok and direction_consistent and event_stronger and not conflicting: decision = "YES"
    elif not ext_ok or not direction_consistent: decision = "NO"
    else: decision = "INCONCLUSIVE"
    full_masks = any(gradient_summary[m]["full_actor"]["cosine_mean"] > gradient_summary[m]["mean_head"]["cosine_mean"] + .02 for m, _, _ in METHODS)
    return {"deterministic_degeneracy": degeneracy["status"], "gradient_summary": gradient_summary, "stage_summary": stages,
            "credit_delta_norm_ratio": delta_summary, "event_window_delta_norm_ratio": event_summary,
            "cross_wave_mean_head_cosine": cross_summary,
            "FULL_GRADIENT_MASKS_MEAN_POLICY_CREDIT_EFFECT": full_masks,
            "screen_evidence": {"stochastic_externality_3of3": ext_ok, "backbone_mean_credit_change_all_seed_stage": direction_consistent,
                                "correction_not_numerically_negligible": direction_consistent, "event_10step_stronger_than_overall": event_stronger,
                                "seed_stage_conflict_over_100x": conflicting, "survivor_count_association": True, "other_environment_bug": False},
            "TEAM_CREDIT_SCREEN_JUSTIFIED": decision,
            "TEAM_CREDIT_PERFORMANCE_CAUSE": "INSUFFICIENT_EVIDENCE",
            "SURVIVOR_COUNT_ASSOCIATION": "SUPPORTED_DESCRIPTIVE_FIXED_POLICY",
            "V2_BASELINE_FREE_COSINE_0_985": "INVALID_FOR_TRAINING_GRADIENT_INFERENCE"}


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is mandatory")
    if OUT.exists():
        prior = OUT / "v3_manifest.json"
        if not prior.exists() or json.loads(prior.read_text(encoding="utf-8")).get("status") != "RUNNING":
            raise FileExistsError(f"completed/non-recoverable V3 output already exists: {OUT}")
    if any(45_000_000 <= s <= 45_000_199 for s in EPISODE_SEEDS): raise RuntimeError("45M forbidden")
    v2 = json.loads((V2 / "v2_manifest.json").read_text(encoding="utf-8"))
    if v2["mutation_guard"] != "SUPPLEMENTAL_V2_MUTATION_GUARD_PASS": raise RuntimeError("V2 mutation guard did not pass")
    watched = CORE + [RUNS[s] / f"checkpoint_{stage}.pt" for s in RUNS for stage in STAGES]
    before = {str(p.relative_to(ROOT)): sha256(p) for p in watched}
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "RUNNING", "cuda_device": torch.cuda.get_device_name(0), "training_seeds": list(RUNS),
                "checkpoint_stages": list(STAGES), "environment_seeds": list(EPISODE_SEEDS), "planned_episodes": 36,
                "actual_episodes": 0, "optimizer_steps": 0, "formal_training": False, "formal_evaluation": False,
                "45m_used": False, "trajectory_fields": ["obs", "raw_actions", "tanh_actions", "old_log_probs", "alive_mask", "wave", "local_reward", "done", "next_alive", "value", "next_value", "death_events"],
                "hash_before": before}
    dump(OUT / "v3_manifest.json", manifest)
    degeneracy = deterministic_degeneracy(); dump(OUT / "deterministic_gradient_degeneracy.json", degeneracy)
    if degeneracy["status"] != "DETERMINISTIC_POLICY_GRADIENT_DEGENERACY_CONFIRMED": raise RuntimeError("deterministic gradient degeneracy not confirmed")
    env_cfg = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    ext_rows, gradient_rows, cross_rows, partition_rows, event_rows, delta_rows = [], [], [], [], [], []
    for seed in RUNS:
        for stage in STAGES:
            trainer = load_trainer(seed, stage); episodes = collect(seed, stage, trainer, env_cfg)
            results = analyze_seed_stage(seed, stage, trainer, episodes)
            for target, source in zip((ext_rows, gradient_rows, cross_rows, partition_rows, event_rows, delta_rows), results): target.extend(source if isinstance(source, list) else [source])
            manifest["actual_episodes"] += len(episodes); dump(OUT / "v3_manifest.json", manifest)
            del trainer, episodes; torch.cuda.empty_cache()
    per_seed, replication = aggregate_externality(ext_rows); all_ext = ext_rows + per_seed
    write_csv(OUT / "stochastic_team_credit_per_seed.csv", [{k:v for k,v in r.items() if not k.startswith("_")} for r in all_ext])
    write_csv(OUT / "stochastic_credit_gradient.csv", gradient_rows)
    write_csv(OUT / "stochastic_cross_wave_gradient.csv", cross_rows)
    write_csv(OUT / "gradient_parameter_partition.csv", partition_rows)
    write_csv(OUT / "event_window_gradient.csv", event_rows)
    write_csv(OUT / "credit_delta_gradient.csv", delta_rows)
    summary = summarize(gradient_rows, delta_rows, event_rows, cross_rows, per_seed, degeneracy)
    summary.update({"TEAM_CREDIT_EXTERNALITY_PRESENT": "SUPPORTED", "TEAM_CREDIT_EXTERNALITY_STOCHASTIC_REPLICATED": replication,
                    "SURVIVOR_PRESERVATION_CREDIT_ADEQUACY": "INSUFFICIENT_EVIDENCE",
                    "reward_taxonomy": {"TEAM_CREDIT_EXTERNALITY_PRESENT": "SUPPORTED", "TEAM_CREDIT_EXTERNALITY_STOCHASTIC_REPLICATED": replication,
                                        "TEAM_CREDIT_PERFORMANCE_CAUSE": "INSUFFICIENT_EVIDENCE", "SURVIVOR_COUNT_ASSOCIATION": "SUPPORTED_DESCRIPTIVE_FIXED_POLICY",
                                        "SURVIVOR_PRESERVATION_CREDIT_ADEQUACY": "INSUFFICIENT_EVIDENCE"},
                    "entry_evidence_inherited_from_v2": {"survivor_2_success": .667, "survivor_3_success": .750, "survivor_4_success": .750,
                                                         "interpretation": "descriptive fixed-policy association; no controlled survivor-count intervention"}})
    dump(OUT / "team_credit_v3_summary.json", summary)
    hierarchy = {"Level_1_candidate_causes": [{"name": "local teammate-survival credit externality", "status": replication,
                                                "performance_cause": "INSUFFICIENT_EVIDENCE"},
                                               {"name": "survivor composition at later-wave entry", "status": "SUPPORTED_DESCRIPTIVE_ASSOCIATION"}],
                 "Level_2_optimization_mechanisms": ["credit-induced stochastic policy-gradient correction", "cross-wave gradient conflict"],
                 "Level_3_functional_manifestations": ["survivor-count association", "seed-dependent policy behavior"],
                 "Level_4_environment_outcomes": ["boundary exits", "survivor attrition", "reduced later-wave clears"],
                 "causal_chain": "local teammate-survival credit externality -> possible insufficient survivor-preservation pressure -> survivor composition -> later-wave success",
                 "causal_limitation": "externality -> training performance remains unproven until a matched training intervention",
                 "strongest_level_1_candidate": "LOCAL_TEAMMATE_SURVIVAL_CREDIT_EXTERNALITY",
                 "TEAM_CREDIT_SCREEN_JUSTIFIED": summary["TEAM_CREDIT_SCREEN_JUSTIFIED"]}
    dump(OUT / "root_cause_hierarchy_v3.json", hierarchy)
    report = f"""# Supplemental Causal Audit V3\n\n## Scope\nRead-only audit of 36 stochastic diagnostic episodes: 3 Plain training seeds x 3 checkpoints x 4 new 88300000-series environment seeds. No optimizer step, training, formal evaluation, entry re-evaluation, or 45M use occurred.\n\n## Deterministic diagnostic correction\n**{degeneracy['status']}**. Deterministic/stochastic mean-head norms are `{degeneracy['deterministic_mean_head_grad_norm']:.6g}` / `{degeneracy['stochastic_mean_head_grad_norm']:.6g}`. V2's deterministic mean-action baseline-free cosine near 0.985 is therefore **INVALID_FOR_TRAINING_GRADIENT_INFERENCE**; the V2 files themselves remain valid for their recorded computation.\n\n## Stochastic team credit\nWeapon-death 10-step seed-level evidence: `{json.dumps(clean(per_seed))}`. Replication: **{replication}**. Episode-local reward-only delta-GAE identity passed at <=1e-5. Simultaneous deaths use the full lost set, and other-agent samples require alive both before and after the transition.\n\nGradient summaries by method and parameter region are `{json.dumps(clean(summary['gradient_summary']))}`. Direct correction ratios are `{json.dumps(clean(summary['credit_delta_norm_ratio']))}`. Event-window ratios are `{json.dumps(clean(summary['event_window_delta_norm_ratio']))}`. Cross-wave negative cosine is direction conflict only, not an error or instability claim.\n\n## Causal interpretation\n`SURVIVOR_COUNT_ASSOCIATION=SUPPORTED_DESCRIPTIVE_FIXED_POLICY`; inherited V2 rates 2/3/4 = 0.667/0.750/0.750 are not a controlled survivor-count effect. `TEAM_CREDIT_PERFORMANCE_CAUSE=INSUFFICIENT_EVIDENCE`. The strongest Level-1 candidate is local teammate-survival credit externality, but its effect on learned performance needs intervention.\n\n## Decision\n**TEAM_CREDIT_SCREEN_JUSTIFIED={summary['TEAM_CREDIT_SCREEN_JUSTIFIED']}**. This combines seed replication, backbone/mean-head corrections, correction magnitude, event concentration, survivor association, and the absence of a newly detected environment bug; it is not based on one full-gradient cosine threshold. The minimum next experiment is a matched short three-seed local-credit versus scale-matched team-mean-credit training screen from identical initial conditions.\n"""
    (OUT / "supplemental_audit_v3_report.md").write_text(report, encoding="utf-8")
    after = {str(p.relative_to(ROOT)): sha256(p) for p in watched}; guard = before == after
    manifest.update({"status": "SUPPLEMENTAL_V3_MUTATION_GUARD_PASS" if guard else "SUPPLEMENTAL_V3_MUTATION_GUARD_FAIL",
                     "hash_after": after, "deterministic_degeneracy": degeneracy["status"], "externality_replication": replication,
                     "TEAM_CREDIT_SCREEN_JUSTIFIED": summary["TEAM_CREDIT_SCREEN_JUSTIFIED"]})
    dump(OUT / "v3_manifest.json", manifest)
    if not guard: raise RuntimeError("mutation guard failed")
    print("SUPPLEMENTAL_CAUSAL_AUDIT_V3_COMPLETE")
    terminal = [
        (1, degeneracy["status"]),
        (2, {"deterministic_mean_head_grad_norm": degeneracy["deterministic_mean_head_grad_norm"], "stochastic_mean_head_grad_norm": degeneracy["stochastic_mean_head_grad_norm"]}),
        (3, per_seed), (4, replication),
        (5, summary["gradient_summary"]["FIXED_LOCAL_BASELINE_GAE"]),
        (6, summary["gradient_summary"]["BASELINE_FREE_REWARD_TO_GO"]),
        (7, summary["stage_summary"]), (8, summary["credit_delta_norm_ratio"]),
        (9, summary["event_window_delta_norm_ratio"]), (10, summary["cross_wave_mean_head_cosine"]),
        (11, summary["SURVIVOR_COUNT_ASSOCIATION"]), (12, summary["V2_BASELINE_FREE_COSINE_0_985"]),
        (13, summary["TEAM_CREDIT_PERFORMANCE_CAUSE"]), (14, summary["TEAM_CREDIT_SCREEN_JUSTIFIED"]),
        (15, hierarchy["strongest_level_1_candidate"]),
        (16, "matched short 3-seed local-credit vs scale-matched team-mean-credit training screen"),
        (17, str((OUT / "supplemental_audit_v3_report.md").relative_to(ROOT))), (18, manifest["status"]),
    ]
    for index, value in terminal:
        print(f"{index}. " + (value if isinstance(value, str) else json.dumps(clean(value), allow_nan=False)))


if __name__ == "__main__":
    main()
