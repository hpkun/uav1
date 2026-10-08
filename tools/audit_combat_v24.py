"""Bounded-worker mirrored scripted audits of v2.3 and v2.4; no RL training."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from env.combat_env import MultiUAVCombatEnv
from env.geometry import engagement_geometry
from env.fixed_policy import NearestTargetPursuitPolicy
from env.config import load_config


class AuditedCombatEnv(MultiUAVCombatEnv):
    """Instrumentation records existing decisions without changing RNG or policy."""
    def _entry_attempts(self, attackers, targets, fire_states, side):
        attempts = super()._entry_attempts(attackers, targets, fire_states, side)
        for attacker_index, target_index, hit in attempts:
            attacker, target = attackers[attacker_index], targets[target_index]
            geometry = engagement_geometry(attacker, target)
            self.audit_attempts.append({"step": self.steps, "side": side,
                "attacker_index": attacker_index, "target_index": target_index,
                "distance": geometry.distance, "off_boresight": geometry.off_boresight,
                "target_aspect": geometry.target_aspect, "attacker_speed": attacker.v,
                "target_speed": target.v, "hit": bool(hit),
                "attacker_position": [attacker.x, attacker.y, attacker.z],
                "target_position": [target.x, target.y, target.z],
                "attacker_heading": attacker.psi, "target_heading": target.psi})
        return attempts

    def _resolve_combat(self, red_attempts, blue_attempts):
        red, blue = super()._resolve_combat(red_attempts, blue_attempts)
        for side, credited in (("red", red), ("blue", blue)):
            for target, attackers in credited.items():
                self.audit_kills.append({"step": self.steps, "side": side,
                    "target_index": target, "attacker_indices": attackers})
        return red, blue


def state_row(env):
    return {side: {"positions": [[a.x,a.y,a.z] for a in states],
        "speed": [a.v for a in states], "heading": [a.psi for a in states],
        "pitch": [a.theta for a in states], "alive": [bool(a.alive) for a in states]}
        for side,states in (("red",env.red),("blue",env.blue))}


def run_episode(task):
    config, seed, trace_path = task
    env = AuditedCombatEnv(config)
    env.reset(seed)
    # Deliberately instantiate the same scripted policy for Red.
    red_policy = NearestTargetPursuitPolicy(config["blue_policy"],config["action"])
    episode_attempts, episode_kills = [], []
    trace = Path(trace_path).open("w",encoding="utf-8") if trace_path else None
    if trace:
        trace.write(json.dumps({"step":0,**state_row(env),"attempts":[],"kills":[]})+"\n")
    try:
        while True:
            env.audit_attempts, env.audit_kills = [], []
            actions = red_policy.team_actions(env.red, env.blue)
            _, _, terminated, truncated, info = env.step(actions)
            episode_attempts.extend(env.audit_attempts)
            episode_kills.extend(env.audit_kills)
            if trace:
                trace.write(json.dumps({"step":env.steps,**state_row(env),
                    "attempts":env.audit_attempts,"kills":env.audit_kills})+"\n")
            if terminated or truncated:
                break
    finally:
        if trace: trace.close()
    fields = ("red_success","blue_win","draw","termination_reason","episode_length")
    for side in ("red","blue"):
        fields += tuple(f"{side}_{key}" for key in ("survivors","losses","fire_attempts","weapon_hits",
            "attack_kills","boundary_exits","ground_losses","first_fire_window_step","first_attempt_step",
            "first_hit_step","first_kill_step","fire_window_steps","fire_window_pair_steps"))
    return {"seed":seed,**{key:info[key] for key in fields},
            "attempts":episode_attempts,"kills":episode_kills}


def numeric_summary(values, quantiles):
    if not values: return {"count":0, **{key:None for key in ("mean","median","min","max",*quantiles)}}
    array = np.asarray(values,dtype=float)
    return {"count":len(array),"mean":float(array.mean()),"median":float(np.median(array)),
        "min":float(array.min()),"max":float(array.max()),
        **{key:float(np.quantile(array,value)) for key,value in quantiles.items()}}


def summarize(rows,config):
    total=len(rows)
    result={"episodes":total,"team_size":config["scenario"]["team_size"],
        "environment_version":config["environment_version"],
        "red_wins":sum(r["red_success"] for r in rows),"blue_wins":sum(r["blue_win"] for r in rows),
        "draws":sum(r["draw"] for r in rows),
        "timeouts":sum(r["termination_reason"]=="red_failure_timeout" for r in rows),
        "average_episode_length":float(np.mean([r["episode_length"] for r in rows]))}
    for count,rate in (("red_wins","red_win_rate"),("blue_wins","blue_win_rate"),
                       ("draws","draw_rate"),("timeouts","timeout_rate")):
        result[rate]=result[count]/total
    for side in ("red","blue"):
        metrics={}
        for field in ("survivors","losses","fire_attempts","weapon_hits","attack_kills","boundary_exits","ground_losses"):
            metrics[f"average_{field}"]=float(np.mean([r[f"{side}_{field}"] for r in rows]))
            metrics[f"total_{field}"]=sum(r[f"{side}_{field}"] for r in rows)
        for field in ("fire_window_steps","fire_window_pair_steps"):
            if all(f"{side}_{field}" in row for row in rows):
                metrics[f"average_{field}"]=float(np.mean([r[f"{side}_{field}"] for r in rows]))
        for event,label in (("fire_window","fire_window"),("attempt","attempt"),("hit","hit"),("kill","kill")):
            valid=[r[f"{side}_first_{event}_step"] for r in rows if r[f"{side}_first_{event}_step"] is not None]
            metrics[f"{label}_episode_rate"]=len(valid)/total
            metrics[f"average_first_{label}_step"]=float(np.mean(valid)) if valid else None
            metrics[f"first_{label}_observed_episodes"]=len(valid)
        metrics["per_uav_fire_attempts"]=metrics["average_fire_attempts"]/result["team_size"]
        metrics["per_uav_attack_kills"]=metrics["average_attack_kills"]/result["team_size"]
        result[side]=metrics
    # Paired episode outcome difference: timeouts/draws contribute zero.
    outcome=np.asarray([int(row["red_success"])-int(row["blue_win"]) for row in rows],dtype=float)
    sem=float(outcome.std(ddof=1)/np.sqrt(total)) if total>1 else None
    result["side_win_difference_95pct_normal_interval"]=(
        [float(outcome.mean()-1.96*sem),float(outcome.mean()+1.96*sem)] if sem is not None else None)
    result["red_minus_blue"]={"win_rate":result["red_win_rate"]-result["blue_win_rate"],
        **{key:result["red"][key]-result["blue"][key] for key in result["red"]
           if key.startswith("average_") and result["red"][key] is not None and result["blue"][key] is not None}}
    attempts=[event for row in rows for event in row["attempts"]]
    result["attack_geometry"]={
        "distance":numeric_summary([a["distance"] for a in attempts],{"p10":.1,"p90":.9}),
        "off_boresight":numeric_summary([a["off_boresight"] for a in attempts],{"p90":.9}),
        "target_aspect":numeric_summary([a["target_aspect"] for a in attempts],{"p90":.9}),
        "speed_advantage":numeric_summary([a["attacker_speed"]-a["target_speed"] for a in attempts],{"p10":.1})}
    violations=[]
    if config["environment_version"]=="2.4":
        for row in rows:
            for event in row["attempts"]:
                if not (config["weapon"]["range_min"]-1e-9 <= event["distance"] <= config["weapon"]["range_max"]+1e-9
                    and event["off_boresight"] <= config["weapon"]["off_boresight_angle_max"]+1e-10
                    and event["target_aspect"] <= config["weapon"]["target_aspect_angle_max"]+1e-10):
                    violations.append({"seed":row["seed"],**event})
    repeats=0; repeat_episodes=0
    for row in rows:
        seen=set(); this_repeats=0
        for event in row["attempts"]:
            key=(event["side"],event["attacker_index"],event["target_index"])
            this_repeats+=key in seen; seen.add(key)
        repeats+=this_repeats; repeat_episodes+=this_repeats>0
    result.update(attack_qualification_violations=violations,
        repeated_pair_attempts=repeats,episodes_with_repeated_pair_attempts=repeat_episodes,
        head_on_attempts=sum(a["target_aspect"]>np.pi/2 for a in attempts),
        episodes_ending_by_step_50=sum(r["episode_length"]<=50 for r in rows))
    # Diagnostic flags do not retune rules. A substantial bias triggers review.
    flags=[]
    if violations: flags.append("attack_qualification_violation")
    if abs(result["red_minus_blue"]["win_rate"])>.20: flags.append("large_side_win_bias")
    if max(result["red"]["attempt_episode_rate"],result["blue"]["attempt_episode_rate"])<.10: flags.append("almost_no_attack_opportunities")
    if result["timeout_rate"]>.90: flags.append("almost_all_episodes_timeout")
    if result["episodes_ending_by_step_50"]/total>.50: flags.append("predominantly_immediate_endings")
    losses=sum(result[s]["average_losses"] for s in ("red","blue"))
    noncombat=sum(result[s]["average_boundary_exits"]+result[s]["average_ground_losses"] for s in ("red","blue"))
    result["noncombat_loss_fraction"]=noncombat/max(losses,1e-12)
    if losses and result["noncombat_loss_fraction"]>.50: flags.append("noncombat_losses_dominate")
    result["difficulty_review_flags"]=flags
    result["audit_passed"]=not flags
    return result


def audit(config,episodes,seed_base,workers,output,trace_count):
    version=config["environment_version"]
    folder=output/f"v{version.replace('.','')}"
    folder.mkdir(parents=True,exist_ok=True)
    traces=folder/"traces"; traces.mkdir(exist_ok=True)
    tasks=[(config,seed_base+i,str(traces/f"seed_{seed_base+i}.jsonl") if i<trace_count else None)
           for i in range(episodes)]
    rows=[]; start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn")) as pool:
        for index,row in enumerate(pool.map(run_episode,tasks,chunksize=1),1):
            rows.append(row)
            if index%25==0 or index==episodes:
                progress={"version":version,"completed_episodes":index,"episodes":episodes,
                    "elapsed_seconds":time.perf_counter()-start}
                (folder/"progress.json").write_text(json.dumps(progress,indent=2))
                print(json.dumps(progress),flush=True)
    with (folder/"episodes.jsonl").open("w",encoding="utf-8") as stream:
        for row in rows: stream.write(json.dumps(row)+"\n")
    result=summarize(rows,config)
    result.update(seed_base=seed_base,seed_end=seed_base+episodes-1,workers=workers,
        elapsed_seconds=time.perf_counter()-start,
        config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        trace_seeds=list(range(seed_base,seed_base+min(trace_count,episodes))))
    (folder/"audit_report.json").write_text(json.dumps(result,indent=2))
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--episodes",type=int,default=1000)
    parser.add_argument("--seed-base",type=int,default=40_000_000)
    parser.add_argument("--workers",type=int,choices=(1,2),default=2)
    parser.add_argument("--trace-count",type=int,default=10)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/v24_combat_audit")
    parser.add_argument("--versions",nargs="+",choices=("2.3","2.4"),default=("2.4","2.3"))
    args=parser.parse_args()
    if args.episodes<=0 or not 0<=args.trace_count<=args.episodes:
        parser.error("episodes positive; trace-count must be between zero and episodes")
    # Only the coordinator imports torch; spawned simulation workers stay light.
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable for repository audit")
    print("CUDA available: "+torch.cuda.get_device_name(),flush=True)
    reports={}
    for version in args.versions:
        config=load_config(ROOT/("configs/combat_environment.yaml" if version=="2.3" else "configs/combat_environment_v24.yaml"))
        reports[version]=audit(config,args.episodes,args.seed_base,args.workers,args.output_dir,args.trace_count)
        (args.output_dir/"audit_report.json").write_text(json.dumps({"versions":reports},indent=2))
    if "2.3" in reports and "2.4" in reports:
        comparison={version:{"timeout_rate":r["timeout_rate"],"average_episode_length":r["average_episode_length"],
            **{side:r[side] for side in ("red","blue")}} for version,r in reports.items()}
        (args.output_dir/"comparison.json").write_text(json.dumps(comparison,indent=2))
    print(json.dumps({version:{k:r[k] for k in ("red_win_rate","blue_win_rate","timeout_rate","audit_passed","difficulty_review_flags")}
                      for version,r in reports.items()}),flush=True)
    if any(not report["audit_passed"] for report in reports.values()):
        raise SystemExit("Difficulty audit failed; preserve formal parameters and review saved episodes")


if __name__=="__main__": main()
