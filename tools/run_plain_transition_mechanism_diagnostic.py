"""Run the matched 44M Plain-MAPPO transition mechanism diagnostic."""
from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithm.common.evaluator import episode_return_metrics
from algorithm.modular_mappo.evaluation import per_wave_episode_diagnostics
from algorithm.modules.wave_survival_pbrs import mission_context_numpy
from env.factory import make_combat_environment
from tools.plain_transition_diagnostic_common import (
    CHECKPOINTS, EVALUATION_SEEDS, OUTPUT_DIR, POLICY_SEEDS,
    canonical_spawn_seed, canonicalize_spawn, classify_death, continuation_should_stop,
    dump_death_trace, future_rng_seed, load_trainer, new_ring_buffers, step_trace_row,
    transition_snapshot,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def action_for(trainer, env, observation, alive, actor_hidden, episode_mask):
    wave = np.asarray([env.wave_index]); total = np.asarray([env.total_waves])
    context = mission_context_numpy(
        trainer, wave, total, env.blue_alive_mask[None],
        np.asarray([env.steps]), env.max_steps,
    )
    actions, actor_hidden = trainer.act(
        observation[None], alive[None], True, False, context,
        actor_hidden, episode_mask,
    )
    return actions[0], actor_hidden


def _wave_row(record: dict, wave: int) -> dict:
    source = next((x for x in record.get("per_wave_metrics", []) if int(x["wave_index"]) == wave), None)
    names = {
        "recorded": None, "cleared": "wave_cleared", "duration": "duration_steps",
        "red_survivors_start": "red_survivors_start", "red_survivors_end": "red_survivors_end",
        "ground_losses": "red_ground_losses", "boundary_losses": "red_boundary_exits",
        "red_attack_kills": "red_attack_kills", "team_return": "team_return",
    }
    out = {}
    for name, key in names.items():
        out[f"wave_{wave}_{name}"] = (source is not None) if name == "recorded" else (None if source is None else source.get(key))
    return out


def _risk_template() -> dict:
    return {"living":0,"risk":0,"activations":0,"max_consecutive":0,
            "min_boundary_margin":None,"min_positive_time_to_boundary":None}


def _update_risk(bucket: dict, row: dict, previous: bool, consecutive: int) -> tuple[bool, int]:
    risk = bool(row["would_trigger_blue_ground_guard"])
    bucket["living"] += 1; bucket["risk"] += int(risk)
    if risk and not previous: bucket["activations"] += 1
    consecutive = consecutive + 1 if risk else 0
    bucket["max_consecutive"] = max(bucket["max_consecutive"], consecutive)
    margin = float(row["boundary_margin"])
    bucket["min_boundary_margin"] = margin if bucket["min_boundary_margin"] is None else min(bucket["min_boundary_margin"], margin)
    ttb = row["time_to_boundary"]
    if ttb is not None and ttb >= 0:
        bucket["min_positive_time_to_boundary"] = ttb if bucket["min_positive_time_to_boundary"] is None else min(bucket["min_positive_time_to_boundary"], ttb)
    return risk, consecutive


def run_direct_episode(trainer, contract: dict, policy_seed: int, evaluation_seed: int):
    env = make_combat_environment(contract["environment_config"])
    observation, _ = env.reset(evaluation_seed)
    alive = env.red_alive_mask.copy(); actor_hidden, _ = trainer.initial_hidden(1)
    episode_mask = np.zeros(1, np.float32); returns = np.zeros(4, dtype=float)
    rings = new_ring_buffers(); deaths=[]; transitions=[]; transition_agents=[]; clones=[]
    risk = {wave:_risk_template() for wave in (1,2,3)}
    previous = {(wave,agent):False for wave in (1,2,3) for agent in range(4)}
    consecutive = {(wave,agent):0 for wave in (1,2,3) for agent in range(4)}
    transition_indices=[]
    while True:
        wave = int(env.wave_index); actions, actor_hidden = action_for(
            trainer, env, observation, alive, actor_hidden, episode_mask
        )
        pre_alive = np.asarray(env.red_alive_mask, dtype=bool).copy()
        for agent in range(4):
            if not pre_alive[agent]:
                continue
            trace = step_trace_row(env, agent, actions[agent], wave)
            rings[agent].append(trace)
            key=(wave,agent)
            previous[key],consecutive[key]=_update_risk(
                risk[wave],trace,previous[key],consecutive[key]
            )
        observation, reward, terminated, truncated, info = env.step(actions)
        returns += reward
        post_alive = np.asarray(env.red_alive_mask, dtype=bool).copy()
        for agent in np.flatnonzero(pre_alive & ~post_alive):
            cause=classify_death(env.red[int(agent)],env.arena_radius)
            deaths.append(dump_death_trace(rings[int(agent)],policy_seed,evaluation_seed,wave,int(agent),cause,
                int(env.steps),env.red[int(agent)],env.arena_radius))
        if info.get("spawned_next_wave",False):
            summary, agents = transition_snapshot(env,policy_seed,evaluation_seed,wave)
            summary["spawn_radial_angle"] = info.get("wave_spawn_radial_angle")
            summary["source_wave_ground_risk_ratio"] = risk[wave]["risk"] / max(risk[wave]["living"],1)
            summary["source_wave_ground_risk_steps"] = risk[wave]["risk"]
            summary["source_wave_living_agent_steps"] = risk[wave]["living"]
            summary["next_wave_clear"] = None
            transitions.append(summary); transition_agents.extend(agents)
            transition_indices.append(len(transitions)-1)
            clones.append((deepcopy(env),dict(summary)))
            for agent in range(4):
                previous[(wave+1,agent)]=False;consecutive[(wave+1,agent)]=0
        alive=np.asarray(info["red_alive_mask"],np.float32)
        actor_hidden=trainer.recurrent.apply_alive(actor_hidden,alive[None])
        episode_mask[:]=1
        if terminated or truncated:
            team_return,_=episode_return_metrics(returns)
            flat=per_wave_episode_diagnostics(info,3)
            direct={"policy_training_seed":policy_seed,"evaluation_seed":evaluation_seed,
                    "waves_cleared":int(info["waves_cleared"]),
                    "clear_wave_1":int(info["waves_cleared"]>=1),"clear_wave_2":int(info["waves_cleared"]>=2),"clear_wave_3":int(info["waves_cleared"]>=3),
                    "episode_return":team_return,"red_losses":int(info["red_losses"]),"blue_losses":int(info["blue_losses"]),
                    "red_attack_kills":int(info["red_attack_kills"]),"boundary_losses":int(info["red_boundary_exits"]),
                    "ground_losses":int(info["red_ground_losses"]),"timeout":int(info["termination_reason"]=="red_failure_timeout"),
                    "episode_length":int(info["episode_length"]),"termination_reason":info["termination_reason"]}
            for w in (1,2,3): direct.update(_wave_row(info,w))
            for transition in transitions:
                nxt=int(transition["next_wave"])
                row=next((x for x in info.get("per_wave_metrics",[]) if int(x["wave_index"])==nxt),None)
                transition["next_wave_clear"]=bool(row and row.get("wave_cleared",False))
            risk_rows=[]
            all_bucket=_risk_template()
            for w in (1,2,3):
                b=risk[w]
                risk_rows.append({"policy_training_seed":policy_seed,"evaluation_seed":evaluation_seed,"wave":w,
                                  "living_agent_decision_steps":b["living"],"ground_risk_steps":b["risk"],
                                  "ground_risk_ratio":b["risk"]/max(b["living"],1),"ground_risk_activation_count":b["activations"],
                                  "max_consecutive_ground_risk_steps":b["max_consecutive"],"minimum_boundary_margin":b["min_boundary_margin"],
                                  "minimum_positive_time_to_boundary":b["min_positive_time_to_boundary"]})
                all_bucket["living"]+=b["living"];all_bucket["risk"]+=b["risk"];all_bucket["activations"]+=b["activations"]
                all_bucket["max_consecutive"]=max(all_bucket["max_consecutive"],b["max_consecutive"])
                for k in ("min_boundary_margin","min_positive_time_to_boundary"):
                    if b[k] is not None: all_bucket[k]=b[k] if all_bucket[k] is None else min(all_bucket[k],b[k])
            risk_rows.append({"policy_training_seed":policy_seed,"evaluation_seed":evaluation_seed,"wave":0,
                              "living_agent_decision_steps":all_bucket["living"],"ground_risk_steps":all_bucket["risk"],
                              "ground_risk_ratio":all_bucket["risk"]/max(all_bucket["living"],1),"ground_risk_activation_count":all_bucket["activations"],
                              "max_consecutive_ground_risk_steps":all_bucket["max_consecutive"],"minimum_boundary_margin":all_bucket["min_boundary_margin"],
                              "minimum_positive_time_to_boundary":all_bucket["min_positive_time_to_boundary"]})
            return direct,transitions,transition_agents,deaths,risk_rows,clones


def aggregate_direct(rows: list[dict]) -> dict:
    mean=lambda key:float(np.mean([float(x[key]) for x in rows]))
    w1,w2,w3=(mean(f"clear_wave_{w}") for w in (1,2,3))
    return {"W1":w1,"W2":w2,"W3":w3,"Q2":w2/w1 if w1 else None,"Q3":w3/w2 if w2 else None,
            "average_waves":mean("waves_cleared"),"average_return":mean("episode_return"),
            "average_red_loss":mean("red_losses"),"average_blue_loss":mean("blue_losses"),
            "average_boundary":mean("boundary_losses"),"average_ground":mean("ground_losses"),
            "timeout_rate":mean("timeout"),"average_episode_length":mean("episode_length")}


def historical_row(policy_seed: int) -> dict:
    path=ROOT/f"outputs/diag_mappo_learnability/l3_seed{policy_seed}/evaluation_history.csv"
    with path.open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    row=next(x for x in rows if int(x["sampled_steps"])==3_000_000)
    w1=float(row["clear_wave_1_probability"]);w2=float(row["clear_wave_2_probability"]);w3=float(row["clear_wave_3_probability"])
    return {"W1":w1,"W2":w2,"W3":w3,"Q2":w2/w1 if w1 else None,"Q3":w3/w2 if w2 else None,
            "average_waves":float(row["average_waves_cleared"]),"average_return":float(row["average_return"]),
            "average_red_loss":float(row["average_red_loss"]),"average_blue_loss":float(row["average_blue_loss"]),
            "average_boundary":float(row["average_red_boundary_exits"]),"average_ground":float(row["average_red_ground_losses"]),
            "timeout_rate":float(row["timeout_rate"]),"average_episode_length":float(row["average_episode_length"])}


def run_continuation(source_env, entry: dict, trainers: dict[int,object], mode="native") -> list[dict]:
    results=[]; initial=None
    for continuation_seed,trainer in trainers.items():
        env=deepcopy(source_env)
        spawn_seed=None;spawn_angle=None
        if mode == "canonical_spawn":
            spawn_seed,spawn_angle=canonicalize_spawn(env,int(entry["evaluation_seed"]),int(entry["next_wave"]))
        future_seed=future_rng_seed(int(entry["evaluation_seed"]),int(entry["next_wave"]))
        env.rng=np.random.default_rng(future_seed)
        obs=env._observations()
        if initial is None: initial=obs.copy()
        elif not np.array_equal(initial,obs): raise RuntimeError("same-state policies received different initial observations")
        alive=env.red_alive_mask.copy();hidden,_=trainer.initial_hidden(1);ep=np.zeros(1,np.float32)
        start_step=int(env.steps);start_alive=int(env.red_alive_mask.sum())
        counts={name:int(env.combat_counts["red"][name]) for name in ("attack_kills","boundary_exits","ground_losses")}
        ret=0.0; success=False;reason="unknown"
        while True:
            wave=int(env.wave_index); actions,hidden=action_for(trainer,env,obs,alive,hidden,ep)
            obs,reward,terminated,truncated,info=env.step(actions);ret+=float(np.sum(reward))
            alive=np.asarray(info["red_alive_mask"],np.float32);hidden=trainer.recurrent.apply_alive(hidden,alive[None]);ep[:]=1
            stop,success=continuation_should_stop(wave,int(entry["next_wave"]),info,terminated,truncated)
            if stop:
                reason="next_wave_clear" if success else str(info["termination_reason"]);break
        ground=int(env.combat_counts["red"]["ground_losses"])-counts["ground_losses"]
        boundary=int(env.combat_counts["red"]["boundary_exits"])-counts["boundary_exits"]
        red_lost=start_alive-int(env.red_alive_mask.sum());combat=max(0,red_lost-ground-boundary)
        row={"source_policy_seed":entry["source_policy_seed"],"evaluation_seed":entry["evaluation_seed"],
             "source_cleared_wave":entry["cleared_wave"],"next_wave":entry["next_wave"],
             "continuation_policy_seed":continuation_seed,
             "entry_mode":mode,"canonical_spawn_seed":spawn_seed,"canonical_spawn_radial_angle":spawn_angle,
             "future_rng_seed":future_seed,
             "entry_global_step":start_step,"entry_remaining_horizon":env.max_steps-start_step,"entry_survivors":start_alive,
             "next_wave_clear":int(success),"continuation_steps":int(env.steps-start_step),"red_survivors_end":int(env.red_alive_mask.sum()),
             "red_losses_during_continuation":red_lost,"ground_losses":ground,"boundary_losses":boundary,"combat_losses":combat,
             "blue_kills":int(env.combat_counts["red"]["attack_kills"])-counts["attack_kills"],
             "incremental_return":ret,"termination_reason":reason,
             "altitude_min":entry["red_altitude_min"],"boundary_margin_min":entry["red_boundary_margin_min"],
             "formation_spread":entry["red_formation_spread"],"pairwise_mean":entry["red_pairwise_distance_mean"],
             "minimum_spawn_distance":entry["minimum_spawn_distance"],"minimum_red_blue_distance":entry["minimum_red_blue_distance"]}
        results.append(row)
    return results


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",default=str(OUTPUT_DIR));parser.add_argument("--device",choices=("cuda",),default="cuda");parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args();out=Path(args.output_dir)
    if out.exists() and not args.overwrite: raise FileExistsError(f"diagnostic output exists: {out}")
    out.mkdir(parents=True,exist_ok=True)
    trainers={};contracts={}
    for seed in POLICY_SEEDS: trainers[seed],contracts[seed]=load_trainer(CHECKPOINTS[seed],args.device)
    direct=[];transitions=[];agents=[];deaths=[];risks=[];clones=[]
    for seed in POLICY_SEEDS:
        for case in EVALUATION_SEEDS:
            values=run_direct_episode(trainers[seed],contracts[seed],seed,case)
            d,t,a,x,r,c=values;direct.append(d);transitions.extend(t);agents.extend(a);deaths.extend(x);risks.extend(r);clones.extend(c)
            print(f"[DIRECT] policy={seed} case={case} waves={d['waves_cleared']}",flush=True)
    write_csv(out/"direct_case_results.csv",direct);write_csv(out/"transition_states.csv",transitions)
    write_csv(out/"transition_agents.csv",agents);write_csv(out/"ground_risk_cases.csv",risks)
    with (out/"death_pretrace.jsonl").open("w",encoding="utf-8") as stream:
        for row in deaths: stream.write(json.dumps(row,separators=(",",":"))+"\n")
    integrity={"status":"DIAGNOSTIC_REPLAY_MATCHED","policies":{}}
    for seed in POLICY_SEEDS:
        observed=aggregate_direct([x for x in direct if x["policy_training_seed"]==seed]);history=historical_row(seed)
        delta={k:observed[k]-history[k] for k in observed if observed[k] is not None and history[k] is not None}
        matched=all(abs(delta[k])<=.02 for k in ("W1","W2","W3")) and abs(delta["average_waves"])<=.05
        integrity["policies"][str(seed)]={"historical":history,"diagnostic":observed,"delta":delta,"matched":matched}
        if not matched: integrity["status"]="DIAGNOSTIC_REPLAY_MISMATCH"
    (out/"replay_integrity.json").write_text(json.dumps(integrity,indent=2),encoding="utf-8")
    if integrity["status"]!="DIAGNOSTIC_REPLAY_MATCHED":
        print(json.dumps(integrity,indent=2));raise RuntimeError("DIAGNOSTIC_REPLAY_MISMATCH; counterfactual stopped")
    continuation=[]
    for index,(source,entry) in enumerate(clones,1):
        continuation.extend(run_continuation(source,entry,trainers))
        print(f"[CONT] {index}/{len(clones)} source={entry['source_policy_seed']} case={entry['evaluation_seed']} next={entry['next_wave']}",flush=True)
    write_csv(out/"continuation_results.csv",continuation)
    canonical_transitions=[];canonical_continuation=[]
    for index,(source,entry) in enumerate(clones,1):
        canonical=deepcopy(source);spawn_seed,spawn_angle=canonicalize_spawn(canonical,int(entry["evaluation_seed"]),int(entry["next_wave"]))
        summary, _ = transition_snapshot(canonical,int(entry["source_policy_seed"]),int(entry["evaluation_seed"]),int(entry["cleared_wave"]))
        summary.update({"entry_mode":"canonical_spawn","canonical_spawn_seed":spawn_seed,
                        "canonical_spawn_radial_angle":spawn_angle,"canonical_source_red_state_unchanged":True,
                        "source_wave_ground_risk_ratio":entry["source_wave_ground_risk_ratio"],
                        "source_wave_ground_risk_steps":entry["source_wave_ground_risk_steps"],
                        "source_wave_living_agent_steps":entry["source_wave_living_agent_steps"],
                        "next_wave_clear":entry["next_wave_clear"]})
        canonical_transitions.append(summary)
        canonical_continuation.extend(run_continuation(source,entry,trainers,"canonical_spawn"))
        print(f"[CANONICAL] {index}/{len(clones)} source={entry['source_policy_seed']} case={entry['evaluation_seed']} next={entry['next_wave']}",flush=True)
    if len(direct) != len(POLICY_SEEDS) * len(EVALUATION_SEEDS):
        raise RuntimeError(f"direct episode count mismatch: {len(direct)}")
    if len(canonical_transitions) != len(transitions):
        raise RuntimeError(f"canonical transition count mismatch: {len(canonical_transitions)} != {len(transitions)}")
    expected_continuations = len(POLICY_SEEDS) * len(transitions)
    if len(continuation) != expected_continuations:
        raise RuntimeError(f"native continuation count mismatch: {len(continuation)} != {expected_continuations}")
    if len(canonical_continuation) != expected_continuations:
        raise RuntimeError(f"canonical continuation count mismatch: {len(canonical_continuation)} != {expected_continuations}")
    write_csv(out/"canonical_transition_states.csv",canonical_transitions)
    write_csv(out/"canonical_continuation_results.csv",canonical_continuation)
    armed_ok=all((not bool(x["alive"])) or bool(x["fire_armed"]) for x in agents)
    simultaneous=sum(bool(row.get("simultaneous_boundary_ground",False)) for row in deaths)
    metadata={"status":"DIAGNOSTIC_COMPLETE","diagnostic_protocol_version":2,
              "replay_integrity":integrity["status"],"transition_weapon_state_reset":"TRANSITION_WEAPON_STATE_RESET_PASS" if armed_ok else "TRANSITION_WEAPON_STATE_RESET_FAIL",
              "policy_seeds":list(POLICY_SEEDS),"evaluation_seed_start":EVALUATION_SEEDS[0],"evaluation_seed_end":EVALUATION_SEEDS[-1],
              "direct_episode_count":len(direct),"native_transition_count":len(transitions),"canonical_transition_count":len(canonical_transitions),
              "native_continuation_count":len(continuation),"canonical_continuation_count":len(canonical_continuation),
              "transition_count":len(transitions),"continuation_count":len(continuation),
              "simultaneous_boundary_ground_count":simultaneous,"canonical_spawn_mode":"policy-invariant diagnostic-only Blue respawn",
              "future_rng_mode":"matched initial future RNG stream, not event-wise coupled randomness",
              "matched_future_rng_note":"matched initial future RNG stream, not event-wise coupled randomness",
              "training":False,"policy_update":False,"future_final_45m_used":False}
    (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(metadata,indent=2))


if __name__=="__main__": main()
