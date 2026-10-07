"""Offline paired analysis for L3 MAPPO versus actor mission context; never evaluates."""
from __future__ import annotations

import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from algorithm.common.protocol import config_sha256
from tools.analyze_mappo_baseline_learnability import read_eval,select_endpoint,safe_ratio,audit_actual_environment

MANIFEST=ROOT/"experiments/actor_mission_context_development_manifest.json"
DEST=ROOT/"outputs/actor_mission_context_analysis"
ENDPOINTS=(900_000,1_500_000,2_000_000,3_000_000)
METRICS=("w1","w2","w3","q2","q3","average_waves","return","red_loss","blue_loss",
         "kill_loss_ratio","boundary","ground","timeout","episode_length",
         "red_survivors_entering_w2","red_survivors_entering_w3",
         "wave1_clear_time","wave2_clear_time","wave3_clear_time")

def finite(row,key):
    value=row.get(key)
    try:value=float(value)
    except (TypeError,ValueError):return None
    return value if math.isfinite(value) else None

def endpoint_row(method,seed,row):
    w1,w2,w3=(finite(row,f"clear_wave_{k}_probability") for k in (1,2,3))
    return {"method":method,"training_seed":seed,"sampled_steps":int(row["sampled_steps"]),
        "w1":w1,"w2":w2,"w3":w3,"q2":safe_ratio(w2,w1),"q3":safe_ratio(w3,w2),
        "average_waves":finite(row,"average_waves_cleared"),"return":finite(row,"average_return"),
        "red_loss":finite(row,"average_red_loss"),"blue_loss":finite(row,"average_blue_loss"),
        "kill_loss_ratio":finite(row,"kill_loss_ratio"),"boundary":finite(row,"average_red_boundary_exits"),
        "ground":finite(row,"average_red_ground_losses"),"timeout":finite(row,"timeout_rate"),
        "episode_length":finite(row,"average_episode_length"),
        "red_survivors_entering_w2":finite(row,"red_survivors_after_wave_1"),
        "red_survivors_entering_w3":finite(row,"red_survivors_after_wave_2"),
        **{f"wave{k}_clear_time":finite(row,f"wave_{k}_duration") for k in (1,2,3)}}

def write_csv(path,rows):
    fields=sorted({k for row in rows for k in row})
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def locate_baseline(root,seed):
    for candidate in (root/f"l3_seed{seed}",root/f"seed{seed}"):
        if candidate.is_dir():return candidate
    return root/f"l3_seed{seed}"

def audit_run(directory,method,seed,manifest):
    required=("env_config.yaml","runtime_env_config.yaml","run_config.json","run_summary.json",
              "evaluation_history.csv","training_metrics.jsonl","optimization_metrics.jsonl","latest.pt","final.pt","checkpoint_3000000.pt")
    missing=[name for name in required if not (directory/name).is_file()]
    if missing:raise RuntimeError(f"incomplete {method} seed{seed}: {missing}")
    run=json.loads((directory/"run_config.json").read_text(encoding="utf-8"))
    summary=json.loads((directory/"run_summary.json").read_text(encoding="utf-8"))
    state=torch.load(directory/"latest.pt",map_location="cpu",weights_only=False);extra=state.get("extra",{})
    if int(state.get("sampled_steps",-1))!=3_000_000 or int(summary.get("sampled_steps",-1))!=3_000_000:
        raise RuntimeError(f"incomplete 3M endpoint: {directory}")
    if not all(torch.isfinite(value).all().item() for group in (state.get("actor",{}),state.get("critic",{})) for value in group.values()):
        raise RuntimeError(f"non-finite checkpoint tensors: {directory}")
    pseudo={"conditions":{"L3":{"total_waves":3,"max_steps":3000}}}
    env_audit=audit_actual_environment(directory,{"condition":"L3"},pseudo,state)
    if env_audit["status"]!="VALID_FULL_3_WAVE_BASELINE":raise RuntimeError(f"invalid runtime environment: {directory}")
    if method=="actor_mission_context":
        expected={"development_method":"actor_mission_context","wave_context_target":"actor_only",
                  "wave_context_encoding":"mission_markov","mission_context_dim":5,
                  "actor_input_dim":57,"critic_context_dim":0}
        for key,value in expected.items():
            if run.get(key)!=value or extra.get(key)!=value:raise RuntimeError(f"method provenance mismatch {key}: {directory}")
    elif run.get("enabled_modules")!=["actor_lr_decay"]:
        raise RuntimeError(f"baseline is not valid Plain MAPPO L3: {directory}")
    rows=read_eval(directory/"evaluation_history.csv")
    return rows,{"method":method,"training_seed":seed,"directory":str(directory),
        "sampled_steps":int(state["sampled_steps"]),"evaluation_rows":len(rows),
        "checkpoint_map_location":"cpu","finite_checkpoint":True,"environment":env_audit}

def summarize(values):
    clean=[float(v) for v in values if v is not None]
    return {"mean":float(np.mean(clean)) if clean else None,
            "sample_std":float(np.std(clean,ddof=1)) if len(clean)>1 else None,
            "min":min(clean) if clean else None,"max":max(clean) if clean else None}

def development_label(aggregate,complete=True):
    d_aw=aggregate["delta_average_waves"]["mean"];d_w1=aggregate["delta_w1"]["mean"]
    q_improved=(aggregate["delta_q2"]["mean"] is not None and aggregate["delta_q2"]["mean"]>0) or (aggregate["delta_q3"]["mean"] is not None and aggregate["delta_q3"]["mean"]>0)
    promising=(complete and d_aw is not None and d_aw>=.15 and aggregate["delta_average_waves"]["paired_wins_of_3"]>=2 and d_w1 is not None and d_w1>=-.05 and q_improved)
    if promising:return "ACTOR_MISSION_CONTEXT_PROMISING"
    if d_aw is None or d_aw<=0 or (d_w1 is not None and d_w1<-.05):return "ACTOR_MISSION_CONTEXT_NOT_SUPPORTED"
    return "ACTOR_MISSION_CONTEXT_WEAK_OR_MIXED"

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--baseline-root",default="outputs/diag_mappo_learnability");
    parser.add_argument("--method-root",default="outputs/dev_actor_mission_context_3m");parser.add_argument("--output-dir",default=str(DEST));args=parser.parse_args()
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"));baseline_root=Path(args.baseline_root);method_root=Path(args.method_root)
    if not baseline_root.is_absolute():baseline_root=ROOT/baseline_root
    if not method_root.is_absolute():method_root=ROOT/method_root
    output=Path(args.output_dir);output=output if output.is_absolute() else ROOT/output
    histories={};integrity=[];endpoints=[]
    for seed in manifest["training_seeds"]:
        for method,directory in (("baseline",locate_baseline(baseline_root,seed)),
                                 ("actor_mission_context",method_root/f"seed{seed}")):
            history,audit=audit_run(directory,method,seed,manifest);histories[(method,seed)]=history;integrity.append(audit)
            for target in ENDPOINTS:
                row=endpoint_row(method,seed,select_endpoint(history,target));row["target_endpoint"]=target;endpoints.append(row)
    paired=[]
    for target in ENDPOINTS:
        for seed in manifest["training_seeds"]:
            base=next(r for r in endpoints if r["method"]=="baseline" and r["training_seed"]==seed and r["target_endpoint"]==target)
            new=next(r for r in endpoints if r["method"]=="actor_mission_context" and r["training_seed"]==seed and r["target_endpoint"]==target)
            row={"training_seed":seed,"target_endpoint":target,"baseline_sampled_steps":base["sampled_steps"],"method_sampled_steps":new["sampled_steps"]}
            for key in ("w1","w2","w3","q2","q3","average_waves","return","red_loss","boundary","ground"):
                row[f"baseline_{key}"]=base[key];row[f"method_{key}"]=new[key]
                row[f"delta_{key}"]=None if base[key] is None or new[key] is None else new[key]-base[key]
            paired.append(row)
    final=[r for r in paired if r["target_endpoint"]==3_000_000]
    aggregate={}
    for key in ("w1","w2","w3","q2","q3","average_waves","return","red_loss","boundary","ground"):
        values=[r[f"delta_{key}"] for r in final];lower_is_better=key in {"red_loss","boundary","ground"}
        wins=sum(v is not None and (v<0 if lower_is_better else v>0) for v in values)
        aggregate[f"delta_{key}"]={**summarize(values),"paired_wins_of_3":wins,
                                    "improvement_direction":"negative" if lower_is_better else "positive"}
    complete=len(integrity)==6
    label=development_label(aggregate,complete)
    result={"status":"ANALYSIS_COMPLETE","development_label":label,"replication_unit":"training_seed_n3",
        "paired_initialization_note":manifest["paired_seed_note"],"primary_metric":"3M average_waves_cleared",
        "gate":manifest["development_gate"],"aggregate_3m":aggregate,"integrity":integrity,
        "no_episode_level_significance_test":True,"no_policy_evaluation_performed":True}
    output.mkdir(parents=True,exist_ok=True);write_csv(output/"endpoints.csv",endpoints);write_csv(output/"paired_deltas.csv",paired)
    (output/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    (output/"report.md").write_text("# Actor mission-context development analysis\n\n"+f"Decision: **{label}**\n\n```json\n"+json.dumps(result,indent=2)+"\n```\n",encoding="utf-8")
    print(json.dumps({"status":"ANALYSIS_COMPLETE","development_label":label,"output_dir":str(output)},indent=2))

if __name__=="__main__":main()
