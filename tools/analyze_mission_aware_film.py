"""CPU-only offline analysis of Plain, raw-context, and Mission-Aware FiLM MAPPO."""
from __future__ import annotations

import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.analyze_actor_mission_context import endpoint_row,finite,locate_baseline,summarize,write_csv
from tools.analyze_mappo_baseline_learnability import audit_actual_environment,read_eval,select_endpoint

MANIFEST=ROOT/"experiments/mission_aware_film_development_manifest.json";DEST=ROOT/"outputs/mission_aware_film_analysis"
ENDPOINTS=(900_000,1_500_000,2_000_000,3_000_000);DIAG_TARGETS=("early",900_000,1_500_000,2_000_000,3_000_000)

def locate(root,seed,baseline=False):return locate_baseline(root,seed) if baseline else root/f"seed{seed}"
def audit_run(directory,method,seed):
    required=("env_config.yaml","runtime_env_config.yaml","run_config.json","run_summary.json","evaluation_history.csv","training_metrics.jsonl","optimization_metrics.jsonl","latest.pt","final.pt","checkpoint_3000000.pt")
    missing=[x for x in required if not (directory/x).is_file()]
    if missing:raise RuntimeError(f"incomplete {method} seed{seed}: {missing}")
    run=json.loads((directory/"run_config.json").read_text(encoding="utf-8"));summary=json.loads((directory/"run_summary.json").read_text(encoding="utf-8"))
    state=torch.load(directory/"latest.pt",map_location="cpu",weights_only=False);extra=state.get("extra",{})
    if int(state.get("sampled_steps",-1))!=3_000_000 or int(summary.get("sampled_steps",-1))!=3_000_000:raise RuntimeError(f"incomplete endpoint: {directory}")
    if not all(torch.isfinite(v).all() for group in (state.get("actor",{}),state.get("critic",{})) for v in group.values()):raise RuntimeError(f"nonfinite checkpoint: {directory}")
    env_audit=audit_actual_environment(directory,{"condition":"L3"},{"conditions":{"L3":{"total_waves":3,"max_steps":3000}}},state)
    if env_audit["status"]!="VALID_FULL_3_WAVE_BASELINE":raise RuntimeError(f"environment mismatch: {directory}")
    expectations={
      "baseline":{"actor_input_dim":52,"actor_context_dim":0,"critic_context_dim":0},
      "raw_context":{"development_method":"actor_mission_context","actor_input_dim":57,"actor_context_dim":5,"critic_context_dim":0},
      "mission_aware_film":{"development_method":"mission_aware_film","actor_input_dim":52,"actor_context_dim":5,"critic_context_dim":0,"mission_film_enabled":True,"mission_film_mode":"bounded_augmented_film","mission_encoder_hidden_dim":32,"mission_film_alpha":.2,"mission_film_identity_init":True,"mission_film_augmented_residual":True}}
    for key,value in expectations[method].items():
        actual=run.get(key,run.get("network_architecture",{}).get(key));actual_extra=extra.get(key,extra.get("network_architecture",{}).get(key))
        if actual!=value or actual_extra!=value:raise RuntimeError(f"{method} provenance mismatch {key}: {actual}/{actual_extra}")
    return read_eval(directory/"evaluation_history.csv"),{"method":method,"training_seed":seed,"directory":str(directory),"sampled_steps":3_000_000,"evaluation_rows":len(read_eval(directory/"evaluation_history.csv")),"checkpoint_map_location":"cpu","finite_checkpoint":True,"environment":env_audit}

def select_diagnostics(directory,seed):
    rows=[json.loads(x) for x in (directory/"optimization_metrics.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    required=("film_delta_gamma_abs_mean","film_beta_abs_mean","film_residual_abs_mean","film_saturation_fraction")
    if not rows or any(k not in row for row in rows for k in required):raise RuntimeError(f"Film diagnostics missing: {directory}")
    selected=[]
    for target in DIAG_TARGETS:
        row=rows[0] if target=="early" else min(rows,key=lambda x:abs(int(x["sampled_steps"])-target))
        selected.append({"training_seed":seed,"target_endpoint":target,"sampled_steps":int(row["sampled_steps"]),**{k:finite(row,k) for k in required}})
    return selected

def label(aggregate,complete):
    aw=aggregate["delta_average_waves"]["mean"];w1=aggregate["delta_w1"]["mean"]
    qi=any(aggregate[k]["mean"] is not None and aggregate[k]["mean"]>0 for k in ("delta_q2","delta_q3"))
    if complete and aw is not None and aw>=.15 and aggregate["delta_average_waves"]["paired_wins_of_3"]>=2 and w1 is not None and w1>=-.05 and qi:return "MISSION_AWARE_FILM_PROMISING"
    if aw is None or aw<=0 or (w1 is not None and w1<-.05):return "MISSION_AWARE_FILM_NOT_SUPPORTED"
    return "MISSION_AWARE_FILM_WEAK_OR_MIXED"

def main():
    p=argparse.ArgumentParser();p.add_argument("--baseline-root",default="outputs/diag_mappo_learnability");p.add_argument("--raw-context-root",default="outputs/dev_actor_mission_context_3m");p.add_argument("--film-root",default="outputs/dev_mission_aware_film_3m");p.add_argument("--output-dir",default=str(DEST));args=p.parse_args()
    roots={"baseline":Path(args.baseline_root),"raw_context":Path(args.raw_context_root),"mission_aware_film":Path(args.film_root)}
    roots={k:(v if v.is_absolute() else ROOT/v) for k,v in roots.items()};out=Path(args.output_dir);out=out if out.is_absolute() else ROOT/out
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"));endpoints=[];integrity=[];diagnostics=[]
    for seed in manifest["training_seeds"]:
      for method in ("baseline","raw_context","mission_aware_film"):
        directory=locate(roots[method],seed,method=="baseline");history,audit=audit_run(directory,method,seed);integrity.append(audit)
        for target in ENDPOINTS:
            row=endpoint_row(method,seed,select_endpoint(history,target));row["target_endpoint"]=target;endpoints.append(row)
        if method=="mission_aware_film":diagnostics.extend(select_diagnostics(directory,seed))
    paired=[]
    for target in ENDPOINTS:
      for seed in manifest["training_seeds"]:
        base=next(r for r in endpoints if r["method"]=="baseline" and r["training_seed"]==seed and r["target_endpoint"]==target);film=next(r for r in endpoints if r["method"]=="mission_aware_film" and r["training_seed"]==seed and r["target_endpoint"]==target)
        row={"training_seed":seed,"target_endpoint":target,"baseline_sampled_steps":base["sampled_steps"],"film_sampled_steps":film["sampled_steps"]}
        for key in ("w1","w2","w3","q2","q3","average_waves","return","red_loss","boundary","ground"):
            row[f"baseline_{key}"]=base[key];row[f"film_{key}"]=film[key];row[f"delta_{key}"]=None if base[key] is None or film[key] is None else film[key]-base[key]
        paired.append(row)
    final=[r for r in paired if r["target_endpoint"]==3_000_000];aggregate={}
    for key in ("w1","w2","w3","q2","q3","average_waves","return","red_loss","boundary","ground"):
        values=[r[f"delta_{key}"] for r in final];lower=key in {"red_loss","boundary","ground"};aggregate[f"delta_{key}"]={**summarize(values),"paired_wins_of_3":sum(v is not None and (v<0 if lower else v>0) for v in values),"improvement_direction":"negative" if lower else "positive"}
    result={"status":"ANALYSIS_COMPLETE","development_label":label(aggregate,len(integrity)==9),"replication_unit":"training_seed_n3","primary_gate_baseline":"Plain MAPPO L3","raw_context_role":"descriptive ablation only","gate":manifest["development_gate"],"aggregate_3m":aggregate,"integrity":integrity,"no_episode_level_significance_test":True,"no_policy_evaluation_performed":True}
    out.mkdir(parents=True,exist_ok=True);write_csv(out/"three_method_endpoints.csv",endpoints);write_csv(out/"film_vs_plain_paired_deltas.csv",paired);write_csv(out/"film_diagnostics.csv",diagnostics);(out/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8");(out/"report.md").write_text(f"# Mission-Aware FiLM development analysis\n\nDecision: **{result['development_label']}**\n\n```json\n{json.dumps(result,indent=2)}\n```\n",encoding="utf-8");print(json.dumps({"status":result["status"],"development_label":result["development_label"],"output_dir":str(out)},indent=2))
if __name__=="__main__":main()
