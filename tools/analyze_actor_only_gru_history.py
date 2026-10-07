"""Strictly offline CPU analysis of Actor-only GRU History versus Plain MAPPO."""
from __future__ import annotations

import argparse,csv,json,math,statistics
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1];SEEDS=(5301,5302,5303);TARGETS=(900_000,1_500_000,2_000_000,3_000_000)

def read_csv(path:Path)->list[dict[str,Any]]:
    with path.open(encoding="utf-8",newline="") as stream:
        rows=[]
        for raw in csv.DictReader(stream):
            row={}
            for key,value in raw.items():
                try:row[key]=float(value) if value not in (None,"") else None
                except ValueError:row[key]=value
            rows.append(row)
    if not rows:raise RuntimeError(f"empty evaluation history: {path}")
    return rows

def read_jsonl(path:Path)->list[dict[str,Any]]:return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
def ratio(num,den):return None if num is None or den in (None,0) else num/den
def metrics(row):
    w1,w2,w3=(row.get(f"clear_wave_{i}_probability") for i in (1,2,3))
    return {"sampled_steps":int(row["sampled_steps"]),"w1":w1,"w2":w2,"w3":w3,"q2":ratio(w2,w1),"q3":ratio(w3,w2),"average_waves":row.get("average_waves_cleared"),"return":row.get("average_return"),"red_loss":row.get("average_red_loss"),"blue_loss":row.get("average_blue_loss"),"boundary":row.get("average_red_boundary_exits"),"ground":row.get("average_red_ground_losses"),"timeout":row.get("timeout_rate"),"episode_length":row.get("average_episode_length")}
def select(rows,target):return min(rows,key=lambda row:abs(int(row["sampled_steps"])-target))
def mean(values):
    values=[float(x) for x in values if x is not None];return None if not values else statistics.fmean(values)
def summarize(values):
    values=[float(x) for x in values if x is not None];return {"mean":mean(values),"sample_std":statistics.stdev(values) if len(values)>1 else 0.,"min":min(values),"max":max(values)}
def write_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf-8",newline="") as stream:w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(rows)

def audit(root:Path,method:str,seed:int):
    directory=root/(f"l3_seed{seed}" if method=="plain" else f"seed{seed}")
    required=("evaluation_history.csv","run_summary.json","run_config.json","optimization_metrics.jsonl","latest.pt","final.pt","checkpoint_3000000.pt")
    missing=[x for x in required if not (directory/x).is_file()]
    if missing:raise RuntimeError(f"incomplete {method} seed{seed}: {missing}")
    summary=json.loads((directory/"run_summary.json").read_text(encoding="utf-8"));run=json.loads((directory/"run_config.json").read_text(encoding="utf-8"))
    if int(summary.get("sampled_steps",-1))!=3_000_000:raise RuntimeError(f"non-final run: {directory}")
    if method=="gru":
        expected={"development_method":"actor_only_gru_history","actor_input_dim":52,"actor_context_dim":0,"critic_context_dim":0}
        if any(run.get(k)!=v for k,v in expected.items()) or run.get("enabled_modules")!=["actor_lr_decay","recurrent_memory"]:raise RuntimeError(f"GRU protocol mismatch: {directory}")
    history=read_csv(directory/"evaluation_history.csv");final=metrics(history[-1]);peak=max(history,key=lambda x:float(x.get("average_waves_cleared") or 0));peak_metric=metrics(peak)
    diagnostics={}
    if method=="gru":
        rows=read_jsonl(directory/"optimization_metrics.jsonl");keys=("actor_hidden_norm","actor_hidden_norm_max","actor_gru_grad_norm","critic_gru_grad_norm","hidden_reset_count_total")
        if any(any(k not in row or not math.isfinite(float(row[k])) for k in keys) for row in rows):raise RuntimeError(f"invalid recurrent diagnostics: {directory}")
        diagnostics={"training_seed":seed,"updates":len(rows),"actor_hidden_norm_mean":mean([x["actor_hidden_norm"] for x in rows]),"actor_hidden_norm_max":max(x["actor_hidden_norm_max"] for x in rows),"actor_gru_grad_norm_mean":mean([x["actor_gru_grad_norm"] for x in rows]),"actor_gru_grad_norm_max":max(x["actor_gru_grad_norm"] for x in rows),"critic_gru_grad_norm_max":max(x["critic_gru_grad_norm"] for x in rows),"hidden_reset_count":rows[-1]["hidden_reset_count_total"]}
        if diagnostics["actor_hidden_norm_max"]<=0 or diagnostics["actor_gru_grad_norm_max"]<=0 or diagnostics["critic_gru_grad_norm_max"]!=0:raise RuntimeError(f"inactive/misconfigured recurrent path: {directory}")
    return {"method":method,"seed":seed,"directory":str(directory),"history":history,"final":final,"peak":peak_metric,"retention":None if peak_metric["average_waves"] in (None,0) else final["average_waves"]/peak_metric["average_waves"],"diagnostics":diagnostics}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--plain-root",default="outputs/diag_mappo_learnability");parser.add_argument("--gru-root",default="outputs/dev_actor_only_gru_history_3m");parser.add_argument("--output-dir",default="outputs/actor_only_gru_history_analysis");args=parser.parse_args()
    roots={k:(Path(v) if Path(v).is_absolute() else ROOT/v) for k,v in {"plain":args.plain_root,"gru":args.gru_root}.items()};records=[audit(roots[m],m,s) for s in SEEDS for m in ("plain","gru")]
    get=lambda method,seed:next(x for x in records if x["method"]==method and x["seed"]==seed)
    endpoints=[]
    for record in records:
        for target in TARGETS:endpoints.append({"method":record["method"],"training_seed":record["seed"],"target_step":target,**metrics(select(record["history"],target))})
    paired=[]
    for seed in SEEDS:
        plain,gru=get("plain",seed)["final"],get("gru",seed)["final"];row={"training_seed":seed}
        for key in ("w1","w2","w3","q2","q3","average_waves","return","red_loss","blue_loss","boundary","ground","timeout","episode_length"):
            row[f"plain_{key}"]=plain[key];row[f"gru_{key}"]=gru[key];row[f"delta_{key}"]=None if plain[key] is None or gru[key] is None else gru[key]-plain[key]
        paired.append(row)
    aggregate={key:{**summarize([r[f"delta_{key}"] for r in paired]),"paired_wins":sum(r[f"delta_{key}"] is not None and r[f"delta_{key}"]>0 for r in paired)} for key in ("w1","w2","w3","q2","q3","average_waves","return")}
    gate=(aggregate["average_waves"]["mean"]>=.15 and aggregate["average_waves"]["paired_wins"]>=2 and aggregate["w1"]["mean"]>=-.05 and (aggregate["q2"]["mean"]>0 or aggregate["q3"]["mean"]>0))
    retention=[{"method":r["method"],"training_seed":r["seed"],"peak_step":r["peak"]["sampled_steps"],"peak_average_waves":r["peak"]["average_waves"],"final_average_waves":r["final"]["average_waves"],"retention":r["retention"]} for r in records]
    result={"status":"ANALYSIS_COMPLETE","primary_endpoint":3_000_000,"replication_unit":"training_seed_n3","evaluation_episodes_are_replicates":False,"no_new_evaluation":True,"success_gate":{"mean_delta_average_waves":aggregate["average_waves"]["mean"],"paired_average_waves_wins":aggregate["average_waves"]["paired_wins"],"mean_delta_w1":aggregate["w1"]["mean"],"mean_delta_q2":aggregate["q2"]["mean"],"mean_delta_q3":aggregate["q3"]["mean"],"label":"ACTOR_ONLY_GRU_HISTORY_SUPPORTED" if gate else "ACTOR_ONLY_GRU_HISTORY_NOT_SUPPORTED"},"aggregate":aggregate,"history_diagnostics":[r["diagnostics"] for r in records if r["method"]=="gru"]}
    out=Path(args.output_dir);out=out if out.is_absolute() else ROOT/out;out.mkdir(parents=True,exist_ok=True);write_csv(out/"endpoints.csv",endpoints);write_csv(out/"paired_3m.csv",paired);write_csv(out/"peak_retention.csv",retention);write_csv(out/"history_diagnostics.csv",result["history_diagnostics"]);(out/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8");(out/"report.md").write_text(f"# Actor-only GRU History analysis\n\nPrimary label: **{result['success_gate']['label']}**\n",encoding="utf-8");print(json.dumps({"status":result["status"],"label":result["success_gate"]["label"],"output_dir":str(out)},indent=2))

if __name__=="__main__":main()
