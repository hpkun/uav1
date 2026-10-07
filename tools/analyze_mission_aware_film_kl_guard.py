"""Small CPU-only offline analyzer for FiLM Actor-KL-Guard development runs."""
from __future__ import annotations

import argparse,csv,json,statistics
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(5301,5302,5303)

def read_rows(path:Path)->list[dict[str,float]]:
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

def q(num,den):return None if num is None or den in (None,0) else num/den
def metrics(row:dict[str,Any])->dict[str,Any]:
    w1,w2,w3=(row.get(f"clear_wave_{i}_probability") for i in (1,2,3))
    return {"sampled_steps":int(row["sampled_steps"]),"w1":w1,"w2":w2,"w3":w3,"q2":q(w2,w1),"q3":q(w3,w2),
      "average_waves":row.get("average_waves_cleared"),"return":row.get("average_return"),
      "boundary":row.get("average_red_boundary_exits"),"ground":row.get("average_red_ground_losses")}

def directory(root:Path,method:str,seed:int)->Path:
    if method=="plain":return root/f"l3_seed{seed}"
    return root/f"seed{seed}"

def audit(root:Path,method:str,seed:int)->dict[str,Any]:
    path=directory(root,method,seed);required=("run_summary.json","evaluation_history.csv","latest.pt","final.pt","checkpoint_3000000.pt")
    missing=[name for name in required if not (path/name).is_file()]
    if missing:raise RuntimeError(f"incomplete {method} seed{seed}: {missing}")
    summary=json.loads((path/"run_summary.json").read_text(encoding="utf-8"))
    if int(summary.get("sampled_steps",-1))!=3_000_000:raise RuntimeError(f"non-final run: {path}")
    history=read_rows(path/"evaluation_history.csv");final=metrics(history[-1]);peak_row=max(history,key=lambda r:float(r.get("average_waves_cleared") or 0));peak=metrics(peak_row)
    retention=None if peak["average_waves"] in (None,0) else final["average_waves"]/peak["average_waves"]
    guard=summary.get("actor_kl_guard_summary",{})
    if method=="guard" and not guard.get("enabled"):raise RuntimeError(f"guard provenance missing: {path}")
    return {"method":method,"training_seed":seed,"directory":str(path),"final":final,"peak":peak,"retention":retention,"guard":guard}

def mean(values):
    valid=[float(v) for v in values if v is not None]
    return None if not valid else statistics.fmean(valid)

def delta(candidate:dict[str,Any],reference:dict[str,Any],label:str)->dict[str,Any]:
    row={"training_seed":candidate["training_seed"],"comparison":label}
    for key in ("w1","w2","w3","q2","q3","average_waves","return","boundary","ground"):
        a,b=candidate["final"][key],reference["final"][key];row[f"guard_{key}"]=a;row[f"reference_{key}"]=b;row[f"delta_{key}"]=None if a is None or b is None else a-b
    return row

def write_csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:return
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w",encoding="utf-8",newline="") as stream:w=csv.DictWriter(stream,fieldnames=keys);w.writeheader();w.writerows(rows)

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--guard-root",default="outputs/dev_mission_aware_film_kl_guard_3m");parser.add_argument("--film-root",default="outputs/dev_mission_aware_film_3m");parser.add_argument("--plain-root",default="outputs/diag_mappo_learnability");parser.add_argument("--output-dir",default="outputs/mission_aware_film_kl_guard_analysis");args=parser.parse_args()
    roots={k:(Path(v) if Path(v).is_absolute() else ROOT/v) for k,v in {"guard":args.guard_root,"film":args.film_root,"plain":args.plain_root}.items()}
    records=[audit(roots[method],method,seed) for seed in SEEDS for method in ("guard","film","plain")]
    get=lambda method,seed:next(r for r in records if r["method"]==method and r["training_seed"]==seed)
    paired=[]
    for seed in SEEDS:
        paired.extend((delta(get("guard",seed),get("film",seed),"guard_minus_film"),delta(get("guard",seed),get("plain",seed),"guard_minus_plain")))
    guard=[get("guard",seed) for seed in SEEDS];film_rows=[r for r in paired if r["comparison"]=="guard_minus_film"];plain_rows=[r for r in paired if r["comparison"]=="guard_minus_plain"]
    retention_mean=mean([r["retention"] for r in guard])
    gate_a=(mean([r["delta_average_waves"] for r in film_rows])>=.15 and sum(r["delta_average_waves"]>0 for r in film_rows)>=2 and retention_mean is not None and retention_mean>=.90)
    plain_q2=mean([r["delta_q2"] for r in plain_rows]);plain_q3=mean([r["delta_q3"] for r in plain_rows])
    gate_b=(mean([r["delta_average_waves"] for r in plain_rows])>=.15 and sum(r["delta_average_waves"]>0 for r in plain_rows)>=2 and mean([r["delta_w1"] for r in plain_rows])>=-.05 and ((plain_q2 is not None and plain_q2>0) or (plain_q3 is not None and plain_q3>0)))
    result={"status":"ANALYSIS_COMPLETE","replication_unit":"training_seed_n3","no_new_evaluation":True,
      "gate_a_label":"KL_GUARD_RETENTION_SUPPORTED" if gate_a else "KL_GUARD_RETENTION_NOT_SUPPORTED",
      "gate_b_label":"MISSION_AWARE_FILM_KL_GUARD_PROMISING" if gate_b else "MISSION_AWARE_FILM_KL_GUARD_NOT_PROMISING",
      "mean_final_to_observed_peak_retention":retention_mean,
      "guard_diagnostics":[{"training_seed":r["training_seed"],**r["guard"]} for r in guard]}
    out=Path(args.output_dir);out=out if out.is_absolute() else ROOT/out;out.mkdir(parents=True,exist_ok=True)
    endpoints=[];retention=[]
    for r in records:
        endpoints.append({"method":r["method"],"training_seed":r["training_seed"],**r["final"]})
        retention.append({"method":r["method"],"training_seed":r["training_seed"],"peak_step":r["peak"]["sampled_steps"],"peak_average_waves":r["peak"]["average_waves"],"final_average_waves":r["final"]["average_waves"],"retention":r["retention"]})
    write_csv(out/"three_method_3m.csv",endpoints);write_csv(out/"paired_deltas.csv",paired);write_csv(out/"peak_retention.csv",retention);write_csv(out/"guard_diagnostics.csv",result["guard_diagnostics"])
    (out/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8");(out/"report.md").write_text(f"# Mission-Aware FiLM Actor KL Guard analysis\n\nGate A: **{result['gate_a_label']}**  \nGate B: **{result['gate_b_label']}**\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"gate_a":result["gate_a_label"],"gate_b":result["gate_b_label"],"output_dir":str(out)},indent=2))

if __name__=="__main__":main()
