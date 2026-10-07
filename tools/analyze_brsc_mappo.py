"""Offline-only paired Plain versus BRSC-MAPPO V1 development analyzer."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];SEEDS=(5301,5302,5303);STEPS=(900000,1500000,2000000,2400000,2700000,3000000)
TASKS=("w1_to_w2","w1_to_w3","w2_to_w3")

def csv_rows(path):
    with Path(path).open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))
def nearest(values,step):return min(values,key=lambda row:abs(int(row["sampled_steps"])-step))
def metric(row,key):return None if row.get(key) in (None,"") else float(row[key])
def endpoint(run_dir,values):
    summary=Path(run_dir)/"run_summary.json"
    if not summary.exists() or int(json.loads(summary.read_text(encoding="utf-8")).get("sampled_steps",-1))!=3000000:raise RuntimeError(f"BRSC_DEVELOPMENT_INCOMPLETE: {run_dir} is not exact 3M")
    row=next((item for item in values if int(item["sampled_steps"])==3000000),None)
    if row is None:raise RuntimeError(f"BRSC_DEVELOPMENT_INCOMPLETE: {run_dir} lacks exact 3M evaluation")
    return row
def summarize(row):
    w1,w2,w3=(metric(row,f"clear_wave_{wave}_probability") for wave in (1,2,3))
    return {"sampled_steps":int(row["sampled_steps"]),"W1":w1,"W2":w2,"W3":w3,
            "Q2":None if not w1 else w2/w1,"Q3":None if not w2 else w3/w2,
            "average_waves":metric(row,"average_waves_cleared"),"return":metric(row,"average_return"),
            "red_loss":metric(row,"average_red_loss"),"blue_loss":metric(row,"average_blue_loss"),
            "ground":metric(row,"average_red_ground_losses"),"boundary":metric(row,"average_red_boundary_exits"),
            "episode_length":metric(row,"average_episode_length")}
def mechanism(path):
    rows=[json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    def values(key,selected=None):return np.asarray([float(row.get(key,0)) for row in (rows if selected is None else selected)],float)
    result={"updates":len(rows)}
    for task in TASKS:
        validation=[row for row in rows if float(row.get(f"brsc_{task}_validation_segments",0))>0]
        ready=values(f"brsc_{task}_ready")
        result[task]={"first_ready_step":next((int(row["sampled_steps"]) for row in rows if float(row.get(f"brsc_{task}_ready",0))>.5),None),
          "final_ready":bool(ready[-1]>.5) if len(ready) else False,"ready_fraction":float(ready.mean()) if len(ready) else None,
          "validation_checks":len(validation),
          "validation_auroc_mean":float(values(f"brsc_{task}_validation_auroc",validation).mean()) if validation else None,
          "validation_brier_skill_mean":float(values(f"brsc_{task}_validation_brier_skill",validation).mean()) if validation else None}
    for key in ("brsc_actor_active","brsc_adv_abs_mean","brsc_segment_length_mean","brsc_credit_distance_mean","brsc_boundary_credit_abs_mean_wave1","brsc_boundary_credit_abs_mean_wave2","brsc_conflict","brsc_projection_applied","brsc_trust_cap_applied","brsc_aux_to_tactical_ratio_post","brsc_aux_induced_clip"):
        data=values(key);result[f"{key}_mean"]=float(data.mean()) if len(data) else None
    for wave in (1,2):
        events=values(f"brsc_boundary_events_wave{wave}").sum();credited=values(f"brsc_credited_boundaries_wave{wave}").sum()
        result[f"brsc_credited_boundary_fraction_wave{wave}"]=float(credited/events) if events else None
    maximum=float(values("brsc_aux_to_tactical_ratio_post").max()) if rows else 0.;result["brsc_aux_to_tactical_ratio_post_max"]=maximum
    if maximum>.250001:result["protocol_anomaly"]="BRSC_PROTOCOL_ANOMALY"
    return result
def gate(paired):
    means={key:float(np.mean([row[key] for row in paired])) for key in paired[0] if key.startswith("delta_")}
    passed=(means["delta_average_waves"]>=.15 and sum(row["delta_average_waves"]>0 for row in paired)>=2 and means["delta_W1"]>=-.05 and means["delta_Q2"]>0 and means["delta_Q3"]>0 and max(means["delta_Q2"],means["delta_Q3"])>=.05)
    return passed,means
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--brsc-root",default="outputs/dev_brsc_mappo_v1_3m");args=parser.parse_args()
    plain=ROOT/"outputs/diag_mappo_learnability";brsc=ROOT/args.brsc_root;curves={};mechanisms={};paired=[]
    try:
        for seed in SEEDS:
            pd=plain/f"l3_seed{seed}";bd=brsc/f"seed{seed}";pr=csv_rows(pd/"evaluation_history.csv");br=csv_rows(bd/"evaluation_history.csv");pe=endpoint(pd,pr);be=endpoint(bd,br)
            curves[str(seed)]={"plain":[{"requested_step":s,**summarize(nearest(pr,s) if s<3000000 else pe)} for s in STEPS],"brsc":[{"requested_step":s,**summarize(nearest(br,s) if s<3000000 else be)} for s in STEPS]}
            mechanisms[str(seed)]=mechanism(bd/"optimization_metrics.jsonl");x,y=curves[str(seed)]["plain"][-1],curves[str(seed)]["brsc"][-1]
            paired.append({"seed":seed,**{f"delta_{key}":y[key]-x[key] for key in ("W1","W2","W3","Q2","Q3","average_waves","return","red_loss","blue_loss","ground","boundary","episode_length")}})
    except (FileNotFoundError,RuntimeError) as error:
        print(json.dumps({"offline_only":True,"result":"BRSC_DEVELOPMENT_INCOMPLETE","reason":str(error)},indent=2));raise SystemExit(2)
    passed,means=gate(paired);anomaly=any("protocol_anomaly" in row for row in mechanisms.values())
    print(json.dumps({"offline_only":True,"all_three_complete":True,"primary_endpoint":"exact_3000000","curves":curves,"mechanism_metrics":mechanisms,"paired_deltas":paired,"mean_deltas":means,"primary_gate":"BRSC_PROTOCOL_ANOMALY" if anomaly else ("BRSC_SUPPORTED" if passed else "BRSC_NOT_SUPPORTED")},indent=2))
if __name__=="__main__":main()
