"""Offline-only paired Plain versus CAIW-MAPPO V2 development analyzer."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];STEPS=(900000,1500000,2000000,2400000,2700000,3000000);SEEDS=(5301,5302,5303)
def rows(path):
    with Path(path).open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))
def nearest(values,step):return min(values,key=lambda x:abs(int(x["sampled_steps"])-step))
def exact_row(values,step):
    row=next((x for x in values if int(x["sampled_steps"])==int(step)),None)
    if row is None:raise RuntimeError("CAIW_PRIMARY_ENDPOINT_3M_MISSING")
    return row
def primary_endpoint(run_dir,values):
    path=Path(run_dir)/"run_summary.json"
    if not path.exists():raise RuntimeError("CAIW_DEVELOPMENT_INCOMPLETE: run_summary.json missing")
    if int(json.loads(path.read_text(encoding="utf-8")).get("sampled_steps",-1))!=3000000:raise RuntimeError("CAIW_DEVELOPMENT_INCOMPLETE: run_summary sampled_steps != 3000000")
    try:return exact_row(values,3000000)
    except RuntimeError as error:raise RuntimeError(f"CAIW_DEVELOPMENT_INCOMPLETE: {error}") from error
def metric(row,key):return None if row.get(key) in (None,"") else float(row[key])
def summarize(row):
    w1,w2,w3=(metric(row,f"clear_wave_{w}_probability") for w in (1,2,3))
    return {"sampled_steps":int(row["sampled_steps"]),"W1":w1,"W2":w2,"W3":w3,"Q2":None if not w1 else w2/w1,"Q3":None if not w2 else w3/w2,"average_waves":metric(row,"average_waves_cleared"),"return":metric(row,"average_return"),"red_loss":metric(row,"average_red_loss"),"blue_loss":metric(row,"average_blue_loss"),"ground":metric(row,"average_red_ground_losses"),"boundary":metric(row,"average_red_boundary_exits"),"episode_length":metric(row,"average_episode_length")}
def primary_gate(paired):
    means={k:float(np.mean([row[k] for row in paired])) for k in paired[0] if k.startswith("delta_")}
    passed=(means["delta_average_waves"]>=.15 and sum(row["delta_average_waves"]>0 for row in paired)>=2 and means["delta_W1"]>=-.05 and means["delta_Q2"]>0 and means["delta_Q3"]>0 and max(means["delta_Q2"],means["delta_Q3"])>=.05)
    return passed,means
def mechanism(path):
    values=[json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    def series(key):return np.asarray([float(row.get(key,0)) for row in values],float)
    result={"updates":len(values)}
    for task in ("w1_to_w2","w1_to_w3","w2_to_w3"):
        ready=series(f"caiw_{task}_ready");result[task]={"first_ready_step":next((int(r["sampled_steps"]) for r in values if float(r.get(f"caiw_{task}_ready",0))>.5),None),"ready_fraction":float(ready.mean()) if len(ready) else None,"final_readiness":bool(ready[-1]>.5) if len(ready) else False,"validation_pass_fraction":float(series(f"caiw_{task}_validation_pass").mean()) if values else None,"validation_auroc_mean":float(series(f"caiw_{task}_validation_auroc").mean()) if values else None,"validation_brier_skill_mean":float(series(f"caiw_{task}_validation_brier_skill").mean()) if values else None,"freshness_accepted_fraction_mean":float(series(f"caiw_{task}_freshness_accepted_fraction").mean()) if values else None}
    for key in ("caiw_actor_active","caiw_active_wave1","caiw_active_wave2","caiw_adv_abs_mean","caiw_action_sensitivity_abs_mean","caiw_aux_to_tactical_ratio_post","caiw_conflict","caiw_projection_applied","caiw_trust_cap_applied","caiw_aux_induced_clip"):
        x=series(key);result[f"{key}_mean"]=float(x.mean()) if len(x) else None
    if values and float(series("caiw_aux_to_tactical_ratio_post").max())>.250001:result["protocol_anomaly"]="CAIW_AUX_RATIO_CAP_EXCEEDED"
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument("--caiw-root",default="outputs/dev_caiw_mappo_v2_3m");a=p.parse_args();plain=ROOT/"outputs/diag_mappo_learnability";caiw=ROOT/a.caiw_root;paired=[];curves={};mechanisms={}
    try:
        for seed in SEEDS:
            pd=plain/f"l3_seed{seed}";cd=caiw/f"seed{seed}";pr=rows(pd/"evaluation_history.csv");cr=rows(cd/"evaluation_history.csv");pp=primary_endpoint(pd,pr);cp=primary_endpoint(cd,cr)
            curves[str(seed)]={"plain":[{"requested_step":s,**summarize(nearest(pr,s) if s<3000000 else pp)} for s in STEPS],"caiw":[{"requested_step":s,**summarize(nearest(cr,s) if s<3000000 else cp)} for s in STEPS]};mechanisms[str(seed)]=mechanism(cd/"optimization_metrics.jsonl")
            x,y=curves[str(seed)]["plain"][-1],curves[str(seed)]["caiw"][-1];paired.append({"seed":seed,**{f"delta_{k}":y[k]-x[k] for k in ("W1","W2","W3","Q2","Q3","average_waves","return","red_loss","blue_loss","ground","boundary","episode_length")}})
    except (FileNotFoundError,RuntimeError) as error:
        print(json.dumps({"offline_only":True,"result":"CAIW_DEVELOPMENT_INCOMPLETE","reason":str(error)},indent=2));raise SystemExit(2)
    passed,means=primary_gate(paired);print(json.dumps({"offline_only":True,"all_three_complete":True,"primary_endpoint":"exact_3000000","curves":curves,"mechanism_metrics":mechanisms,"paired_deltas":paired,"mean_deltas":means,"primary_gate":"CAIW_SUPPORTED" if passed else "CAIW_NOT_SUPPORTED"},indent=2))
if __name__=="__main__":main()
