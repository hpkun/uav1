"""Offline-only paired Plain versus IWSC development analyzer."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STEPS=(900000,1500000,2000000,2400000,2700000,3000000)

def rows(path):
    with path.open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))
def nearest(values,step):return min(values,key=lambda x:abs(int(x["sampled_steps"])-step))
def exact_row(values,sampled_steps):
    row=next((row for row in values if int(row["sampled_steps"])==int(sampled_steps)),None)
    if row is None:raise RuntimeError("IWSC_PRIMARY_ENDPOINT_3M_MISSING")
    return row
def primary_endpoint(run_dir,values):
    summary_path=Path(run_dir)/"run_summary.json"
    if not summary_path.exists():raise RuntimeError("IWSC_DEVELOPMENT_INCOMPLETE: run_summary.json missing")
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("sampled_steps",-1))!=3000000:raise RuntimeError("IWSC_DEVELOPMENT_INCOMPLETE: run_summary sampled_steps != 3000000")
    try:return exact_row(values,3000000)
    except RuntimeError as error:raise RuntimeError(f"IWSC_DEVELOPMENT_INCOMPLETE: {error}") from error
def metric(row,key):
    value=row.get(key);return None if value in (None,"") else float(value)
def summarize(row):
    w1,w2,w3=(metric(row,f"clear_wave_{i}_probability") for i in (1,2,3))
    return {"sampled_steps":int(row["sampled_steps"]),"W1":w1,"W2":w2,"W3":w3,
      "Q2":None if not w1 else w2/w1,"Q3":None if not w2 else w3/w2,
      "average_waves":metric(row,"average_waves_cleared"),"return":metric(row,"average_return"),
      "red_loss":metric(row,"average_red_loss"),"blue_loss":metric(row,"average_blue_loss"),
      "ground":metric(row,"average_red_ground_losses"),"boundary":metric(row,"average_red_boundary_exits"),
      "episode_length":metric(row,"average_episode_length")}
def mechanism(path):
    values=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    def first(key):
        row=next((x for x in values if float(x.get(key,0))>.5),None);return None if row is None else int(row["sampled_steps"])
    def mean(key):return float(np.mean([float(x.get(key,0)) for x in values])) if values else None
    last=values[-1] if values else {}
    return {"first_ready_wave1":first("iw_ready_wave1"),"first_ready_wave2":first("iw_ready_wave2"),
      "first_actor_active_wave1":first("iw_active_wave1"),"first_actor_active_wave2":first("iw_active_wave2"),
      "actor_active_update_fraction":mean("iw_actor_active"),"projection_fraction":mean("iw_projection_applied"),
      "gradient_conflict_fraction":mean("iw_gradient_conflict"),
      "mean_aux_to_tactical_grad_ratio":float(np.mean([float(x.get("iw_aux_grad_norm",0))/max(float(x.get("iw_tactical_grad_norm",0)),1e-12) for x in values])),
      "mean_delta_wave1":mean("iw_delta_mean_wave1"),"mean_delta_wave2":mean("iw_delta_mean_wave2"),
      "final_critic_loss":last.get("iw_critic_loss"),"final_mae_wave1":last.get("iw_mae_wave1"),"final_mae_wave2":last.get("iw_mae_wave2"),
      "final_target_mean_wave1":last.get("iw_target_mean_wave1"),"final_target_std_wave1":last.get("iw_target_std_wave1"),
      "final_target_mean_wave2":last.get("iw_target_mean_wave2"),"final_target_std_wave2":last.get("iw_target_std_wave2"),
      "final_replay_segments_wave1":last.get("iw_replay_segments_wave1"),"final_replay_segments_wave2":last.get("iw_replay_segments_wave2")}
def main():
    p=argparse.ArgumentParser();p.add_argument("--iw-root",default="outputs/dev_iwsc_mappo_3m");args=p.parse_args()
    plain_root=ROOT/"outputs/diag_mappo_learnability";iw_root=ROOT/args.iw_root;paired=[];curves={};mechanisms={};complete=True
    try:
      for seed in (5301,5302,5303):
        plain_dir=plain_root/f"l3_seed{seed}";iw_dir=iw_root/f"seed{seed}"
        pr=rows(plain_dir/"evaluation_history.csv");ir=rows(iw_dir/"evaluation_history.csv")
        plain_primary=primary_endpoint(plain_dir,pr);iw_primary=primary_endpoint(iw_dir,ir)
        intermediate=STEPS[:-1]
        curves[str(seed)]={"plain":[{"requested_step":s,**summarize(nearest(pr,s))} for s in intermediate]+[{"requested_step":3000000,**summarize(plain_primary)}],
                           "iwsc":[{"requested_step":s,**summarize(nearest(ir,s))} for s in intermediate]+[{"requested_step":3000000,**summarize(iw_primary)}]}
        mechanisms[str(seed)]=mechanism(iw_root/f"seed{seed}"/"optimization_metrics.jsonl")
        a,b=curves[str(seed)]["plain"][-1],curves[str(seed)]["iwsc"][-1]
        paired.append({"seed":seed,**{f"delta_{k}":b[k]-a[k] for k in ("W1","W2","W3","Q2","Q3","average_waves","return")}})
    except (FileNotFoundError,RuntimeError) as error:
      print(json.dumps({"offline_only":True,"result":"IWSC_DEVELOPMENT_INCOMPLETE","reason":str(error)},indent=2));raise SystemExit(2)
    means={k:float(np.mean([x[k] for x in paired])) for k in paired[0] if k.startswith("delta_")}
    gate=(means["delta_average_waves"]>=.15 and sum(x["delta_average_waves"]>0 for x in paired)>=2 and
          means["delta_W1"]>=-.05 and means["delta_Q2"]>0 and means["delta_Q3"]>0 and
          max(means["delta_Q2"],means["delta_Q3"])>=.05)
    print(json.dumps({"offline_only":True,"all_three_complete":True,"primary_endpoint":"exact_3000000","curves":curves,"mechanism_metrics":mechanisms,"paired_deltas":paired,"mean_deltas":means,
      "primary_gate":"IWSC_SUPPORTED" if gate else "IWSC_NOT_SUPPORTED"},indent=2))
if __name__=="__main__":main()
