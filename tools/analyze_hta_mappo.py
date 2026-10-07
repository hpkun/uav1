"""Offline-only paired Plain versus HTA-MAPPO V1 development analyzer."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];SEEDS=(5301,5302,5303)

def rows(path):
    with Path(path).open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))
def value(row,key):return None if row.get(key) in (None,"") else float(row[key])
def exact(run_dir):
    summary=Path(run_dir)/"run_summary.json"
    if not summary.exists() or int(json.loads(summary.read_text(encoding="utf-8")).get("sampled_steps",-1))!=3_000_000:raise RuntimeError(f"incomplete run: {run_dir}")
    return next((row for row in rows(Path(run_dir)/"evaluation_history.csv") if int(row["sampled_steps"])==3_000_000),None)
def metrics(row):
    if row is None:raise RuntimeError("exact-3M evaluation is absent")
    w1,w2,w3=(value(row,f"clear_wave_{i}_probability") for i in (1,2,3))
    return {"W1":w1,"W2":w2,"W3":w3,"Q2":None if not w1 else w2/w1,"Q3":None if not w2 else w3/w2,
      "average_waves":value(row,"average_waves_cleared"),"return":value(row,"average_return"),"red_loss":value(row,"average_red_loss"),
      "blue_loss":value(row,"average_blue_loss"),"ground":value(row,"average_red_ground_losses"),"boundary":value(row,"average_red_boundary_exits"),
      "episode_length":value(row,"average_episode_length")}
def mechanism(path):
    data=[json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    keys=("hta_manager_entropy","hta_manager_approx_kl","hta_manager_explained_variance","hta_manager_max_option_fraction",
          "hta_option_switch_fraction","hta_macro_duration_mean","hta_wave_end_fraction","hta_rollout_truncation_fraction",
          "hta_option_residual_abs_mean","hta_option_residual_abs_max")
    result={key:float(np.mean([float(row[key]) for row in data if key in row])) if any(key in row for row in data) else None for key in keys}
    result["updates"]=len(data)
    for option in range(4):
        key=f"hta_option_{option}_fraction";result[key]=float(np.mean([float(row[key]) for row in data if key in row])) if any(key in row for row in data) else None
    return result
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--hta-root",default="outputs/dev_hta_mappo_v1_3m");args=parser.parse_args()
    plain=ROOT/"outputs/diag_mappo_learnability";hta=ROOT/args.hta_root;paired=[];mechanisms={};endpoints={}
    try:
        for seed in SEEDS:
            p=metrics(exact(plain/f"l3_seed{seed}"));h=metrics(exact(hta/f"seed{seed}"));endpoints[str(seed)]={"plain":p,"hta":h}
            paired.append({"seed":seed,**{f"delta_{key}":h[key]-p[key] for key in p}})
            mechanisms[str(seed)]=mechanism(hta/f"seed{seed}"/"optimization_metrics.jsonl")
    except (FileNotFoundError,RuntimeError) as error:
        print(json.dumps({"offline_only":True,"result":"HTA_DEVELOPMENT_INCOMPLETE","reason":str(error)},indent=2));raise SystemExit(2)
    means={key:float(np.mean([row[key] for row in paired])) for key in paired[0] if key.startswith("delta_")}
    supported=(means["delta_average_waves"]>=.15 and sum(row["delta_average_waves"]>0 for row in paired)>=2 and
      means["delta_W1"]>=-.05 and means["delta_Q2"]>0 and means["delta_Q3"]>0 and max(means["delta_Q2"],means["delta_Q3"])>=.05)
    print(json.dumps({"offline_only":True,"primary_endpoint":"exact_3000000","endpoints":endpoints,"paired_deltas":paired,
      "mean_deltas":means,"mechanism_metrics":mechanisms,"primary_gate":"HTA_SUPPORTED" if supported else "HTA_NOT_SUPPORTED"},indent=2))

if __name__=="__main__":main()
