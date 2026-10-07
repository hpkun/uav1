#!/usr/bin/env python3
"""Offline-only analysis for the matched Actor clip 0.5 versus 1.0 screen."""
from __future__ import annotations
import csv,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SEEDS=(5301,5302,5303);TARGET=1_805_280;EVAL_STEPS=(1_603_584,1_701_888,1_800_192,1_805_280);OUT=ROOT/"outputs/actor_grad_clip_300k_analysis"
RUNS={"Control05":lambda s:ROOT/f"outputs/dev_actor_clip05_seed{s}_300k","Treatment10":lambda s:ROOT/f"outputs/dev_actor_clip10_seed{s}_300k","HistoricalPlain":lambda s:ROOT/f"outputs/dev_pwtr_plain_seed{s}_300k"}
FIELDS={"AW":"average_waves_cleared","W1":"clear_wave_1_probability","W2":"clear_wave_2_probability","W3":"clear_wave_3_probability","Return":"average_return","RedLoss":"average_red_loss","Boundary":"average_red_boundary_exits","Ground":"average_red_ground_losses","EpisodeLength":"average_episode_length"}
WINDOWS={"EARLY":(1_505_280,1_603_584),"MIDDLE":(1_603_584,1_701_888),"LATE":(1_701_888,1_805_280)}
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
 return h.hexdigest()
def rows(path):
 with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def jsonl(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def write_csv(path,data):
 data=list(data);fields=[]
 for row in data:
  for key in row:
   if key not in fields:fields.append(key)
 with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
def extract(row):
 out={key:float(row[value]) for key,value in FIELDS.items()};out["Q2"]=None if out["W1"]==0 else out["W2"]/out["W1"];out["Q3"]=None if out["W2"]==0 else out["W3"]/out["W2"];return out
def delta(high,low,key):return None if high[key] is None or low[key] is None else high[key]-low[key]
def summary(values):
 values=[float(v) for v in values];return {"n":len(values),"mean":statistics.mean(values),"median":statistics.median(values),"min":min(values),"max":max(values)}
def main():
 if OUT.exists():raise FileExistsError(OUT)
 endpoint={};paired=[];learning=[];clipping=[];historical=[];protocol={};trainer_shas={}
 for seed in SEEDS:
  source=ROOT/f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_1505280.pt";source_sha=sha(source);endpoint[seed]={}
  for method,pathfn in RUNS.items():
   path=pathfn(seed)
   if not path.is_dir():raise RuntimeError(f"missing run: {path}")
   if not all((path/name).is_file() for name in ("run_config.json","run_summary.json","branch_from.json","evaluation_history.csv","optimization_metrics.jsonl","latest.pt","final.pt")):raise RuntimeError(f"incomplete run: {path}")
   run=json.loads((path/"run_config.json").read_text());branch=json.loads((path/"branch_from.json").read_text());history=rows(path/"evaluation_history.csv");by_step={int(r["sampled_steps"]):r for r in history}
   expected={"Control05":"actor_grad_clip_05_control","Treatment10":"actor_grad_clip_10","HistoricalPlain":"pwtr_plain_matched_control"}[method]
   if run["development_method"]!=expected or int(run["seed"])!=seed or int(run["total_sampled_steps"])!=TARGET:raise RuntimeError(f"run identity mismatch: {path}")
   if method!="HistoricalPlain" and (branch["parent_checkpoint_sha256"]!=source_sha or int(branch["source_training_seed"])!=seed):raise RuntimeError(f"parent mismatch: {path}")
   final=by_step.get(TARGET)
   if final is None or (int(final["evaluation_episodes"]),int(final["evaluation_seed_base"]),int(final["evaluation_seed_end"]))!=(50,44_000_000,44_000_049):raise RuntimeError(f"endpoint mismatch: {path}")
   if any(int(row["evaluation_seed_base"])>=45_000_000 for row in history):raise RuntimeError(f"45M evaluation seed found: {path}")
   endpoint[seed][method]=extract(final);manifest={r["path"]:r["sha256"] for r in run["runtime_source_manifest_files"]};trainer_shas[(method,seed)]=manifest["algorithm/modular_mappo/trainer.py"]
   protocol[f"{method}_seed{seed}"]={"status":"PASS","parent_sha256":branch["parent_checkpoint_sha256"],"trainer_sha256":trainer_shas[(method,seed)]}
   if method!="HistoricalPlain":
    opt=jsonl(path/"optimization_metrics.jsonl")
    for window,(lower,upper) in WINDOWS.items():
     selected=[r for r in opt if lower<int(r["sampled_steps"])<=upper]
     clipping.append({"method":method,"seed":seed,"window":window,"rows":len(selected),**{f"{key}_mean":statistics.mean(float(r[key]) for r in selected) for key in ("actor_grad_clip_preclip_norm","actor_grad_clip_postclip_norm","actor_grad_clip_exact_scale","actor_grad_clip_pressure_fraction","actor_grad_clip_critic_preclip_norm","actor_grad_clip_critic_exact_scale","actor_grad_clip_critic_pressure_fraction","approx_kl","clip_fraction","entropy","ratio_underflow_fraction","max_abs_log_ratio")}})
  for step in EVAL_STEPS:
   current={method:extract({int(r["sampled_steps"]):r for r in rows(pathfn(seed)/"evaluation_history.csv")}[step]) for method,pathfn in RUNS.items()}
   learning.append({"seed":seed,"sampled_steps":step,**{f"{method}_{key}":value for method,metrics in current.items() for key,value in metrics.items()},**{f"delta10minus05_{key}":delta(current["Treatment10"],current["Control05"],key) for key in FIELDS|{"Q2":"","Q3":""}}})
  for key in ("AW","W1","W2","W3","Q2","Q3","Return","RedLoss","Boundary","Ground","EpisodeLength"):paired.append({"seed":seed,"metric":key,"control":endpoint[seed]["Control05"][key],"treatment":endpoint[seed]["Treatment10"][key],"delta":delta(endpoint[seed]["Treatment10"],endpoint[seed]["Control05"],key)})
  for key in ("AW","W1","W2","W3"):historical.append({"seed":seed,"metric":key,"control05_minus_historical":delta(endpoint[seed]["Control05"],endpoint[seed]["HistoricalPlain"],key),"treatment10_minus_historical":delta(endpoint[seed]["Treatment10"],endpoint[seed]["HistoricalPlain"],key),"historical_source_sha_diff":trainer_shas[("Control05",seed)]!=trainer_shas[("HistoricalPlain",seed)]})
 if any(trainer_shas[("Control05",s)]!=trainer_shas[("Treatment10",s)] for s in SEEDS):raise RuntimeError("current matched runtime trainer SHA mismatch")
 per_metric={key:{"per_seed":{str(s):delta(endpoint[s]["Treatment10"],endpoint[s]["Control05"],key) for s in SEEDS}} for key in ("AW","W1","W2","W3","Q2","Q3")}
 for item in per_metric.values():
  values=[v for v in item["per_seed"].values() if v is not None];item.update({"mean":statistics.mean(values) if values else None,"wins":sum(v>0 for v in values),"n_defined":len(values)})
 safety=all(per_metric["AW"]["per_seed"][str(s)]>-.50 and per_metric["W1"]["per_seed"][str(s)]>-.25 for s in SEEDS) and per_metric["W1"]["mean"]>=-.05
 efficacy=per_metric["AW"]["wins"]>=2 and per_metric["AW"]["mean"]>0 and per_metric["W3"]["wins"]>=2 and per_metric["W3"]["mean"]>0
 label="SAFETY_FAIL" if not safety else "PROMISING" if efficacy else "NOT_SUPPORTED"
 control_integrity={key:statistics.mean(row["control05_minus_historical"] for row in historical if row["metric"]==key) for key in ("AW","W1","W2","W3")}
 retention={}
 for seed in SEEDS:
  retention[str(seed)]={}
  for method in ("Control05","Treatment10"):
   points=[row for row in learning if row["seed"]==seed];best=max(points,key=lambda row:row[f"{method}_AW"]);final=next(row for row in points if row["sampled_steps"]==TARGET);retention[str(seed)][method]={"best_step":best["sampled_steps"],"best_AW":best[f"{method}_AW"],"final_AW":final[f"{method}_AW"],"final_minus_best_AW":final[f"{method}_AW"]-best[f"{method}_AW"]}
 report={"protocol_validation":protocol,"replication_unit":"training_seed","n":3,"endpoint":endpoint,"primary_treatment10_minus_current_control05":per_metric,"gates":{"A_safety":safety,"B_primary_efficacy":efficacy},"ACTOR_GRAD_CLIP_SCREEN":label,"CONTROL05_INTEGRITY":{"rows":historical,"mean_delta":control_integrity,"historical_source_sha_diff":True},"current_runtime_trainer_sha_matched":True,"learning_dynamics":learning,"best_to_final_retention":retention,"clipping_dynamics":clipping,"evaluation_rerun":False,"uses_45m":False}
 OUT.mkdir(parents=True);(OUT/"analysis.json").write_text(json.dumps(report,indent=2));write_csv(OUT/"paired_endpoint.csv",paired);write_csv(OUT/"learning_dynamics.csv",learning);write_csv(OUT/"clipping_dynamics.csv",clipping);write_csv(OUT/"control_historical_context.csv",historical);(OUT/"decision_support.txt").write_text(f"ACTOR_GRAD_CLIP_SCREEN={label}\nGateA={safety}\nGateB={efficacy}\n")
 print(json.dumps({"status":"ACTOR_GRAD_CLIP_ANALYSIS_COMPLETE","label":label,"output":str(OUT)},indent=2))
if __name__=="__main__":main()
