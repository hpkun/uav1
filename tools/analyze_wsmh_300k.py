#!/usr/bin/env python3
"""Strictly offline WSMH paired analysis; never evaluates a policy."""
from __future__ import annotations
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SEEDS=(5301,5302,5303);ENDPOINT=1_805_280
RUNS={"WSMH":lambda s:ROOT/f"outputs/dev_wsmh_seed{s}_300k","Plain":lambda s:ROOT/f"outputs/dev_pwtr_plain_seed{s}_300k","Stratified":lambda s:ROOT/f"outputs/dev_pwtr_stratified_seed{s}_300k","WSAI":lambda s:ROOT/f"outputs/dev_wsai_seed{s}_300k"};OUT=ROOT/"outputs/wsmh_300k_analysis"
FIELDS={"AW":"average_waves_cleared","W1":"clear_wave_1_probability","W2":"clear_wave_2_probability","W3":"clear_wave_3_probability","Return":"average_return","RedLoss":"average_red_loss","Boundary":"average_red_boundary_exits","Ground":"average_red_ground_losses","EpisodeLength":"average_episode_length"}
def read(path):
 with (path/"evaluation_history.csv").open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def extract(row):
 out={k:float(row[v]) for k,v in FIELDS.items()};out["Q2"]=None if out["W1"]==0 else out["W2"]/out["W1"];out["Q3"]=None if out["W2"]==0 else out["W3"]/out["W2"];out["sampled_steps"]=int(row["sampled_steps"]);return out
def exact(path):
 rows=[r for r in read(path) if int(r["sampled_steps"])==ENDPOINT]
 if len(rows)!=1 or (int(rows[0]["evaluation_episodes"]),int(rows[0]["evaluation_seed_base"]),int(rows[0]["evaluation_seed_end"]))!=(50,44_000_000,44_000_049):raise RuntimeError(f"invalid exact endpoint: {path}")
 return extract(rows[0])
def aggregate(values):
 defined={s:v for s,v in values.items() if v is not None};return {"mean":None if not defined else statistics.mean(defined.values()),"n_defined":len(defined),"undefined_seeds":[s for s in values if s not in defined],"per_seed":values}
def delta(a,b,k):return None if a[k] is None or b[k] is None else a[k]-b[k]
def main():
 if OUT.exists():raise FileExistsError(OUT)
 data={};dynamics={};specialization={}
 for seed in SEEDS:
  for method,pathfn in RUNS.items():
   if not pathfn(seed).is_dir():raise RuntimeError(f"missing {method} seed{seed}")
  run=json.loads((RUNS["WSMH"](seed)/"run_config.json").read_text())
  if (run.get("development_method"),int(run["seed"]),int(run["total_sampled_steps"]))!=("wsmh_mappo",seed,ENDPOINT):raise RuntimeError(f"WSMH identity mismatch: {seed}")
  data[seed]={name:exact(pathfn(seed)) for name,pathfn in RUNS.items()}
  history=read(RUNS["WSMH"](seed));dynamics[seed]=[]
  for target in (1_603_584,1_701_888,1_800_192,1_805_280):
   rows=[r for r in history if int(r["sampled_steps"])==target]
   if len(rows)!=1:raise RuntimeError(f"missing exact WSMH evaluation {seed}@{target}")
   dynamics[seed].append(extract(rows[0]))
  with (RUNS["WSMH"](seed)/"optimization_metrics.jsonl").open(encoding="utf-8") as f:opt=[json.loads(line) for line in f if line.strip()]
  specialization[seed]={k:v for k,v in opt[-1].items() if k.startswith("wsmh_")}
 def comparison(low):return {metric:aggregate({s:delta(data[s]["WSMH"],data[s][low],metric) for s in SEEDS}) for metric in ("AW","W1","W2","W3","Q2","Q3","Return","Boundary","Ground")}
 primary=comparison("Plain");vs_wsai=comparison("WSAI");vs_strat=comparison("Stratified")
 safety=all(primary["AW"]["per_seed"][s]>-.50 and primary["W1"]["per_seed"][s]>-.25 for s in SEEDS) and primary["W1"]["mean"]>=-.05
 efficacy=sum(primary["AW"]["per_seed"][s]>0 for s in SEEDS)>=2 and primary["AW"]["mean"]>0 and sum(primary["W3"]["per_seed"][s]>0 for s in SEEDS)>=2 and primary["W3"]["mean"]>0
 label="SAFETY_FAIL" if not safety else ("PROMISING" if efficacy else "NOT_SUPPORTED")
 report={"analysis":"WSMH_300K_OFFLINE","replication_unit":"training_seed","n":3,"exact_endpoint":ENDPOINT,"endpoint":data,"primary_wsmh_minus_plain":primary,"secondary_wsmh_minus_stratified":vs_strat,"secondary_wsmh_minus_wsai":vs_wsai,"gates":{"A_safety":safety,"B_primary_efficacy":efficacy},"WSMH_SCREEN":label,"learning_dynamics":dynamics,"mean_head_specialization":specialization,"evaluation_rerun":False,"uses_45m":False}
 OUT.mkdir(parents=True);(OUT/"analysis.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print(f"WSMH_SCREEN={label}")
if __name__=="__main__":main()
