#!/usr/bin/env python3
"""Strictly offline WSAI 300k paired analysis; never evaluates a policy."""
from __future__ import annotations
import csv,json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];SEEDS=(5301,5302,5303);ENDPOINT=1_805_280
WSAI=lambda seed:ROOT/f"outputs/dev_wsai_seed{seed}_300k"
PLAIN=lambda seed:ROOT/f"outputs/dev_pwtr_plain_seed{seed}_300k"
STRAT=lambda seed:ROOT/f"outputs/dev_pwtr_stratified_seed{seed}_300k"
OUT=ROOT/"outputs/wsai_300k_analysis"

FIELDS={"AW":"average_waves_cleared","W1":"clear_wave_1_probability","W2":"clear_wave_2_probability","W3":"clear_wave_3_probability",
 "Return":"average_return","RedLoss":"average_red_loss","Boundary":"average_red_boundary_exits","Ground":"average_red_ground_losses","EpisodeLength":"average_episode_length"}
def qmetrics(row):
 w1=row["W1"];w2=row["W2"];return {"Q2":None if w1==0 else w2/w1,"Q3":None if w2==0 else row["W3"]/w2}
def rows(path):
 with (path/"evaluation_history.csv").open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))
def extract(row):
 value={name:float(row[key]) for name,key in FIELDS.items()};value.update(qmetrics(value));value["sampled_steps"]=int(row["sampled_steps"]);return value
def exact(path):
 candidates=[row for row in rows(path) if int(row["sampled_steps"])==ENDPOINT]
 if len(candidates)!=1 or int(candidates[0]["evaluation_episodes"])!=50 or int(candidates[0]["evaluation_seed_base"])!=44_000_000 or int(candidates[0]["evaluation_seed_end"])!=44_000_049:raise RuntimeError(f"invalid exact endpoint: {path}")
 return extract(candidates[0])
def paired(high,low,metric):return None if high[metric] is None or low[metric] is None else high[metric]-low[metric]
def aggregate(values):
 defined=[value for value in values.values() if value is not None]
 return {"mean":None if not defined else statistics.mean(defined),"n_defined":len(defined),"undefined_seeds":[seed for seed,value in values.items() if value is None],"per_seed":values}

def main():
 if OUT.exists():raise FileExistsError(OUT)
 data={};dynamics={};specialization={}
 for seed in SEEDS:
  for path in (WSAI(seed),PLAIN(seed),STRAT(seed)):
   if not path.is_dir():raise RuntimeError(f"missing run: {path}")
  run=json.loads((WSAI(seed)/"run_config.json").read_text());summary=json.loads((WSAI(seed)/"run_summary.json").read_text())
  if run.get("development_method")!="wsai_mappo" or int(run["seed"])!=seed or int(run["total_sampled_steps"])!=ENDPOINT:raise RuntimeError(f"WSAI identity mismatch: {seed}")
  data[seed]={"WSAI":exact(WSAI(seed)),"Plain":exact(PLAIN(seed)),"Stratified":exact(STRAT(seed))}
  history=rows(WSAI(seed));dynamics[seed]=[]
  for target in (1_603_584,1_701_888,1_800_192,1_805_280):
   row=min(history,key=lambda item:abs(int(item["sampled_steps"])-target));item=extract(row);item["requested_step"]=target;dynamics[seed].append(item)
  opt=[]
  with (WSAI(seed)/"optimization_metrics.jsonl").open(encoding="utf-8") as stream:
   for line in stream:
    if line.strip():opt.append(json.loads(line))
  last=opt[-1];specialization[seed]={key:last.get(key) for key in last if key.startswith("wsai_")}
 primary={metric:aggregate({seed:paired(data[seed]["WSAI"],data[seed]["Plain"],metric) for seed in SEEDS}) for metric in ("AW","W1","W2","W3","Q2","Q3")}
 secondary={metric:aggregate({seed:paired(data[seed]["WSAI"],data[seed]["Stratified"],metric) for seed in SEEDS}) for metric in ("AW","W1","W2","W3","Q2","Q3")}
 safety=all(primary["AW"]["per_seed"][s]>-.50 and primary["W1"]["per_seed"][s]>-.25 for s in SEEDS) and primary["W1"]["mean"]>=-.05
 efficacy=sum(primary["AW"]["per_seed"][s]>0 for s in SEEDS)>=2 and primary["AW"]["mean"]>0 and sum(primary["W3"]["per_seed"][s]>0 for s in SEEDS)>=2 and primary["W3"]["mean"]>0
 conditional=any(primary[m]["n_defined"]>=2 and primary[m]["mean"]>0 for m in ("Q2","Q3"))
 label="SAFETY_FAIL" if not safety else ("PROMISING" if efficacy else "NOT_SUPPORTED")
 report={"analysis":"WSAI_300K_OFFLINE","replication_unit":"training_seed","n":3,"exact_endpoint":ENDPOINT,
  "endpoint":data,"primary_wsai_minus_plain":primary,"secondary_wsai_minus_stratified":secondary,
  "gates":{"A_safety":safety,"B_primary_efficacy":efficacy,"C_conditional_progression":conditional},
  "WSAI_SCREEN":label,"CONDITIONAL_PROGRESSION_SIGNAL":"YES" if conditional else "NO",
  "learning_dynamics":dynamics,"actor_specialization":specialization,"evaluation_rerun":False,"uses_45m":False}
 OUT.mkdir(parents=True);(OUT/"analysis.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print(f"WSAI_SCREEN={label}")

if __name__=="__main__":main()
