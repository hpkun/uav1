"""Strict offline analyzer for the matched W1SG three-seed screen."""
from __future__ import annotations
import argparse,csv,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SEEDS=(5301,5302,5303);TARGET=1_805_280
METHODS={"W1SG":"w1sg_current_actor","CurrentActor":"pwtr_current_actor_only","Stratified":"pwtr_stratified","Plain":"pwtr_plain_matched_control"}
DIRS={"W1SG":"dev_w1sg_current_actor_seed{seed}_300k","CurrentActor":"dev_pwtr_current_actor_only_seed{seed}_300k","Stratified":"dev_pwtr_stratified_seed{seed}_300k","Plain":"dev_pwtr_plain_seed{seed}_300k"}
COLS={"AW":"average_waves_cleared","W1":"clear_wave_1_probability","W2":"clear_wave_2_probability","W3":"clear_wave_3_probability","Return":"average_return","RedLoss":"average_red_loss","Boundary":"average_red_boundary_exits","Ground":"average_red_ground_losses","EpisodeLength":"average_episode_length"}
def q(v):v["Q2"]=None if v["W1"]==0 else v["W2"]/v["W1"];v["Q3"]=None if v["W2"]==0 else v["W3"]/v["W2"];return v
def read(method,seed):
 p=ROOT/"outputs"/DIRS[method].format(seed=seed)
 with (p/"evaluation_history.csv").open(newline="",encoding="utf-8-sig") as f:rows=list(csv.DictReader(f))
 row=[x for x in rows if int(float(x["sampled_steps"]))==TARGET]
 if len(row)!=1:raise RuntimeError(f"{p}: exact endpoint missing")
 run=json.loads((p/"run_config.json").read_text());summary=json.loads((p/"run_summary.json").read_text())
 if run.get("development_method")!=METHODS[method] or int(summary.get("sampled_steps",-1))!=TARGET:raise RuntimeError(f"{p}: identity mismatch")
 values=q({k:float(row[0][c]) for k,c in COLS.items()})
 opt=[json.loads(x) for x in (p/"optimization_metrics.jsonl").read_text().splitlines() if x.strip()]
 if any(not math.isfinite(float(v)) for x in opt for v in x.values() if isinstance(v,(int,float))):raise RuntimeError(f"{p}: nonfinite")
 return values,opt
def paired(data,high,low,metric):
 values=[];undefined=[]
 for seed in SEEDS:
  a,b=data[high,seed][0].get(metric),data[low,seed][0].get(metric)
  if a is None or b is None:undefined.append(seed)
  else:values.append(a-b)
 return values,undefined
def stats(v):return {"mean":statistics.mean(v),"sample_sd":statistics.stdev(v) if len(v)>1 else 0.,"min":min(v),"max":max(v)}
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--output-dir",default="outputs/w1sg_300k_analysis");args=parser.parse_args();out=ROOT/args.output_dir
 if out.exists():raise FileExistsError(out)
 data={(m,s):read(m,s) for m in METHODS for s in SEEDS};comparisons={"rescue":("W1SG","CurrentActor"),"safety_efficacy":("W1SG","Stratified"),"baseline":("W1SG","Plain")};paired_rows=[];summary={}
 for label,(high,low) in comparisons.items():
  summary[label]={}
  for metric in (*COLS,"Q2","Q3"):
   values,undefined=paired(data,high,low,metric);summary[label][metric]={"values":values,"n_defined":len(values),"undefined_seeds":undefined,**(stats(values) if values else {"mean":None,"sample_sd":None,"min":None,"max":None})}
  for seed in SEEDS:
   row={"comparison":label,"seed":seed}
   for metric in (*COLS,"Q2","Q3"):
    a,b=data[high,seed][0].get(metric),data[low,seed][0].get(metric);row[f"delta_{metric}"]=None if a is None or b is None else a-b
   paired_rows.append(row)
 rescue=summary["rescue"];safe=summary["safety_efficacy"]
 gate_a=sum(x>0 for x in rescue["AW"]["values"])>=2 and rescue["AW"]["mean"]>0 and sum(x>0 for x in rescue["W3"]["values"])>=2 and rescue["W3"]["mean"]>0
 gate_b=not any(x<=-.5 for x in safe["AW"]["values"]) and not any(x<=-.25 for x in safe["W1"]["values"]) and safe["W1"]["mean"]>=-.05
 gate_c=sum(x>0 for x in safe["AW"]["values"])>=2 and safe["AW"]["mean"]>0 and sum(x>0 for x in safe["W3"]["values"])>=2 and safe["W3"]["mean"]>0
 screen="SAFETY_FAIL" if not gate_b else "NOT_SUPPORTED" if not gate_a else "MECHANISM_RESCUE_ONLY" if not gate_c else "PROMISING"
 diag=[]
 for seed in SEEDS:
  rows=data["W1SG",seed][1]
  for key in ("w1sg_backbone_importance_share","w1sg_mean_head_importance_share","w1sg_log_std_head_importance_share","w1sg_raw_protected_gradient_cosine","w1sg_norm_preservation_relative_error","w1sg_renorm_scale"):
   values=[float(r[key]) for r in rows if isinstance(r.get(key),(int,float)) and float(r.get("w1sg_active",0))>0];diag.append({"seed":seed,"metric":key,"mean":statistics.mean(values) if values else None,"max":max(values) if values else None})
 out.mkdir(parents=True);report={"status":screen,"gate_A_rescue":gate_a,"gate_B_safety":gate_b,"gate_C_efficacy":gate_c,"paired":summary,"training_seed_n":3,"primary_endpoint":TARGET,"primary_metric":"AverageWaves","no_significance_test":True,"reevaluation":False,"45m_used":False}
 (out/"analysis.json").write_text(json.dumps(report,indent=2));
 for name,rows in (("paired_endpoint.csv",paired_rows),("mechanism_diagnostics.csv",diag)):
  with (out/name).open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()
