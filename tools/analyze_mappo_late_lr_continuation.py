#!/usr/bin/env python3
"""Strict read-only descriptive analysis of matched MAPPO late-LR branches."""
from __future__ import annotations
import csv,hashlib,json,statistics,sys
from copy import deepcopy
from pathlib import Path
from typing import Any
import torch,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.common.protocol import config_sha256
from tools.preflight_mappo_late_lr_continuation import ENV,SOURCE,SOURCE_STEP,CONTROL,TREATMENT,load_yaml,validate_configs

TARGET=1_500_000;OUT=ROOT/"outputs/mappo_late_lr_continuation_analysis"
ARMS={"Control":(ROOT/"outputs/mappo_attn_lr3e4_cont_seed5303_1p5m",CONTROL,.0003),"Treatment":(ROOT/"outputs/mappo_attn_lr1e4_cont_seed5303_1p5m",TREATMENT,.0001)}
FIELDS={"W1":"clear_wave_1_probability","W2":"clear_wave_2_probability","W3":"clear_wave_3_probability","AW":"average_waves_cleared","Return":"average_return","RedLoss":"average_red_loss","Boundary":"average_red_boundary_exits","Ground":"average_red_ground_losses","EpisodeLength":"average_episode_length"}

def sha256(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def csv_rows(path:Path):
 with path.open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))
def jsonl(path:Path):return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def qmetrics(row:dict[str,Any])->dict[str,float|None]:
 result={key:float(row[column]) for key,column in FIELDS.items()};result["Q2"]=None if result["W1"]==0 else result["W2"]/result["W1"];result["Q3"]=None if result["W2"]==0 else result["W3"]/result["W2"];return result
def validate_evaluations(name:str,rows:list[dict])->None:
 if not rows:raise RuntimeError(f"{name} evaluation history empty")
 for index,row in enumerate(rows):
  required=("evaluation_seed_base","evaluation_seed_end","evaluation_episodes")
  missing=[key for key in required if row.get(key) in (None,"")]
  if missing:raise RuntimeError(f"{name} evaluation row {index} missing provenance: {missing}")
  actual=tuple(int(float(row[key])) for key in required)
  if actual!=(47_000_000,47_000_049,50):raise RuntimeError(f"{name} evaluation row {index} protocol mismatch: {actual}")
 endpoints=[row for row in rows if int(float(row["sampled_steps"]))==TARGET]
 if len(endpoints)!=1:raise RuntimeError(f"{name} requires exactly one 1.5M endpoint")
def retention(rows:list[dict])->dict[str,float|None]:
 endpoint=next(row for row in rows if int(float(row["sampled_steps"]))==TARGET);pre=[row for row in rows if int(float(row["sampled_steps"]))<TARGET]
 if not pre:raise RuntimeError("retention requires pre-endpoint evaluations")
 best_w3=max(float(row[FIELDS["W3"]]) for row in pre);best_aw=max(float(row[FIELDS["AW"]]) for row in pre);end_w3=float(endpoint[FIELDS["W3"]]);end_aw=float(endpoint[FIELDS["AW"]])
 return {"best_pre_endpoint_W3":best_w3,"best_pre_endpoint_AW":best_aw,"endpoint_W3":end_w3,"endpoint_AW":end_aw,"W3_retention":None if best_w3==0 else end_w3/best_w3,"AW_retention":None if best_aw==0 else end_aw/best_aw,"W3_drop":end_w3-best_w3,"AW_drop":end_aw-best_aw}
def matched_configs(control:dict,treatment:dict)->bool:
 a,b=deepcopy(control),deepcopy(treatment)
 for value in (a,b):value.pop("development_method",None);value["training"].pop("actor_learning_rate",None)
 return a==b
def mean_existing(rows:list[dict],key:str)->float|None:
 values=[float(row[key]) for row in rows if row.get(key) is not None]
 return None if not values else statistics.mean(values)
def validate_run(name:str,path:Path,cfg_path:Path,actor_lr:float,source_sha:str,env_sha:str)->dict:
 required=("run_config.json","run_summary.json","branch_from.json","algorithm_config.yaml","env_config.yaml","evaluation_history.csv","optimization_metrics.jsonl","latest.pt","final.pt","checkpoint_1500000.pt")
 missing=[item for item in required if not (path/item).is_file()]
 if missing:raise RuntimeError(f"incomplete {name} run: {missing}")
 cfg=load_yaml(path/"algorithm_config.yaml");expected_cfg=load_yaml(cfg_path)
 if cfg!=expected_cfg:raise RuntimeError(f"{name} algorithm snapshot mismatch")
 if load_yaml(path/"env_config.yaml")!=load_yaml(ENV):raise RuntimeError(f"{name} environment snapshot mismatch")
 run=json.loads((path/"run_config.json").read_text(encoding="utf-8"));summary=json.loads((path/"run_summary.json").read_text(encoding="utf-8"));branch=json.loads((path/"branch_from.json").read_text(encoding="utf-8"))
 expected={"algorithm":"MAPPO","seed":5303,"source_sampled_steps":SOURCE_STEP,"total_sampled_steps":TARGET,"environment_variant":"persistent_wave_v2","critic_type":"attention","actor_learning_rate":actor_lr,"critic_learning_rate":.0003,"parent_checkpoint_sha256":source_sha,"environment_config_sha256":env_sha,"algorithm_config_sha256":config_sha256(cfg)}
 bad={key:(run.get(key),value) for key,value in expected.items() if run.get(key)!=value}
 if bad:raise RuntimeError(f"{name} run identity mismatch: {bad}")
 if branch.get("parent_checkpoint_sha256")!=source_sha or int(branch.get("source_sampled_steps",-1))!=SOURCE_STEP or branch.get("environment_config_sha256")!=env_sha:raise RuntimeError(f"{name} branch provenance mismatch")
 if int(summary.get("sampled_steps",-1))!=TARGET or summary.get("parent_checkpoint_sha256")!=source_sha:raise RuntimeError(f"{name} summary mismatch")
 rows=csv_rows(path/"evaluation_history.csv");validate_evaluations(name,rows);opt=jsonl(path/"optimization_metrics.jsonl")
 for checkpoint_name in ("latest.pt","final.pt","checkpoint_1500000.pt"):
  state=torch.load(path/checkpoint_name,map_location="cuda",weights_only=False)
  if state.get("algorithm")!="MAPPO" or state.get("critic_type")!="attention" or int(state.get("sampled_steps",-1))!=TARGET:raise RuntimeError(f"{name} {checkpoint_name} mismatch")
 endpoint=next(row for row in rows if int(float(row["sampled_steps"]))==TARGET)
 best_state=torch.load(path/"best_eval.pt",map_location="cuda",weights_only=False) if (path/"best_eval.pt").is_file() else None
 best=None if best_state is None else best_state.get("extra",{}).get("best_evaluation")
 optimization={key:mean_existing(opt,key) for key in ("actor_loss","value_loss","entropy","approx_kl","clip_fraction","actor_grad_norm","critic_grad_norm","explained_variance","actor_learning_rate","critic_learning_rate","policy_log_std_mean_psi","policy_log_std_mean_theta","policy_log_std_mean_v")}
 return {"history":rows,"endpoint":qmetrics(endpoint),"retention":retention(rows),"best_diagnostic":None if best is None else qmetrics(best),"best_sampled_steps":None if best is None else int(best["sampled_steps"]),"optimization":optimization,"run":run}
def write_csv(path:Path,rows:list[dict]):
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with path.open("w",newline="",encoding="utf-8") as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def main():
 if OUT.exists():raise FileExistsError(OUT)
 if not torch.cuda.is_available():raise RuntimeError("CUDA mandatory for checkpoint audit")
 control_cfg,treatment_cfg=load_yaml(CONTROL),load_yaml(TREATMENT);validate_configs(control_cfg,treatment_cfg)
 if not matched_configs(control_cfg,treatment_cfg):raise RuntimeError("branch protocols differ beyond Actor LR and identity")
 source_sha=sha256(SOURCE);env_sha=config_sha256(load_yaml(ENV));results={}
 for name,(path,cfg,lr) in ARMS.items():results[name]=validate_run(name,path,cfg,lr,source_sha,env_sha)
 control_steps=[int(float(row["sampled_steps"])) for row in results["Control"]["history"]];treatment_steps=[int(float(row["sampled_steps"])) for row in results["Treatment"]["history"]]
 if control_steps!=treatment_steps:raise RuntimeError("branches do not share identical evaluation checkpoints")
 curve=[]
 for name in ARMS:
  for row in results[name]["history"]:curve.append({"method":name,"sampled_steps":int(float(row["sampled_steps"])),**qmetrics(row)})
 delta={key:None if results["Treatment"]["endpoint"][key] is None or results["Control"]["endpoint"][key] is None else results["Treatment"]["endpoint"][key]-results["Control"]["endpoint"][key] for key in (*FIELDS,"Q2","Q3")}
 public={name:{key:value for key,value in result.items() if key not in {"history","run"}} for name,result in results.items()}
 report={"status":"ANALYSIS_COMPLETE","research_question":"Does reducing only late-training Actor LR from 3e-4 to 1e-4 improve retention?","descriptive_only":True,"winner_label":None,"protocol_validation":{"common_source_checkpoint_sha256":source_sha,"source_sampled_steps":SOURCE_STEP,"training_seed":5303,"target_sampled_steps":TARGET,"environment_sha_match_all":True,"matched_except_actor_lr":True,"common_evaluation_scenarios":True,"evaluation_seed_range":[47_000_000,47_000_049],"common_evaluation_steps":control_steps,"bitwise_continuation_from_original_run_claimed":False},"results":public,"treatment_minus_control_endpoint":delta,"uses_44m":False,"uses_45m":False,"uses_46m":False}
 OUT.mkdir(parents=True);(OUT/"analysis.json").write_text(json.dumps(report,indent=2),encoding="utf-8");write_csv(OUT/"evaluation_curve.csv",curve);write_csv(OUT/"endpoint_comparison.csv",[{"method":name,**result["endpoint"],**result["retention"]} for name,result in results.items()]);(OUT/"decision_support.txt").write_text("ANALYSIS_COMPLETE\nDescriptive matched continuation; no winner gate is defined.\n",encoding="utf-8");print(json.dumps({"status":"ANALYSIS_COMPLETE","output":str(OUT)},indent=2))
if __name__=="__main__":main()
