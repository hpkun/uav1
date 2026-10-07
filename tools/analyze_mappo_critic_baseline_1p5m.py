#!/usr/bin/env python3
"""Read-only descriptive analysis of the matched MAPPO critic baseline."""
from __future__ import annotations
import csv,json,statistics,sys
from copy import deepcopy
from pathlib import Path
from typing import Any
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.common.protocol import config_sha256
from tools.preflight_mappo_critic_baseline_1p5m import validate_configs
TARGET=1_500_000;OUT=ROOT/"outputs/mappo_critic_baseline_analysis"
ENV=ROOT/"configs/persistent_wave_v2_environment.yaml"
ARMS={"MLP":(ROOT/"configs/mappo_mlp_baseline_1p5m.yaml",ROOT/"outputs/mappo_mlp_seed5303_1p5m","mlp"),"Attention":(ROOT/"configs/mappo_attention_baseline_1p5m.yaml",ROOT/"outputs/mappo_attention_seed5303_1p5m","attention")}
FIELDS={"W1":"clear_wave_1_probability","W2":"clear_wave_2_probability","W3":"clear_wave_3_probability","AW":"average_waves_cleared","Return":"average_return","RedLoss":"average_red_loss","Boundary":"average_red_boundary_exits","Ground":"average_red_ground_losses","EpisodeLength":"average_episode_length"}
def load_yaml(p):return yaml.safe_load(p.read_text(encoding="utf-8"))
def csv_rows(p):
 with p.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def jsonl(p):return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def qmetrics(row):
 result={k:float(row[v]) for k,v in FIELDS.items()};result["Q2"]=None if result["W1"]==0 else result["W2"]/result["W1"];result["Q3"]=None if result["W2"]==0 else result["W3"]/result["W2"];return result
def write_csv(path,rows):
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with path.open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def validate_evaluation_provenance(name,history):
 required=("evaluation_seed_base","evaluation_seed_end","evaluation_episodes")
 for index,row in enumerate(history):
  missing=[key for key in required if key not in row or row[key] in (None,"")]
  if missing:raise RuntimeError(f"{name} evaluation row {index} missing provenance: {missing}")
  actual=tuple(int(float(row[key])) for key in required)
  if actual!=(46_000_000,46_000_049,50):raise RuntimeError(f"{name} evaluation row {index} protocol mismatch: {actual}")
 return True
def validate_run_identity(name,run,expected):
 missing=[key for key in expected if key not in run]
 if missing:raise RuntimeError(f"{name} run identity missing fields: {missing}")
 bad={key:(run[key],value) for key,value in expected.items() if run[key]!=value}
 if bad:raise RuntimeError(f"{name} run identity mismatch: {bad}")
 return True
def validate_run(name,path,cfg,critic_type,environment_sha=None):
 required=("run_config.json","run_summary.json","evaluation_history.csv","optimization_metrics.jsonl","latest.pt")
 if not path.is_dir() or any(not (path/x).is_file() for x in required):raise RuntimeError(f"incomplete {name} run: {path}")
 run=json.loads((path/"run_config.json").read_text());summary=json.loads((path/"run_summary.json").read_text());history=csv_rows(path/"evaluation_history.csv");opt=jsonl(path/"optimization_metrics.jsonl")
 expected={"seed":5303,"total_sampled_steps":TARGET,"environment_variant":"persistent_wave_v2","algorithm":"MAPPO","critic_type":critic_type,"algorithm_config_sha256":config_sha256(cfg),"environment_config_sha256":environment_sha if environment_sha is not None else config_sha256(load_yaml(ENV))}
 validate_run_identity(name,run,expected)
 endpoint=[r for r in history if int(float(r["sampled_steps"]))==TARGET]
 if len(endpoint)!=1:raise RuntimeError(f"{name} requires one exact 1.5M evaluation")
 validate_evaluation_provenance(name,history)
 state=torch.load(path/"latest.pt",map_location="cuda",weights_only=False)
 stored=str(state.get("critic_type",state.get("extra",{}).get("critic_type","attention")))
 if stored!=critic_type or int(state.get("sampled_steps",-1))!=TARGET:raise RuntimeError(f"{name} checkpoint identity mismatch")
 if int(summary.get("sampled_steps",-1))!=TARGET or summary.get("critic_type")!=critic_type:raise RuntimeError(f"{name} summary identity mismatch")
 best_state=torch.load(path/"best_eval.pt",map_location="cuda",weights_only=False) if (path/"best_eval.pt").is_file() else None
 best=None if best_state is None else best_state.get("extra",{}).get("best_evaluation")
 return run,summary,history,opt,state,endpoint[0],best
def main():
 if OUT.exists():raise FileExistsError(OUT)
 if not torch.cuda.is_available():raise RuntimeError("CUDA mandatory for checkpoint audit")
 env=load_yaml(ENV);environment_sha=config_sha256(env);configs={name:load_yaml(value[0]) for name,value in ARMS.items()};validate_configs(configs["MLP"],configs["Attention"])
 results={};curve=[];params=[]
 for name,(cfg_path,path,critic_type) in ARMS.items():
  run,summary,history,opt,state,endpoint,best=validate_run(name,path,configs[name],critic_type,environment_sha)
  endpoint_metrics=qmetrics(endpoint);best_metrics=None if best is None else qmetrics(best)
  optimization={key:statistics.mean(float(r[key]) for r in opt if r.get(key) is not None) for key in ("value_loss","explained_variance")}
  results[name]={"exact_endpoint":endpoint_metrics,"best_diagnostic":best_metrics,"best_sampled_steps":None if best is None else int(best["sampled_steps"]),"optimization":optimization,"critic_type":critic_type}
  params.append({"method":name,"actor_parameter_count":run["actor_parameter_count"],"critic_parameter_count":run["critic_parameter_count"],"total_parameter_count":run["total_parameter_count"]})
  for row in history:curve.append({"method":name,"sampled_steps":int(float(row["sampled_steps"])),**qmetrics(row)})
 delta={key:None if results["Attention"]["exact_endpoint"][key] is None or results["MLP"]["exact_endpoint"][key] is None else results["Attention"]["exact_endpoint"][key]-results["MLP"]["exact_endpoint"][key] for key in (*FIELDS,"Q2","Q3")}
 report={"status":"ANALYSIS_COMPLETE","research_question":"Centralized Attention Critic versus Centralized MLP Critic","training_seed":5303,"evaluation_seed_range":[46_000_000,46_000_049],"paired_by_training_seed_and_evaluation_scenarios":True,"common_action_noise_after_initialization":False,"protocol_validation":{"training_seed":5303,"target_sampled_steps":TARGET,"environment_sha_match_all":True,"evaluation_seed_provenance_match_all":True,"common_evaluation_scenarios":True,"observation_dim":52,"wave_information_used":False},"results":results,"attention_minus_mlp":delta,"parameter_counts":params,"uses_44m":False,"uses_45m":False,"winner_label":None}
 OUT.mkdir(parents=True);(OUT/"analysis.json").write_text(json.dumps(report,indent=2),encoding="utf-8");write_csv(OUT/"endpoint_comparison.csv",[{"method":n,**v["exact_endpoint"]} for n,v in results.items()]);write_csv(OUT/"learning_curve_comparison.csv",curve);write_csv(OUT/"parameter_counts.csv",params);(OUT/"decision_support.txt").write_text("ANALYSIS_COMPLETE\nNo winner gate is defined.\n",encoding="utf-8");print(json.dumps({"status":"ANALYSIS_COMPLETE","output":str(OUT)},indent=2))
if __name__=="__main__":main()
