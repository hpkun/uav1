"""Offline analyzer for completed clean single-wave trajectory diagnostics."""
from __future__ import annotations
import argparse,json,math,statistics,sys
from pathlib import Path
from typing import Any,Callable
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.single_wave_stability_common import BEST_STEPS,TRAINING_SEEDS,optimization_phase_predicates,read_csv,run_dir,write_csv
OPT_FIELDS=("actor_learning_rate","critic_learning_rate","actor_loss","value_loss","explained_variance","entropy","policy_log_std_mean_psi","policy_log_std_mean_theta","policy_log_std_mean_v","approx_kl","clip_fraction","first_minibatch_approx_kl","first_minibatch_clip_fraction","ratio_p1","ratio_p50","ratio_p99","ratio_min","ratio_max","actor_grad_norm","critic_grad_norm")

def number(row,key,default=None):
 try:return float(row[key])
 except (KeyError,TypeError,ValueError):return default
def truth(value):return str(value).lower() in {"true","1","yes"}
def stats(values):
 x=np.asarray([v for v in values if v is not None and math.isfinite(v)],float)
 if not len(x):return {k:None for k in ("mean","median","p10","p90","p95","p99","max")}
 return {"mean":float(x.mean()),"median":float(np.median(x)),"p10":float(np.quantile(x,.1)),"p90":float(np.quantile(x,.9)),"p95":float(np.quantile(x,.95)),"p99":float(np.quantile(x,.99)),"max":float(x.max())}
def empirical_gt(a,b):
 if not a or not b:return None
 return float(np.mean(np.asarray(a)[:,None]>np.asarray(b)[None,:]))
def groups(rows):
 return {"WIN":[r for r in rows if truth(r["red_success"])],"NON_WIN":[r for r in rows if not truth(r["red_success"])],"BOUNDARY_ANY":[r for r in rows if number(r,"red_boundary_exits",0)>=1],"GROUND_ANY":[r for r in rows if number(r,"red_ground_losses",0)>=1],"TIMEOUT":[r for r in rows if r["termination_reason"]=="red_failure_timeout"],"BOUNDARY_FAILURE":[r for r in rows if not truth(r["red_success"]) and number(r,"red_boundary_exits",0)>=1],"PURE_COMBAT_DEFEAT":[r for r in rows if r["termination_reason"]=="blue_win" and number(r,"red_boundary_exits",0)==0 and number(r,"red_ground_losses",0)==0],"MUTUAL_DESTRUCTION_DRAW":[r for r in rows if r["termination_reason"]=="draw_mutual_destruction"]}

def optimization_summary():
 out=[]
 for seed in TRAINING_SEEDS:
  records=[json.loads(line) for line in (run_dir(seed)/"optimization_metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()];best=BEST_STEPS[seed]
  phases=optimization_phase_predicates(best)
  for phase,pred in phases.items():
   subset=[r for r in records if pred(int(r["sampled_steps"]))]
   for metric in OPT_FIELDS:
    summary=stats([number(r,metric) for r in subset]);out.append({"training_seed":seed,"phase":phase,"metric":metric,"n_updates":len(subset),**summary,"note":"actor_grad_norm is update-level minibatch aggregate, not clipping fraction" if metric=="actor_grad_norm" else ""})
 return out

def reward_outputs(outcomes):
 ranking=[];component=[];prob=[]
 for seed in TRAINING_SEEDS:
  for role in ("best","final"):
   sub=[r for r in outcomes if int(r["training_seed"])==seed and r["checkpoint_role"]==role];g=groups(sub)
   for name,items in g.items():
    for metric in ("episode_return","R1","R2","R3","R4","episode_length","red_attack_kills","blue_attack_kills","red_losses","blue_losses","red_boundary_exits","red_ground_losses"):
     ranking.append({"training_seed":seed,"checkpoint_role":role,"outcome_class":name,"metric":metric,"N":len(items),**stats([number(r,metric) for r in items])})
    absolute={f"R{i}":sum(abs(number(r,f"R{i}",0)) for r in items) for i in range(1,5)};den=sum(absolute.values())
    for i in range(1,5):component.append({"training_seed":seed,"checkpoint_role":role,"outcome_class":name,"component":f"R{i}","N":len(items),**stats([number(r,f"R{i}") for r in items]),"episode_level_absolute_reward_component_share":absolute[f"R{i}"]/den if den else None})
   prob.append({"training_seed":seed,"checkpoint_role":role,"n_boundary_failure":len(g["BOUNDARY_FAILURE"]),"n_pure_combat_defeat":len(g["PURE_COMBAT_DEFEAT"]),"n_win":len(g["WIN"]),"n_mutual_destruction_draw":len(g["MUTUAL_DESTRUCTION_DRAW"]),"P_Return_boundary_failure_gt_pure_combat_defeat":empirical_gt([number(r,"episode_return") for r in g["BOUNDARY_FAILURE"]],[number(r,"episode_return") for r in g["PURE_COMBAT_DEFEAT"]]),"P_Return_boundary_failure_gt_win":empirical_gt([number(r,"episode_return") for r in g["BOUNDARY_FAILURE"]],[number(r,"episode_return") for r in g["WIN"]]),"P_Return_pure_combat_defeat_gt_win":empirical_gt([number(r,"episode_return") for r in g["PURE_COMBAT_DEFEAT"]],[number(r,"episode_return") for r in g["WIN"]]),"confidence_warning":"empirical pair probability; small groups have wide uncertainty; BOUNDARY_ANY is descriptive overlap only"})
 return ranking,component,prob

def counterfactual_summary(scores,outcomes):
 outcomes_key={(r["training_seed"],r["checkpoint_role"],r["episode_seed"]):r for r in outcomes};bucket={}
 for row in scores:
  key=tuple(row[k] for k in ("training_seed","checkpoint_role","boundary_penalty","timeout_penalty","win_bonus","mission_loss_penalty"));bucket.setdefault(key,[]).append(row)
 out=[]
 for key,items in bucket.items():
  classified=[(r,outcomes_key[(r["training_seed"],r["checkpoint_role"],r["episode_seed"])]) for r in items];win=[number(r,"fixed_trajectory_score") for r,o in classified if truth(o["red_success"])];boundary=[number(r,"fixed_trajectory_score") for r,o in classified if not truth(o["red_success"]) and number(o,"red_boundary_exits",0)>=1];combat=[number(r,"fixed_trajectory_score") for r,o in classified if o["termination_reason"]=="blue_win" and number(o,"red_boundary_exits",0)==0 and number(o,"red_ground_losses",0)==0]
  out.append(dict(zip(("training_seed","checkpoint_role","boundary_penalty","timeout_penalty","win_bonus","mission_loss_penalty"),key),n=len(items),mean_score=statistics.mean(number(r,"fixed_trajectory_score") for r in items),P_boundary_failure_gt_pure_combat_defeat=empirical_gt(boundary,combat),P_boundary_failure_gt_win=empirical_gt(boundary,win),warning="fixed-trajectory ranking only; no retraining-policy claim"))
 return out

def blue_guard(outcomes):
 rows=[]
 for seed in TRAINING_SEEDS:
  for role in ("best","final"):
   for cls,pred in (("ALL",lambda r:True),("BOUNDARY",lambda r:number(r,"red_boundary_exits",0)>0),("NO_BOUNDARY",lambda r:number(r,"red_boundary_exits",0)==0)):
    sub=[r for r in outcomes if int(r["training_seed"])==seed and r["checkpoint_role"]==role and pred(r)];acts=[number(r,"blue_ground_guard_activations",0) for r in sub]
    rows.append({"training_seed":seed,"checkpoint_role":role,"state_class":cls,"N":len(sub),"activation_episode_rate":float(np.mean(np.asarray(acts)>0)) if acts else None,"mean_activation_ratio":statistics.mean(number(r,"blue_ground_guard_activation_ratio",0) for r in sub) if sub else None,"max_activation_duration":max((number(r,"blue_ground_guard_max_duration_steps",0) for r in sub),default=None),"causal_warning":"association only"})
 return rows

def main():
 p=argparse.ArgumentParser();p.add_argument("--trajectory-dir",default="outputs/single_wave_checkpoint_trajectory_audit");a=p.parse_args();out=ROOT/a.trajectory_dir
 manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
 if manifest.get("status")!="TRAJECTORY_COLLECTION_COMPLETE" or manifest.get("formal_48m_seeds_accessed") is not False:raise RuntimeError("invalid trajectory manifest")
 targets=("optimization_phase_summary.csv","reward_ranking_by_outcome.csv","episode_reward_component_summary.csv","offline_counterfactual_ranking_summary.csv","blue_guard_summary.csv","root_cause_evidence_table.csv","final_stability_diagnostic_report.md")
 existing=[x for x in targets if (out/x).exists()]
 if existing:raise FileExistsError(f"refusing to overwrite analyzer outputs: {existing}")
 outcomes=read_csv(out/"episode_outcomes.csv");ranking,components,probs=reward_outputs(outcomes);opt=optimization_summary();guard=blue_guard(outcomes);cf=counterfactual_summary(read_csv(out/"offline_counterfactual_reward_scores.csv"),outcomes)
 write_csv(out/"optimization_phase_summary.csv",opt);write_csv(out/"reward_ranking_by_outcome.csv",ranking+[{**r,"outcome_class":"PAIRWISE_EMPIRICAL_PROBABILITY","metric":"RETURN_RANKING"} for r in probs]);write_csv(out/"episode_reward_component_summary.csv",components);write_csv(out/"offline_counterfactual_ranking_summary.csv",cf);write_csv(out/"blue_guard_summary.csv",guard)
 boundary=read_csv(out/"boundary_behavior_summary.csv");drift=read_csv(out/"policy_drift_summary.csv");sat=read_csv(out/"action_saturation_summary.csv");cross=read_csv(out/"cross_horizon_evaluation.csv")
 def find(rows,**want):return next((r for r in rows if all(str(r.get(k))==str(v) for k,v in want.items())),None)
 boundary_rise=[]
 for seed in TRAINING_SEEDS:
  b=find(boundary,training_seed=seed,checkpoint_role="best");f=find(boundary,training_seed=seed,checkpoint_role="final")
  if b and f:boundary_rise.append(number(f,"boundary_episode_rate",0)-number(b,"boundary_episode_rate",0))
 tactical_rise=[]
 escape_rise=[]
 for seed in TRAINING_SEEDS:
  b=find(boundary,training_seed=seed,checkpoint_role="best");f=find(boundary,training_seed=seed,checkpoint_role="final")
  if b and f:
   tactical_rise.append(number(f,"tactical_overshoot_fraction",0)-number(b,"tactical_overshoot_fraction",0))
   escape_rise.append(number(f,"escape_like_fraction",0)-number(b,"escape_like_fraction",0))
 heading_sat_rise=[]
 for seed in TRAINING_SEEDS:
  b=find(sat,training_seed=seed,checkpoint_role="best",scope="boundary_precursors",threshold="0.9");f=find(sat,training_seed=seed,checkpoint_role="final",scope="boundary_precursors",threshold="0.9")
  if b and f:heading_sat_rise.append(number(f,"heading",0)-number(b,"heading",0))
 drift_boundary=[number(r,"deterministic_action_l2",0) for r in drift if r.get("state_class")=="boundary_precursor"]
 drift_other=[number(r,"deterministic_action_l2",0) for r in drift if r.get("state_class")=="non_boundary"]
 guard_assoc=[]
 for seed in TRAINING_SEEDS:
  for role in ("best","final"):
   b=find(guard,training_seed=seed,checkpoint_role=role,state_class="BOUNDARY");n=find(guard,training_seed=seed,checkpoint_role=role,state_class="NO_BOUNDARY")
   if b and n and number(b,"activation_episode_rate") is not None and number(n,"activation_episode_rate") is not None:guard_assoc.append(number(b,"activation_episode_rate")-number(n,"activation_episode_rate"))
 paired_horizon=[]
 for seed in TRAINING_SEEDS:
  for role in ("best","final"):
   h3=[r for r in cross if int(r["training_seed"])==seed and r["checkpoint_role"]==role and int(float(r["evaluation_max_steps"]))==3000];h1=[r for r in cross if int(r["training_seed"])==seed and r["checkpoint_role"]==role and int(float(r["evaluation_max_steps"]))==1000]
   if h3 and h1:paired_horizon.append(abs(np.mean([truth(r["red_success"]) for r in h3])-np.mean([truth(r["red_success"]) for r in h1])))
 def phase_mean(seed,phase,metric):
  row=find(opt,training_seed=seed,phase=phase,metric=metric);return None if row is None else number(row,"mean")
 kl_pattern=[];critic_pattern=[]
 for seed in TRAINING_SEEDS:
  before,late=phase_mean(seed,"BEST_PRECEDING_200K","approx_kl"),phase_mean(seed,"BEST_TO_FINAL","approx_kl")
  vb,vl=phase_mean(seed,"BEST_PRECEDING_200K","value_loss"),phase_mean(seed,"BEST_TO_FINAL","value_loss")
  if before is not None and late is not None:kl_pattern.append(late-before)
  if vb is not None and vl is not None:critic_pattern.append(vl-vb)
 reward_boundary_not_worse=[]
 for seed in (5401,5402):
  row=next((r for r in probs if r["training_seed"]==seed and r["checkpoint_role"]=="final"),None)
  reward_boundary_not_worse.append(bool(row and row["n_boundary_failure"]>=5 and row["n_pure_combat_defeat"]>=5 and row["P_Return_boundary_failure_gt_pure_combat_defeat"] is not None and row["P_Return_boundary_failure_gt_pure_combat_defeat"]>=.5))
 local_optimum_supported=(len(escape_rise)>=2 and escape_rise[0]>.05 and escape_rise[1]>.05 and all(reward_boundary_not_worse))
 evidence=[
  {"candidate":"Environment implementation bug","level":"NOT_SUPPORTED","evidence":"strict identities pass; prior mechanics regression audit passed"},
  {"candidate":"Reward mission-alignment mismatch","level":"SUPPORTED_PHENOMENON","evidence":"local R1/R2 and no shared terminal team reward; episode ranking in reward_ranking_by_outcome.csv"},
  {"candidate":"Boundary-associated late degradation","level":"STRONGLY_SUPPORTED_ASSOCIATION" if len(boundary_rise)==3 and boundary_rise[0]>.2 and boundary_rise[1]>.2 else "SUPPORTED_ASSOCIATION","evidence":f"paired boundary episode-rate deltas={boundary_rise}; phenotype association, not reward causality"},
  {"candidate":"Reward-driven boundary local optimum","level":"SUPPORTED_ASSOCIATION" if local_optimum_supported else "PLAUSIBLE_NOT_ESTABLISHED","evidence":f"final boundary-failure return not-worse flags for 5401/5402={reward_boundary_not_worse}; paired escape-like fraction deltas={escape_rise}; requires both conditions"},
  {"candidate":"Tactical overshoot","level":"SUPPORTED_ASSOCIATION" if any(x>.05 for x in tactical_rise[:2]) else "SUPPORTED_PHENOMENON" if any(number(r,"tactical_overshoot_count",0)>0 for r in boundary) else "NOT_SUPPORTED","evidence":f"paired tactical fraction deltas={tactical_rise}; no intent inference"},
  {"candidate":"Action saturation","level":"SUPPORTED_ASSOCIATION" if any(x>.05 for x in heading_sat_rise[:2]) else "NOT_SUPPORTED","evidence":f"boundary precursor heading >.9 paired deltas={heading_sat_rise}"},
  {"candidate":"Policy mean drift","level":"SUPPORTED_ASSOCIATION" if drift_boundary and drift_other and statistics.mean(drift_boundary)>1.25*max(statistics.mean(drift_other),1e-12) else "SUPPORTED_PHENOMENON" if drift else "DATA_INSUFFICIENT","evidence":"best/final actors evaluated on same union fixed-state banks"},
  {"candidate":"Policy variance drift","level":"SUPPORTED_PHENOMENON" if drift else "DATA_INSUFFICIENT","evidence":"log_std best/final deltas on the same banks; association requires state-class contrast"},
  {"candidate":"PPO KL/update instability","level":"SUPPORTED_ASSOCIATION" if len(kl_pattern)==3 and kl_pattern[0]>0 and kl_pattern[1]>0 and kl_pattern[2]<max(kl_pattern[0],kl_pattern[1]) else "PLAUSIBLE_NOT_ESTABLISHED","evidence":f"late-minus-prebest KL means={kl_pattern}; observational"},
  {"candidate":"Critic/value drift","level":"SUPPORTED_ASSOCIATION" if len(critic_pattern)==3 and critic_pattern[0]>0 and critic_pattern[1]>0 and critic_pattern[2]<max(critic_pattern[0],critic_pattern[1]) else "PLAUSIBLE_NOT_ESTABLISHED","evidence":f"late-minus-prebest value-loss means={critic_pattern}"},
  {"candidate":"Weapon stochasticity","level":"SUPPORTED_PHENOMENON","evidence":"static attempt-level probabilities; not equivalent to episode win variance"},
  {"candidate":"Blue ground-guard interaction","level":"SUPPORTED_ASSOCIATION" if any(abs(x)>.1 for x in guard_assoc) else "NOT_SUPPORTED","evidence":f"boundary-minus-nonboundary activation episode-rate deltas={guard_assoc}; association only"},
  {"candidate":"3000-step direct horizon effect","level":"SUPPORTED_ASSOCIATION" if any(x>.05 for x in paired_horizon) else "NOT_SUPPORTED","evidence":f"paired H3000/H1000 win-rate absolute deltas={paired_horizon}; deployment truncation only"},
  {"candidate":"3000-step training-horizon effect","level":"DATA_INSUFFICIENT","evidence":"requires fresh matched H1000 retraining"}]
 write_csv(out/"root_cause_evidence_table.csv",evidence)
 report="# Clean MAPPO single-wave stability diagnostic\n\nThe report is observational and does not claim `ROOT_CAUSE_FOUND`.\n\n## Protocol\n\nSix clean MAPPO checkpoints, deterministic `tanh(mean)`, paired 89.2M environment seeds. H1000 rows are explicitly cross-horizon deployment evaluation of H3000-trained policies.\n\n## Evidence\n\n|Candidate|Level|\n|---|---|\n"+"\n".join(f"|{r['candidate']}|{r['level']}|" for r in evidence)+"\n\nReward ranking uses newly collected individual episodes. Counterfactual scores re-rank fixed trajectories only. See the CSV artifacts for paired seed/checkpoint statistics and confidence limitations.\n"
 (out/"final_stability_diagnostic_report.md").write_text(report,encoding="utf-8");print(json.dumps({"status":"SINGLE_WAVE_STABILITY_ANALYSIS_COMPLETE","trajectory_dir":str(out),"episodes":len(outcomes)},indent=2))
if __name__=="__main__":main()
