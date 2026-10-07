#!/usr/bin/env python3
"""Offline, read-only MAPPO persistent-wave root-cause audit."""
from __future__ import annotations
import csv,json,math,statistics,sys
from pathlib import Path
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/mappo_root_cause_audit"
RUNS={
 "MLP-1.5M":ROOT/"outputs/mappo_mlp_seed5303_1p5m",
 "Attention-fresh-1.5M":ROOT/"outputs/mappo_attention_seed5303_1p5m",
 "Control-3e-4":ROOT/"outputs/mappo_attn_lr3e4_cont_seed5303_1p5m",
 "Treatment-1e-4":ROOT/"outputs/mappo_attn_lr1e4_cont_seed5303_1p5m",
 "Plain-3M-seed5301":ROOT/"outputs/diag_mappo_learnability/l3_seed5301",
 "Plain-3M-seed5302":ROOT/"outputs/diag_mappo_learnability/l3_seed5302",
 "Plain-3M-seed5303":ROOT/"outputs/diag_mappo_learnability/l3_seed5303",
}
CONTINUATIONS=("Control-3e-4","Treatment-1e-4")
METRICS={"W1":"clear_wave_1_probability","W2":"clear_wave_2_probability","W3":"clear_wave_3_probability","AW":"average_waves_cleared","Return":"average_return","RedLoss":"average_red_loss","BlueLoss":"average_blue_loss","Boundary":"average_red_boundary_exits","Ground":"average_red_ground_losses","EpisodeLength":"average_episode_length","SurvivorsAfterW1":"average_red_survivors_after_wave_1","SurvivorsAfterW2":"average_red_survivors_after_wave_2","SurvivorsAfterW3":"average_red_survivors_after_wave_3"}

def csv_rows(path:Path):
 with path.open(newline="",encoding="utf-8-sig") as stream:return list(csv.DictReader(stream))
def jsonl(path:Path):return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def number(value):
 try:return float(value)
 except (TypeError,ValueError):return None
def percentile(values:list[float],q:float):return None if not values else float(np.quantile(np.asarray(values,float),q))
def summary(values:list[float]):
 return {"n":len(values),"median":percentile(values,.5),"p90":percentile(values,.9),"p95":percentile(values,.95),"p99":percentile(values,.99),"max":None if not values else max(values)}
def endpoint(rows:list[dict],target:int):
 exact=[row for row in rows if int(float(row["sampled_steps"]))==target]
 if len(exact)!=1:raise RuntimeError(f"expected one endpoint {target}, found {len(exact)}")
 row=exact[0];result={key:number(row.get(field)) for key,field in METRICS.items()}
 result["Q2"]=None if not result["W1"] else result["W2"]/result["W1"];result["Q3"]=None if not result["W2"] else result["W3"]/result["W2"]
 return result
def closest_eval(rows:list[dict],step:int,direction:str):
 candidates=[row for row in rows if (int(float(row["sampled_steps"]))<=step if direction=="before" else int(float(row["sampled_steps"]))>=step)]
 if not candidates:return None
 return max(candidates,key=lambda r:int(float(r["sampled_steps"]))) if direction=="before" else min(candidates,key=lambda r:int(float(r["sampled_steps"])))
def corr(rows:list[dict],a:str,b:str):
 pairs=[(number(row.get(a)),number(row.get(b))) for row in rows];pairs=[p for p in pairs if p[0] is not None and p[1] is not None]
 if len(pairs)<3:return None
 x=np.asarray([p[0] for p in pairs]);y=np.asarray([p[1] for p in pairs])
 if x.std()==0 or y.std()==0:return None
 return float(np.corrcoef(x,y)[0,1])
def stage(step:int,target:int)->str:
 if target<=1_500_000:
  return "0-0.5M" if step<=500_000 else ("0.5-1.0M" if step<=1_000_000 else "1.0-1.5M")
 return "0-0.5M" if step<=500_000 else ("0.5-1.0M" if step<=1_000_000 else ("1.0-1.5M" if step<=1_500_000 else ("1.5-2.0M" if step<=2_000_000 else ("2.0-2.5M" if step<=2_500_000 else "2.5-3.0M"))))
def validate_inputs():
 missing={name:[file for file in ("evaluation_history.csv","optimization_metrics.jsonl","run_summary.json","training_metrics.jsonl") if not (path/file).is_file()] for name,path in RUNS.items()}
 bad={name:value for name,value in missing.items() if value}
 if bad:raise RuntimeError(f"incomplete audit inputs: {bad}")

def optimization_audit(name:str,rows:list[dict],evaluations:list[dict]):
 kl=[float(row["approx_kl"]) for row in rows];clip=[float(row["clip_fraction"]) for row in rows];grad=[float(row["actor_grad_norm"]) for row in rows]
 record={"method":name,**{f"kl_{k}":v for k,v in summary(kl).items()},"kl_gt_0.03_fraction":float(np.mean(np.asarray(kl)>.03)),"kl_gt_0.05_fraction":float(np.mean(np.asarray(kl)>.05)),"kl_gt_0.10_fraction":float(np.mean(np.asarray(kl)>.10)),"clip_fraction_mean":statistics.mean(clip),"actor_grad_norm_mean":statistics.mean(grad)}
 outliers=[]
 for row in rows:
  if float(row["approx_kl"])<=.05:continue
  epochs=[(index,float(row.get(f"epoch_{index}_approx_kl",float("nan")))) for index in range(10) if row.get(f"epoch_{index}_approx_kl") is not None]
  max_epoch,max_epoch_kl=max(epochs,key=lambda x:x[1]) if epochs else (None,None);step=int(row["sampled_steps"]);before=closest_eval(evaluations,step,"before");after=closest_eval(evaluations,step,"after")
  outliers.append({"method":name,"sampled_steps":step,"approx_kl":row["approx_kl"],"max_epoch":max_epoch,"max_epoch_kl":max_epoch_kl,"clip_fraction":row.get("clip_fraction"),"actor_grad_norm":row.get("actor_grad_norm"),"entropy":row.get("entropy"),"log_std_psi":row.get("policy_log_std_mean_psi"),"log_std_theta":row.get("policy_log_std_mean_theta"),"log_std_v":row.get("policy_log_std_mean_v"),"ratio_p1":row.get("ratio_p1"),"ratio_p99":row.get("ratio_p99"),"ratio_min":row.get("ratio_min"),"ratio_max":row.get("ratio_max"),"first_minibatch_kl":row.get("first_minibatch_approx_kl"),"evaluation_before_step":None if before is None else int(float(before["sampled_steps"])),"evaluation_before_AW":None if before is None else before.get("average_waves_cleared"),"evaluation_before_W3":None if before is None else before.get("clear_wave_3_probability"),"evaluation_after_step":None if after is None else int(float(after["sampled_steps"])),"evaluation_after_AW":None if after is None else after.get("average_waves_cleared"),"evaluation_after_W3":None if after is None else after.get("clear_wave_3_probability")})
 clipping={"method":name,"updates":len(grad),"pre_clip_norm_mean":statistics.mean(grad),"pre_clip_norm_median":statistics.median(grad),**{f"fraction_gt_{str(threshold).replace('.','_')}":float(np.mean(np.asarray(grad)>threshold)) for threshold in (.5,1,2,5,10)}}
 return record,outliers,clipping
def policy_std_rows(name:str,rows:list[dict],target:int):
 result=[]
 for phase in sorted({stage(int(row["sampled_steps"]),target) for row in rows}):
  selected=[row for row in rows if stage(int(row["sampled_steps"]),target)==phase]
  for action in ("psi","theta","v"):
   values=[float(row[f"policy_log_std_mean_{action}"]) for row in selected if row.get(f"policy_log_std_mean_{action}") is not None]
   result.append({"method":name,"stage":phase,"action":action,"statistic_scope":"distribution_of_per-update_live-state_means",**{f"log_std_{key}":value for key,value in {"p1":percentile(values,.01),"p10":percentile(values,.1),"median":percentile(values,.5),"p90":percentile(values,.9),"p99":percentile(values,.99),"min":None if not values else min(values),"max":None if not values else max(values),"mean":None if not values else statistics.mean(values)}.items()}})
 return result
def write_csv(path:Path,rows:list[dict]):
 fields=[]
 for row in rows:
  for key in row:
   if key not in fields:fields.append(key)
 with path.open("w",newline="",encoding="utf-8") as stream:writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def md_table(rows,columns):
 head="| "+" | ".join(columns)+" |\n|"+"|".join(["---"]*len(columns))+"|\n"
 return head+"".join("| "+" | ".join(str(row.get(column,"")) for column in columns)+" |\n" for row in rows)

def main():
 validate_inputs()
 if OUT.exists():raise FileExistsError(OUT)
 data={};optimization=[];outliers=[];clipping=[];std_rows=[];endpoints=[];reward_corr=[];reward_endpoints=[];critic_rows=[]
 for name,path in RUNS.items():
  evals=csv_rows(path/"evaluation_history.csv");opt=jsonl(path/"optimization_metrics.jsonl");summary_json=json.loads((path/"run_summary.json").read_text(encoding="utf-8"));target=int(summary_json["sampled_steps"]);data[name]={"eval":evals,"opt":opt,"target":target,"summary":summary_json}
  ep=endpoint(evals,target);endpoints.append({"method":name,"sampled_steps":target,**ep})
  exact=next(row for row in evals if int(float(row["sampled_steps"]))==target);reward_endpoints.append({"method":name,"sampled_steps":target,"R1":number(exact.get("average_episode_r1_total")),"R2":number(exact.get("average_episode_r2_total")),"R3":number(exact.get("average_episode_r3_total")),"R4":number(exact.get("average_episode_r4_total")),"Return":number(exact.get("average_return")),"AW":number(exact.get("average_waves_cleared"))})
  oa,ko,gc=optimization_audit(name,opt,evals);optimization.append(oa);outliers.extend(ko);clipping.append(gc);std_rows.extend(policy_std_rows(name,opt,target))
  critic_rows.append({"method":name,"updates":len(opt),"value_loss_mean":statistics.mean(float(row["value_loss"]) for row in opt),"value_loss_median":statistics.median(float(row["value_loss"]) for row in opt),"explained_variance_mean":statistics.mean(float(row["explained_variance"]) for row in opt),"explained_variance_median":statistics.median(float(row["explained_variance"]) for row in opt),"critic_grad_norm_mean":statistics.mean(float(row["critic_grad_norm"]) for row in opt),"per_wave_value_error":"NOT_OBSERVABLE_FROM_EXISTING_LOGS"})
  reward_corr.append({"method":name,"n_evaluations":len(evals),"return_vs_waves":corr(evals,"average_return","average_waves_cleared"),"return_vs_blue_losses":corr(evals,"average_return","average_blue_loss"),"return_vs_red_losses":corr(evals,"average_return","average_red_loss"),"return_vs_boundary":corr(evals,"average_return","average_red_boundary_exits"),"return_vs_ground":corr(evals,"average_return","average_red_ground_losses"),"return_vs_episode_length":corr(evals,"average_return","average_episode_length")})
 continuation=json.loads((ROOT/"outputs/mappo_late_lr_continuation_analysis/analysis.json").read_text(encoding="utf-8"))
 evidence=[
  {"candidate":"MAPPO implementation bug","classification":"IMPLEMENTATION_CORRECT","evidence":"Static audit and directed GAE/transition tests found no contract violation in tanh log-prob, PPO ratio/clipping, bootstrap, masks, or update loop."},
  {"candidate":"Actor LR","classification":"SUPPORTED_CONTRIBUTOR","evidence":"Treatment improved endpoint W3/Q3 and Red survival relative to matched Control, but AW was identical and Control did not reproduce fresh collapse."},
  {"candidate":"policy std dynamics","classification":"SUPPORTED_ASSOCIATION","evidence":"Treatment has materially lower logged state-dependent log_std, but Actor LR jointly changes the shared backbone, mean head, and log_std head. Variance is not causally isolated."},
  {"candidate":"KL heavy-tail","classification":"SUPPORTED_PHENOMENON","evidence":"Treatment has a heavy KL tail (p95/p99/max about .0888/.1313/.2551), while its endpoint W3 is not worse. Existence is supported; performance causality is not."},
  {"candidate":"KL causal role","classification":"PLAUSIBLE_NOT_ESTABLISHED","evidence":"KL spikes do not map monotonically to endpoint quality and the intervention did not isolate KL."},
  {"candidate":"gradient clipping","classification":"PLAUSIBLE_NOT_ESTABLISHED","evidence":"Logged actor_grad_norm is an update-level mean of minibatch pre-clip norms. Mean >0.5 suggests frequent activation but cannot establish a per-minibatch clipping fraction or causal loss."},
  {"candidate":"GAE implementation","classification":"IMPLEMENTATION_CORRECT","evidence":"Intermediate wave transitions bootstrap; episode done and individual death stop recursion; terminal transition observation is retained across auto-reset."},
  {"candidate":"GAE long-horizon adequacy","classification":"PLAUSIBLE_NOT_ESTABLISHED","evidence":"gamma*lambda=.94905. Correct code does not establish adequate credit across hundreds to thousands of steps."},
  {"candidate":"wave sample scarcity","classification":"PLAUSIBLE_NOT_ESTABLISHED","evidence":"Per-step wave_index was not persisted in these runs; direct W1/W2/W3 transition exposure is NOT_OBSERVABLE_FROM_EXISTING_LOGS."},
  {"candidate":"Attention Critic","classification":"BENEFICIAL / NOT_SUPPORTED_AS_PRIMARY_CAUSE","evidence":"MLP-vs-Attention comparison changes value behavior/performance but does not remove late-wave decline; per-wave value calibration was not logged."},
  {"candidate":"observation aliasing","classification":"FORMAL_ALIASING_EXISTS / PRIMARY_CAUSE_NOT_ESTABLISHED","evidence":"52D policy input omits wave identity, waves remaining, global remaining horizon, and weapon readiness. Empirical magnitude needs the fixed-bank audit."},
  {"candidate":"reward pathology","classification":"NOT_SUPPORTED","evidence":"R1 directly rewards kills/losses and dominates event outcomes; existing evidence does not show pathological shaping dominance."},
  {"candidate":"mission-level reward limitation","classification":"PLAUSIBLE_NOT_PRIMARY","evidence":"There is no wave-clear, survivor, final-mission, or time bonus, but aggregate correlations do not establish this as primary."},
  {"candidate":"environment lifecycle bug","classification":"NOT_SUPPORTED","evidence":"Lifecycle is internally consistent and explicit: Red state persists, Blue respawns, global time continues, fire states reset, and intermediate clear is nonterminal."},
  {"candidate":"entry-state quality","classification":"PLAUSIBLE_WITH_SUPPORTING_ASSOCIATION","evidence":"Survivor/duration aggregates are consistent with entry-state amplification, but complete W2/W3 entry geometry was not persisted."},
  {"candidate":"closed-loop basin sensitivity","classification":"STRONGLY_SUPPORTED_PHENOMENON","evidence":"Large seed dispersion and continuation divergence establish a strong system property, not its underlying mechanism."},
  {"candidate":"underlying basin mechanism","classification":"UNDERLYING_MECHANISM_NOT_FULLY_LOCALIZED","evidence":"Candidate mechanisms include on-policy distribution shift, exploration, state-dependent variance, hidden context, later-wave scarcity, entry-state amplification, and repeated PPO updates."},
  {"candidate":"continuation RNG/state reset","classification":"SUPPORTED_SOURCE_OF_TRAJECTORY_DIVERGENCE","evidence":"Branches restore weights/optimizers/counters/episode indices but not live environment state or full RNG. This supports divergence, not a standalone performance cause."},
 ]
 implementation=[
  ("Actor tanh-Gaussian","CORRECT","Normal latent, rsample, tanh action."),("latent raw action保存","CORRECT","RolloutBatch stores raw_actions and PPO passes them back."),("tanh Jacobian log-prob","CORRECT","Stable latent-space 2*(log2-x-softplus(-2x)) correction."),("old/new log-prob ratio","CORRECT","new-old then exp; old values collected on-policy."),("deterministic evaluation","CORRECT","tanh(distribution.mean)."),("state-dependent log_std","REASONABLE_BUT_NONSTANDARD","Linear state head instead of global parameter; may redistribute/collapse variance by state."),("log_std clamp","CORRECT","[-5,2] before exp; aggregate logs do not show boundary contact, state tails unobserved."),("squashed entropy estimator","REASONABLE_BUT_NONSTANDARD","One reparameterized Monte-Carlo sample of -log pi(a|s)."),("entropy gradient","POTENTIAL_RISK","Pathwise MC gradient is mathematically meaningful but adds sampling variance and RNG consumption per minibatch."),("optimizer order","CORRECT","Actor backward/step then Critic backward/step on separate networks."),("PPO clipping","CORRECT","min(rA,clip(r)A) with alive mask."),("value clipping","CORRECT","max(unclipped squared error, clipped-value squared error)."),("advantage normalization","CORRECT","Global live-agent normalization, then dead slots masked."),("gradient clipping","CORRECT","clip_grad_norm_=.5; logged value is pre-clip norm."),("minibatch sampling","CORRECT","Fresh NumPy permutation per epoch over time×env transitions."),("10 epochs","CORRECT","All configured epochs execute; each transition revisited once per epoch."),("rollout/update boundary","CORRECT","Fixed on-policy rollout, then update; final partial rollout exact target."),("old value timing","CORRECT","Computed once before PPO update under no_grad."),("Critic bootstrap","CORRECT","Uses transition_next_observations and next_alive mask."),("checkpoint/resume","POTENTIAL_RISK","Restores parameters/optimizers/counters, but old checkpoint lacks full RNG and live 24-env dynamic state.")]
 reward={"R1":"+10 shared among credited Red attackers per Blue kill; -10 to Red killed by weapon or ground","R2":"-10 to each Red boundary exit","R3":"+0.001 approach shaping for nearest alive Blue under angular/distance condition","R4":"tiered nearest-target advantage +0.1/+0.02/+0.01 or threat -0.15/-0.025/-0.015","wave_clear_bonus":False,"survivor_bonus":False,"final_mission_bonus":False,"time_penalty":False,"per_wave_R1_R4":"Recorded inside per-wave episode info during evaluation, but raw episode records are not persisted in evaluation_history; NOT_OBSERVABLE_FROM_EXISTING_LOGS","fixed_wave_count_return_vs_episode_length":"Requires per-episode evaluation records; NOT_OBSERVABLE_FROM_EXISTING_LOGS","classification":"REWARD_LIMITATION_BUT_NOT_PRIMARY"}
 observation={"dimension":52,"included":["own x/y/altitude/speed/last executed bank/heading/pitch","three ally slots: relative 3D position, speed, relative heading, pitch, alive","four enemy slots: distance, speed, AA, ATA, HA, alive"],"omitted":["wave index","waves cleared/remaining","global elapsed/remaining horizon","own weapon FireState readiness","teammate/opponent weapon readiness","explicit mission progress"],"formal_aliasing":"FORMAL_ALIASING_EXISTS","primary_cause":"PRIMARY_CAUSE_NOT_ESTABLISHED"}
 environment={"red_physical_state_across_waves":"preserved","red_alive_dead_across_waves":"preserved","blue_respawn":"complete 4-UAV formation at candidate direction maximizing minimum Red distance","global_steps":"continuous across waves","timeout":"single 3000-step mission horizon","fire_state_boundary":"both Red and Blue FireState reset/armed; Blue controller transient reset","clearing_step_new_blue_attack":"no; spawn occurs after combat resolution and new Blue acts next step","intermediate_clear":"terminated=false,truncated=false when time remains","final_clear":"episode termination retained","mutual_destruction":"draw; no next wave","boundary_ground":"resolved before combat each step","auto_reset":"terminal transition_next_observation retained separately from reset observation","classification":"EXPLICIT_DESIGN_WITH_POTENTIAL_TASK_DIFFICULTY"}
 gae={"implementation_correctness":"IMPLEMENTATION_CORRECT","long_horizon_temporal_credit_adequacy":"PLAUSIBLE_NOT_ESTABLISHED","gamma_lambda_product":.999*.95,"ordinary_live":"bootstrap and lambda recursion continue","individual_death":"death-step reward retained; next_alive=0 stops bootstrap/recursion; later dead slots masked","red_all_dead_final_clear_timeout":"done=1 stops bootstrap","intermediate_wave_clear":"done=0 and next observation is freshly spawned wave, so credit propagates across boundary","auto_reset":"vector stores transition_next_observations before replacing current observation with reset","tests":"existing and added directed tests cover death/done/intermediate transition semantics"}
 summary_json={"status":"AUDIT_COMPLETE","scope":"read-only existing evidence; no training or formal reevaluation","continuation_key_facts":continuation,"endpoints":endpoints,"optimization_summary":optimization,"gradient_clipping_summary":clipping,"critic_summary":critic_rows,"policy_std_scope_note":"Percentiles are across logged per-update live-state mean log_std, not per-state or per-wave tails.","wave_exposure":"NOT_RECORDED","entry_state_full_geometry":"NOT_OBSERVABLE_FROM_EXISTING_LOGS","per_wave_value_calibration":"NOT_OBSERVABLE_FROM_EXISTING_LOGS","implementation_audit":[{"item":a,"classification":b,"evidence":c} for a,b,c in implementation],"gae":gae,"observation":observation,"environment":environment,"reward":reward,"reward_endpoint_components":reward_endpoints,"reward_correlations":reward_corr,"root_cause_evidence":evidence}
 OUT.mkdir(parents=True);(OUT/"audit_summary.json").write_text(json.dumps(summary_json,indent=2),encoding="utf-8")
 write_csv(OUT/"optimization_audit.csv",optimization);write_csv(OUT/"kl_outliers.csv",outliers);write_csv(OUT/"gradient_clipping_summary.csv",clipping);write_csv(OUT/"policy_std_summary.csv",std_rows);write_csv(OUT/"endpoint_summary.csv",endpoints);write_csv(OUT/"reward_correlations.csv",reward_corr);write_csv(OUT/"reward_endpoint_components.csv",reward_endpoints);write_csv(OUT/"critic_summary.csv",critic_rows)
 (OUT/"implementation_audit.md").write_text("# MAPPO implementation audit\n\n"+md_table(summary_json["implementation_audit"],["item","classification","evidence"]),encoding="utf-8")
 (OUT/"gae_semantics.md").write_text("# GAE semantics\n\n"+"\n".join(f"- **{k}**: {v}" for k,v in gae.items()),encoding="utf-8")
 (OUT/"observation_aliasing.md").write_text("# Observation aliasing\n\nFORMAL_ALIASING_EXISTS; PRIMARY_CAUSE_NOT_ESTABLISHED.\n\nIncluded: "+"; ".join(observation["included"])+".\n\nOmitted: "+"; ".join(observation["omitted"])+".\n\nA legal fixed-state-bank audit is required for causal magnitude.\n",encoding="utf-8")
 (OUT/"environment_audit.md").write_text("# Environment lifecycle audit\n\n"+"\n".join(f"- **{k}**: {v}" for k,v in environment.items()),encoding="utf-8")
 (OUT/"reward_audit.md").write_text("# Reward audit\n\n"+"\n".join(f"- **{k}**: {v}" for k,v in reward.items())+"\n\nEvaluation-history correlations are descriptive across checkpoints, not independent experimental units.\n",encoding="utf-8")
 (OUT/"entry_state_plan.md").write_text("# Entry-state and exposure plan\n\nExisting formal logs do not persist per-step wave index or full W2/W3 entry geometry. These fields are **NOT_OBSERVABLE_FROM_EXISTING_LOGS**. Run `tools/diagnose_mappo_fixed_state_bank.py` later on independent 88M diagnostic seeds to collect legal states, per-wave log-std tails, entry geometry, alias nearest-neighbours, and small deterministic/stochastic deployment comparisons.\n",encoding="utf-8")
 report="# MAPPO root-cause report (recalibrated)\n\n## Evidence classification\n\n"+md_table(evidence,["candidate","classification","evidence"])+"\n## Main conclusion\n\nNo explicit MAPPO implementation bug was found. Closed-loop trajectory/basin sensitivity is a **strongly supported phenomenon**, not a localized mechanistic root cause. Actor LR is a supported contributor; policy-std dynamics are an association; KL heavy tails are a phenomenon whose causal role is not established. GAE code correctness is distinct from long-horizon credit adequacy. Observation aliasing, wave exposure, entry-state amplification, and PPO/data-distribution interactions remain mechanisms to localize.\n"
 (OUT/"root_cause_report_recalibrated.md").write_text(report,encoding="utf-8")
 print(json.dumps({"status":"AUDIT_COMPLETE","output":str(OUT),"kl_outliers":len(outliers),"runs":len(RUNS)},indent=2))
if __name__=="__main__":main()
