"""Bounded, signal-based M1–M8 capability benchmark."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np,torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner

MODULES=[("M1","pw_m1_wave_context.yaml"),("M2","pw_m2_gru.yaml"),("M3","pw_m3_popart.yaml"),("M4","pw_m4_wave_reward.yaml"),("M5","pw_m5_wave_balance.yaml"),("M6","pw_m6_warm_start.yaml"),("M7","pw_m7_curriculum.yaml"),("M8","pw_m8_policy_anchor.yaml")]

def optimization_rows(run):
 path=run/"optimization_metrics.jsonl";return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def expected_signal(mid,summary,updates):
 final=summary["final_optimization_metrics"]
 if mid=="M1":return summary["module_protocol"]["module_config"]["wave_context"]["context_target"]=="critic_only"
 if mid=="M2":return final.get("hidden_norm_max",0)>0 and final.get("actor_optimizer_steps_this_update")==final.get("recurrent_minibatches_per_epoch",0)*2
 if mid=="M3":return final.get("popart_count",0)>0 and final.get("popart_std",1)>0
 if mid=="M4":return summary["reward_adapter_totals"]["reward_bonus_total"]>0
 if mid=="M5":return any(abs(row.get("effective_wave_weight_mean",0)-1)<1e-6 and max(row.get(f"weight_wave_{k}",0) for k in (1,2,3))<=3.000001 and len({round(row.get(f"weight_wave_{k}",0),6) for k in (1,2,3) if row.get(f"alive_agent_samples_wave_{k}",0)>0})>1 for row in updates)
 if mid=="M6":return summary["warm_start_provenance"].get("actor",{}).get("exact_loaded_count",0)>0
 if mid=="M7":return [row["total_waves"] for row in summary["curriculum_transitions"]]==[1,2,3]
 if mid=="M8":return final.get("anchor_kl",0)>0 and final.get("anchor_effective_coefficient",0)>0
 return False

def main():
 p=argparse.ArgumentParser();p.add_argument("--pilot-steps",type=int,default=5000);p.add_argument("--device",default="cpu");p.add_argument("--output-dir",default="outputs/modular_hardening_study");a=p.parse_args()
 if not 1<=a.pilot_steps<=20000:raise ValueError("pilot-steps must be in [1,20000]")
 if a.pilot_steps*8>180000:raise ValueError("aggregate pilot budget exceeds 180k")
 out=ROOT/a.output_dir;out.mkdir(parents=True,exist_ok=True);env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text());source=ROOT/"outputs/d999_seed2023/best_eval.pt";rows=[]
 for mid,file in MODULES:
  run=out/f"pilot_{mid.lower()}";summary_path=run/"run_summary.json"
  if not summary_path.exists():
   cfg=load_config(ROOT/"configs"/file);cfg["training"].update({"seed":42,"num_train_envs":4,"total_sampled_steps":a.pilot_steps,"evaluation_episodes":3,"rollout_steps":128,"ppo_epochs":2,"minibatch_size":256})
   if mid=="M7":cfg["modules"]["curriculum"].update({"stage1_end":a.pilot_steps//3,"stage2_end":2*a.pilot_steps//3})
   if mid in {"M6","M8"} and not source.exists():rows.append({"module":mid,"status":"NOT_READY","reason":"source checkpoint missing"});continue
   runner=ModularMAPPOTrainingRunner(env,cfg,4,a.pilot_steps,a.device,42,run,False,str(source) if mid=="M6" else None,str(source) if mid=="M8" else None);runner.run()
  summary=json.loads(summary_path.read_text());updates=optimization_rows(run);values=[value for row in updates for value in row.values() if isinstance(value,(int,float))];finite=bool(values) and np.all(np.isfinite(values));artifacts=all((run/name).exists() for name in ("latest.pt","best_eval.pt","evaluation_history.csv","training_metrics.jsonl","optimization_metrics.jsonl"));state=torch.load(run/"latest.pt",map_location="cpu",weights_only=False);protocol=all(key in state.get("extra",{}) for key in ("environment_version","network_architecture","training_seed","training_num_envs"));signal=expected_signal(mid,summary,updates);status="READY" if finite and artifacts and protocol and signal else ("READY_WITH_CAUTION" if finite and artifacts and protocol else "NOT_READY")
  tf=summary["wave_transition_fractions"];af=summary["wave_alive_agent_sample_fractions"];ev=summary["latest_evaluation"] or {}
  rows.append({"module":mid,"status":status,"expected_signal":signal,"sampled_steps":summary["sampled_steps"],**{f"transition_w{k}":tf[f"wave_{k}"] for k in (1,2,3)},**{f"alive_agent_w{k}":af[f"wave_{k}"] for k in (1,2,3)},"average_return":ev.get("average_return"),"average_waves_cleared":ev.get("average_waves_cleared",0),"clear_wave_3_probability":ev.get("clear_wave_3_probability",0),"anchor_kl":summary["final_optimization_metrics"].get("anchor_kl",0),"popart_std":summary["final_optimization_metrics"].get("popart_std",1),"reward_bonus_total":summary["reward_adapter_totals"]["reward_bonus_total"]})
 fields=list(rows[0]);
 with (out/"module_micro_pilot.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 wavefields=["module"]+[f"transition_w{k}" for k in (1,2,3)]+[f"alive_agent_w{k}" for k in (1,2,3)]
 with (out/"module_capability_wave_samples.csv").open("w",newline="",encoding="utf-8") as f:w=csv.DictWriter(f,fieldnames=wavefields);w.writeheader();w.writerows({k:r.get(k) for k in wavefields} for r in rows)
 payload={"purpose":"functional/capability evidence, not performance","seed":42,"num_envs":4,"pilot_steps_per_module":a.pilot_steps,"aggregate_pilot_steps":a.pilot_steps*8,"readiness_is_signal_based":True,"rows":rows};(out/"module_capability_summary.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
 table="\n".join(f"| {r['module']} | {r['status']} | {r.get('transition_w1',0):.4f}/{r.get('transition_w2',0):.4f}/{r.get('transition_w3',0):.4f} | {r.get('alive_agent_w1',0):.4f}/{r.get('alive_agent_w2',0):.4f}/{r.get('alive_agent_w3',0):.4f} | {r.get('expected_signal')} |" for r in rows)
 (out/"module_capability_report.md").write_text(f"# Hardened modular capability report\n\n5k/module, seed42, 4 envs; mechanism evidence only.\n\n|Module|Readiness|Transition W1/W2/W3|Alive-agent W1/W2/W3|Expected signal|\n|---|---|---|---|---|\n{table}\n",encoding="utf-8");print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
