"""Strict preflight for the three-seed W1SG 300k development screen."""
from __future__ import annotations
import csv,hashlib,json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.modular_mappo.protocol import validate_w1sg_branch
from algorithm.train_modular_mappo import load_config
SEEDS=(5301,5302,5303);SOURCE=1_505_280;TARGET=1_805_280
REFS={"plain":("pwtr_plain_matched_control",None),"stratified":("pwtr_stratified",(True,False,"recent_uniform",False,False,False,False)),"current_actor_only":("pwtr_current_actor_only",(True,True,"current",False,False,True,False))}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check_ref(path,kind,seed,parent_sha):
 required=("run_summary.json","run_config.json","algorithm_config.yaml","evaluation_history.csv","latest.pt","final.pt")
 if any(not (path/n).is_file() for n in required):raise RuntimeError(f"{path}: incomplete files")
 summary=json.loads((path/"run_summary.json").read_text());run=json.loads((path/"run_config.json").read_text());algorithm=yaml.safe_load((path/"algorithm_config.yaml").read_text())
 if int(summary.get("sampled_steps",-1))!=TARGET or int(run.get("seed",-1))!=seed:raise RuntimeError(f"{path}: endpoint/seed mismatch")
 provenance=run.get("branch_provenance",{});actual_parent=provenance.get("parent_checkpoint_sha256")
 if actual_parent!=parent_sha:raise RuntimeError(f"{path}: parent SHA mismatch")
 method,expected=REFS[kind]
 if algorithm.get("development_method")!=method:raise RuntimeError(f"{path}: method mismatch")
 module=algorithm.get("modules",{}).get("persistent_wave_trajectory_replay",{})
 if expected is None:
  if module.get("enabled",False):raise RuntimeError(f"{path}: Plain enables PWTR")
 else:
  actual=(bool(module.get("fresh_wave_stratification")),bool(module.get("replay_enabled")),module.get("replay_source"),bool(module.get("priority_enabled")),bool(module.get("bridge_enabled")),bool(module.get("actor_replay")),bool(module.get("critic_replay")))
  fixed=tuple(int(module.get(k,-1)) for k in ("sequence_length","bridge_half_length","min_segment_length","partition_capacity","actor_max_age_updates"))
  if actual!=expected or fixed!=(128,64,32,32,2):raise RuntimeError(f"{path}: branch identity mismatch")
 with (path/"evaluation_history.csv").open(newline="",encoding="utf-8-sig") as f:rows=list(csv.DictReader(f))
 exact=[r for r in rows if int(float(r["sampled_steps"]))==TARGET]
 if len(exact)!=1 or (int(float(exact[0]["evaluation_episodes"])),int(float(exact[0]["evaluation_seed_base"])),int(float(exact[0]["evaluation_seed_end"])))!=(50,44_000_000,44_000_049):raise RuntimeError(f"{path}: evaluation mismatch")
 return {"kind":kind,"seed":seed,"parent_sha_match":True,"endpoint":TARGET}
def main():
 if not torch.cuda.is_available():raise RuntimeError("CUDA mandatory")
 env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text());cfg=load_config(ROOT/"configs/dev_w1sg_current_actor_300k.yaml")
 if (env["environment_variant"],env["persistent_waves"]["total_waves"],env["simulation"]["max_steps"])!=("persistent_wave_v2",3,3000):raise RuntimeError("environment identity mismatch")
 sources={};validations={};refs=[]
 for seed in SEEDS:
  checkpoint=ROOT/f"outputs/diag_mappo_learnability/l3_seed{seed}/checkpoint_{SOURCE}.pt";state=torch.load(checkpoint,map_location="cpu",weights_only=False);source_sha=sha(checkpoint)
  if int(state.get("sampled_steps",-1))!=SOURCE or int(state["extra"].get("training_seed",-1))!=seed or state.get("enabled_modules")!=["actor_lr_decay"] or float(state["actor_optimizer"]["param_groups"][0]["lr"])!=1e-4:raise RuntimeError(f"seed{seed}: source mismatch")
  sources[str(seed)]={"path":str(checkpoint),"sha256":source_sha};validations[str(seed)]=validate_w1sg_branch(state,env,cfg,{"training_seed":seed,"training_num_envs":24,"training_smoke":False})
  for kind in REFS:refs.append(check_ref(ROOT/f"outputs/dev_pwtr_{kind}_seed{seed}_300k",kind,seed,source_sha))
 historical=json.loads((ROOT/"outputs/dev_pwtr_stratified_seed5301_300k/run_config.json").read_text());old={x["path"]:x["sha256"] for x in historical["runtime_source_manifest_files"]}
 frozen=("algorithm/modules/persistent_wave_trajectory_replay.py","env/combat_env.py","env/persistent_env.py");hashes={p:{"historical":old.get(p),"current":sha(ROOT/p)} for p in frozen}
 if any(v["historical"]!=v["current"] for v in hashes.values()):raise RuntimeError("frozen PWTR/environment SHA mismatch")
 outputs=[ROOT/f"outputs/dev_w1sg_current_actor_seed{seed}_300k" for seed in SEEDS];existing=[str(p) for p in outputs if p.exists()]
 if existing:raise RuntimeError(f"formal outputs exist: {existing}")
 report={"status":"READY_FOR_W1SG_300K_SCREEN","cuda":torch.cuda.get_device_name(0),"sources":sources,"validations":validations,"references":refs,"reference_count":len(refs),"frozen_sha":hashes,"formal_outputs_absent":True,"formal_output_count":3,"evaluation_range":[44_000_000,44_000_049],"45m_used":False};print(json.dumps(report,indent=2))
if __name__=="__main__":main()
