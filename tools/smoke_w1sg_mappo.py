"""CUDA-only CurrentActor versus W1SG tiny integration smoke."""
from __future__ import annotations
from copy import deepcopy
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import validate_pwtr_branch,validate_w1sg_branch
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.train_modular_mappo import load_config

SOURCE=ROOT/"outputs/diag_mappo_learnability/l3_seed5302/checkpoint_1505280.pt"
def digest(x):
 h=hashlib.sha256()
 def add(v):
  if torch.is_tensor(v):h.update(v.detach().cpu().numpy().tobytes())
  elif isinstance(v,np.ndarray):h.update(v.tobytes())
  elif isinstance(v,dict):
   for k in sorted(v,key=str):h.update(str(k).encode());add(v[k])
  elif isinstance(v,(list,tuple)):
   for q in v:add(q)
  else:h.update(repr(v).encode())
 add(x);return h.hexdigest()
def fsha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def natural(trainer,wave):
 t=128;obs=np.random.default_rng(wave).normal(size=(t,1,4,52)).astype("f")
 with torch.no_grad():
  o=torch.as_tensor(obs[:,0],device="cuda");alive=torch.ones(t,4,device="cuda");dist,_=trainer.actor.distribution_step(o,None,None,None,alive);raw=dist.sample();log=trainer.actor._squashed_log_prob(dist,raw,torch.tanh(raw))
 return type("B",(),{"observations":obs,"next_observations":obs+.01,"raw_actions":raw.cpu().numpy()[:,None],"old_log_probs":log.cpu().numpy()[:,None],"rewards":np.full((t,1,4),.1,"f"),"dones":np.zeros((t,1),"f"),"alive_masks":np.ones((t,1,4),"f"),"next_alive_masks":np.ones((t,1,4),"f"),"wave_indices":np.full((t,1),wave,np.int64),"wave_transition_flags":np.zeros((t,1),"f")})()

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--output-dir",default="outputs/smoke_w1sg_mappo");args=parser.parse_args();out=ROOT/args.output_dir
 if not torch.cuda.is_available():raise RuntimeError("CUDA mandatory")
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text());state=torch.load(SOURCE,map_location="cpu",weights_only=False);sha=fsha(SOURCE)
 ccfg=load_config(ROOT/"configs/dev_pwtr_current_actor_only_300k.yaml");wcfg=load_config(ROOT/"configs/dev_w1sg_current_actor_300k.yaml");runtime={"training_seed":5302,"training_num_envs":24,"training_smoke":False}
 cp=validate_pwtr_branch(state,env,ccfg,runtime);wp=validate_w1sg_branch(state,env,wcfg,runtime);rollouts={}
 for name,cfg,prov in (("control",ccfg,cp),("w1sg",wcfg,wp)):
  runner=ModularMAPPOTrainingRunner(env,cfg,2,1_805_280,"cuda",5302,out/name,False,resume_mode=True,branch_provenance={**prov,"parent_checkpoint_sha256":sha});runner.branch_from(SOURCE,prov["intervention"],sha);rollouts[name]=runner.collect_rollout(2);runner.vector.close()
 fields=("observations","actions","raw_actions","old_log_probs","raw_environment_rewards","alive_masks","next_alive_masks","wave_indices","dones");physical={k:np.array_equal(getattr(rollouts["control"],k),getattr(rollouts["w1sg"],k)) for k in fields}
 if not all(physical.values()):raise RuntimeError(f"rollout mismatch {physical}")
 fresh={}
 for name,cfg in (("control",ccfg),("w1sg",wcfg)):
  tr=build_modular_mappo_trainer(cfg,"cuda",256,1_805_280);tr.load(SOURCE,strict_protocol=False,restore_rng=True);m=tr.update(deepcopy(rollouts[name]));fresh[name]={"actor":digest(tr.actor.state_dict()),"critic":digest(tr.critic.state_dict()),"ao":digest(tr.actor_optimizer.state_dict()),"co":digest(tr.critic_optimizer.state_dict()),"rng":digest(tr.capture_rng_state()),"replay":m["pwtr_replay_batches"]}
 if fresh["control"]!=fresh["w1sg"]:raise RuntimeError("Fresh PPO/sensitivity is not bitwise inert")
 trainers={name:build_modular_mappo_trainer(cfg,"cuda",256,1_805_280) for name,cfg in (("control",ccfg),("w1sg",wcfg))}
 for tr in trainers.values():tr.load(SOURCE,strict_protocol=False,restore_rng=True)
 w1=natural(trainers["control"],1);w2=natural(trainers["control"],2);w3=natural(trainers["control"],3)
 # Deterministic sensitivity with normalized synthetic W1 advantages.
 wt=trainers["w1sg"];before=(digest(wt.actor.state_dict()),digest(wt.critic.state_dict()),digest(wt.actor_optimizer.state_dict()),digest(wt.critic_optimizer.state_dict()),digest(wt.capture_rng_state()))
 obs=torch.as_tensor(w1.observations,device="cuda");raw=torch.as_tensor(w1.raw_actions,device="cuda");old=torch.as_tensor(w1.old_log_probs,device="cuda");alive=torch.as_tensor(w1.alive_masks,device="cuda");waves=torch.as_tensor(w1.wave_indices,device="cuda");adv=torch.linspace(-1,1,128,device="cuda")[:,None,None].expand(128,1,4)
 wt.wave1_sensitivity_gating.compute_sensitivity(wt.actor,obs,raw,old,alive,waves,adv,.2);after=(digest(wt.actor.state_dict()),digest(wt.critic.state_dict()),digest(wt.actor_optimizer.state_dict()),digest(wt.critic_optimizer.state_dict()),digest(wt.capture_rng_state()))
 if before!=after:raise RuntimeError("sensitivity computation mutated training state/RNG")
 reward=digest([w2.rewards,w3.rewards]);results={}
 for name,tr in trainers.items():
  mod=tr.persistent_wave_trajectory_replay;mod.ingest_rollout(deepcopy(w2),1);mod.ingest_rollout(deepcopy(w3),1);mod.current_rollout_generation=1;mod.current_later_states=512;mod.fresh_rollout_count=2
  critic_before=digest(tr.critic.state_dict());metrics=tr._pwtr_replay_phase();results[name]={"loss":metrics["pwtr_actor_replay_loss"],"raw_grad":metrics["pwtr_actor_replay_grad_norm_preclip"],"budget":metrics["pwtr_replay_batches_budget"],"batches":metrics["pwtr_replay_batches"],"actor_steps":metrics["pwtr_replay_actor_optimizer_steps_this_phase"],"critic_steps":metrics["pwtr_replay_critic_optimizer_steps_this_phase"],"critic_unchanged":critic_before==digest(tr.critic.state_dict()),"reward_unchanged":reward==digest([w2.rewards,w3.rewards])}
 if results["control"]["loss"]!=results["w1sg"]["loss"] or abs(results["control"]["raw_grad"]-results["w1sg"]["raw_grad"])>1e-5:raise RuntimeError(f"raw replay mismatch {results}")
 diag=wt.wave1_sensitivity_gating.gradient_metrics()
 if not (diag["w1sg_raw_protected_gradient_cosine"]<1 and diag["w1sg_norm_preservation_relative_error"]<=1e-5):raise RuntimeError(f"gating ineffective {diag}")
 if any(r["critic_steps"]!=0 or not r["critic_unchanged"] for r in results.values()):raise RuntimeError("critic replay changed")
 report={"status":"W1SG_CUDA_SMOKE_PASS","cuda":torch.cuda.get_device_name(0),"physical_rollout_bitwise":physical,"fresh_update_bitwise":True,"sensitivity_state_rng_inert":True,"replay":results,"w1sg":diag,"source_unchanged":sha==fsha(SOURCE),"reward_unchanged":True,"all_finite":all(np.isfinite(v) for v in diag.values()),"45m_used":False,"formal_training_started":False}
 (out/"smoke_report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
