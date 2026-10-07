"""CUDA-only matched smoke for Actor clipping 0.5 versus 1.0."""
from __future__ import annotations
from copy import deepcopy
import argparse,hashlib,json,random,sys
from pathlib import Path
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modular_mappo.protocol import validate_pwtr_branch,validate_actor_grad_clip_branch
SOURCE=ROOT/"outputs/diag_mappo_learnability/l3_seed5302/checkpoint_1505280.pt"
def digest(value):
 h=hashlib.sha256()
 def add(x):
  if torch.is_tensor(x):h.update(x.detach().cpu().numpy().tobytes())
  elif isinstance(x,np.ndarray):h.update(x.tobytes())
  elif isinstance(x,dict):
   for key in sorted(x,key=str):h.update(str(key).encode());add(x[key])
  elif isinstance(x,(list,tuple)):
   for item in x:add(item)
  else:h.update(repr(x).encode())
 add(value);return h.hexdigest()
def fsha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def global_rng():return {"python":random.getstate(),"numpy":np.random.get_state(),"cpu":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all()}
def restore_rng(state):random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["cpu"]);torch.cuda.set_rng_state_all(state["cuda"])
def raw_gradient_identity(left,right,rollout):
 def gradients(trainer,rng):
  restore_rng(rng);obs=torch.as_tensor(rollout.observations.reshape(-1,4,52)[:16],device="cuda");act=torch.as_tensor(rollout.actions.reshape(-1,4,3)[:16],device="cuda");raw=torch.as_tensor(rollout.raw_actions.reshape(-1,4,3)[:16],device="cuda");old=torch.as_tensor(rollout.old_log_probs.reshape(-1,4)[:16],device="cuda");alive=torch.as_tensor(rollout.alive_masks.reshape(-1,4)[:16],device="cuda");count=obs.shape[0];adv=torch.linspace(-1,1,count*4,device="cuda").reshape(count,4)*alive;oldv=torch.zeros(count,4,device="cuda");target=torch.ones(count,4,device="cuda");weights=torch.ones(count,device="cuda");ctx=torch.zeros(count,0,device="cuda")
  losses=trainer._loss_step(obs,act,raw,old,alive,adv,oldv,target,weights,ctx);trainer.actor_optimizer.zero_grad();(losses[0]-trainer.entropy_coefficient*losses[2]+losses[3]).backward();actor=[p.grad.detach().clone() for p in trainer.actor.trainable_policy_parameters()];trainer.actor_optimizer.zero_grad();trainer.critic_optimizer.zero_grad();(trainer.value_loss_coefficient*losses[1]).backward();critic=[p.grad.detach().clone() for p in trainer.critic.parameters()];return losses,actor,critic
 rng=global_rng();a,ag,ac=gradients(left,rng);b,bg,bc=gradients(right,rng)
 return {"actor_loss":bool(torch.equal(a[0],b[0])),"entropy":bool(torch.equal(a[2],b[2])),"value_loss":bool(torch.equal(a[1],b[1])),"actor_gradient_tensors":all(torch.equal(x,y) for x,y in zip(ag,bg)),"critic_gradient_tensors":all(torch.equal(x,y) for x,y in zip(ac,bc))}
def main():
 parser=argparse.ArgumentParser();parser.add_argument("--output-dir",default="outputs/smoke_actor_grad_clip_mappo");args=parser.parse_args();out=ROOT/args.output_dir
 if not torch.cuda.is_available():raise RuntimeError("CUDA mandatory")
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text());state=torch.load(SOURCE,map_location="cpu",weights_only=False);source_sha=fsha(SOURCE)
 configs={"Plain":load_config(ROOT/"configs/dev_pwtr_plain_300k.yaml"),"Clip05":load_config(ROOT/"configs/dev_actor_grad_clip_05_control_300k.yaml"),"Clip10":load_config(ROOT/"configs/dev_actor_grad_clip_10_300k.yaml")};runtime={"training_seed":5302,"training_num_envs":24,"training_smoke":False}
 provenance={"Plain":validate_pwtr_branch(state,env,configs["Plain"],runtime),"Clip05":validate_actor_grad_clip_branch(state,env,configs["Clip05"],runtime),"Clip10":validate_actor_grad_clip_branch(state,env,configs["Clip10"],runtime)}
 runners={};rollouts={};initial={}
 for name in ("Plain","Clip05","Clip10"):
  runner=ModularMAPPOTrainingRunner(env,configs[name],2,1_805_280,"cuda",5302,out/name,False,resume_mode=True,branch_provenance={**provenance[name],"parent_checkpoint_sha256":source_sha});runner.branch_from(SOURCE,provenance[name]["intervention"],source_sha);runners[name]=runner
  initial[name]={"actor":digest(runner.trainer.actor.state_dict()),"critic":digest(runner.trainer.critic.state_dict()),"actor_optimizer":digest(runner.trainer.actor_optimizer.state_dict()),"critic_optimizer":digest(runner.trainer.critic_optimizer.state_dict()),"rng":digest(runner.trainer.capture_rng_state())};rollouts[name]=runner.collect_rollout(2)
 fields=("observations","actions","raw_actions","old_log_probs","raw_environment_rewards","alive_masks","wave_indices","dones");physical={field:all(np.array_equal(getattr(rollouts["Plain"],field),getattr(rollouts[name],field)) for name in ("Clip05","Clip10")) for field in fields}
 for runner in runners.values():runner.vector.close()
 branch_identity={key:len({initial[name][key] for name in initial})==1 for key in initial["Plain"]}
 if not all(branch_identity.values()) or not all(physical.values()):raise RuntimeError(f"branch/rollout identity failure: {branch_identity}/{physical}")
 raw_identity=raw_gradient_identity(runners["Clip05"].trainer,runners["Clip10"].trainer,rollouts["Plain"])
 if not all(raw_identity.values()):raise RuntimeError(f"raw gradient identity failed: {raw_identity}")
 # Reload all three after the gradient-only probe, then execute the same update under identical global RNG.
 trainers={name:build_modular_mappo_trainer(configs[name],"cuda",256,1_805_280) for name in configs}
 for value in trainers.values():value.load(SOURCE,strict_protocol=False,restore_rng=False);value.ppo_epochs=1
 rng=global_rng();results={};post={}
 for name in ("Plain","Clip05","Clip10"):
  restore_rng(rng);results[name]=trainers[name].update(deepcopy(rollouts[name]));post[name]={"actor":digest(trainers[name].actor.state_dict()),"critic":digest(trainers[name].critic.state_dict()),"actor_optimizer":digest(trainers[name].actor_optimizer.state_dict()),"critic_optimizer":digest(trainers[name].critic_optimizer.state_dict()),"rng":digest(global_rng())}
 control_noop={key:post["Plain"][key]==post["Clip05"][key] for key in post["Plain"]};control_noop.update({key:results["Plain"][key]==results["Clip05"][key] for key in ("actor_loss","value_loss","actor_grad_norm","critic_grad_norm")})
 critic_identity={key:post["Clip05"][key]==post["Clip10"][key] for key in ("critic","critic_optimizer")};critic_identity["preclip"]=results["Clip05"]["actor_grad_clip_critic_preclip_norm"]==results["Clip10"]["actor_grad_clip_critic_preclip_norm"];critic_identity["postclip"]=results["Clip05"]["actor_grad_clip_critic_postclip_norm"]==results["Clip10"]["actor_grad_clip_critic_postclip_norm"]
 if not all(control_noop.values()) or not all(critic_identity.values()):raise RuntimeError(f"single-variable identity failure: {control_noop}/{critic_identity}")
 if results["Clip05"]["actor_grad_clip_preclip_norm"]<=1.:raise RuntimeError("smoke Actor norm did not exercise both thresholds")
 if not (results["Clip05"]["actor_grad_clip_postclip_norm"]<=.50001 and .999<results["Clip10"]["actor_grad_clip_postclip_norm"]<=1.00001 and results["Clip10"]["actor_grad_clip_exact_scale"]>results["Clip05"]["actor_grad_clip_exact_scale"]):raise RuntimeError("Actor clip intervention did not produce expected geometry")
 checkpoints={}
 for name in ("Clip05","Clip10"):
  path=out/f"{name}.pt";trainers[name].save(path);restored=build_modular_mappo_trainer(configs[name],"cuda",256,1_805_280);restored.load(path,strict_protocol=True,restore_rng=False);checkpoints[name]=digest(restored.actor.state_dict())==digest(trainers[name].actor.state_dict())
 cross_rejected=[]
 for checkpoint_name,config_name in (("Clip10","Clip05"),("Clip05","Clip10")):
  try:build_modular_mappo_trainer(configs[config_name],"cuda",256,1_805_280).load(out/f"{checkpoint_name}.pt",strict_protocol=True,restore_rng=False);cross_rejected.append(False)
  except RuntimeError:cross_rejected.append(True)
 if not all(checkpoints.values()) or not all(cross_rejected):raise RuntimeError("resume protocol failure")
 finite=all(np.isfinite(value) for name in ("Clip05","Clip10") for value in results[name].values() if isinstance(value,(int,float)))
 report={"status":"ACTOR_GRAD_CLIP_CUDA_SMOKE_PASS","cuda":torch.cuda.get_device_name(0),"branch_identity":branch_identity,"first_physical_rollout_bitwise":physical,"raw_matched_minibatch_identity":raw_identity,"control05_plain_noop":control_noop,"critic_identity_clip05_clip10":critic_identity,"actor_geometry":{"preclip":results["Clip05"]["actor_grad_clip_preclip_norm"],"clip05_post":results["Clip05"]["actor_grad_clip_postclip_norm"],"clip10_post":results["Clip10"]["actor_grad_clip_postclip_norm"],"clip05_exact_scale":results["Clip05"]["actor_grad_clip_exact_scale"],"clip10_exact_scale":results["Clip10"]["actor_grad_clip_exact_scale"]},"critic_limit_clip05":results["Clip05"]["actor_grad_clip_critic_limit"],"critic_limit_clip10":results["Clip10"]["actor_grad_clip_critic_limit"],"strict_resume":checkpoints,"cross_config_resume_rejected":cross_rejected,"all_finite":finite,"source_unchanged":source_sha==fsha(SOURCE),"uses_45m":False,"formal_training_started":False}
 (out/"smoke_report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
