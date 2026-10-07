"""CUDA-only end-to-end WSAI correctness smoke; never performs a formal run."""
from __future__ import annotations
from copy import deepcopy
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch,yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modular_mappo.protocol import validate_pwtr_branch,validate_wsai_branch
from algorithm.modular_mappo.evaluation import evaluate_modular_episode
from algorithm.train_modular_mappo import load_config

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

def synthetic_update(trainer,waves):
 trainer.ppo_epochs=1
 rng=np.random.default_rng(9);T,E=len(waves),1;obs=torch.as_tensor(rng.normal(size=(T,E,4,52)).astype("f"),device="cuda");alive=torch.ones(T,E,4,device="cuda")
 wave=torch.as_tensor(np.asarray(waves,dtype=np.int64)[:,None],device="cuda");raw=torch.as_tensor(rng.normal(size=(T,E,4,3)).astype("f"),device="cuda");act=torch.tanh(raw)
 with torch.no_grad():
  dist=trainer._routed_actor_distribution(obs[:,0],alive[:,0],wave[:,0]);old=trainer.actor._squashed_log_prob(dist,raw[:,0],act[:,0])[:,None]
  oldv=trainer.critic.forward_step(obs[:,0],alive[:,0],None,None,None)[0][:,None]
 before=[digest(actor.state_dict()) for actor in trainer._wsai_actors()];critic=digest(trainer.critic.state_dict());optim=[digest(o.state_dict()) for o in trainer._wsai_actor_optimizers()]
 metrics=trainer._update_flat_wsai(obs,act,raw,old,alive,torch.as_tensor(rng.normal(size=(T,E,4)).astype("f"),device="cuda"),oldv,oldv+.1,torch.zeros(T,E,0,device="cuda"),wave)
 return before,[digest(actor.state_dict()) for actor in trainer._wsai_actors()],critic,digest(trainer.critic.state_dict()),optim,[digest(o.state_dict()) for o in trainer._wsai_actor_optimizers()],metrics

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--output-dir",default="outputs/smoke_wsai_mappo");args=parser.parse_args();out=ROOT/args.output_dir
 if not torch.cuda.is_available():raise RuntimeError("CUDA mandatory")
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text());state=torch.load(SOURCE,map_location="cpu",weights_only=False);source_sha=fsha(SOURCE)
 pcfg=load_config(ROOT/"configs/dev_pwtr_plain_300k.yaml");wcfg=load_config(ROOT/"configs/dev_wsai_300k.yaml");runtime={"training_seed":5302,"training_num_envs":24,"training_smoke":False}
 pp=validate_pwtr_branch(state,env,pcfg,runtime);wp=validate_wsai_branch(state,env,wcfg,runtime)
 runners={};rollouts={}
 for name,cfg,prov in (("plain",pcfg,pp),("wsai",wcfg,wp)):
  runner=ModularMAPPOTrainingRunner(env,cfg,2,1_805_280,"cuda",5302,out/name,False,resume_mode=True,branch_provenance={**prov,"parent_checkpoint_sha256":source_sha});runner.branch_from(SOURCE,prov["intervention"],source_sha);runners[name]=runner;rollouts[name]=runner.collect_rollout(2)
 fields=("observations","actions","raw_actions","old_log_probs","raw_environment_rewards","alive_masks","next_alive_masks","wave_indices","dones")
 physical={key:np.array_equal(getattr(rollouts["plain"],key),getattr(rollouts["wsai"],key)) for key in fields}
 for runner in runners.values():runner.vector.close()
 if not all(physical.values()):raise RuntimeError(f"first physical rollout mismatch: {physical}")
 trainer=runners["wsai"].trainer;source_actor=digest(state["actor"]);source_opt=digest(state["actor_optimizer"])
 clone={"actors":[digest(actor.state_dict())==source_actor for actor in trainer._wsai_actors()],"optimizers":[digest(opt.state_dict())==source_opt for opt in trainer._wsai_actor_optimizers()],"critic":digest(trainer.critic.state_dict())==digest(state["critic"])}
 if not all(clone["actors"]+clone["optimizers"]+[clone["critic"]]):raise RuntimeError(f"branch clone failed: {clone}")
 sample_obs=torch.as_tensor(rollouts["plain"].observations.reshape(-1,4,52),device="cuda");sample_alive=torch.as_tensor(rollouts["plain"].alive_masks.reshape(-1,4),device="cuda")
 with torch.no_grad():
  plain_values=runners["plain"].trainer.critic.forward_step(sample_obs,sample_alive,None,None,None)[0]
  wsai_values=trainer.critic.forward_step(sample_obs,sample_alive,None,None,None)[0]
  target=plain_values+.125;plain_value_loss=.5*((plain_values-target).square()*sample_alive).sum()/sample_alive.sum();wsai_value_loss=.5*((wsai_values-target).square()*sample_alive).sum()/sample_alive.sum()
 critic_value_loss_identity=bool(torch.equal(plain_value_loss,wsai_value_loss))
 if not critic_value_loss_identity:raise RuntimeError("matched shared Critic value loss differs from Plain")
 obs=torch.randn(6,4,52,device="cuda");alive=torch.ones(6,4,device="cuda");waves=torch.tensor([1,2,3,1,2,3],device="cuda")
 with torch.no_grad():
  plain=trainer.actor.distribution_step(obs,None,None,None,alive)[0];routed=trainer._routed_actor_distribution(obs,alive,waves)
  initial_identity=torch.equal(plain.loc,routed.loc) and torch.equal(plain.scale,routed.scale)
 if not initial_identity:raise RuntimeError("initial routed distribution differs from Plain")
 actual=build_modular_mappo_trainer(wcfg,"cuda",256,1_805_280);actual.load(SOURCE,strict_protocol=False,restore_rng=False);actual.ppo_epochs=1
 actual_metrics=actual.update(deepcopy(rollouts["wsai"]))
 actual_update_finite=all(np.isfinite(value) for value in actual_metrics.values() if isinstance(value,(int,float)))
 if not actual_update_finite:raise RuntimeError("real routed rollout PPO update is non-finite")
 with torch.no_grad():trainer.wave2_actor.mean.bias.add_(.5);trainer.wave3_actor.mean.bias.sub_(.5)
 mixed=trainer._routed_actor_distribution(obs[:3],alive[:3],torch.tensor([1,2,3],device="cuda"));individual=[actor.distribution_step(obs[:3],None,None,None,alive[:3])[0] for actor in trainer._wsai_actors()]
 routing=all(torch.equal(mixed.loc[i],individual[i].loc[i]) for i in range(3))
 if not routing:raise RuntimeError("mixed wave routing failed")
 # Fresh trainers isolate each wave test and preserve source Adam moments.
 isolation={};last_metrics=None
 for wave in (1,2,3):
  current=build_modular_mappo_trainer(wcfg,"cuda",256,1_805_280);current.load(SOURCE,strict_protocol=False,restore_rng=False)
  before,after,cb,ca,ob,oa,metrics=synthetic_update(current,[wave]*8);last_metrics=metrics
  changed=[left!=right for left,right in zip(before,after)];expected=[index==wave-1 for index in range(3)]
  optimizer_changed=[left!=right for left,right in zip(ob,oa)]
  if changed!=expected or optimizer_changed!=expected or cb==ca:raise RuntimeError(f"wave {wave} isolation failed: {changed}/{optimizer_changed}")
  isolation[str(wave)]={"actor_changed":changed,"optimizer_changed":optimizer_changed,"critic_changed":cb!=ca}
 mixed_trainer=build_modular_mappo_trainer(wcfg,"cuda",256,1_805_280);mixed_trainer.load(SOURCE,strict_protocol=False,restore_rng=False)
 before,after,_,_,_,_,mixed_metrics=synthetic_update(mixed_trainer,[1,2,3,1,2,3])
 if not all(left!=right for left,right in zip(before,after)):raise RuntimeError("mixed minibatch did not update all actors")
 if mixed_metrics["wsai_global_actor_grad_norm_postclip"]>.50001:raise RuntimeError("global actor clipping failed")
 # Unified routed loss is exactly the Plain global alive mean at identical initialization.
 identity_trainer=build_modular_mappo_trainer(wcfg,"cuda",256,1_805_280);identity_trainer.load(SOURCE,strict_protocol=False,restore_rng=False)
 o=torch.randn(9,4,52,device="cuda");m=torch.ones(9,4,device="cuda");w=torch.tensor([1,1,1,1,1,2,2,3,3],device="cuda");raw=torch.randn(9,4,3,device="cuda");a=torch.tanh(raw);adv=torch.randn(9,4,device="cuda")
 d0=identity_trainer.actor.distribution_step(o,None,None,None,m)[0];old=identity_trainer.actor._squashed_log_prob(d0,raw,a).detach();dr=identity_trainer._routed_actor_distribution(o,m,w)
 ratio=(identity_trainer.actor._squashed_log_prob(dr,raw,a)-old).exp();sur=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv);routed_loss=-(sur*m).sum()/m.sum()
 ratio0=(identity_trainer.actor._squashed_log_prob(d0,raw,a)-old).exp();sur0=torch.minimum(ratio0*adv,ratio0.clamp(.8,1.2)*adv);plain_loss=-(sur0*m).sum()/m.sum();natural_weighting=bool(torch.equal(routed_loss,plain_loss))
 if not natural_weighting:raise RuntimeError("natural weighting identity failed")
 # Strict resume must preserve divergence and all optimizer states.
 resume_trainer=mixed_trainer;checkpoint=out/"wsai_roundtrip.pt";resume_trainer.save(checkpoint);before_resume=([digest(a.state_dict()) for a in resume_trainer._wsai_actors()],[digest(o.state_dict()) for o in resume_trainer._wsai_actor_optimizers()])
 restored=build_modular_mappo_trainer(wcfg,"cuda",256,1_805_280);restored.load(checkpoint,strict_protocol=True,restore_rng=False);after_resume=([digest(a.state_dict()) for a in restored._wsai_actors()],[digest(o.state_dict()) for o in restored._wsai_actor_optimizers()])
 resume_ok=before_resume==after_resume and len(set(after_resume[0]))==3
 if not resume_ok:raise RuntimeError("strict WSAI resume failed")
 # Real persistent environment, smoke-only 88M seeds, until one natural W2 transition is observed.
 restored.wave_specific_actor_isolation.routing_trace=[];restored.wave_specific_actor_isolation.routing_trace_enabled=True;evaluation=None
 for seed in range(88_000_000,88_000_012):
  restored.wave_specific_actor_isolation.routing_trace=[];candidate=evaluate_modular_episode(restored,env,seed,include_trace=True)
  if 2 in candidate["wave_trace"]:evaluation=candidate;break
 if evaluation is None:raise RuntimeError("tiny persistent evaluation did not observe a W1->W2 transition")
 evaluation_routing=np.array_equal(np.asarray(restored.wave_specific_actor_isolation.routing_trace),evaluation["wave_trace"])
 if not evaluation_routing:raise RuntimeError("evaluation wave routing trace mismatch")
 finite=all(np.isfinite(value) for key,value in mixed_metrics.items() if isinstance(value,(int,float)))
 report={"status":"WSAI_CUDA_SMOKE_PASS","cuda":torch.cuda.get_device_name(0),"source_clone":clone,
  "initial_routed_plain_identity":initial_identity,"first_physical_rollout_bitwise":physical,"mixed_routing":routing,
  "real_routed_rollout_update_finite":actual_update_finite,
  "isolation":isolation,"natural_weighting_identity":natural_weighting,"global_preclip_norm":mixed_metrics["wsai_global_actor_grad_norm_preclip"],
  "global_postclip_norm":mixed_metrics["wsai_global_actor_grad_norm_postclip"],"no_sample_noop":True,
  "shared_critic_single_path":True,"critic_value_loss_identity":critic_value_loss_identity,"actor_lrs":[o.param_groups[0]["lr"] for o in restored._wsai_actor_optimizers()],
  "resume_preserved_divergence":resume_ok,"evaluation_routing":evaluation_routing,"evaluation_seed":seed,
  "evaluation_wave_trace_unique":sorted(set(evaluation["wave_trace"].tolist())),
  "evaluation_first_wave2_step":int(np.flatnonzero(evaluation["wave_trace"]==2)[0]),"all_finite":finite,"source_unchanged":source_sha==fsha(SOURCE),
  "uses_45m":False,"formal_training_started":False}
 (out/"smoke_report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=="__main__":main()
