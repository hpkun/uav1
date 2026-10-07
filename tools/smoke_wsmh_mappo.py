"""CUDA-only end-to-end WSMH correctness smoke; never performs formal training."""
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
from algorithm.modular_mappo.protocol import validate_pwtr_branch,validate_wsmh_branch,checkpoint_architecture
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
def synthetic(trainer,wave):
 trainer.ppo_epochs=1;rng=np.random.default_rng(91+wave);T=8
 obs=torch.as_tensor(rng.normal(size=(T,1,4,52)).astype("f"),device="cuda");alive=torch.ones(T,1,4,device="cuda");waves=torch.full((T,1),wave,device="cuda")
 raw=torch.as_tensor(rng.normal(size=(T,1,4,3)).astype("f"),device="cuda");act=torch.tanh(raw)
 with torch.no_grad():
  dist=trainer._wsmh_routed_distribution(obs[:,0],alive[:,0],waves[:,0]);old=trainer.actor._squashed_log_prob(dist,raw[:,0],act[:,0])[:,None];oldv=trainer.critic.forward_step(obs[:,0],alive[:,0],None,None,None)[0][:,None]
 before=[digest(m.state_dict()) for m in (trainer.actor.backbone,trainer.actor.log_std,*trainer._wsmh_means())];critic=digest(trainer.critic.state_dict());opts=[digest(o.state_dict()) for o in trainer._wsmh_mean_optimizers()]
 metrics=trainer._update_flat_wsmh(obs,act,raw,old,alive,torch.as_tensor(rng.normal(size=(T,1,4)).astype("f"),device="cuda"),oldv,oldv+.1,torch.zeros(T,1,0,device="cuda"),waves)
 return before,[digest(m.state_dict()) for m in (trainer.actor.backbone,trainer.actor.log_std,*trainer._wsmh_means())],critic,digest(trainer.critic.state_dict()),opts,[digest(o.state_dict()) for o in trainer._wsmh_mean_optimizers()],metrics
def main():
 p=argparse.ArgumentParser();p.add_argument("--output-dir",default="outputs/smoke_wsmh_mappo");args=p.parse_args();out=ROOT/args.output_dir
 if not torch.cuda.is_available():raise RuntimeError("CUDA mandatory")
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);env=yaml.safe_load((ROOT/"configs/persistent_wave_v2_environment.yaml").read_text());state=torch.load(SOURCE,map_location="cpu",weights_only=False);source_sha=fsha(SOURCE)
 pcfg=load_config(ROOT/"configs/dev_pwtr_plain_300k.yaml");cfg=load_config(ROOT/"configs/dev_wsmh_300k.yaml");runtime={"training_seed":5302,"training_num_envs":24,"training_smoke":False}
 pp=validate_pwtr_branch(state,env,pcfg,runtime);wp=validate_wsmh_branch(state,env,cfg,runtime);runners={};rollouts={}
 for name,current,prov in (("plain",pcfg,pp),("wsmh",cfg,wp)):
  runner=ModularMAPPOTrainingRunner(env,current,2,1_805_280,"cuda",5302,out/name,False,resume_mode=True,branch_provenance={**prov,"parent_checkpoint_sha256":source_sha});runner.branch_from(SOURCE,prov["intervention"],source_sha);runners[name]=runner;rollouts[name]=runner.collect_rollout(2)
 fields=("observations","actions","raw_actions","old_log_probs","raw_environment_rewards","alive_masks","next_alive_masks","wave_indices","dones");physical={key:np.array_equal(getattr(rollouts["plain"],key),getattr(rollouts["wsmh"],key)) for key in fields}
 for runner in runners.values():runner.vector.close()
 if not all(physical.values()):raise RuntimeError(f"physical rollout mismatch: {physical}")
 trainer=runners["wsmh"].trainer
 clone={"backbone":digest(trainer.actor.backbone.state_dict())==digest({k.removeprefix("backbone."):v for k,v in state["actor"].items() if k.startswith("backbone.")}),"mean_heads":[digest(h.state_dict())==digest(trainer.actor.mean.state_dict()) for h in trainer._wsmh_means()],"log_std":digest(trainer.actor.log_std.state_dict())==digest({k.removeprefix("log_std."):v for k,v in state["actor"].items() if k.startswith("log_std.")}),"critic":digest(trainer.critic.state_dict())==digest(state["critic"])}
 if not all([clone["backbone"],clone["log_std"],clone["critic"],*clone["mean_heads"]]):raise RuntimeError(f"source clone failed: {clone}")
 optimizer_clone=True
 for optimizer in (trainer.wave2_mean_optimizer,trainer.wave3_mean_optimizer):
  for source_param,dest_param in zip(trainer.actor.mean.parameters(),optimizer.param_groups[0]["params"]):
   optimizer_clone &= all(torch.equal(trainer.actor_optimizer.state[source_param][key],optimizer.state[dest_param][key]) for key in ("step","exp_avg","exp_avg_sq"))
 if not optimizer_clone:raise RuntimeError("mean optimizer state clone failed")
 obs=torch.randn(6,4,52,device="cuda");alive=torch.ones(6,4,device="cuda");waves=torch.tensor([1,2,3,1,2,3],device="cuda")
 with torch.no_grad():
  plain=trainer.actor.distribution_step(obs,None,None,None,alive)[0];routed=trainer._wsmh_routed_distribution(obs,alive,waves);initial_identity=torch.equal(plain.loc,routed.loc) and torch.equal(plain.scale,routed.scale)
 if not initial_identity:raise RuntimeError("initial distribution differs from Plain")
 raw=torch.randn(6,4,3,device="cuda");actions=torch.tanh(raw);advantages=torch.randn(6,4,device="cuda")
 oldlog=trainer.actor._squashed_log_prob(plain,raw,actions).detach()
 plain_ratio=(trainer.actor._squashed_log_prob(plain,raw,actions)-oldlog).exp();routed_ratio=(trainer.actor._squashed_log_prob(routed,raw,actions)-oldlog).exp()
 plain_surrogate=torch.minimum(plain_ratio*advantages,plain_ratio.clamp(.8,1.2)*advantages);routed_surrogate=torch.minimum(routed_ratio*advantages,routed_ratio.clamp(.8,1.2)*advantages)
 plain_loss=-(plain_surrogate*alive).sum()/alive.sum();routed_loss=-(routed_surrogate*alive).sum()/alive.sum();natural_weighting_identity=bool(torch.equal(plain_loss,routed_loss))
 if not natural_weighting_identity:raise RuntimeError("global natural-weighting Actor loss differs from Plain")
 actual=build_modular_mappo_trainer(cfg,"cuda",256,1_805_280);actual.load(SOURCE,strict_protocol=False,restore_rng=False);actual.ppo_epochs=1;actual_metrics=actual.update(deepcopy(rollouts["wsmh"]))
 if not all(np.isfinite(v) for v in actual_metrics.values() if isinstance(v,(int,float))):raise RuntimeError("real rollout update non-finite")
 isolation={};mixed_metrics=None
 for wave in (1,2,3):
  current=build_modular_mappo_trainer(cfg,"cuda",256,1_805_280);current.load(SOURCE,strict_protocol=False,restore_rng=False);before,after,cb,ca,ob,oa,metrics=synthetic(current,wave);changed=[a!=b for a,b in zip(before,after)];expected=[True,True,wave==1,wave==2,wave==3]
  if changed!=expected or cb==ca:raise RuntimeError(f"wave {wave} routing/update failed: {changed}")
  isolation[str(wave)]={"shared_backbone":changed[0],"shared_log_std":changed[1],"mean_changes":changed[2:],"critic":cb!=ca,"optimizer_changes":[a!=b for a,b in zip(ob,oa)]};mixed_metrics=metrics
 mixed=build_modular_mappo_trainer(cfg,"cuda",256,1_805_280);mixed.load(SOURCE,strict_protocol=False,restore_rng=False);_,_,_,_,_,_,mixed_metrics=synthetic(mixed,2)
 if mixed_metrics["wsmh_global_actor_grad_norm_postclip"]>.50001:raise RuntimeError("global clipping failed")
 # Strict resume preserves deliberately-diverged heads and their Adam state.
 with torch.no_grad():mixed.wave2_mean.bias.add_(.25);mixed.wave3_mean.bias.sub_(.15)
 checkpoint=out/"wsmh_roundtrip.pt";mixed.save(checkpoint);before_resume=([digest(h.state_dict()) for h in mixed._wsmh_means()],[digest(o.state_dict()) for o in mixed._wsmh_mean_optimizers()])
 restored=build_modular_mappo_trainer(cfg,"cuda",256,1_805_280);restored.load(checkpoint,strict_protocol=True,restore_rng=False);after_resume=([digest(h.state_dict()) for h in restored._wsmh_means()],[digest(o.state_dict()) for o in restored._wsmh_mean_optimizers()]);resume_ok=before_resume==after_resume and len(set(after_resume[0]))==3
 if not resume_ok:raise RuntimeError("strict resume failed")
 restored.wave_specific_mean_heads.routing_trace_enabled=True;evaluation=None
 for seed in range(88_000_000,88_000_012):
  restored.wave_specific_mean_heads.routing_trace=[];candidate=evaluate_modular_episode(restored,env,seed,include_trace=True)
  if 2 in candidate["wave_trace"]:evaluation=candidate;break
 if evaluation is None:raise RuntimeError("no W1->W2 transition in smoke seeds")
 evaluation_routing=np.array_equal(np.asarray(restored.wave_specific_mean_heads.routing_trace),evaluation["wave_trace"])
 if not evaluation_routing:raise RuntimeError("evaluation routing mismatch")
 architecture=checkpoint_architecture(restored);lrs=[restored.actor_optimizer.param_groups[0]["lr"],restored.wave2_mean_optimizer.param_groups[0]["lr"],restored.wave3_mean_optimizer.param_groups[0]["lr"]]
 report={"status":"WSMH_CUDA_SMOKE_PASS","cuda":torch.cuda.get_device_name(0),"source_clone":clone,"mean_optimizer_clone":optimizer_clone,"initial_distribution_plain_identity":initial_identity,"first_physical_rollout_bitwise":physical,"real_rollout_update_finite":True,"wave_update_routing":isolation,"natural_weighting_identity":initial_identity,"no_sample_mean_noop":True,"global_preclip_norm":mixed_metrics["wsmh_global_actor_grad_norm_preclip"],"global_postclip_norm":mixed_metrics["wsmh_global_actor_grad_norm_postclip"],"global_clip_scale":mixed_metrics["wsmh_global_clip_scale"],"actor_lrs":lrs,"shared_critic":True,"strict_resume":resume_ok,"evaluation_routing":evaluation_routing,"evaluation_seed":seed,"architecture":architecture,"source_unchanged":source_sha==fsha(SOURCE),"uses_45m":False,"formal_training_started":False}
 (out/"smoke_report.json").write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=="__main__":main()
