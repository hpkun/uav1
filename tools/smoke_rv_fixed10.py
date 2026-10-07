#!/usr/bin/env python3
"""CUDA-only RV V1 branch/parity/update smoke; performs zero environment steps."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer

SOURCE=ROOT/"outputs/diag_mappo_learnability/l3_seed5302/checkpoint_1505280.pt"
DEFAULT_OUTPUT=ROOT/"outputs/rv_fixed10_cuda_smoke"

def digest(value):
 h=hashlib.sha256()
 def add(item):
  if torch.is_tensor(item):h.update(item.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(item,dict):
   for key in sorted(item,key=str):h.update(str(key).encode());add(item[key])
  elif isinstance(item,(list,tuple)):
   for child in item:add(child)
  else:h.update(repr(item).encode())
 add(value);return h.hexdigest()

def main():
 parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT);args=parser.parse_args()
 output=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
 if output.exists():raise FileExistsError(output)
 if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 cc=load_config(ROOT/"configs/dev_rv_fixed10_control_300k.yaml");rc=load_config(ROOT/"configs/dev_rv_mappo_v1_300k.yaml")
 control=build_modular_mappo_trainer(cc,"cuda",256,1_805_280);rv=build_modular_mappo_trainer(rc,"cuda",256,1_805_280)
 control.load(SOURCE,strict_protocol=False,restore_rng=False);rv.load(SOURCE,strict_protocol=False,restore_rng=False)
 initial={"actor":digest(control.actor.state_dict())==digest(rv.actor.state_dict()),"critic":digest(control.critic.state_dict())==digest(rv.critic.state_dict()),"actor_optimizer":digest(control.actor_optimizer.state_dict())==digest(rv.actor_optimizer.state_dict()),"critic_optimizer":digest(control.critic_optimizer.state_dict())==digest(rv.critic_optimizer.state_dict())}
 obs=np.random.default_rng(7).normal(size=(3,4,52)).astype("f");alive=np.ones((3,4),"f");waves=np.asarray([1,2,3])
 before_rng=torch.cuda.get_rng_state().clone();torch.cuda.set_rng_state(before_rng.clone());c=control.act(obs,alive,False,True,wave_indices=waves);c_end=torch.cuda.get_rng_state().clone();torch.cuda.set_rng_state(before_rng.clone());r=rv.act(obs,alive,False,True,wave_indices=waves);r_end=torch.cuda.get_rng_state().clone()
 stochastic=[float(np.max(np.abs(c[i]-r[i]))) for i in range(3)];det=float(np.max(np.abs(control.act(obs,alive,True,True,wave_indices=waves)[0]-rv.act(obs,alive,True,True,wave_indices=waves)[0])))
 ot=torch.as_tensor(obs,device="cuda");at=torch.as_tensor(alive,device="cuda");wt=torch.as_tensor(waves,device="cuda")
 base,_=rv.actor.distribution_step(ot,None,None,None,at);behavior=rv._behavior_actor_distribution(base,ot,at,wt);reference,_=rv.reference_variance_actor.distribution_step(ot,None,None,None,at)
 raw=behavior.rsample();action=torch.tanh(raw);old=rv.actor._squashed_log_prob(behavior,raw,action);ones=torch.ones(3,4,device="cuda");zeros=torch.zeros(3,4,device="cuda")
 losses=rv._loss_step(ot,action,raw,old,at,ones,zeros,zeros,ones,torch.zeros(3,0,device="cuda"),wave_indices=wt)
 ratio_error=float((losses[4]-1).abs().max().detach());reference_hash=rv.reference_variance_actor_sha256();logstd_before=[p.detach().clone() for p in rv.actor.log_std.parameters()];actor_before={n:p.detach().clone() for n,p in rv.actor.named_parameters() if p.requires_grad}
 rv.actor_optimizer.zero_grad();(losses[0]-.01*losses[2]).backward();logstd_grad_none=all(p.grad is None for p in rv.actor.log_std.parameters());reference_grad_none=all(p.grad is None for p in rv.reference_variance_actor.parameters());rv.actor_optimizer.step()
 trainable_changed=any(not torch.equal(actor_before[n],p) for n,p in rv.actor.named_parameters() if p.requires_grad)
 report={"status":"RV_FIXED10_CUDA_SMOKE_PASS","device":torch.cuda.get_device_name(0),"initial_state_parity":initial,"same_rng_raw_action_max_abs_error":stochastic[1],"same_rng_bounded_action_max_abs_error":stochastic[0],"same_rng_old_log_prob_max_abs_error":stochastic[2],"same_rng_final_state_equal":bool(torch.equal(c_end,r_end)),"deterministic_max_abs_error":det,"ratio_identity_max_abs_error":ratio_error,"reference_sigma_behavior_max_abs_error":float((behavior.scale-reference.scale).abs().max()),"reference_hash_unchanged":reference_hash==rv.reference_variance_actor_sha256(),"current_log_std_grad_none":logstd_grad_none,"current_log_std_weight_unchanged":all(torch.equal(a,b) for a,b in zip(logstd_before,rv.actor.log_std.parameters())),"reference_all_grad_none":reference_grad_none,"backbone_or_mean_changed":trainable_changed,"actor_optimizer_membership_preserved":all(any(p is q for g in rv.actor_optimizer.param_groups for q in g["params"]) for p in rv.actor.log_std.parameters()),"environment_steps":0,"formal_training":False,"uses_44m":False,"uses_45m":False}
 checks=list(initial.values())+[max(stochastic)<1e-6,report["same_rng_final_state_equal"],det==0.,ratio_error<1e-6,report["reference_sigma_behavior_max_abs_error"]==0.,report["reference_hash_unchanged"],logstd_grad_none,report["current_log_std_weight_unchanged"],reference_grad_none,trainable_changed,report["actor_optimizer_membership_preserved"]]
 if not all(checks):raise RuntimeError(report)
 output.mkdir(parents=True);(output/"smoke_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
