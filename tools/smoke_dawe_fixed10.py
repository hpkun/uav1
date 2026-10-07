#!/usr/bin/env python3
"""CUDA-only synthetic correctness smoke for DAWE V1; never trains an environment."""
from __future__ import annotations
import argparse,hashlib,json,sys
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.trainer import stable_ratio_terms
from algorithm.modules import DeploymentAlignedWaveExplorationModule

SOURCE=ROOT/"outputs/diag_mappo_learnability/l3_seed5302/checkpoint_1505280.pt"
OUT=ROOT/"outputs/dawe_fixed10_cuda_smoke"
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
 parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,default=OUT);args=parser.parse_args();output=args.output_dir
 if not output.is_absolute():output=ROOT/output
 if output.exists():raise FileExistsError(output)
 if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
 control_cfg=load_config(ROOT/"configs/dev_dawe_fixed10_control_300k.yaml")
 dawe_cfg=load_config(ROOT/"configs/dev_dawe_fixed10_v1_300k.yaml")
 control=build_modular_mappo_trainer(control_cfg,"cuda",256,1_805_280)
 dawe=build_modular_mappo_trainer(dawe_cfg,"cuda",256,1_805_280)
 control.load(SOURCE,strict_protocol=False,restore_rng=False);dawe.load(SOURCE,strict_protocol=False,restore_rng=False)
 initial={"actor":digest(control.actor.state_dict())==digest(dawe.actor.state_dict()),
  "critic":digest(control.critic.state_dict())==digest(dawe.critic.state_dict()),
  "actor_optimizer":digest(control.actor_optimizer.state_dict())==digest(dawe.actor_optimizer.state_dict()),
  "critic_optimizer":digest(control.critic_optimizer.state_dict())==digest(dawe.critic_optimizer.state_dict())}
 obs=np.random.default_rng(7).normal(size=(3,4,52)).astype("f");alive=np.ones((3,4),"f");waves=np.asarray([1,2,3])
 det_c=control.act(obs,alive,True,True,wave_indices=waves)[0]
 det_d=dawe.act(obs,alive,True,True,wave_indices=waves)[0]
 rng=torch.cuda.get_rng_state().clone();torch.cuda.set_rng_state(rng.clone())
 c=control.act(obs,alive,False,True,wave_indices=waves)
 torch.cuda.set_rng_state(rng.clone());d=dawe.act(obs,alive,False,True,wave_indices=waves)
 raw_c=torch.as_tensor(c[1],device="cuda");raw_d=torch.as_tensor(d[1],device="cuda")
 with torch.no_grad():
  base,_=dawe.actor.distribution_step(torch.as_tensor(obs,device="cuda"),None,None,None,torch.as_tensor(alive,device="cuda"))
  residual_ratio=(raw_d-base.loc)/(raw_c-base.loc)
  identity={}
  for wave in (1,2,3):
   b=torch.distributions.Normal(base.loc[wave-1:wave],base.scale[wave-1:wave])
   eff=dawe._effective_actor_distribution(b,torch.tensor([wave],device="cuda"));raw=eff.rsample();act=torch.tanh(raw)
   old=dawe.actor._squashed_log_prob(eff,raw,act);recomputed=dawe.actor._squashed_log_prob(dawe._effective_actor_distribution(b,torch.tensor([wave],device="cuda")),raw,act)
   _,ratio=stable_ratio_terms(recomputed,old);identity[str(wave)]={"max_logprob_error":float((recomputed-old).abs().max()),"max_ratio_error":float((ratio-1).abs().max())}
 scaling={"W1":float((residual_ratio[0]-0.25).abs().max()),"W2":float((residual_ratio[1]-0.25).abs().max()),
          "W3":float((raw_d[2]-raw_c[2]).abs().max())}
 T,E,A=3,1,4;batch_obs=obs[:,None];batch_alive=np.ones((T,E,A),"f");batch_waves=np.asarray([[1],[2],[3]])
 with torch.no_grad():
  actions,raw_actions,logs,_=dawe.act(obs,alive,False,True,wave_indices=waves)
 rollout=ModularRolloutBatch(batch_obs,actions[:,None],raw_actions[:,None],logs[:,None],np.ones((T,E,A),"f")*.01,np.ones((T,E,A),"f")*.01,np.zeros((T,E),"f"),batch_alive,batch_obs.copy(),batch_alive.copy(),batch_waves,np.full((T,E),3),np.zeros((T,E,0),"f"),np.zeros((T,E,0),"f"),episode_masks=np.ones((T,E),"f"))
 state_before=torch.cuda.get_rng_state().clone()
 _=dawe.deployment_aligned_wave_exploration.effective_distribution(base,torch.tensor([1,2,3],device="cuda"))
 _=dawe._dawe_diagnostics(rollout,torch.as_tensor(batch_obs,device="cuda"),torch.as_tensor(raw_actions[:,None],device="cuda"),torch.as_tensor(batch_alive,device="cuda"),torch.zeros((T,E,0),device="cuda"),torch.as_tensor(batch_waves,device="cuda"))
 state_after=torch.cuda.get_rng_state().clone()
 dawe.ppo_epochs=1;dawe.minibatch_size=3;metrics=dawe.update(rollout)
 ratios=[metrics[f"dawe_wave{wave}_effective_to_base_std_ratio"] for wave in (1,2,3)]
 report={"status":"DAWE_CUDA_SMOKE_PASS","device":torch.cuda.get_device_name(0),"source":str(SOURCE.relative_to(ROOT)),
  "initial_state_parity":initial,"deterministic_max_abs_error":float(np.max(np.abs(det_c-det_d))),
  "same_rng_scaling_max_errors":scaling,"logprob_ratio_identity":identity,
  "distribution_and_diagnostics_rng_unchanged":bool(torch.equal(state_before,state_after)),
  "synthetic_ppo_finite":all(np.isfinite(value) for value in metrics.values() if isinstance(value,(int,float))),
  "dawe_diagnostic_ratios":ratios,"environment_steps":0,"formal_training":False,"uses_45m":False}
 checks=list(initial.values())+[report["deterministic_max_abs_error"]==0,scaling["W1"]<2e-5,scaling["W2"]<2e-5,scaling["W3"]<2e-5,
  all(item["max_ratio_error"]<1e-6 for item in identity.values()),report["distribution_and_diagnostics_rng_unchanged"],report["synthetic_ppo_finite"],
  np.allclose(ratios,[.25,.25,1.0],atol=1e-7)]
 if not all(checks):raise RuntimeError(f"DAWE smoke check failed: {report}")
 output.mkdir(parents=True);(output/"smoke_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
 print(json.dumps(report,indent=2))
if __name__=="__main__":main()
