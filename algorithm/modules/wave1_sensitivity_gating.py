"""Wave-1 PPO-gradient sensitivity gated plasticity for PWTR Actor replay."""
from __future__ import annotations
from copy import deepcopy
from typing import Iterable
import numpy as np
import torch

W1SG_MAPPO_VERSION=1
EXPECTED_CONFIG={"enabled":True,"source_wave":1,"importance_objective":"ppo_surrogate","importance_chunk_states":512,"gate_transform":"inverse_sqrt","preserve_global_gradient_norm":True,"epsilon":1e-12}

def normalize_importance(values,epsilon=1e-12):
 count=sum(x.numel() for x in values);mean=float(sum(x.double().sum() for x in values)/max(count,1)) if values else 0.
 return ([torch.zeros_like(x) for x in values] if mean<=epsilon else [x/(mean+epsilon) for x in values]),mean

def inverse_sqrt_gate(normalized):return torch.rsqrt(1.+normalized)

class Wave1SensitivityGatingModule:
 name="wave1_sensitivity_gating";version=W1SG_MAPPO_VERSION
 def __init__(self,config=None):
  self.config=deepcopy(config or {"enabled":False});self.enabled=bool(self.config.get("enabled",False))
  if self.enabled and self.config!=EXPECTED_CONFIG:raise ValueError(f"W1SG V1 requires exact config: {EXPECTED_CONFIG}")
  self.source_wave=int(self.config.get("source_wave",1));self.chunk_states=int(self.config.get("importance_chunk_states",512));self.epsilon=float(self.config.get("epsilon",1e-12))
  self.active_update_count=0;self.identity_fallback_count=0;self.replay_gradient_step_count=0;self._importance={};self._normalized={};self._gradient_rows=[];self._metrics=self.default_metrics()
 @staticmethod
 def parameters(actor):return [(n,p) for n,p in actor.named_parameters() if p.requires_grad]
 @staticmethod
 def family(name):
  if name.startswith("backbone."):return "backbone"
  if name.startswith("mean."):return "mean_head"
  if name.startswith("log_std."):return "log_std_head"
  raise RuntimeError(f"unrecognized W1SG Actor parameter family: {name}")
 def default_metrics(self):
  out={"w1sg_enabled":float(self.enabled),"w1sg_active":0.,"w1sg_w1_alive_samples":0.,"w1sg_importance_raw_mean":0.,"w1sg_importance_raw_max":0.,"w1sg_importance_normalized_mean":0.,"w1sg_importance_normalized_max":0.,"w1sg_importance_nonzero_fraction":0.,"w1sg_gate_mean":1.,"w1sg_gate_min":1.,"w1sg_gate_p10":1.,"w1sg_gate_p50":1.,"w1sg_gate_p90":1.,"w1sg_actor_grad_norm_raw":0.,"w1sg_actor_grad_norm_after_gate_before_renorm":0.,"w1sg_actor_grad_norm_after_renorm":0.,"w1sg_norm_preservation_relative_error":0.,"w1sg_raw_protected_gradient_cosine":1.,"w1sg_renorm_scale":1.,"w1sg_active_update_count":float(self.active_update_count),"w1sg_identity_fallback_count":float(self.identity_fallback_count),"w1sg_replay_gradient_step_count":float(self.replay_gradient_step_count)}
  for f in ("backbone","mean_head","log_std_head"):out[f"w1sg_{f}_importance_mean"]=0.;out[f"w1sg_{f}_importance_share"]=0.
  return out
 def compute_sensitivity(self,actor,observations,raw_actions,old_log_probs,alive_masks,wave_indices,advantages,clip_ratio):
  self._importance={};self._normalized={};self._gradient_rows=[];self._metrics=self.default_metrics()
  if not self.enabled:return self._metrics
  flat=lambda x:x.reshape(x.shape[0]*x.shape[1],*x.shape[2:]);obs,raw,oldlog,alive,adv=map(flat,(observations,raw_actions,old_log_probs,alive_masks,advantages));waves=wave_indices.reshape(-1)
  indices=torch.nonzero(waves==self.source_wave,as_tuple=False).flatten();self._metrics["w1sg_w1_alive_samples"]=float(alive[indices].sum()) if indices.numel() else 0.
  params=self.parameters(actor);accum=[torch.zeros_like(p) for _,p in params];total=0.
  for start in range(0,indices.numel(),self.chunk_states):
   ix=indices[start:start+self.chunk_states];mask=alive[ix];n=float(mask.sum())
   if n<=0:continue
   dist,_=actor.distribution_step(obs[ix],None,None,None,mask);newlog=actor._squashed_log_prob(dist,raw[ix],torch.tanh(raw[ix]));ratio=(newlog-oldlog[ix]).exp();sur=torch.minimum(ratio*adv[ix],ratio.clamp(1-clip_ratio,1+clip_ratio)*adv[ix]);loss=-(sur*mask).sum()/mask.sum().clamp_min(1)
   grads=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
   for i,g in enumerate(grads):
    if g is not None:accum[i].add_(g.detach().square(),alpha=n)
   total+=n
  if total<=0:return self._fallback()
  importance=[x/total for x in accum];normalized,raw_mean=normalize_importance(importance,self.epsilon)
  if raw_mean<=self.epsilon:return self._fallback()
  self._importance={n:x for (n,_),x in zip(params,importance)};self._normalized={n:x for (n,_),x in zip(params,normalized)};raw_all=torch.cat([x.reshape(-1) for x in importance]);norm_all=torch.cat([x.reshape(-1) for x in normalized]);gates=inverse_sqrt_gate(norm_all);self.active_update_count+=1
  self._metrics.update({"w1sg_active":1.,"w1sg_importance_raw_mean":raw_mean,"w1sg_importance_raw_max":float(raw_all.max()),"w1sg_importance_normalized_mean":float(norm_all.double().mean()),"w1sg_importance_normalized_max":float(norm_all.max()),"w1sg_importance_nonzero_fraction":float((raw_all>0).float().mean()),"w1sg_gate_mean":float(gates.mean()),"w1sg_gate_min":float(gates.min()),"w1sg_gate_p10":float(torch.quantile(gates,.1)),"w1sg_gate_p50":float(torch.quantile(gates,.5)),"w1sg_gate_p90":float(torch.quantile(gates,.9)),"w1sg_active_update_count":float(self.active_update_count)})
  isum=float(raw_all.double().sum())
  for f in ("backbone","mean_head","log_std_head"):
   joined=torch.cat([x.reshape(-1) for (n,_),x in zip(params,importance) if self.family(n)==f]);self._metrics[f"w1sg_{f}_importance_mean"]=float(joined.double().mean());self._metrics[f"w1sg_{f}_importance_share"]=float(joined.double().sum()/(isum+self.epsilon))
  return dict(self._metrics)
 def _fallback(self):
  self.identity_fallback_count+=1;self._metrics["w1sg_identity_fallback_count"]=float(self.identity_fallback_count);return dict(self._metrics)
 def gate_actor_gradients(self,named_parameters:Iterable):
  params=[(n,p) for n,p in named_parameters if p.requires_grad and p.grad is not None]
  if not self.enabled or not self._normalized or not params:self._gradient_rows.append({"raw":0.,"gated":0.,"renorm":0.,"error":0.,"cosine":1.,"scale":1.});return self.gradient_metrics()
  raw={n:p.grad.detach().clone() for n,p in params};raw_norm=float(torch.sqrt(sum(x.double().square().sum() for x in raw.values())))
  for n,p in params:
   if n not in self._normalized:raise RuntimeError(f"W1SG parameter alignment missing: {n}")
   p.grad.mul_(inverse_sqrt_gate(self._normalized[n]))
  gated=float(torch.sqrt(sum(p.grad.detach().double().square().sum() for _,p in params)));scale=raw_norm/gated if raw_norm>self.epsilon and gated>self.epsilon else 1.
  if scale!=1.:
   for _,p in params:p.grad.mul_(scale)
  renorm=float(torch.sqrt(sum(p.grad.detach().double().square().sum() for _,p in params)));error=abs(renorm-raw_norm)/max(raw_norm,self.epsilon);dot=sum((raw[n].double()*p.grad.detach().double()).sum() for n,p in params);cos=float(dot/(max(raw_norm,self.epsilon)*max(renorm,self.epsilon))) if raw_norm>self.epsilon and renorm>self.epsilon else 1.
  if error>1e-5:raise FloatingPointError(f"W1SG norm preservation failed: {error}")
  self.replay_gradient_step_count+=1;self._gradient_rows.append({"raw":raw_norm,"gated":gated,"renorm":renorm,"error":error,"cosine":cos,"scale":scale});return self.gradient_metrics()
 def gradient_metrics(self):
  out=dict(self._metrics)
  if self._gradient_rows:
   avg=lambda k:float(np.mean([r[k] for r in self._gradient_rows]));out.update({"w1sg_actor_grad_norm_raw":avg("raw"),"w1sg_actor_grad_norm_after_gate_before_renorm":avg("gated"),"w1sg_actor_grad_norm_after_renorm":avg("renorm"),"w1sg_norm_preservation_relative_error":max(r["error"] for r in self._gradient_rows),"w1sg_raw_protected_gradient_cosine":avg("cosine"),"w1sg_renorm_scale":avg("scale")})
  out.update({"w1sg_active_update_count":float(self.active_update_count),"w1sg_identity_fallback_count":float(self.identity_fallback_count),"w1sg_replay_gradient_step_count":float(self.replay_gradient_step_count)});return out
 def state_dict(self):return {"version":self.version,"config":deepcopy(self.config),"active_update_count":self.active_update_count,"identity_fallback_count":self.identity_fallback_count,"replay_gradient_step_count":self.replay_gradient_step_count}
 def load_state_dict(self,state,branch_from_plain=False):
  self._importance={};self._normalized={};self._gradient_rows=[]
  if state is None:
   if branch_from_plain:return
   raise RuntimeError("W1SG checkpoint state missing")
  if int(state.get("version",-1))!=self.version or state.get("config")!=self.config:raise RuntimeError("W1SG checkpoint version/config mismatch")
  for k in ("active_update_count","identity_fallback_count","replay_gradient_step_count"):setattr(self,k,int(state.get(k,0)))
  self._metrics=self.default_metrics()

__all__=["W1SG_MAPPO_VERSION","Wave1SensitivityGatingModule","normalize_importance","inverse_sqrt_gate"]
