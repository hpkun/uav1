"""Protocol state for Wave-Specific Mean Heads MAPPO (WSMH-MAPPO)."""
from __future__ import annotations
from copy import deepcopy
import numpy as np

WSMH_MAPPO_VERSION = 1
EXPECTED_WSMH_CONFIG = {"enabled":True,"total_waves":3,"routing":"environment_wave",
 "shared_backbone":True,"shared_log_std":True,"wave_specific_mean":True,
 "mean_initialization":"clone_source_mean","mean_optimizer_initialization":"clone_source_mean_optimizer_state",
 "natural_wave_weighting":True,"global_actor_grad_clip":True}

class WaveSpecificMeanHeadsModule:
 name="wave_specific_mean_heads";version=WSMH_MAPPO_VERSION
 def __init__(self,config=None):
  self.config=deepcopy(config or {"enabled":False});self.enabled=bool(self.config.get("enabled",False))
  if self.enabled and self.config!=EXPECTED_WSMH_CONFIG:raise ValueError(f"WSMH V1 requires exact config: {EXPECTED_WSMH_CONFIG}")
  self.mean_optimizer_steps=[0,0,0];self.shared_actor_optimizer_steps=0;self.routed_alive_samples=[0,0,0]
  self.routing_trace=[];self.routing_trace_enabled=False
 def record_routing(self,waves):
  if self.routing_trace_enabled:self.routing_trace.extend(int(v) for v in waves)
 def record_minibatch(self,counts,stepped):
  self.shared_actor_optimizer_steps+=1
  for i in range(3):self.routed_alive_samples[i]+=int(counts[i]);self.mean_optimizer_steps[i]+=int(bool(stepped[i]))
 def diagnostics(self):
  out={"wsmh_enabled":float(self.enabled),"wsmh_shared_actor_optimizer_steps":float(self.shared_actor_optimizer_steps)}
  for i in range(3):
   out[f"wsmh_wave{i+1}_mean_optimizer_steps"]=float(self.mean_optimizer_steps[i]);out[f"wsmh_wave{i+1}_routed_alive_samples"]=float(self.routed_alive_samples[i])
  return out
 def state_dict(self):
  return {"version":self.version,"config":deepcopy(self.config),"mean_optimizer_steps":list(self.mean_optimizer_steps),
   "shared_actor_optimizer_steps":self.shared_actor_optimizer_steps,"routed_alive_samples":list(self.routed_alive_samples)}
 def load_state_dict(self,state,branch_from_plain=False):
  if state is None:
   if not branch_from_plain:raise RuntimeError("WSMH checkpoint state missing")
   self.mean_optimizer_steps=[0,0,0];self.shared_actor_optimizer_steps=0;self.routed_alive_samples=[0,0,0]
  else:
   if int(state.get("version",-1))!=self.version or state.get("config")!=self.config:raise RuntimeError("WSMH checkpoint version/config mismatch")
   self.mean_optimizer_steps=[int(v) for v in state.get("mean_optimizer_steps",[])];self.shared_actor_optimizer_steps=int(state.get("shared_actor_optimizer_steps",0));self.routed_alive_samples=[int(v) for v in state.get("routed_alive_samples",[])]
   if len(self.mean_optimizer_steps)!=3 or len(self.routed_alive_samples)!=3:raise RuntimeError("WSMH checkpoint counter shape mismatch")
  self.routing_trace=[];self.routing_trace_enabled=False

def pairwise_mean_l2_distances(heads):
 vectors=[np.concatenate([p.detach().cpu().numpy().reshape(-1) for p in head.parameters()]) for head in heads];out={}
 for left,right,label in ((0,1,"12"),(0,2,"13"),(1,2,"23")):
  distance=float(np.linalg.norm(vectors[left]-vectors[right]));denominator=max(float(np.linalg.norm(vectors[left])),float(np.linalg.norm(vectors[right])),1e-12)
  out[f"wsmh_mean{label}_parameter_l2"]=distance;out[f"wsmh_mean{label}_relative_l2"]=distance/denominator
 return out

__all__=["WSMH_MAPPO_VERSION","EXPECTED_WSMH_CONFIG","WaveSpecificMeanHeadsModule","pairwise_mean_l2_distances"]
