"""Fixed actor-only gradient clipping intervention for AOGC-MAPPO."""
from __future__ import annotations
from copy import deepcopy

ACTOR_GRAD_CLIP_VERSION=1
ALLOWED_ACTOR_LIMITS=(.5,1.0)

class ActorGradientClippingModule:
 name="actor_gradient_clipping";version=ACTOR_GRAD_CLIP_VERSION
 def __init__(self,config=None):
  self.config=deepcopy(config or {"enabled":False});self.enabled=bool(self.config.get("enabled",False))
  self.actor_optimizer_steps=0;self.actor_clipped_minibatches=0
  if self.enabled:
   required={"enabled":True,"mode":"actor_only_fixed_norm","actor_max_grad_norm":self.config.get("actor_max_grad_norm"),"critic_max_grad_norm_unchanged":True,"critic_max_grad_norm":.5}
   if self.config!=required or float(self.config["actor_max_grad_norm"]) not in ALLOWED_ACTOR_LIMITS:raise ValueError("Actor gradient clipping V1 accepts only fixed actor limits 0.5 or 1.0 with critic limit 0.5")
   self.mode="actor_only_fixed_norm";self.actor_max_grad_norm=float(self.config["actor_max_grad_norm"]);self.critic_max_grad_norm=.5
  else:self.mode="disabled";self.actor_max_grad_norm=None;self.critic_max_grad_norm=.5
 def record_step(self,preclip):
  self.actor_optimizer_steps+=1;self.actor_clipped_minibatches+=int(float(preclip)>self.actor_max_grad_norm)
 def state_dict(self):return {"version":self.version,"config":deepcopy(self.config),"actor_optimizer_steps":self.actor_optimizer_steps,"actor_clipped_minibatches":self.actor_clipped_minibatches}
 def load_state_dict(self,state,branch_from_plain=False):
  if state is None:
   if not branch_from_plain:raise RuntimeError("actor gradient clipping checkpoint state missing")
   self.actor_optimizer_steps=0;self.actor_clipped_minibatches=0;return
  if int(state.get("version",-1))!=self.version or state.get("config")!=self.config:raise RuntimeError("actor gradient clipping checkpoint version/config mismatch")
  self.actor_optimizer_steps=int(state.get("actor_optimizer_steps",0));self.actor_clipped_minibatches=int(state.get("actor_clipped_minibatches",0))

__all__=["ACTOR_GRAD_CLIP_VERSION","ALLOWED_ACTOR_LIMITS","ActorGradientClippingModule"]
