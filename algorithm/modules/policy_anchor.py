"""Closed-form pre-tanh Gaussian policy anchor."""
from __future__ import annotations

from typing import Any
import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from .base import CapabilityModule


class PolicyAnchorRegularizer(CapabilityModule):
    name="policy_anchor"
    def __init__(self,config:dict[str,Any]|None=None)->None:
        super().__init__(config); self.coefficient=float(self.config.get("coefficient",0.01)); self.schedule=str(self.config.get("schedule","constant")); self.decay_steps=int(self.config.get("decay_steps",1_000_000)); self.reference_actor:nn.Module|None=None; self.reference_checkpoint:str|None=None
        if self.schedule not in {"constant","linear_decay"}:raise ValueError("invalid anchor schedule")
    def attach(self,actor:nn.Module,checkpoint:str|None=None)->None:
        self.reference_actor=actor.eval(); self.reference_checkpoint=checkpoint
        for parameter in self.reference_actor.parameters():parameter.requires_grad_(False)
    def effective_coefficient(self,sampled_steps:int)->float:
        if not self.enabled:return 0.0
        return self.coefficient if self.schedule=="constant" else self.coefficient*max(0.0,1.0-sampled_steps/max(self.decay_steps,1))
    def loss(self,current:Normal,reference:Normal,sampled_steps:int,mask:torch.Tensor)->tuple[torch.Tensor,dict[str,float]]:
        # tanh is a deterministic bijection on the open interval, so KL is
        # invariant and can be evaluated exactly in pre-tanh Normal space.
        per_agent=kl_divergence(current,reference).sum(-1); mean=(per_agent*mask).sum()/mask.sum().clamp_min(1.0); loss=mean*self.effective_coefficient(sampled_steps)
        return loss,{"anchor_kl":float(mean.detach()),"anchor_loss":float(loss.detach())}

__all__=["PolicyAnchorRegularizer"]
