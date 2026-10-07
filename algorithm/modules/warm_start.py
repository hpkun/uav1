"""Explicit exact/partial actor and critic warm-start loading."""
from __future__ import annotations

from typing import Any
import hashlib
import torch
from algorithm.common.protocol import config_sha256
from .base import CapabilityModule


class WarmStartInitializer(CapabilityModule):
    name="warm_start"
    def __init__(self,config:dict[str,Any]|None=None)->None:
        super().__init__(config); self.mode=str(self.config.get("mode","none"))
        if self.mode not in {"none","actor_only","actor_critic"}:raise ValueError("invalid warm-start mode")
    @staticmethod
    def _copy(target:torch.nn.Module,source:dict[str,torch.Tensor],actor:bool)->dict[str,list[str]]:
        target_state=target.state_dict(); loaded=[]; partial=[]; missing=[]
        aliases={}
        if actor:
            aliases={"encoder.0.weight":"backbone.0.weight","encoder.0.bias":"backbone.0.bias","encoder.2.weight":"backbone.2.weight","encoder.2.bias":"backbone.2.bias"}
        for key,value in target_state.items():
            source_key=key if key in source else aliases.get(key)
            if source_key is None or source_key not in source:missing.append(key);continue
            incoming=source[source_key]
            if incoming.shape==value.shape:value.copy_(incoming);loaded.append(key)
            elif incoming.ndim==value.ndim==2 and incoming.shape[0]==value.shape[0] and incoming.shape[1]<=value.shape[1]:
                value.zero_();value[:,:incoming.shape[1]].copy_(incoming);partial.append(key)
            else:missing.append(key)
        target.load_state_dict(target_state);return {"loaded":loaded,"partially_loaded":partial,"not_loaded":missing}
    def initialize(self,trainer,checkpoint:str)->dict[str,Any]:
        state=torch.load(checkpoint,map_location="cpu",weights_only=False);extra=state.get("extra",{})
        digest=hashlib.sha256(open(checkpoint,"rb").read()).hexdigest()
        result={"source_checkpoint":str(checkpoint),"source_algorithm":state.get("algorithm"),"source_environment_variant":extra.get("environment_variant"),"source_training_seed":extra.get("training_seed"),"source_sampled_steps":int(state.get("sampled_steps",0)),"pretraining_sampled_steps":int(state.get("sampled_steps",0)),"source_config_fingerprint":extra.get("algorithm_config_sha256"),"source_checkpoint_sha256":digest,"mode":self.mode}
        if self.mode=="none":return {**result,"actor":{"loaded":[],"partially_loaded":[],"not_loaded":[]}}
        result["actor"]=self._copy(trainer.actor,state["actor"],True)
        if self.mode=="actor_critic":result["critic"]=self._copy(trainer.critic,state["critic"],False)
        for key in ("actor","critic"):
            if key in result:
                result[key]["exact_loaded_count"]=len(result[key]["loaded"]);result[key]["partial_loaded_count"]=len(result[key]["partially_loaded"]);result[key]["not_loaded_count"]=len(result[key]["not_loaded"])
        return result

__all__=["WarmStartInitializer"]
