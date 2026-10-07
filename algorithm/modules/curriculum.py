"""Fixed-step 1→2→3 wave curriculum without mutating source config."""
from __future__ import annotations

from copy import deepcopy
from typing import Any
from .base import CapabilityModule


class CurriculumController(CapabilityModule):
    name="curriculum"
    def __init__(self,config:dict[str,Any]|None=None)->None:
        super().__init__(config); self.stage_mode=str(self.config.get("stage_mode","fixed_steps"))
        if self.stage_mode!="fixed_steps": raise ValueError("only fixed_steps curriculum is supported")
        self.stage1_end=int(self.config.get("stage1_end",500_000)); self.stage2_end=int(self.config.get("stage2_end",1_000_000))
        if not 0<=self.stage1_end<=self.stage2_end: raise ValueError("invalid curriculum boundaries")
    def stage(self,sampled_steps:int)->tuple[int,int]:
        # Disabled means a true no-op.  In particular, do not smuggle a
        # three-wave default into callers that supplied an L1/L2 environment.
        if not self.enabled:return 0,0
        if sampled_steps<self.stage1_end:return 1,1
        if sampled_steps<self.stage2_end:return 2,2
        return 3,3
    def runtime_config(self,environment_config:dict[str,Any],sampled_steps:int)->dict[str,Any]:
        config=deepcopy(environment_config)
        if not self.enabled:return config
        _,waves=self.stage(sampled_steps)
        if "persistent_waves" in config:config["persistent_waves"]["total_waves"]=waves
        return config

__all__=["CurriculumController"]
