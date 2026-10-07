"""Wave-specific actor isolation for the WSAI-MAPPO development screen."""
from __future__ import annotations

from copy import deepcopy
import numpy as np


WSAI_MAPPO_VERSION = 1
EXPECTED_WSAI_CONFIG = {
    "enabled": True,
    "total_waves": 3,
    "actor_routing": "environment_wave",
    "shared_critic": True,
    "initialization": "clone_source_actor",
    "optimizer_initialization": "clone_source_actor_optimizer",
    "natural_wave_weighting": True,
    "global_actor_grad_clip": True,
}


class WaveSpecificActorIsolationModule:
    """Protocol state and cumulative diagnostics for three isolated actors."""

    name = "wave_specific_actor_isolation"
    version = WSAI_MAPPO_VERSION

    def __init__(self, config=None):
        self.config = deepcopy(config or {"enabled": False})
        self.enabled = bool(self.config.get("enabled", False))
        if self.enabled and self.config != EXPECTED_WSAI_CONFIG:
            raise ValueError(f"WSAI V1 requires exact config: {EXPECTED_WSAI_CONFIG}")
        self.optimizer_steps = [0, 0, 0]
        self.routed_alive_samples = [0, 0, 0]
        self.routing_trace = []
        self.routing_trace_enabled = False

    def record_routing(self, waves):
        if self.routing_trace_enabled:
            self.routing_trace.extend(int(value) for value in waves)

    def record_minibatch(self, counts, stepped):
        for index in range(3):
            self.routed_alive_samples[index] += int(counts[index])
            self.optimizer_steps[index] += int(bool(stepped[index]))

    def diagnostics(self):
        out = {"wsai_enabled": float(self.enabled)}
        for index in range(3):
            wave = index + 1
            out[f"wsai_wave{wave}_optimizer_steps"] = float(self.optimizer_steps[index])
            out[f"wsai_wave{wave}_routed_alive_samples"] = float(self.routed_alive_samples[index])
        return out

    def state_dict(self):
        return {"version": self.version, "config": deepcopy(self.config),
                "optimizer_steps": list(self.optimizer_steps),
                "routed_alive_samples": list(self.routed_alive_samples)}

    def load_state_dict(self, state, branch_from_plain=False):
        if state is None:
            if branch_from_plain:
                self.optimizer_steps = [0, 0, 0]
                self.routed_alive_samples = [0, 0, 0]
                self.routing_trace = []
                self.routing_trace_enabled = False
                return
            raise RuntimeError("WSAI checkpoint state missing")
        if int(state.get("version", -1)) != self.version or state.get("config") != self.config:
            raise RuntimeError("WSAI checkpoint version/config mismatch")
        self.optimizer_steps = [int(value) for value in state.get("optimizer_steps", [0, 0, 0])]
        self.routed_alive_samples = [int(value) for value in state.get("routed_alive_samples", [0, 0, 0])]
        self.routing_trace = []
        self.routing_trace_enabled = False
        if len(self.optimizer_steps) != 3 or len(self.routed_alive_samples) != 3:
            raise RuntimeError("WSAI checkpoint counter shape mismatch")


def pairwise_actor_l2_distances(actors):
    vectors = [np.concatenate([parameter.detach().cpu().numpy().reshape(-1)
                               for parameter in actor.parameters()]) for actor in actors]
    result={}
    for left,right,label in ((0,1,"12"),(0,2,"13"),(1,2,"23")):
        distance=float(np.linalg.norm(vectors[left]-vectors[right]))
        denominator=max(float(np.linalg.norm(vectors[left])),float(np.linalg.norm(vectors[right])),1e-12)
        result[f"wsai_actor{label}_parameter_l2"]=distance
        result[f"wsai_actor{label}_relative_l2"]=distance/denominator
    return result


__all__ = ["WSAI_MAPPO_VERSION", "EXPECTED_WSAI_CONFIG",
           "WaveSpecificActorIsolationModule", "pairwise_actor_l2_distances"]
