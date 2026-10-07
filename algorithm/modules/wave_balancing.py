"""Inverse-frequency wave weights for policy/value losses."""
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from .base import CapabilityModule


class WaveBalancingModule(CapabilityModule):
    name = "wave_balancing"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.target = str(self.config.get("loss_target", "actor_critic"))
        if self.target not in {"actor_only", "critic_only", "actor_critic"}:
            raise ValueError(f"invalid wave-balance target: {self.target}")
        self.frequency_basis = str(self.config.get("frequency_basis", "alive_agent"))
        if self.frequency_basis not in {"transition", "alive_agent"}:
            raise ValueError("frequency_basis must be transition or alive_agent")
        self.max_weight = float(self.config.get("max_weight", 3.0)); self.epsilon = float(self.config.get("epsilon", 1e-6))
        if self.max_weight < 1.0: raise ValueError("max_weight must be >= 1")

    def compute_numpy(self, wave_indices: np.ndarray, alive_masks: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, float]]:
        wave = np.asarray(wave_indices, dtype=np.int64)
        transition_counts=np.asarray([(wave==k).sum() for k in (1,2,3)],dtype=np.float64)
        transition_total=max(float(transition_counts.sum()),1.0);transition_fractions=transition_counts/transition_total
        if alive_masks is None:
            alive=np.ones((*wave.shape,1),dtype=np.float64)
        else:
            alive=np.asarray(alive_masks,dtype=np.float64)
            if alive.shape[:-1]!=wave.shape:raise ValueError("alive_masks must be wave shape + [agents]")
        alive_counts=np.asarray([alive[wave==k].sum() for k in (1,2,3)],dtype=np.float64)
        alive_total=max(float(alive_counts.sum()),1.0);alive_fractions=alive_counts/alive_total
        counts=alive_counts if self.frequency_basis=="alive_agent" else transition_counts
        total=max(float(counts.sum()),1.0);fractions=counts/total
        if not self.enabled or np.count_nonzero(counts) <= 1:
            weights=np.ones_like(wave,dtype=np.float32); per=np.ones(3,dtype=float)
        else:
            raw=np.zeros(3,dtype=float);present=counts>0;raw[present]=1.0/np.maximum(fractions[present],self.epsilon)
            # Solve mean_c[min(scale*r_k,max_weight)] == 1.  This retains a
            # hard cap and the baseline loss scale simultaneously.
            lo,hi=0.0,1.0
            mean=lambda scale:float(np.sum(counts*np.minimum(scale*raw,self.max_weight))/total)
            while mean(hi)<1.0:hi*=2.0
            for _ in range(80):
                mid=(lo+hi)/2.0
                if mean(mid)<1.0:lo=mid
                else:hi=mid
            per=np.minimum(hi*raw,self.max_weight)
            weights=np.asarray([per[int(k)-1] if 1<=int(k)<=3 else 1.0 for k in wave.reshape(-1)],dtype=np.float32).reshape(wave.shape)
        effective=float(np.sum(alive_counts*per)/alive_total) if self.frequency_basis=="alive_agent" else float(np.sum(transition_counts*per)/transition_total)
        metrics={
            **{f"transition_samples_wave_{k}":float(transition_counts[k-1]) for k in (1,2,3)},
            **{f"transition_fraction_wave_{k}":float(transition_fractions[k-1]) for k in (1,2,3)},
            **{f"alive_agent_samples_wave_{k}":float(alive_counts[k-1]) for k in (1,2,3)},
            **{f"alive_agent_fraction_wave_{k}":float(alive_fractions[k-1]) for k in (1,2,3)},
            **{f"samples_wave_{k}":float(counts[k-1]) for k in (1,2,3)},
            **{f"fraction_wave_{k}":float(fractions[k-1]) for k in (1,2,3)},
            **{f"weight_wave_{k}":float(per[k-1] if counts[k-1]>0 else 0.0) for k in (1,2,3)},
            "effective_wave_weight_mean":effective,
        }
        return weights,metrics

    def compute_tensor(self, wave_indices: torch.Tensor, alive_masks: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        weights,metrics=self.compute_numpy(wave_indices.detach().cpu().numpy(),None if alive_masks is None else alive_masks.detach().cpu().numpy())
        return torch.as_tensor(weights,dtype=torch.float32,device=wave_indices.device),metrics

    @property
    def actor_enabled(self): return self.enabled and self.target in {"actor_only","actor_critic"}
    @property
    def critic_enabled(self): return self.enabled and self.target in {"critic_only","actor_critic"}


__all__=["WaveBalancingModule"]
