"""Identity bookkeeping is discrete; aligned hidden tensors retain BPTT gradients."""
from dataclasses import dataclass, fields
import torch


@dataclass
class MemoryState:
    self_hidden: torch.Tensor
    peer_hidden: torch.Tensor
    obstacle_hidden: torch.Tensor
    peer_ids: torch.Tensor
    obstacle_ids: torch.Tensor
    fusion_hidden: torch.Tensor

    def detach(self):
        return MemoryState(**{f.name: getattr(self, f.name).detach() for f in fields(self)})

    def clone(self):
        return MemoryState(**{f.name: getattr(self, f.name).clone() for f in fields(self)})

    def numpy(self):
        return {f.name: getattr(self, f.name).detach().cpu().numpy().copy() for f in fields(self)}

    @classmethod
    def from_numpy(cls, values, device):
        return cls(**{key: torch.as_tensor(value, device=device) for key, value in values.items()})


@torch.no_grad()
def identity_matches(previous_ids, current_ids, valid):
    """Return a current-slot -> previous-slot map; -1 denotes a new identity.

    Prefix dimensions are independent environments/observers, never a pool.
    Invalid slots must carry -1, and active IDs must be unique per observer.
    """
    if previous_ids.shape[:-1] != current_ids.shape[:-1] or valid.shape != current_ids.shape:
        raise ValueError('identity mapping shape mismatch')
    for ids in (previous_ids, current_ids):
        if ids.dtype != torch.long or (ids < -1).any():
            raise ValueError('entity IDs must be int64 and >= -1')
        duplicate = (ids.unsqueeze(-1) == ids.unsqueeze(-2)) & (ids.unsqueeze(-1) >= 0)
        eye = torch.eye(ids.shape[-1], dtype=torch.bool, device=ids.device)
        if (duplicate & ~eye).any():
            raise ValueError('duplicate entity ID within one observer/type')
    if valid.dtype != torch.bool or ((current_ids >= 0) != valid).any():
        raise ValueError('valid entity IDs and validity masks disagree')
    matches = (current_ids.unsqueeze(-1) == previous_ids.unsqueeze(-2)) & valid.unsqueeze(-1)
    matches &= previous_ids.unsqueeze(-2) >= 0
    if previous_ids.shape[-1] == 0:
        return torch.full_like(current_ids, -1)
    return torch.where(matches.any(-1), matches.long().argmax(-1), -1)


def align_hidden(hidden, previous_ids, current_ids, valid):
    if hidden.shape[:-1] != previous_ids.shape:
        raise ValueError('hidden-state/identity shape mismatch')
    indices = identity_matches(previous_ids, current_ids, valid)
    if hidden.shape[-2] == 0:
        return hidden.new_zeros((*current_ids.shape, hidden.shape[-1]))
    aligned = hidden.gather(-2, indices.clamp_min(0).unsqueeze(-1).expand(*indices.shape, hidden.shape[-1]))
    return aligned * (indices >= 0).unsqueeze(-1)

