"""Read-only diagnostics; clipping is performed before each optimizer step."""
import torch
import math

ACTION_NAMES = ('heading', 'pitch', 'speed')


def finite_grad_norm(module):
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:return 0.
    norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])))
    if not math.isfinite(norm):raise FloatingPointError('non-finite MADDPG gradient')
    return norm


def clip_gradients(module, max_norm):
    pre = finite_grad_norm(module)
    torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm, error_if_nonfinite=True)
    post = finite_grad_norm(module)
    return pre, post, pre > max_norm


def scalar_statistics(values, alive):
    selected = values.detach()[alive.bool()]
    if not selected.numel():
        return dict.fromkeys(('mean','std','min','max','abs_mean','abs_max'))
    numbers = torch.stack((selected.mean(), selected.std(unbiased=False), selected.min(),
        selected.max(), selected.abs().mean(), selected.abs().max())).cpu().tolist()
    return dict(zip(('mean','std','min','max','abs_mean','abs_max'), numbers))


def action_statistics(actions, alive):
    selected = actions.detach()[alive.bool()]
    if not selected.numel():
        return {f'{metric}_{name}': None for metric in ('action_mean','action_std','saturation_95_ratio','saturation_99_ratio') for name in ACTION_NAMES}
    matrix = torch.stack((selected.mean(0), selected.std(0,unbiased=False),
        (selected.abs()>=.95).float().mean(0), (selected.abs()>=.99).float().mean(0))).cpu().tolist()
    return {f'{metric}_{name}': matrix[i][j] for i,metric in enumerate(
        ('action_mean','action_std','saturation_95_ratio','saturation_99_ratio')) for j,name in enumerate(ACTION_NAMES)}


@torch.no_grad()
def parameter_distances(modules, initial, targets):
    drift, lag = [], []
    for module, reference, target in zip(modules, initial, targets):
        current = torch.nn.utils.parameters_to_vector(module.parameters())
        target_vector = torch.nn.utils.parameters_to_vector(target.parameters())
        drift.append(float(torch.linalg.vector_norm(current-reference)))
        lag.append(float(torch.linalg.vector_norm(current-target_vector)))
    return drift, lag
