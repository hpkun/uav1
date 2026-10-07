from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from tools.audit_wave_gradient_conflict import (
    DIAGNOSTIC_ENV_SEED_BASE,
    DIAGNOSTIC_TORCH_SEED,
    RAW_REQUIRED_FIELDS,
    clipped_value_error,
    cosine_metrics,
    flatten_gradients,
    global_live_advantage_normalization,
    mutation_fingerprint,
    ppo_clipped_surrogate,
    seed_is_forbidden,
    select_checkpoint,
    split_half_masks,
    state_sha256,
    sufficient_wave_samples,
    wave_agent_mask,
)


TOOL = Path(__file__).resolve().parents[1] / "tools/audit_wave_gradient_conflict.py"


def test_cosine_same_direction_is_positive_one():
    assert cosine_metrics(torch.tensor([1., 2.]), torch.tensor([2., 4.]))["cosine"] == pytest.approx(1.)


def test_cosine_opposite_direction_is_negative_one():
    assert cosine_metrics(torch.tensor([1., 2.]), torch.tensor([-2., -4.]))["cosine"] == pytest.approx(-1.)


def test_cosine_orthogonal_is_zero():
    assert cosine_metrics(torch.tensor([1., 0.]), torch.tensor([0., 2.]))["cosine"] == pytest.approx(0.)


def test_none_gradient_is_replaced_by_parameter_sized_zero():
    used = torch.nn.Parameter(torch.tensor([2., 3.])); unused = torch.nn.Parameter(torch.tensor([7.]))
    vector = flatten_gradients(used.square().sum(), [used, unused])
    assert vector.tolist() == [4., 6., 0.]


def test_wave_mask_selects_only_requested_wave():
    alive = torch.ones(2, 2, 2); waves = torch.tensor([[1, 2], [3, 2]])
    mask = wave_agent_mask(alive, waves, 2)
    assert mask.sum().item() == 4 and mask[0, 0].sum().item() == 0 and mask[1, 1].sum().item() == 2


def test_dead_agents_are_excluded_from_wave_mask():
    alive = torch.tensor([[[1., 0., 1., 0.]]]); waves = torch.tensor([[1]])
    assert wave_agent_mask(alive, waves, 1).tolist() == alive.tolist()


def test_global_normalization_uses_all_waves_not_each_wave():
    advantages = torch.tensor([[[0., 2.]], [[10., 12.]]]); alive = torch.ones_like(advantages)
    normalized = global_live_advantage_normalization(advantages, alive)
    assert normalized[0].mean() < 0 and normalized[1].mean() > 0
    assert normalized.mean().item() == pytest.approx(0., abs=1e-6)


def test_global_normalization_masks_dead_agents():
    advantages = torch.tensor([[[1., 999.]]]); alive = torch.tensor([[[1., 0.]]])
    assert global_live_advantage_normalization(advantages, alive)[0, 0, 1].item() == 0


def test_actor_surrogate_matches_plain_clipped_formula():
    old = torch.zeros(3); new = torch.log(torch.tensor([1.5, .5, 1.])); advantage = torch.tensor([1., -1., 2.])
    actual = ppo_clipped_surrogate(new, old, advantage, .2)
    expected = torch.minimum(torch.exp(new) * advantage, torch.exp(new).clamp(.8, 1.2) * advantage)
    assert torch.equal(actual, expected)


def test_critic_clipped_error_matches_plain_formula():
    current = torch.tensor([2., -2.]); old = torch.tensor([0., 0.]); target = torch.tensor([1., -1.])
    expected = torch.maximum((current-target).square(), (old+(current-old).clamp(-.2,.2)-target).square())
    assert torch.equal(clipped_value_error(current, old, target, .2, True), expected)


def test_critic_unclipped_error():
    current = torch.tensor([2.]); old = torch.tensor([0.]); target = torch.tensor([1.])
    assert clipped_value_error(current, old, target, .2, False).item() == 1.


def test_sample_gate_accepts_exactly_64():
    assert sufficient_wave_samples(64)


def test_sample_gate_rejects_63():
    assert not sufficient_wave_samples(63)


def test_split_half_is_deterministic():
    mask = torch.ones(3, 4)
    first_a, second_a = split_half_masks(mask, 123)
    first_b, second_b = split_half_masks(mask, 123)
    assert torch.equal(first_a, first_b) and torch.equal(second_a, second_b)


def test_split_half_is_balanced_disjoint_and_complete():
    mask = torch.ones(11)
    first, second = split_half_masks(mask, 4)
    assert abs(first.sum().item() - second.sum().item()) <= 1
    assert not torch.any((first > 0) & (second > 0))
    assert torch.equal(first + second, mask)


class _TrainerStub:
    def __init__(self):
        self.actor = torch.nn.Linear(2, 2)
        self.critic = torch.nn.Linear(2, 1)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4)
        self.actor_update_count = 7; self.critic_update_count = 8; self.ppo_update_count = 9


def test_actor_parameters_bitwise_unchanged_after_autograd_grad():
    trainer = _TrainerStub(); before = state_sha256(trainer.actor.state_dict())
    torch.autograd.grad(trainer.actor(torch.ones(1, 2)).sum(), tuple(trainer.actor.parameters()))
    assert state_sha256(trainer.actor.state_dict()) == before


def test_critic_parameters_bitwise_unchanged_after_autograd_grad():
    trainer = _TrainerStub(); before = state_sha256(trainer.critic.state_dict())
    torch.autograd.grad(trainer.critic(torch.ones(1, 2)).sum(), tuple(trainer.critic.parameters()))
    assert state_sha256(trainer.critic.state_dict()) == before


def test_optimizer_state_unchanged_after_autograd_grad():
    trainer = _TrainerStub(); before = mutation_fingerprint(trainer)
    torch.autograd.grad(trainer.actor(torch.ones(1, 2)).sum(), tuple(trainer.actor.parameters()))
    assert mutation_fingerprint(trainer) == before


def test_tool_has_no_optimizer_or_trainer_update_call_path():
    tree = ast.parse(TOOL.read_text(encoding="utf-8"))
    forbidden = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        rendered = ast.unparse(node.func)
        if rendered.endswith("optimizer.step") or rendered.endswith("trainer.update") or rendered.endswith(".backward"):
            forbidden.append(rendered)
    assert forbidden == []


def test_diagnostic_environment_seed_is_not_44m_or_45m():
    assert not seed_is_forbidden(DIAGNOSTIC_ENV_SEED_BASE)


def test_diagnostic_torch_seed_is_not_44m_or_45m():
    assert not seed_is_forbidden(DIAGNOSTIC_TORCH_SEED)


def test_protected_seed_ranges_are_detected():
    assert seed_is_forbidden(44_000_000) and seed_is_forbidden(45_000_199)


def test_exact_3m_checkpoint_requires_exact_file(tmp_path):
    checkpoints = {2_999_999: tmp_path / "checkpoint_2999999.pt"}
    with pytest.raises(RuntimeError, match="exact"):
        select_checkpoint(checkpoints, 3_000_000, 0, True)


def test_exact_3m_checkpoint_is_selected(tmp_path):
    path = tmp_path / "checkpoint_3000000.pt"
    assert select_checkpoint({3_000_000: path}, 3_000_000, 0, True) == (3_000_000, path)


def test_nearest_checkpoint_selection_for_1p5m(tmp_path):
    values = {1_204_224: tmp_path / "a", 1_505_280: tmp_path / "b", 1_800_192: tmp_path / "c"}
    assert select_checkpoint(values, 1_500_000, 120_000)[0] == 1_505_280


def test_nearest_checkpoint_selection_for_2m(tmp_path):
    values = {1_800_192: tmp_path / "a", 2_101_248: tmp_path / "b"}
    assert select_checkpoint(values, 2_000_000, 120_000)[0] == 2_101_248


def test_nearest_checkpoint_outside_tolerance_fails(tmp_path):
    with pytest.raises(RuntimeError, match="exceeds"):
        select_checkpoint({1_700_000: tmp_path / "a"}, 2_000_000, 120_000)


def test_required_raw_csv_and_report_fields_are_declared_in_tool():
    source = TOOL.read_text(encoding="utf-8")
    required = {"training_seed", "checkpoint_path", "checkpoint_step", "rollout_index",
                "diagnostic_env_seed_base", "diagnostic_torch_seed", "transition_samples_w1",
                "alive_agent_samples_w3", "actor_loss_w1", "critic_grad_norm_w3",
                "actor_normadv_w1_w2_cosine", "actor_rawadv_w2_w3_cosine",
                "critic_w1_w3_cosine", "actor_within_w1_cos", "critic_within_w3_cos"}
    assert required <= set(RAW_REQUIRED_FIELDS)
    assert "wave_gradient_conflict_report.md" in source
    assert "wave_gradient_conflict_summary.json" in source


def test_state_hash_changes_on_parameter_mutation():
    module = torch.nn.Linear(2, 1); before = state_sha256(module.state_dict())
    with torch.no_grad(): module.weight.add_(1.)
    assert state_sha256(module.state_dict()) != before
