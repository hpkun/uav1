from __future__ import annotations

import hashlib
import csv
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from algorithm.modules import MultiWaveRewardAdapter, PopArtValueNormalizer, RecurrentMemoryModule, WaveContextModule
from algorithm.modular_mappo.trainer import ModularMAPPOTrainer
from algorithm.modular_mappo.evaluation import evaluate_modular_episode
from algorithm.mappo.trainer import compute_gae
from algorithm.train_modular_mappo import load_config
from tools.analyze_jiao2025_reproduction import (
    run_episode, validate_evaluation_history, validate_jiao_config,
    validate_jiao_evaluation_seeds, validate_validation_fresh_disjoint,
)
from tools.prepare_jiao2025_screening import preflight

ROOT = Path(__file__).resolve().parents[1]


def info(red, blue, clear=False):
    return {"red_alive_mask": np.asarray(red, np.float32),
            "blue_alive_mask": np.asarray(blue, np.float32),
            "wave_cleared_this_step": clear}


def adapter():
    return MultiWaveRewardAdapter({"enabled": True, "mode": "jiao_r2_replacement"})


def test_scalar_round_encoding_is_exact_for_numpy_and_tensor():
    module = WaveContextModule({"enabled": True, "encoding": "scalar_round", "context_target": "actor_critic"})
    expected = np.asarray([[1.], [2.], [3.]], np.float32)
    assert module.context_dim == 1
    assert np.array_equal(module.encode_numpy([1, 2, 3], [3, 3, 3]), expected)
    actual = module.encode_tensor(torch.tensor([1, 2, 3]), torch.tensor([3, 3, 3]))
    assert torch.equal(actual, torch.as_tensor(expected))


def test_scalar_f_reaches_actor_and_critic_inputs():
    trainer = ModularMAPPOTrainer(hidden_dim=16, modules_config={
        "wave_context": {"enabled": True, "encoding": "scalar_round", "context_target": "actor_critic"},
    })
    obs = torch.zeros(2, 4, 52)
    context = torch.tensor([[1.], [3.]])
    actor_input = trainer.actor._input(obs, context)
    critic_input = trainer.critic._input(obs, context)
    assert trainer.actor.context_dim == trainer.critic.context_dim == 1
    assert torch.equal(actor_input[..., -1], context.expand(2, 4))
    assert torch.equal(critic_input[..., -1], context.expand(2, 4))
    assert actor_input.shape[-1] == critic_input.shape[-1] == 53


def test_actor_and_critic_gru_and_hidden_lifecycle():
    trainer = ModularMAPPOTrainer(hidden_dim=16, modules_config={
        "recurrent_memory": {"enabled": True, "mode": "actor_critic_gru", "hidden_dim": 8, "sequence_length": 4},
    })
    assert isinstance(trainer.actor.gru, torch.nn.GRUCell)
    assert isinstance(trainer.critic.gru, torch.nn.GRUCell)
    actor_hidden, critic_hidden = trainer.initial_hidden(2)
    actor_hidden.fill(1.0); critic_hidden.fill(2.0)
    # A wave change is not represented here: only true episode done resets.
    trainer.recurrent.reset_for_episode(actor_hidden, np.asarray([False, True]))
    trainer.recurrent.reset_for_episode(critic_hidden, np.asarray([False, True]))
    assert actor_hidden[0].all() and critic_hidden[0].all()
    assert not actor_hidden[1].any() and not critic_hidden[1].any()
    masked = trainer.recurrent.apply_alive(actor_hidden[:1], np.asarray([[1, 0, 1, 0]], np.float32))
    assert masked[0, 0].all() and not masked[0, 1].any() and masked[0, 2].all() and not masked[0, 3].any()


def test_full_sequence_matches_chunk_restoration_for_actor_and_critic():
    torch.manual_seed(7)
    trainer = ModularMAPPOTrainer(hidden_dim=16, modules_config={
        "wave_context": {"enabled": True, "encoding": "scalar_round", "context_target": "actor_critic"},
        "recurrent_memory": {"enabled": True, "mode": "actor_critic_gru", "hidden_dim": 8, "sequence_length": 3},
    })
    obs = torch.randn(7, 1, 4, 52); alive = torch.ones(7, 1, 4)
    context = torch.tensor([1., 1., 2., 2., 2., 3., 3.]).view(7, 1, 1)
    episode = torch.ones(7, 1)

    ah = ch = None; actor_full = []; critic_full = []; saved_a = []; saved_c = []
    for t in range(7):
        saved_a.append(None if ah is None else ah.detach().clone())
        saved_c.append(None if ch is None else ch.detach().clone())
        dist, ah = trainer.actor.distribution_step(obs[t], context[t], ah, episode[t], alive[t])
        value, ch = trainer.critic.forward_step(obs[t], alive[t], context[t], ch, episode[t])
        actor_full.append(dist.mean); critic_full.append(value)
    actor_chunk = []; critic_chunk = []
    for start, stop in ((0, 3), (3, 6), (6, 7)):
        ah = saved_a[start]; ch = saved_c[start]
        for t in range(start, stop):
            dist, ah = trainer.actor.distribution_step(obs[t], context[t], ah, episode[t], alive[t])
            value, ch = trainer.critic.forward_step(obs[t], alive[t], context[t], ch, episode[t])
            actor_chunk.append(dist.mean); critic_chunk.append(value)
    assert torch.allclose(torch.stack(actor_full), torch.stack(actor_chunk), atol=1e-7)
    assert torch.allclose(torch.stack(critic_full), torch.stack(critic_chunk), atol=1e-7)


def test_wave_and_rollout_boundaries_preserve_hidden_but_true_done_resets():
    trainer = ModularMAPPOTrainer(hidden_dim=16, modules_config={
        "recurrent_memory": {"enabled": True, "mode": "actor_critic_gru", "hidden_dim": 8, "sequence_length": 4},
    })
    hidden = np.ones((1, 4, 8), np.float32)
    assert np.array_equal(trainer.recurrent.reset_for_episode(hidden.copy(), np.asarray([False])), hidden)
    reset = trainer.recurrent.reset_for_episode(hidden.copy(), np.asarray([True]))
    assert not reset.any()


def test_gae_crosses_wave_clear_and_stops_only_at_done():
    rewards = torch.tensor([[[1.]], [[2.]], [[3.]]])
    values = torch.zeros_like(rewards); next_values = torch.ones_like(rewards)
    alive = torch.ones_like(rewards)
    adv_cross, _ = compute_gae(rewards, values, next_values, torch.zeros(3, 1), alive, alive, .9, .8)
    adv_done, _ = compute_gae(rewards, values, next_values, torch.tensor([[0.], [1.], [0.]]), alive, alive, .9, .8)
    assert adv_cross[0, 0, 0] > adv_done[0, 0, 0]
    assert torch.isclose(adv_done[1, 0, 0], rewards[1, 0, 0])


def test_popart_normalization_preservation_and_checkpoint(tmp_path):
    popart = PopArtValueNormalizer({"enabled": True, "beta": .9})
    layer = torch.nn.Linear(4, 1)
    x = torch.randn(32, 4)
    before = popart.denormalize_values(layer(x)).detach()
    targets = torch.linspace(-10, 30, 100)
    popart.update(targets, layer)
    assert torch.allclose(before, popart.denormalize_values(layer(x)).detach(), atol=2e-5)
    assert torch.allclose(popart.denormalize_values(popart.normalize_targets(targets)), targets, atol=1e-5)
    trainer = ModularMAPPOTrainer(hidden_dim=16, modules_config={"popart": {"enabled": True}})
    trainer.popart.update(targets.to(trainer.device), trainer.critic.output_layer)
    path = tmp_path / "popart.pt"
    trainer.save(path)
    restored = ModularMAPPOTrainer(hidden_dim=16, modules_config={"popart": {"enabled": True}})
    restored.load(path)
    assert torch.equal(restored.popart.mean, trainer.popart.mean)
    assert torch.equal(restored.popart.variance, trainer.popart.variance)
    assert torch.equal(restored.popart.count, trainer.popart.count)


@pytest.mark.parametrize("wave,expected", [(1, 3.), (2, 6.), (3, 9.)])
def test_jiao_r2_blue_multiplier_and_one_time_death(wave, expected):
    module = adapter()
    raw = np.full((1, 4), 99., np.float32)
    red = np.ones((1, 4), np.float32)
    blue_before = np.asarray([[1, 1, 1, 1]], np.float32)
    training, metrics = module.adapt(raw, [info([1, 1, 1, 1], [1, 1, 0, 1])], np.asarray([wave]), red, blue_before)
    assert np.all(training == expected)
    assert metrics["paper_R2_blue_kill_component"] == expected
    # The next transition starts with j=1 already dead, so it cannot pay twice.
    training, metrics = module.adapt(raw, [info([1, 1, 1, 1], [1, 1, 0, 1])], np.asarray([wave]), red, np.asarray([[1, 1, 0, 1]], np.float32))
    assert not training.any() and metrics["paper_R2_blue_kill_component"] == 0


def test_jiao_r2_red_penalty_does_not_scale_with_wave():
    raw = np.ones((1, 4), np.float32)
    for wave in (1, 2, 3):
        training, metrics = adapter().adapt(
            raw, [info([1, 0, 1, 1], [1, 1, 1, 1])], np.asarray([wave]),
            np.ones((1, 4), np.float32), np.ones((1, 4), np.float32),
        )
        assert np.all(training == -2.)
        assert metrics["paper_R2_red_loss_component"] == -2.


def test_jiao_r2_simultaneous_mutual_deaths_and_transition_start_distribution():
    red_before = np.asarray([[1, 1, 0, 1]], np.float32)
    blue_before = np.asarray([[1, 1, 1, 1]], np.float32)
    training, metrics = adapter().adapt(
        np.zeros((1, 4), np.float32),
        [info([0, 1, 0, 0], [0, 1, 0, 1])], np.asarray([2]), red_before, blue_before,
    )
    # Blue j=0,2 => +2*(1+3)=+8; Red i=0,3 => -(1+4)=-5; team=+3.
    assert metrics["paper_R2_blue_kill_component"] == 8
    assert metrics["paper_R2_red_loss_component"] == -5
    assert np.array_equal(training, np.asarray([[3, 3, 0, 3]], np.float32))


def test_jiao_r2_detects_last_death_despite_immediate_blue_respawn():
    training, metrics = adapter().adapt(
        np.zeros((1, 4), np.float32),
        [info([1, 1, 1, 1], [1, 1, 1, 1], clear=True)], np.asarray([3]),
        np.ones((1, 4), np.float32), np.asarray([[0, 0, 1, 0]], np.float32),
    )
    assert metrics["paper_R2_blue_kill_component"] == 9
    assert np.all(training == 9)


def test_jiao_r2_multistep_respawn_and_all_four_index_coefficients():
    module = adapter(); red = np.ones((1, 4), np.float32); raw = np.zeros((1, 4), np.float32)
    before = np.ones((1, 4), np.float32)
    totals = []
    for after in ([0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]):
        reward, _ = module.adapt(raw, [info([1]*4, after)], np.asarray([1]), red, before)
        totals.append(float(reward[0, 0])); before = np.asarray([after], np.float32)
    reward, metrics = module.adapt(raw, [info([1]*4, [1]*4, clear=True)], np.asarray([1]), red, before)
    totals.append(float(reward[0, 0]))
    # One W1 payment per index: 1, 2, 3, then final index 4 despite respawn.
    assert totals == [1., 2., 3., 4.]
    assert metrics["blue_deaths_index_3"] == 1
    reward, _ = module.adapt(raw, [info([1]*4, [0, 1, 1, 1])], np.asarray([2]), red, np.ones((1, 4), np.float32))
    assert np.all(reward == 2.)


def test_noncombat_alive_to_dead_uses_same_formula_mapping():
    # The paper supplies no cause field; direct alive-state mapping is deliberate.
    reward, metrics = adapter().adapt(
        np.zeros((1, 4), np.float32),
        [{**info([1, 0, 1, 1], [1]*4), "red_boundary_exits": 1}],
        np.asarray([3]), np.ones((1, 4), np.float32), np.ones((1, 4), np.float32),
    )
    assert np.all(reward == -2.)
    assert metrics["red_deaths_index_1"] == 1


def test_core_uses_raw_reward_and_full_replaces_it():
    raw = np.arange(4, dtype=np.float32)[None]
    core = MultiWaveRewardAdapter({"enabled": False, "mode": "none"})
    core_reward, _ = core.adapt(raw, [info([1] * 4, [1] * 4)], np.asarray([1]), np.ones((1, 4)), np.ones((1, 4)))
    full_reward, _ = adapter().adapt(raw, [info([1] * 4, [1, 0, 1, 1])], np.asarray([1]), np.ones((1, 4)), np.ones((1, 4)))
    assert np.array_equal(core_reward, raw)
    assert np.all(full_reward == 2.) and not np.array_equal(full_reward, raw + 2.)


def test_resolved_jiao_configs_and_forbidden_modules():
    core = load_config(ROOT / "configs" / "jiao2025_core_1p5m.yaml")
    full = load_config(ROOT / "configs" / "jiao2025_full_1p5m.yaml")
    validate_jiao_config("Jiao-Core", core)
    validate_jiao_config("Jiao-Full", full)
    assert core["modules"]["multi_wave_reward"]["enabled"] is False
    assert full["modules"]["multi_wave_reward"]["enabled"] is True
    assert validate_validation_fresh_disjoint(core) == {
        "validation_start": 10_000_000, "validation_end": 10_000_019,
        "validation_count": 20, "fresh_start": 38_000_000,
        "fresh_end": 38_000_099, "fresh_count": 100, "overlap_count": 0,
    }


def test_four_method_checkpoint_selection_protocol_matches_actual_snapshots():
    paths = {
        "All-Off": ROOT / "outputs" / "pw_alloff_matched_1p5m_seed2023" / "algorithm_config.yaml",
        "WB-MAPPO": ROOT / "outputs" / "pw_m5_wave_balance_1p5m_seed2023" / "algorithm_config.yaml",
        "Jiao-Core": ROOT / "configs" / "jiao2025_core_1p5m.yaml",
        "Jiao-Full": ROOT / "configs" / "jiao2025_full_1p5m.yaml",
    }
    for name, path in paths.items():
        config = load_config(path) if name.startswith("Jiao") else yaml.safe_load(path.read_text(encoding="utf-8"))
        validate_jiao_config(name, config)
        assert config["implementation"]["evaluation_seed_base"] == 10_000_000
        assert config["training"]["evaluation_episodes"] == 20
        assert config["training"]["evaluation_interval_sampled_steps"] == 100_000


def test_training_history_rejects_contaminated_or_fresh_seed_columns(tmp_path):
    config = load_config(ROOT / "configs" / "jiao2025_core_1p5m.yaml")
    for forbidden in (37_000_000, 38_000_000):
        path = tmp_path / "evaluation_history.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("sampled_steps", "evaluation_episodes", "evaluation_seed"))
            writer.writeheader(); writer.writerow({"sampled_steps": 100_000, "evaluation_episodes": 20, "evaluation_seed": forbidden})
        with pytest.raises(RuntimeError):
            validate_evaluation_history(tmp_path, config)


def test_frozen_alloff_m5_files_and_resolved_semantics_unchanged():
    expected = {
        "pw_alloff_matched_1p5m.yaml": "7d74e67462ba5dbe7f463d9ec0db28392a87f2d9c415bbe82923e9b8cee39565",
        "pw_m5_wave_balance.yaml": "8a3d990ef97b95c8780519546b03a296b6f8e3d0a8980550513449df6625a274",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((ROOT / "configs" / name).read_bytes()).hexdigest() == digest
    alloff = load_config(ROOT / "configs" / "pw_alloff_matched_1p5m.yaml")
    m5 = load_config(ROOT / "configs" / "pw_m5_wave_balance.yaml")
    assert not any(module.get("enabled", False) for module in alloff["modules"].values())
    assert {key for key, module in m5["modules"].items() if module.get("enabled", False)} == {"wave_balancing"}
    assert alloff["modules"]["wave_context"].get("encoding", "rich") == "rich"


def test_only_new_38m_seed_range_is_accepted_and_37m_is_rejected():
    assert validate_jiao_evaluation_seeds(range(38_000_000, 38_000_100))[0] == 38_000_000
    for start in (20_000_000, 35_000_000, 36_000_000, 37_000_000):
        with pytest.raises(ValueError):
            validate_jiao_evaluation_seeds(range(start, start + 100))


def test_smoke_and_static_preflight_never_use_validation_comparison_or_holdout():
    forbidden = set(range(10_000_000, 10_000_020)) | set(range(20_000_000, 20_000_200))
    forbidden |= set(range(37_000_000, 38_000_100))
    for name in ("jiao2025_core_smoke.yaml", "jiao2025_full_smoke.yaml"):
        config = load_config(ROOT / "configs" / name)
        seeds = {int(config["training"]["seed"]), int(config["implementation"]["evaluation_seed_base"]),
                 int(config["implementation"]["evaluation_seed_base"]) + 1}
        assert not seeds & forbidden
    manifest = preflight(write_manifest=False)
    assert manifest["fresh_comparison"]["executed"] is False
    assert manifest["fresh_comparison"]["evaluation_seed_start"] == 38_000_000


def test_analysis_and_training_evaluator_paths_are_deterministically_equivalent():
    env_config = yaml.safe_load((ROOT / "configs" / "persistent_wave_v2_environment.yaml").read_text(encoding="utf-8"))
    trainer = ModularMAPPOTrainer(hidden_dim=16, modules_config={
        "wave_context": {"enabled": True, "encoding": "scalar_round", "context_target": "actor_critic"},
        "recurrent_memory": {"enabled": True, "mode": "actor_critic_gru", "hidden_dim": 8, "sequence_length": 4},
    })
    canonical = evaluate_modular_episode(trainer, env_config, 99_999_901, include_trace=True)
    analysis = run_episode(trainer, env_config, 99_999_901, include_trace=True)
    assert np.array_equal(canonical["action_trace"], analysis["action_trace"])
    assert np.array_equal(canonical["wave_trace"], analysis["wave_trace"])
    assert canonical["episode_return"] == analysis["episode_return"]
    assert canonical["waves_cleared"] == analysis["waves_cleared"]
