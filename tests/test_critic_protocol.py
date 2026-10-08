"""Shared formal critic, independent widths and legacy critic isolation."""
import copy
from pathlib import Path

import pytest
import torch
import yaml

from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
from algorithm.common.critic_protocol import checkpoint_widths, validate_mappo_architecture
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.mappo.networks import CentralizedMLPCritic, CentralizedValueCritic
from algorithm.mappo.trainer import MAPPOTrainer
from algorithm.stea_mappo.factory import build_stea_mappo_trainer
from algorithm.stea_mappo.protocol import architecture_from_config, validate_checkpoint
from algorithm.common.protocol import config_sha256

ROOT = Path(__file__).resolve().parents[1]


def config(name):
    return yaml.safe_load((ROOT / f"configs/{name}_5v5.yaml").read_text())


def build(name, cfg=None):
    cfg = config(name) if cfg is None else cfg
    return (build_mappo_trainer(cfg, "cpu") if name == "mappo"
            else build_stea_mappo_trainer(cfg, "cpu", seed=0))


def count(module):
    return sum(p.numel() for p in module.parameters())


def test_formal_critics_identical_structure_and_initial_weights():
    mappo, stea = build("mappo"), build("stea_mappo")
    for trainer in (mappo, stea):
        assert type(trainer.critic) is CentralizedMLPCritic
        assert trainer.critic_hidden_dim == 256
        assert trainer.critic.value_network[0].in_features == 390
        assert not any(k.split('.')[0] in ('wq', 'wk', 'wv', 'embedding') for k in trainer.critic.state_dict())
        assert isinstance(trainer.critic_optimizer, torch.optim.Adam)
        assert trainer.critic_optimizer.param_groups[0]['lr'] == .0003
        assert trainer.value_loss_coefficient == .5 and trainer.clip_value_loss
    assert count(mappo.critic) == count(stea.critic) == 166145
    assert mappo.critic.state_dict().keys() == stea.critic.state_dict().keys()
    for k, value in mappo.critic.state_dict().items():
        assert torch.equal(value, stea.critic.state_dict()[k])
    assert any(k.startswith('ally_attention.') for k in stea.actor.state_dict())
    assert any(k.startswith('enemy_attention.') for k in stea.actor.state_dict())
    assert any(k.startswith('gru.') for k in stea.actor.state_dict())


@pytest.mark.parametrize('name', ['mappo', 'stea_mappo'])
def test_critic_width_changes_only_critic_structure(name):
    baseline = build(name)
    cfg = config(name); cfg['network']['critic_hidden_layers'] = [192, 192]
    changed = build(name, cfg)
    assert changed.critic_hidden_dim == 192
    assert count(changed.critic) != count(baseline.critic)
    assert count(changed.actor) == count(baseline.actor)
    assert {k: v.shape for k, v in changed.actor.state_dict().items()} == {
        k: v.shape for k, v in baseline.actor.state_dict().items()}


def test_mappo_actor_width_does_not_change_critic():
    baseline = build('mappo')
    cfg = config('mappo'); cfg['network']['actor_hidden_layers'] = [192, 192]
    changed = build('mappo', cfg)
    assert changed.actor_hidden_dim == 192 and changed.critic_hidden_dim == 256
    assert count(changed.actor) != count(baseline.actor)
    assert count(changed.critic) == count(baseline.critic)
    assert changed.critic.state_dict().keys() == baseline.critic.state_dict().keys()


@pytest.mark.parametrize('key', ['actor_hidden_layers', 'critic_hidden_layers'])
@pytest.mark.parametrize('layers', [[128], [128, 64], [0, 0], [-1, -1], [1.5, 1.5], [True, True]])
def test_mappo_rejects_invalid_widths(key, layers):
    cfg = config('mappo'); cfg['network'][key] = layers
    with pytest.raises(ValueError, match=key):
        build('mappo', cfg)


@pytest.mark.parametrize('layers', [[128], [128, 64], [0, 0], [-1, -1]])
def test_stea_rejects_invalid_widths(layers):
    cfg = config('stea_mappo'); cfg['network']['critic_hidden_layers'] = layers
    with pytest.raises(ValueError, match='critic_hidden_layers'):
        build('stea_mappo', cfg)


def test_legacy_hidden_dim_default_and_attention_preserved():
    trainer = MAPPOTrainer(hidden_dim=32)
    assert trainer.actor_hidden_dim == trainer.critic_hidden_dim == 32
    assert type(trainer.critic) is CentralizedValueCritic
    for name in ('mappo', 'stea_mappo'):
        cfg = config(name); cfg['network']['critic_type'] = 'attention'
        trainer = build(name, cfg)
        assert type(trainer.critic) is CentralizedValueCritic
        if name != 'mappo':
            assert trainer.network_architecture == architecture_from_config(cfg)
            assert trainer.network_architecture['critic_attention_heads'] == 2


def test_masked_global_and_focal_input():
    critic = build('mappo').critic
    observations = torch.randn(3, 5, 65)
    alive = torch.tensor([[1, 0, 1, 1, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1.]])
    captured = []
    hook = critic.value_network[0].register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
    values = critic(observations, alive)
    mutated = observations.clone(); mutated[alive == 0] = 1e30
    other = critic(mutated, alive); hook.remove()
    assert values.shape == (3, 5) and torch.isfinite(values).all()
    assert torch.equal(values[alive == 0], torch.zeros_like(values[alive == 0]))
    assert torch.equal(values, other) and torch.equal(captured[0], captured[1])
    masked = observations * alive[..., None]
    assert torch.equal(captured[0][..., :325], masked.flatten(1)[:, None].expand(-1, 5, -1))
    assert torch.equal(captured[0][..., 325:], masked)


@pytest.mark.parametrize('name', ['mappo', 'stea_mappo'])
def test_gaussian_ppo_unchanged_and_mlp_metadata(name):
    cfg = config(name); trainer = build(name)
    actor = trainer.actor
    assert actor.log_std_parameter.shape == (3,) and not hasattr(actor, 'log_std')
    assert torch.equal(actor.log_std_parameter, torch.full((3,), -.5))
    assert actor.log_std_min == -5 and actor.log_std_max == .5
    torch.testing.assert_close(actor.mean.weight @ actor.mean.weight.T, torch.eye(3)*.0001)
    assert trainer.entropy_coefficient == .001 and trainer.target_kl == .015 and trainer.ppo_epochs == 10
    assert trainer.gamma == .99 and trainer.gae_lambda == .95 and trainer.clip_ratio == .2
    if name == 'stea_mappo':
        assert trainer.network_architecture == architecture_from_config(cfg)
        assert 'critic_attention_heads' not in trainer.network_architecture


@pytest.mark.parametrize('name', ['mappo', 'stea_mappo'])
@pytest.mark.parametrize('source_type', ['attention', 'mlp'])
def test_cross_critic_direct_load_rejected(tmp_path, name, source_type):
    source = config(name); source['network']['critic_type'] = source_type
    target = copy.deepcopy(source); target['network']['critic_type'] = 'mlp' if source_type == 'attention' else 'attention'
    trainer = build(name, source); path = tmp_path/'checkpoint.pt'; trainer.save(path)
    with pytest.raises(RuntimeError, match='critic_type|network_architecture'):
        build(name, target).load(path)
    env = yaml.safe_load((ROOT/'configs/combat_environment_v25.yaml').read_text())
    extra = dict(environment_version='2.5', observation_dim=65, action_dim=3, num_agents=5,
        training_seed=1, training_gamma=.99, training_num_envs=2, training_total_sampled_steps=512,
        training_smoke=False, effective_hidden_dim=256, environment_config_sha256=config_sha256(env),
        algorithm_config_sha256=config_sha256(source), actor_parameter_count=count(trainer.actor),
        critic_parameter_count=count(trainer.critic), total_parameter_count=count(trainer.actor)+count(trainer.critic))
    if name == 'stea_mappo':
        extra.update(network_architecture=trainer.network_architecture, stea_mappo_impl_version=1)
    state = trainer.checkpoint_state(extra)
    validator = validate_checkpoint_for_evaluation if name == 'mappo' else validate_checkpoint
    with pytest.raises(RuntimeError, match='critic_type'):
        validator(state, env, target)


def test_new_mappo_metadata_checks_independent_widths_and_counts():
    cfg = config('mappo'); cfg['network']['critic_hidden_layers'] = [192, 192]
    trainer = build('mappo', cfg)
    architecture = dict(actor_hidden_dim=256, critic_hidden_dim=192, critic_type='mlp')
    state = trainer.checkpoint_state(dict(network_architecture=architecture, **architecture,
        actor_parameter_count=count(trainer.actor), critic_parameter_count=count(trainer.critic)))
    assert checkpoint_widths(state) == (256, 192)
    validate_mappo_architecture(state, cfg)
    for key in ('critic_hidden_dim', 'actor_hidden_dim', 'critic_type'):
        changed = copy.deepcopy(state); changed['extra']['network_architecture'][key] = 'wrong'
        with pytest.raises(RuntimeError, match='network_architecture'):
            validate_mappo_architecture(changed, cfg)
    changed = copy.deepcopy(state); del changed['extra']['network_architecture']['actor_hidden_dim']
    with pytest.raises(RuntimeError, match='network_architecture'):
        validate_mappo_architecture(changed, cfg)
