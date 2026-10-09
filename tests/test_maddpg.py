"""Classic MADDPG identity, mathematics, OU noise, replay and strict provenance."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from torch import nn
from algorithm.common.protocol import config_sha256
from algorithm.common.vector_env import VectorStep
from algorithm.maddpg.networks import DeterministicActor, CentralizedQCritic
from algorithm.maddpg.trainer import MADDPGTrainer, critic_target, actor_joint_actions, polyak_update, masked_mean
from algorithm.maddpg.replay_buffer import ReplayBatch, JointReplayBuffer
from algorithm.maddpg.noise import OUNoise, exploration_scale
from algorithm.maddpg.factory import build_maddpg_trainer
from algorithm.maddpg.protocol import validate_config, validate_checkpoint, require_cuda
from algorithm.maddpg.runner import MADDPGTrainingRunner
from algorithm.maddpg.evaluation import evaluate, validate_seeds

ROOT = Path(__file__).resolve().parents[1]


def configs(n=8):
    env = yaml.safe_load((ROOT/f'configs/combat_environment_v{24 if n == 8 else 25}.yaml').read_text())
    config = yaml.safe_load((ROOT/f'configs/maddpg_{n}v{n}.yaml').read_text())
    return env, config


def trainer(n=5, device='cpu'):
    return MADDPGTrainer(n*13, 3, n, device=device, seed=2)


def batch(n=5, size=4):
    rng = np.random.default_rng(8)
    return ReplayBatch(rng.normal(size=(size, n, n*13)).astype(np.float32),
        rng.uniform(-1, 1, (size, n, 3)).astype(np.float32), rng.normal(size=(size, n)).astype(np.float32),
        rng.normal(size=(size, n, n*13)).astype(np.float32), np.zeros(size, bool),
        np.ones((size, n), np.float32), np.ones((size, n), np.float32))


def checkpoint_extra(t, env, config):
    return {'environment_version': env['environment_version'], 'environment_config_sha256': config_sha256(env),
        'algorithm_config_sha256': config_sha256(config), **{k: getattr(t, k) for k in ('observation_dim', 'action_dim', 'num_agents', 'gamma', 'tau')},
        'training_seed': t.seed, 'training_num_envs': 16, 'training_total_sampled_steps': 3000000, 'training_smoke': False,
        'network_architecture': t.network_architecture, 'parameter_counts': t.parameter_counts(),
        'ou_noise_state': OUNoise(16, t.num_agents, 3, 4).state_dict()}


@pytest.mark.parametrize('n,actor_count,critic_count', [(5, 83459, 153345), (8, 93443, 285441)])
def test_independent_networks_exact_structures_parameter_counts_and_no_forbidden_modules(n, actor_count, critic_count):
    t = trainer(n)
    for name in ('actors', 'critics', 'target_actors', 'target_critics'):
        modules = getattr(t, name)
        assert len(modules) == len({id(m) for m in modules}) == n
        storage = [{p.data_ptr() for p in m.parameters()} for m in modules]
        assert all(not a.intersection(b) for i, a in enumerate(storage) for b in storage[i+1:])
        assert all(not isinstance(m, (nn.MultiheadAttention, nn.GRU, nn.LSTM, nn.Transformer)) for m in modules.modules())
    for online, targets in ((t.actors, t.target_actors), (t.critics, t.target_critics)):
        assert not {p.data_ptr() for p in online.parameters()}.intersection(p.data_ptr() for p in targets.parameters())
        assert all(torch.equal(p, q) and not q.requires_grad for p, q in zip(online.parameters(), targets.parameters()))
    assert len({id(o) for o in t.actor_optimizers+t.critic_optimizers}) == 2*n
    for i, optimizer in enumerate(t.actor_optimizers):
        assert {id(p) for group in optimizer.param_groups for p in group['params']} == {id(p) for p in t.actors[i].parameters()}
    assert [m.in_features for m in t.actors[0].network if isinstance(m, nn.Linear)] == [13*n, 256, 256]
    assert [m.out_features for m in t.critics[0].network if isinstance(m, nn.Linear)] == [256, 256, 1]
    assert t.critics[0].joint_input_dim == n*(13*n+3)
    assert type(t.actors[0].network[-1]) is nn.Tanh
    assert t.parameter_counts()['actor_per_agent'] == actor_count
    assert t.parameter_counts()['critic_per_agent'] == critic_count
    assert t.parameter_counts()['trainable_total'] == n*(actor_count+critic_count)
    assert not any(hasattr(t, name) for name in ('alpha', 'policy_delay', 'critic2'))
    assert not any('log_std' in name for name, _ in t.actors.named_parameters())


def test_actor_locality_bounds_and_full_joint_critic_influence():
    t = trainer(); b = batch()
    before = t.act(b.observations, b.alive_masks)
    changed = b.observations.copy(); changed[:, 1] += 100
    after = t.act(changed, b.alive_masks)
    np.testing.assert_array_equal(before[:, 0], after[:, 0])
    assert not np.array_equal(before[:, 1], after[:, 1])
    assert np.isfinite(before).all() and (np.abs(before) <= 1).all()
    tensors = t.tensors(b); q = t.critics[0](tensors['observations'], tensors['actions'], tensors['alive_masks'])
    altered_obs, altered_actions = tensors['observations'].clone(), tensors['actions'].clone()
    altered_obs[:, 1] += 1; altered_actions[:, 1] += .2
    assert not torch.equal(q, t.critics[0](altered_obs, tensors['actions'], tensors['alive_masks']))
    assert not torch.equal(q, t.critics[0](tensors['observations'], altered_actions, tensors['alive_masks']))
    assert q.shape == (4,)


def test_target_formula_done_and_individual_death_with_distinct_local_rewards():
    reward = torch.tensor([[1., 2.], [3., 4.], [-10., 6.]])
    alive = torch.tensor([[1., 1.], [1., 1.], [0., 1.]])
    done = torch.tensor([[0.], [1.], [0.]])
    q = torch.tensor([[8., 6.], [10., 10.], [100., 2.]])
    assert torch.equal(critic_target(reward, done, alive, q, .5), torch.tensor([[5., 5.], [3., 4.], [-10., 7.]]))


def test_own_actor_action_replacement_preserves_detached_replay_actions():
    replay = torch.randn(4, 5, 3, requires_grad=True)
    own = torch.randn(4, 3, requires_grad=True)
    result = actor_joint_actions(replay, own, 2)
    for i in (0, 1, 3, 4):
        assert torch.equal(result[:, i], replay[:, i])
    result.sum().backward()
    assert torch.equal(own.grad, torch.ones_like(own)) and replay.grad is None


def test_actor_gradient_isolation_and_critic_freeze():
    t = trainer(); data = t.tensors(batch())
    loss, norm = t.update_actor(2, data)
    assert np.isfinite(loss) and norm > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in t.actors[2].parameters())
    assert all(p.grad is None for i, a in enumerate(t.actors) if i != 2 for p in a.parameters())
    assert all(p.grad is None and p.requires_grad for p in t.critics.parameters())
    assert all(p.grad is None and not p.requires_grad for p in t.target_actors.parameters())


def test_polyak_exact_and_targets_without_gradient():
    online, target = nn.Linear(1, 1).double(), nn.Linear(1, 1).double()
    for p in online.parameters(): p.data.fill_(4.)
    for p in target.parameters(): p.data.fill_(2.)
    polyak_update(online, target, .001)
    assert all(torch.equal(p, torch.full_like(p, 2.002)) for p in target.parameters())
    t = trainer(); result = t.targets(t.tensors(batch()))
    assert not result.requires_grad and result.grad_fn is None


def test_critic_input_masks_dead_slots_and_invariant_to_dead_data():
    t = trainer(); data = t.tensors(batch()); alive = data['alive_masks'].clone(); alive[:, 2] = 0
    x, actions = data['observations'], data['actions']
    joined = CentralizedQCritic.joint_input(x, actions, alive)
    assert not joined[:, 2*65:3*65].any() and not joined[:, 5*65+2*3:5*65+3*3].any()
    expected = t.critics[0](x, actions, alive)
    x[:, 2] = float('nan'); actions[:, 2] = float('inf')
    assert torch.equal(expected, t.critics[0](x, actions, alive))
    assert not t.act(x.numpy(), alive.numpy())[:, 2].any()


def test_death_transition_learns_reward_post_death_is_excluded():
    t = trainer(); b = batch(); alive = b.alive_masks.copy(); next_alive = b.next_alive_masks.copy()
    alive[1:, 0] = 0; next_alive[:, 0] = 0
    rewards = b.rewards.copy(); rewards[0, 0] = -10.; rewards[1:, 0] = 123456
    data = t.tensors(replace(b, alive_masks=alive, next_alive_masks=next_alive, rewards=rewards))
    targets = t.targets(data); assert targets[0, 0] == -10
    q = t.critics[0](data['observations'], data['actions'], data['alive_masks'])
    assert masked_mean((q-targets[:, 0]).square(), data['alive_masks'][:, 0]) == (q[0]+10).square()
    values = torch.tensor([2., 999., 888.], requires_grad=True)
    masked_mean(values, torch.tensor([1., 0., 0.])).backward()
    assert torch.equal(values.grad, torch.tensor([1., 0., 0.]))
    initial = [p.clone() for p in t.critics[0].parameters()]
    t.update(replace(b, alive_masks=alive, next_alive_masks=next_alive, rewards=rewards))
    assert any(not torch.equal(p, before) for p, before in zip(t.critics[0].parameters(), initial))


@pytest.mark.parametrize('n', [5, 8])
def test_all_actors_critics_update_synchronously_every_gradient_step(n):
    t = trainer(n, 'cuda'); data = batch(n)
    old_a, old_c, old_target = ([p.detach().clone() for p in modules.parameters()] for modules in (t.actors, t.critics, t.target_critics))
    for _ in range(2):
        metrics = t.update(data)
        assert all(np.isfinite(metrics[k]).all() for k in ('actor_losses', 'critic_losses', 'actor_grad_norms', 'critic_grad_norms', 'q_means', 'target_mean'))
    assert t.actor_update_count == t.critic_update_count == 2
    assert t.actor_updates_per_agent == t.critic_updates_per_agent == [2]*n
    assert any(not torch.equal(a, b) for a, b in zip(old_a, t.actors.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(old_c, t.critics.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(old_target, t.target_critics.parameters()))
    assert all(p.grad is None for p in t.target_actors.parameters())


def test_all_dead_samples_do_not_update_any_optimizer():
    t = trainer(); b = batch(); b.alive_masks.fill(0)
    before = deepcopy(t.actors.state_dict()); metrics = t.update(b)
    assert t.actor_updates_per_agent == t.critic_updates_per_agent == [0]*5
    assert all(torch.equal(v, t.actors.state_dict()[k]) for k, v in before.items())
    assert metrics['actor_loss'] == metrics['critic_loss'] == 0


def test_replay_environment_transition_units_uniform_sampling_and_wraparound():
    replay = JointReplayBuffer(19, 8, 104, chunk_size=4, seed=8)
    assert replay.storage_nbytes == 0
    b = batch(8, 16); replay.add_batch(*vars(b).values())
    assert len(replay) == 16
    sampled = replay.sample(8)
    assert sampled.observations.shape == (8, 8, 104)
    assert sampled.actions.shape == (8, 8, 3) and sampled.rewards.shape == (8, 8)
    b2 = batch(8, 25); b2.rewards[:] = np.arange(25)[:, None]
    replay.add_batch(*vars(b2).values())
    assert len(replay) == 19 and replay.position == (16+25)%19
    assert set(replay.get_batch(np.arange(19)).rewards[:, 0]) == set(range(6, 25))
    sampled.rewards.fill(999)
    assert not (replay.get_batch(np.arange(19)).rewards == 999).any()


def test_runner_stores_executed_noisy_actions_and_true_terminal_state():
    runner = object.__new__(MADDPGTrainingRunner)
    runner.replay = JointReplayBuffer(32, 5, 65)
    b = batch(5, 16)
    result = VectorStep(np.full((16, 5, 65), 99, np.float32), np.full((16, 5, 65), 7, np.float32), b.rewards,
        np.ones(16, bool), np.zeros(16, bool), [{'executed_red_actions': a.copy()} for a in b.actions], b.alive_masks, b.next_alive_masks)
    runner.store_vector_step(b.observations, b.actions, result)
    stored = runner.replay.get_batch(np.arange(16))
    assert len(runner.replay) == 16 and np.all(stored.next_observations == 7) and stored.dones.all()
    np.testing.assert_array_equal(stored.actions, b.actions)
    result.infos[0]['executed_red_actions'][0, 0] += .1
    with pytest.raises(RuntimeError, match='executed_red_actions'):
        runner.store_vector_step(b.observations, b.actions, result)


def test_ou_reproducible_independent_streams_env_only_reset_and_state_roundtrip():
    a, b = OUNoise(3, 5, 3, 3), OUNoise(3, 5, 3, 3)
    for _ in range(4):
        np.testing.assert_array_equal(a.sample(), b.sample())
    assert not np.array_equal(a.state[0], a.state[1])
    assert not np.array_equal(a.state[0, 0], a.state[0, 1])
    assert not np.array_equal(a.state[..., 0], a.state[..., 1])
    before = a.state.copy(); a.reset([1])
    assert not a.state[1].any(); np.testing.assert_array_equal(a.state[[0, 2]], before[[0, 2]])
    next_a, next_b = a.sample(), b.sample()
    np.testing.assert_array_equal(next_a[[0, 2]], next_b[[0, 2]])
    assert not np.array_equal(next_a[1], next_b[1])
    saved = a.state_dict(); c = OUNoise(3, 5, 3, 999); c.load_state_dict(saved)
    np.testing.assert_array_equal(a.sample(), c.sample())


def test_ou_action_bounds_dead_zero_and_linear_decay():
    runner = object.__new__(MADDPGTrainingRunner)
    runner.trainer = trainer(5); _, runner.algorithm_config = configs(5)
    runner.noise = OUNoise(4, 5, 3, 5, sigma=100.)
    b = batch(); b.alive_masks[:, 1] = 0
    noisy = runner.behavior_actions(b.observations, b.alive_masks)
    assert (np.abs(noisy) <= 1).all() and not noisy[:, 1].any()
    assert not np.array_equal(noisy, runner.trainer.act(b.observations, b.alive_masks))
    assert exploration_scale(0) == 1
    assert exploration_scale(1500000) == pytest.approx(.55)
    assert exploration_scale(3000000) == pytest.approx(.1)
    assert exploration_scale(9000000) == pytest.approx(.1)


@pytest.mark.parametrize('n', [5, 8])
def test_config_formal_values_and_smoke_does_not_change_source(n):
    env, cfg = configs(n); validate_config(env, cfg)
    assert cfg['training']['replay_capacity'] == 1000000
    assert cfg['training']['minibatch_size'] == 1024
    assert cfg['training']['learning_starts'] == 10000
    assert cfg['training']['total_sampled_steps'] == 3000000
    assert cfg['training']['actor_learning_rate'] == cfg['training']['critic_learning_rate'] == 1e-4
    assert cfg['training']['tau'] == .001
    assert cfg['implementation']['ou_sigma'] == .2
    bad = deepcopy(cfg); bad['training']['policy_delay'] = 2
    with pytest.raises(RuntimeError, match='mechanism'):
        validate_config(env, bad)
    with pytest.raises(ValueError, match='70M'):
        validate_seeds(range(70000000, 70000002))


@pytest.mark.parametrize('n', [5, 8])
def test_checkpoint_roundtrip_and_strict_identity_config_architecture(tmp_path, n):
    env, cfg = configs(n); t = build_maddpg_trainer(cfg, 'cuda', seed=2)
    t.update(batch(n)); extra = checkpoint_extra(t, env, cfg)
    path = tmp_path/'maddpg.pt'; t.save(path, extra)
    state = torch.load(path, map_location='cpu', weights_only=False); validate_checkpoint(state, env, cfg)
    loaded = build_maddpg_trainer(cfg, 'cuda', seed=2); loaded.load(path, env, cfg)
    b = batch(n)
    np.testing.assert_array_equal(t.act(b.observations, b.alive_masks), loaded.act(b.observations, b.alive_masks))
    for name in ('actors', 'critics', 'target_actors', 'target_critics'):
        assert all(torch.equal(v, getattr(loaded, name).state_dict()[k]) for k, v in getattr(t, name).state_dict().items())
    assert loaded.actor_updates_per_agent == loaded.critic_updates_per_agent == [1]*n
    with pytest.raises(RuntimeError, match='exact resume unsupported'):
        loaded.resume(path)
    for identity in ('MAPPO', 'RMAPPO', 'EA-MAPPO', 'STEA-MAPPO', 'madsac'):
        bad = dict(state, algorithm=identity)
        with pytest.raises(RuntimeError, match='identity'):
            validate_checkpoint(bad, env, cfg)
    for key, value in (('environment_version', '2.3'), ('observation_dim', 52), ('action_dim', 4), ('num_agents', 4),
                       ('algorithm_config_sha256', 'wrong'), ('environment_config_sha256', 'wrong')):
        bad = dict(state, extra={**state['extra'], key: value})
        with pytest.raises(RuntimeError, match=key):
            validate_checkpoint(bad, env, cfg)
    bad = dict(state, actors=state['actors'][:-1])
    with pytest.raises(RuntimeError, match='independent'):
        validate_checkpoint(bad, env, cfg)
    bad = dict(state, implementation_version=999)
    with pytest.raises(RuntimeError, match='version'):
        validate_checkpoint(bad, env, cfg)
    bad = deepcopy(cfg); bad['training']['gamma'] = .95
    with pytest.raises(RuntimeError, match='(sha256|gamma)'):
        validate_checkpoint(state, env, bad)
    # Existing loaders must independently reject the new checkpoint identity.
    if n == 5:
        from algorithm.mappo.trainer import MAPPOTrainer
        from algorithm.madsac.trainer import MADSACTrainer
        from algorithm.rmappo.factory import build_rmappo_trainer
        from algorithm.ea_mappo.factory import build_ea_mappo_trainer
        from algorithm.stea_mappo.factory import build_stea_mappo_trainer
        for baseline in (MAPPOTrainer(device='cuda'), MADSACTrainer(device='cuda')):
            load = baseline.load if hasattr(baseline, 'load') else baseline.load_for_evaluation
            with pytest.raises(RuntimeError, match='(checkpoint|incompatible)'):
                load(path)
        for name, build in (('rmappo', build_rmappo_trainer), ('ea_mappo', build_ea_mappo_trainer), ('stea_mappo', build_stea_mappo_trainer)):
            config = yaml.safe_load((ROOT/f'configs/{name}_5v5.yaml').read_text())
            with pytest.raises(RuntimeError, match='checkpoint'):
                build(config, 'cuda').load(path)


def test_evaluation_is_reproducible_and_does_not_advance_ou(tmp_path):
    env, cfg = configs(5); env['simulation']['max_steps'] = 3
    t = build_maddpg_trainer(cfg, 'cuda'); noise = OUNoise(2, 5, 3, 5); before = deepcopy(noise.state_dict())
    a = evaluate(t, env, range(10000000, 10000002)); b = evaluate(t, env, range(10000000, 10000002))
    assert a == b and a['evaluation_episodes'] == 2
    np.testing.assert_array_equal(before['state'], noise.state)
    assert before['rng_states'] == noise.state_dict()['rng_states']
    with pytest.raises(RuntimeError, match='CUDA'):
        require_cuda('cpu')
