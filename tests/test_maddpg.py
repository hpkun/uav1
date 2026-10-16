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
    if n == 4:
        return tuple(yaml.safe_load((ROOT/'configs'/name).read_text()) for name in ('combat_environment_v338.yaml','maddpg_4v4_v338.yaml'))
    env = yaml.safe_load((ROOT/f'configs/combat_environment_v{24 if n == 8 else 25}.yaml').read_text())
    config = yaml.safe_load((ROOT/f'configs/maddpg_{n}v{n}.yaml').read_text())
    # Explicit v2 test fixture, not a reinterpretation or edit of historical YAML.
    config['training'].pop('gradient_steps_per_vector_step')
    config['training']['sampled_steps_per_update'] = 100
    return env, config


def trainer(n=5, device='cpu'):
    return MADDPGTrainer(45 if n == 4 else n*13, 3, n, device=device, seed=2)


def batch(n=5, size=4):
    rng = np.random.default_rng(8)
    dim = 45 if n == 4 else n*13
    return ReplayBatch(rng.normal(size=(size, n, dim)).astype(np.float32),
        rng.uniform(-1, 1, (size, n, 3)).astype(np.float32), rng.normal(size=(size, n)).astype(np.float32),
        rng.normal(size=(size, n, dim)).astype(np.float32), np.zeros(size, bool),
        np.ones((size, n), np.float32), np.ones((size, n), np.float32))


def checkpoint_extra(t, env, config):
    from algorithm.maddpg.protocol import schedule_contract
    if t.actor_update_count:
        steps = config['training']['learning_starts']+t.actor_update_count*config['training']['sampled_steps_per_update']
        t.sampled_steps = ((steps+15)//16)*16; t.vector_steps = t.sampled_steps//16
    return {'environment_version': env['environment_version'], 'environment_config_sha256': config_sha256(env),
        'algorithm_config': deepcopy(config), **schedule_contract(config),
        'transitions_since_update': max(0,t.sampled_steps-config['training']['learning_starts'])%config['training']['sampled_steps_per_update'],
        'algorithm_config_sha256': config_sha256(config), **{k: getattr(t, k) for k in ('observation_dim', 'action_dim', 'num_agents', 'gamma', 'tau')},
        'training_seed': t.seed, 'training_num_envs': 16, 'training_total_sampled_steps': 3000000, 'training_smoke': False,
        'network_architecture': t.network_architecture, 'parameter_counts': t.parameter_counts(),
        'ou_noise_state': OUNoise(16, t.num_agents, 3, 4).state_dict()}


@pytest.mark.parametrize('n,actor_count,critic_count', [(4, 78339, 115457), (5, 83459, 153345), (8, 93443, 285441)])
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
    dim = 45 if n == 4 else n*13
    assert [m.in_features for m in t.actors[0].network if isinstance(m, nn.Linear)] == [dim, 256, 256]
    assert [m.out_features for m in t.critics[0].network if isinstance(m, nn.Linear)] == [256, 256, 1]
    assert t.critics[0].joint_input_dim == n*(dim+3)
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


@pytest.mark.parametrize('n', [4, 5, 8])
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


@pytest.mark.parametrize('n', [4, 5, 8])
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


@pytest.mark.parametrize('n', [4, 5, 8])
def test_checkpoint_roundtrip_and_strict_identity_config_architecture(tmp_path, n):
    env, cfg = configs(n); t = build_maddpg_trainer(cfg, 'cuda', seed=2)
    t.update(batch(n)); extra = checkpoint_extra(t, env, cfg)
    path = tmp_path/'maddpg.pt'; t.save(path, extra)
    state = torch.load(path, map_location='cuda', weights_only=False); validate_checkpoint(state, env, cfg)
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
    for key, value in (('environment_version', '2.3'), ('observation_dim', 52), ('action_dim', 4), ('num_agents', n+1),
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


def test_v338_shapes_joint_192_and_own_actor_parameter_isolation():
    t=trainer(4);b=batch(4);data=t.tensors(b)
    assert t.actors[0](data['observations'][:,0]).shape==(4,3)
    assert t.critics[0].joint_input_dim==192
    assert t.critics[0](data['observations'],data['actions'],data['alive_masks']).shape==(4,)
    before=[deepcopy(a.state_dict()) for a in t.actors]
    critic_before=deepcopy(t.critics.state_dict())
    t.update_actor(1,data)
    for i,a in enumerate(t.actors):
        changed=any(not torch.equal(v,a.state_dict()[k]) for k,v in before[i].items())
        assert changed==(i==1)
    assert all(torch.equal(v,t.critics.state_dict()[k]) for k,v in critic_before.items())


def test_v338_each_critic_step_changes_only_its_own_parameters(monkeypatch):
    t=trainer(4)
    for i,opt in enumerate(t.critic_optimizers):
        original=opt.step
        def isolated_step(*args,_i=i,_step=original,**kwargs):
            before=[deepcopy(c.state_dict()) for c in t.critics]
            _step(*args,**kwargs)
            for j,c in enumerate(t.critics):
                changed=any(not torch.equal(v,c.state_dict()[k]) for k,v in before[j].items())
                assert changed==(j==_i)
        monkeypatch.setattr(opt,'step',isolated_step)
    t.update(batch(4))


def test_targets_use_all_target_actors_and_individual_target_critics():
    t=trainer(4);data=t.tensors(batch(4))
    class ConstantActor(nn.Module):
        def __init__(self,value):super().__init__();self.value=value
        def forward(self,x):return x.new_full((*x.shape[:-1],3),self.value)
    class CheckCritic(nn.Module):
        def __init__(self,value):super().__init__();self.value=value
        def forward(self,obs,actions,alive):
            assert torch.allclose(actions[:, :, 0],torch.tensor([.1,.2,.3,.4]).expand(4,4))
            return obs.new_full((len(obs),),self.value)
    t.target_actors=nn.ModuleList([ConstantActor(v) for v in [.1,.2,.3,.4]])
    t.target_critics=nn.ModuleList([CheckCritic(v) for v in [1.,2.,3.,4.]])
    target=t.targets(data)
    assert torch.allclose(target,data['rewards']+.99*torch.arange(1,5))


def test_v338_dead_slot_and_death_done_targets():
    t=trainer(4);d=t.tensors(batch(4));d['alive_masks'][:,2]=0
    baseline=t.critics[0](d['observations'],d['actions'],d['alive_masks'])
    d['observations'][:,2]=float('nan');d['actions'][:,2]=float('inf')
    assert torch.equal(baseline,t.critics[0](d['observations'],d['actions'],d['alive_masks']))
    d['next_alive_masks'][:,2]=0;d['dones'][0]=1
    target=t.targets(d)
    assert torch.equal(target[:,2],d['rewards'][:,2])
    assert torch.equal(target[0],d['rewards'][0])


def test_schedule_remainder_warmup_crossing_and_long_term():
    from algorithm.maddpg.schedule import SampledTransitionSchedule
    s=SampledTransitionSchedule(10000,100);events=0
    for previous in range(0,10000,16):s.add(previous,previous+16)
    assert s.credit==0
    for previous in range(10000,10096,16):s.add(previous,previous+16)
    assert s.credit==96
    s.add(10096,10112);s.consume();events+=1
    assert s.credit==12
    for previous in range(10112,10208,16):s.add(previous,previous+16)
    s.consume();events+=1;assert s.credit==8
    for previous in range(10208,3000000,16):
        s.add(previous,previous+16)
        while s.credit>=100:s.consume();events+=1
    assert events==29900 and s.credit==0
    s=SampledTransitionSchedule(10000,100);s.add(9992,10008);assert s.credit==8
    s=SampledTransitionSchedule(10000,100);s.add(0,10352)
    for _ in range(3):s.consume()
    assert s.credit==52


def test_runner_schedule_fresh_batch_per_event_no_warmup_updates(tmp_path):
    from algorithm.maddpg.schedule import SampledTransitionSchedule
    from types import SimpleNamespace
    r=object.__new__(MADDPGTrainingRunner);_,r.algorithm_config=configs(4);r.output_dir=tmp_path
    r.schedule=SampledTransitionSchedule(10000,100);r.last_metrics={}
    class Replay:
        def __len__(self):return 1024
        def sample(self,n):assert n==1024;return object()
    samples=[]
    def update(b):
        samples.append(b);r.trainer.actor_updates_per_agent=[len(samples)]*4;r.trainer.critic_updates_per_agent=[len(samples)]*4
        return dict(actor_update_count=len(samples),critic_update_count=len(samples))
    r.replay=Replay();r.trainer=SimpleNamespace(sampled_steps=9992,vector_steps=624,update=update)
    r.schedule.add(0,9992);r.update_from_credit();assert not samples
    r.schedule.add(9992,10008);r.update_from_credit();assert not samples and r.schedule.credit==8
    r.trainer.sampled_steps=10352;r.trainer.vector_steps=647;r.schedule.add(10008,10352);r.update_from_credit()
    assert len(samples)==len({id(b) for b in samples})==3 and r.schedule.credit==52
    records=[__import__('json').loads(line) for line in (tmp_path/'optimization_metrics.jsonl').read_text().splitlines()]
    assert len(records)==3 and [x['transitions_since_update'] for x in records]==[252,152,52]


@pytest.mark.parametrize('section,key,value',[
    ('network','observation_dim',46),('network','num_agents',5),('network','action_dim',4),
    ('network','actor_hidden_layers',[128,128]),('network','critic_hidden_layers',[128,128]),
    ('network','attention_heads',4),('training','alpha',.1),('training','policy_delay',2),
    ('implementation','log_std',0),('implementation','evaluation_mode','stochastic'),
    ('training','gradient_steps_per_vector_step',1)])
def test_v338_protocol_rejects_incompatible_config(section,key,value):
    e,c=configs(4);c[section][key]=value
    with pytest.raises((RuntimeError,ValueError)):validate_config(e,c)


def test_v338_identity_is_not_just_dimensions_and_legacy_schedule_is_explicit():
    e,c=configs(4);validate_config(e,c)
    e337=yaml.safe_load((ROOT/'configs/combat_environment_v337.yaml').read_text())
    with pytest.raises(RuntimeError):validate_config(e337,c)
    for identity in [None,{'environment_version':'3.37','team_size':4},{'environment_version':'3.38','team_size':5}]:
        bad=deepcopy(c)
        if identity is None:bad.pop('environment')
        else:bad['environment']=identity
        with pytest.raises(RuntimeError,match='identity'):validate_config(e,bad)
    from algorithm.maddpg.protocol import schedule_contract
    old=yaml.safe_load((ROOT/'configs/maddpg_5v5.yaml').read_text())
    with pytest.raises(RuntimeError,match='legacy'):schedule_contract(old)


@pytest.mark.parametrize('key,value',[
    ('sampled_steps_per_update',112),('learning_starts',9999),('sampled_steps_unit','vector_steps'),
    ('transitions_since_update',-1),('evaluation_mode','stochastic'),('replay_buffer_in_checkpoint',True)])
def test_v2_checkpoint_rejects_wrong_schedule_metadata(key,value):
    e,c=configs(4);t=trainer(4);state=t.checkpoint_state(checkpoint_extra(t,e,c))
    state['extra'][key]=value
    with pytest.raises(RuntimeError):validate_checkpoint(state,e,c)


def test_v1_checkpoint_explicitly_rejected_before_loading(tmp_path):
    e,c=configs(4);t=trainer(4,'cuda');state=t.checkpoint_state(checkpoint_extra(t,e,c));state['implementation_version']=1
    path=tmp_path/'v1.pt';torch.save(state,path)
    with pytest.raises(RuntimeError,match='implementation version'):t.load(path,e,c)
    for name in ['actors','critics','target_actors','target_critics']:
        bad=t.checkpoint_state(checkpoint_extra(t,e,c));bad[name]=bad[name][:-1]
        with pytest.raises(RuntimeError,match='independent'):validate_checkpoint(bad,e,c)


def test_compact_v2_console_formats_include_required_fields_and_no_missing_literals():
    from types import SimpleNamespace
    r=object.__new__(MADDPGTrainingRunner);r.env_config,r.algorithm_config=configs(4)
    r.seed=1;r.device='cuda';r.num_envs=16;r.total_sampled_steps=3000000
    r.trainer=SimpleNamespace(sampled_steps=1200000);r.completed_records=[];r.last_metrics={}
    start=r.startup_log_line();line=r.training_log_line()
    for token in ['MADDPG-v2','env=v3.38','4v4','seed=1','cuda','envs=16','3.0M','update_every=100','deterministic']:assert token in start
    for token in ['40.0%','ep=','win=','Rloss=','Bloss=','fire=','kill=','timeout=','len=','R=','Qloss=','Aloss=','OU=']:assert token in line
    assert 'N/A' in line and 'None' not in line and 'nan' not in line
    ev=dict(sampled_steps=1200000,win_rate=.55,average_red_loss=2.2,average_blue_loss=3.1,
        red_attempt_episode_rate=.95,red_kill_episode_rate=.9,timeout_rate=0,episode_length=398,average_return=12.44)
    for best in [True,False]:
        log=r.evaluation_log_line(ev,best);assert log.startswith('[EVAL*]' if best else '[EVAL]')
        for token in ['deterministic','win=','Rloss=','Bloss=','fire=','kill=','timeout=','len=','R=']:assert token in log
        assert 'None' not in log and 'nan' not in log
