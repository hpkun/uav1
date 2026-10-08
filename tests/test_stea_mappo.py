"""Spatial masking, recurrent lifecycle, sequence PPO and independent CUDA protocol."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from algorithm.mappo.networks import CentralizedValueCritic
from algorithm.mappo.trainer import MAPPOTrainer
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation
from algorithm.stea_mappo.networks import SpatioTemporalEntityAttentionActor, decompose_observations
from algorithm.stea_mappo.factory import build_stea_mappo_trainer, validate_config
from algorithm.stea_mappo.trainer import RecurrentRolloutBatch, sequence_chunks, pack_sequences
from algorithm.stea_mappo.runner import STEAMAPPOTrainingRunner
from algorithm.stea_mappo.evaluation import evaluate_stea_mappo_checkpoint
from algorithm.stea_mappo.protocol import validate_checkpoint, require_cuda
from tools.aggregate_holdout_results import aggregate_holdout_results

ROOT = Path(__file__).resolve().parents[1]


def configs(small=False):
    env = yaml.safe_load((ROOT/'configs/combat_environment.yaml').read_text())
    cfg = yaml.safe_load((ROOT/'configs/stea_mappo.yaml').read_text())
    if small:
        cfg['network'].update(entity_dim=16,spatial_hidden_dim=32,gru_hidden_dim=32,
            recurrent_sequence_length=4,critic_hidden_layers=[32,32])
        cfg['training'].update(rollout_steps=8,minibatch_size=8,ppo_epochs=2,
            evaluation_episodes=2,evaluation_interval_sampled_steps=8)
        cfg['implementation']['checkpoint_interval_sampled_steps'] = 8
        env['simulation']['max_steps'] = 2
    return env,cfg


def observations(*shape):
    obs = torch.randn(*shape,52)
    obs[...,7:28].reshape(*shape,3,7)[...,-1] = 1
    obs[...,28:52].reshape(*shape,4,6)[...,-1] = 1
    return obs


def actor():
    torch.manual_seed(27)
    return SpatioTemporalEntityAttentionActor()


def test_observation_decomposition_exact():
    x = torch.arange(52).float().expand(2,3,4,52)
    own,allies,enemies = decompose_observations(x)
    assert torch.equal(own,x[...,:7])
    assert torch.equal(allies,x[...,7:28].reshape(2,3,4,3,7))
    assert torch.equal(enemies,x[...,28:52].reshape(2,3,4,4,6))
    with pytest.raises(ValueError,match='52'):
        decompose_observations(torch.zeros(51))


@pytest.mark.parametrize('shape',[(4,),(2,4),(2,5,4)])
def test_encoder_shapes(shape):
    embeddings = actor().encode_entities(observations(*shape))
    assert [x.shape for x in embeddings] == [(*shape,64),(*shape,3,64),(*shape,4,64)]


@pytest.mark.parametrize('side,start,end,slots,width',[
    ('ally',7,28,3,7),('enemy',28,52,4,6)])
def test_dead_entities_have_exact_zero_weights_and_empty_context(side,start,end,slots,width):
    model = actor()
    obs = observations(2,4)
    entity = obs[...,start:end].reshape(2,4,slots,width)
    entity[...,0,-1] = 0
    _,diag = model.spatial(obs,torch.ones(2,4))
    weights = diag[f'{side}_attention_weights']
    assert torch.count_nonzero(weights[...,0]) == 0
    assert torch.allclose(weights.sum(-1),torch.ones(2,4,2))
    entity[...,-1] = 0
    _,diag = model.spatial(obs,torch.ones(2,4))
    assert torch.count_nonzero(diag[f'{side}_attention_weights']) == 0
    assert torch.count_nonzero(diag[f'{side}_context']) == 0
    assert torch.isfinite(diag[f'{side}_context']).all()


def test_dead_slot_features_cannot_affect_context_and_entity_permutation_is_equivariant():
    model = actor()
    obs = observations(2,4)
    allies = obs[...,7:28].reshape(2,4,3,7)
    allies[...,0,-1] = 0
    spatial,diag = model.spatial(obs,torch.ones(2,4))
    changed = obs.clone()
    changed[...,7:13] = 1000
    spatial_changed,_ = model.spatial(changed,torch.ones(2,4))
    assert torch.equal(spatial,spatial_changed)
    perm = [2,0,1]
    changed[...,7:28] = allies[...,perm,:].reshape(2,4,21)
    spatial_permuted,diag_permuted = model.spatial(changed,torch.ones(2,4))
    assert torch.allclose(spatial,spatial_permuted,atol=1e-7)
    assert torch.allclose(diag['ally_attention_weights'][...,perm],diag_permuted['ally_attention_weights'])
    assert model.ally_attention.wq.weight.data_ptr() != model.enemy_attention.wq.weight.data_ptr()


@pytest.mark.parametrize('envs',[None,3])
def test_actor_shapes_death_reset_and_episode_reset(envs):
    model = actor()
    shape = (4,) if envs is None else (envs,4)
    obs = observations(*shape)
    hidden = torch.randn(*shape,128)
    alive = torch.ones(*shape)
    start = torch.tensor(0.) if envs is None else torch.zeros(envs)
    actions,next_hidden = model(obs,hidden,alive,start)
    assert actions.shape == (*shape,3) and next_hidden.shape == (*shape,128)
    alive[...,1] = 0
    actions,dead_hidden = model(obs,next_hidden,alive,start)
    assert torch.count_nonzero(actions[...,1,:]) == torch.count_nonzero(dead_hidden[...,1,:]) == 0
    reset_actions,reset_hidden = model(obs,hidden,alive,torch.ones_like(start))
    zero_actions,zero_hidden = model(obs,torch.zeros_like(hidden),alive,start)
    assert torch.equal(reset_actions,zero_actions)
    assert torch.equal(reset_hidden,zero_hidden)


def test_temporal_memory_changes_current_policy_and_no_future_leakage():
    model = actor()
    obs = observations(2,4)
    d0,_,_ = model.distribution_step(obs,torch.zeros(2,4,128),torch.ones(2,4),torch.zeros(2))
    d1,_,_ = model.distribution_step(obs,torch.ones(2,4,128),torch.ones(2,4),torch.zeros(2))
    assert not torch.allclose(d0.mean,d1.mean)
    seq = observations(2,5,4)
    alive,starts = torch.ones(2,5,4),torch.zeros(2,5)
    first,_,_ = model.distribution_sequence(seq,torch.zeros(2,4,128),alive,starts)
    seq[:,3:] += 3
    second,_,_ = model.distribution_sequence(seq,torch.zeros(2,4,128),alive,starts)
    assert torch.equal(first.mean[:,:3],second.mean[:,:3])


def test_sequence_middle_episode_reset_and_death_log_prob_shapes():
    model = actor()
    obs = observations(2,6,4)
    alive,starts = torch.ones(2,6,4),torch.zeros(2,6)
    starts[:,3] = 1
    alive[:,2:5,1] = 0
    distribution,hidden,_ = model.distribution_sequence(obs,torch.randn(2,4,128),alive,starts)
    direct,reset_hidden,_ = model.distribution_step(obs[:,3],torch.zeros(2,4,128),alive[:,3],torch.zeros(2))
    assert torch.equal(distribution.mean[:,3],direct.mean)
    assert torch.equal(hidden[:,3],reset_hidden)
    assert torch.count_nonzero(hidden[:,2:5,1]) == 0
    raw = distribution.rsample()
    log_prob,entropy,*_ = model.evaluate_sequence(obs,raw.tanh(),raw,torch.zeros(2,4,128),alive,starts)
    assert log_prob.shape == entropy.shape == (2,6,4)
    assert torch.isfinite(log_prob).all() and torch.isfinite(entropy).all()
    assert torch.count_nonzero(log_prob[:,2:5,1]) == 0


def test_bptt_carries_history_and_cuts_gradient_at_episode_boundary():
    model = actor()
    obs = observations(1,5,4).requires_grad_()
    alive,starts = torch.ones(1,5,4),torch.zeros(1,5)
    distribution,_,_ = model.distribution_sequence(obs,torch.zeros(1,4,128),alive,starts)
    gradient = torch.autograd.grad(distribution.mean[:,4].sum(),obs)[0]
    assert torch.count_nonzero(gradient[:,0]) > 0
    starts[:,3] = 1
    distribution,_,_ = model.distribution_sequence(obs,torch.zeros(1,4,128),alive,starts)
    gradient = torch.autograd.grad(distribution.mean[:,4].sum(),obs)[0]
    assert torch.count_nonzero(gradient[:,:3]) == 0
    assert torch.count_nonzero(gradient[:,3:]) > 0


def synthetic_rollout(trainer,t=7,e=2):
    rng = np.random.default_rng(25)
    obs = observations(t,e,4).numpy()
    alive = np.ones((t,e,4),dtype=np.float32)
    alive[1:3,0,1] = 0
    starts = np.zeros((t,e),dtype=np.float32)
    starts[0] = 1; starts[3,0] = 1; starts[4,1] = 1
    hidden = np.zeros((e,4,trainer.actor.gru_hidden_dim),dtype=np.float32)
    actions,raw,logs,states = [],[],[],[]
    for step in range(t):
        states.append(hidden.copy())
        a,r,l,hidden = trainer.act(obs[step],alive[step],hidden,starts[step])
        actions.append(a); raw.append(r); logs.append(l)
        if step+1 < t:
            hidden *= alive[step+1,...,None]
            hidden[starts[step+1]>.5] = 0
    dones = np.zeros_like(starts)
    dones[:-1] = starts[1:]
    next_alive = np.concatenate((alive[1:],alive[-1:]),axis=0)
    return RecurrentRolloutBatch(obs,np.stack(actions),np.stack(raw),np.stack(logs),
        rng.normal(size=(t,e,4)).astype(np.float32),dones,alive,obs+.01,next_alive,
        np.stack(states),starts)


def test_chunk_continuity_padding_and_transition_coverage():
    chunks = sequence_chunks(7,2,4)
    assert chunks == [(0,0,4),(0,4,7),(1,0,4),(1,4,7)]
    tensor = torch.arange(14).reshape(7,2)
    packed = pack_sequences(tensor,chunks,4,pad=-1)
    assert packed.tolist() == [[0,2,4,6],[8,10,12,-1],[1,3,5,7],[9,11,13,-1]]
    assert sorted(packed[packed>=0].tolist()) == list(range(14))


def test_pre_update_ratio_audit_recurrent_gradients_and_exact_baseline_critic():
    assert torch.cuda.is_available(), 'CUDA mandatory'
    _,cfg = configs(True)
    trainer = build_stea_mappo_trainer(cfg,'cuda',seed=27)
    baseline = MAPPOTrainer(hidden_dim=32,device='cuda',seed=27)
    assert next(trainer.actor.parameters()).is_cuda and next(trainer.critic.parameters()).is_cuda
    assert type(trainer.critic) is type(baseline.critic) is CentralizedValueCritic
    for key,value in trainer.critic.state_dict().items():
        assert torch.equal(value,baseline.critic.state_dict()[key])
    rollout = synthetic_rollout(trainer)
    audit = trainer.audit_rollout_ratio(rollout)
    assert audit['pre_update_ratio_max_abs_error'] < 1e-5
    critic_before = copy.deepcopy(trainer.critic.state_dict())
    actor_before = copy.deepcopy(trainer.actor.state_dict())
    metrics = trainer.update(rollout)
    assert metrics['valid_environment_transitions'] == 14
    assert metrics['padded_environment_transitions'] == 2
    assert metrics['sequence_chunks'] == 4
    assert metrics['first_minibatch_ratio_mean'] == pytest.approx(1,abs=1e-5)
    assert metrics['first_minibatch_approx_kl'] == pytest.approx(0,abs=1e-7)
    assert all(np.isfinite(list(metrics.values())))
    assert trainer.actor_update_count == trainer.critic_update_count == 4
    for name in ('self_encoder','ally_encoder','enemy_encoder','ally_attention','enemy_attention','spatial_fusion','gru','actor_head','mean','log_std'):
        parameters = list(getattr(trainer.actor,name).parameters())
        assert all(p.grad is not None and p.grad.is_cuda and torch.isfinite(p.grad).all() for p in parameters)
        assert any(torch.count_nonzero(p.grad) for p in parameters)
        assert any(not torch.equal(actor_before[key],value) for key,value in trainer.actor.state_dict().items() if key.startswith(name+'.'))
    assert any(not torch.equal(value,critic_before[key]) for key,value in trainer.critic.state_dict().items())
    corrupted = copy.deepcopy(rollout); corrupted.old_log_probs += .1
    with pytest.raises(RuntimeError,match='ratio mismatch'):
        trainer.audit_rollout_ratio(corrupted)


def test_saturated_raw_actions_preserve_log_probability():
    _,cfg = configs(True)
    trainer = build_stea_mappo_trainer(cfg,'cuda')
    with torch.no_grad():
        trainer.actor.mean.bias.fill_(20.)
    rollout = synthetic_rollout(trainer)
    assert (abs(rollout.actions[rollout.alive_masks>.5])>.999).mean() > .99
    assert trainer.audit_rollout_ratio(rollout)['pre_update_ratio_max_abs_error'] < 1e-5


@pytest.mark.parametrize('field,value',[('recurrent_sequence_length',3),('gru_layers',2),
    ('entity_attention_heads',3),('actor_type','mlp'),('critic_type','unsupported')])
def test_invalid_config_is_rejected(field,value):
    _,cfg = configs()
    cfg['network'][field] = value
    with pytest.raises(ValueError): validate_config(cfg)


def test_formal_hyperparameters_match_baseline_except_explicit_2m_target():
    _,cfg = configs()
    baseline = yaml.safe_load((ROOT/'configs/mappo.yaml').read_text())
    assert {key:value for key,value in cfg['training'].items() if key != 'total_sampled_steps'} == {
        key:value for key,value in baseline['training'].items() if key != 'total_sampled_steps'}
    assert cfg['implementation'] == baseline['implementation']
    assert cfg['training']['total_sampled_steps'] == 2_000_000
    assert cfg['network']['critic_hidden_layers'] == baseline['network']['actor_hidden_layers']


def test_runner_immediate_death_and_episode_reset(tmp_path,monkeypatch):
    env,cfg = configs(True)
    runner = STEAMAPPOTrainingRunner(env,cfg,num_envs=2,total_sampled_steps=12,
        device='cuda',output_dir=tmp_path)
    try:
        result = runner.vector.step_batch(np.zeros((2,4,3),dtype=np.float32))
        result.next_alive_masks[0,1] = 0
        result.terminated[1] = True
        monkeypatch.setattr(runner.vector,'step_batch',lambda actions:result)
        runner.collect_rollout(1)
        assert np.count_nonzero(runner.actor_hidden_states[0,1]) == 0
        assert np.count_nonzero(runner.actor_hidden_states[1]) == 0
        assert np.any(runner.actor_hidden_states[0,0])
        assert np.array_equal(runner.episode_start_masks,[0,1])
    finally:
        runner.vector.close()


def test_cuda_parallel_rollout_smoke_checkpoint_load_evaluation_and_resume(tmp_path):
    env,cfg = configs(True)
    runner = STEAMAPPOTrainingRunner(env,cfg,num_envs=2,total_sampled_steps=12,
        device='cuda',seed=11,output_dir=tmp_path)
    assert runner.vector.num_workers == 2
    assert len(set(runner.vector.worker_pids)) == 2
    rollout = runner.collect_rollout(3)
    assert rollout.actor_hidden_states.shape == (3,2,4,32)
    assert np.count_nonzero(rollout.actor_hidden_states[0]) == 0
    assert np.count_nonzero(rollout.actor_hidden_states[2]) == 0
    assert np.all(rollout.episode_starts[2] == 1)
    assert runner.trainer.audit_rollout_ratio(rollout)['pre_update_ratio_max_abs_error'] < 1e-5
    summary = runner.run()
    assert summary['sampled_steps'] == 12
    assert summary['actor_updates'] == summary['critic_updates'] > 0
    path = tmp_path/'latest.pt'
    assert (tmp_path/'best_eval.pt').is_file()
    state = torch.load(path,map_location='cpu',weights_only=False)
    validate_checkpoint(state,env,cfg)
    assert state['algorithm'] == 'STEA-MAPPO' and 'mappo_impl_version' not in state
    first = evaluate_stea_mappo_checkpoint(path,cfg,env,'cuda',range(10_000_000,10_000_002))
    second = evaluate_stea_mappo_checkpoint(path,cfg,env,'cuda',range(10_000_000,10_000_002))
    assert first == second
    assert first['protocol_complete'] is True
    assert first['checkpoint_training_seed'] == 11
    assert np.isfinite(first['gru_hidden_norm_mean'])
    baseline_cfg = yaml.safe_load((ROOT/'configs/mappo.yaml').read_text())
    with pytest.raises(RuntimeError,match='MAPPO'):
        validate_checkpoint_for_evaluation(state,env,baseline_cfg)
    with pytest.raises(RuntimeError,match='not a MAPPO'):
        MAPPOTrainer(hidden_dim=32,device='cuda').load(path)
    baseline_path = tmp_path/'baseline.pt'
    MAPPOTrainer(hidden_dim=32,device='cuda').save(baseline_path)
    with pytest.raises(RuntimeError,match='not a STEA'):
        runner.trainer.load(baseline_path)
    with pytest.raises(RuntimeError,match='not a STEA'):
        evaluate_stea_mappo_checkpoint(baseline_path,cfg,env,'cuda',[10_000_000])
    resumed = STEAMAPPOTrainingRunner(env,cfg,num_envs=2,total_sampled_steps=16,
        device='cuda',seed=11,output_dir=tmp_path)
    try:
        resumed.actor_hidden_states.fill(1)
        resumed.resume(path)
        assert resumed.trainer.sampled_steps == 12
        assert np.count_nonzero(resumed.actor_hidden_states) == 0
        assert np.all(resumed.episode_start_masks == 1)
        assert resumed.trainer.audit_rollout_ratio(resumed.collect_rollout(1))['pre_update_ratio_max_abs_error'] < 1e-5
    finally:
        resumed.vector.close()


@pytest.mark.parametrize('field', ['actor_type','entity_dim','entity_attention_heads','spatial_hidden_dim',
    'gru_hidden_dim','gru_layers','recurrent_sequence_length','critic_type','critic_hidden_dim','critic_attention_heads'])
def test_each_checkpoint_architecture_field_is_strict(tmp_path,field):
    env,cfg = configs(True)
    runner = STEAMAPPOTrainingRunner(env,cfg,num_envs=1,total_sampled_steps=4,device='cuda',output_dir=tmp_path)
    try:
        runner.save_checkpoint(tmp_path/'checkpoint.pt')
        state = torch.load(tmp_path/'checkpoint.pt',map_location='cpu',weights_only=False)
        state['network_architecture'][field] = 'corrupted'
        with pytest.raises(RuntimeError,match='architecture'):
            validate_checkpoint(state,env,cfg)
    finally:
        runner.vector.close()


def test_checkpoint_fingerprint_and_version_reject(tmp_path):
    env,cfg = configs(True)
    runner = STEAMAPPOTrainingRunner(env,cfg,num_envs=1,total_sampled_steps=4,device='cuda',output_dir=tmp_path)
    try:
        runner.save_checkpoint(tmp_path/'checkpoint.pt')
        state = torch.load(tmp_path/'checkpoint.pt',map_location='cpu',weights_only=False)
        for key,value in [('environment_config_sha256','bad'),('algorithm_config_sha256','bad'),
                          ('stea_mappo_impl_version',0),('observation_dim',53),('actor_parameter_count',1)]:
            bad = copy.deepcopy(state); bad['extra'][key] = value
            with pytest.raises(RuntimeError): validate_checkpoint(bad,env,cfg)
        changed = copy.deepcopy(state); changed['stea_mappo_impl_version'] = 0
        with pytest.raises(RuntimeError,match='version'): validate_checkpoint(changed,env,cfg)
        changed = copy.deepcopy(state); changed['critic_type'] = 'mlp'
        torch.save(changed,tmp_path/'invalid.pt')
        with pytest.raises(RuntimeError,match='critic_type'): runner.trainer.load(tmp_path/'invalid.pt')
        changed = copy.deepcopy(state); changed['extra']['network_architecture']['gru_hidden_dim'] = 64
        torch.save(changed,tmp_path/'invalid.pt')
        with pytest.raises(RuntimeError,match='architecture'): runner.trainer.load(tmp_path/'invalid.pt')
    finally:
        runner.vector.close()


def test_holdout_aggregation_separates_architectures_versions_and_algorithms(tmp_path):
    base = dict(algorithm='STEA-MAPPO',protocol_complete=True,checkpoint_training_seed=1,
        stea_mappo_impl_version=1,network_architecture={'gru_hidden_dim':128},average_return=4.)
    a,b = tmp_path/'a.json',tmp_path/'b.json'
    a.write_text(json.dumps(base))
    other = {**base,'checkpoint_training_seed':2,'average_return':6.}
    b.write_text(json.dumps(other))
    summary,_ = aggregate_holdout_results([a,b],tmp_path/'aggregate')
    assert summary['metrics']['average_return']['mean'] == 5
    for key,value in [('algorithm','MAPPO'),('stea_mappo_impl_version',2),('network_architecture',{'gru_hidden_dim':64})]:
        b.write_text(json.dumps({**other,key:value}))
        with pytest.raises(RuntimeError,match='protocol mismatch'):
            aggregate_holdout_results([a,b],tmp_path/'aggregate')


def test_cuda_enforcement_never_falls_back(monkeypatch):
    with pytest.raises(RuntimeError,match='CUDA is mandatory'): require_cuda('cpu')
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA is mandatory'): require_cuda('cuda')
