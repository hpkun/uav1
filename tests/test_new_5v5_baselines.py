"""Fairness, recurrence, spatial parity and independent checkpoint contracts."""
import copy
import inspect
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
import yaml

from algorithm.mappo.factory import build_mappo_trainer
from algorithm.mappo.networks import CentralizedMLPCritic
from algorithm.mappo.trainer import MAPPOTrainer, RolloutBatch
from algorithm.stea_mappo.factory import build_stea_mappo_trainer
from algorithm.stea_mappo.protocol import validate_checkpoint as validate_stea
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation as validate_mappo
from algorithm.common.protocol import config_sha256
from algorithm.rmappo.factory import build_rmappo_trainer
from algorithm.rmappo.trainer import RecurrentRolloutBatch, sequence_chunks, pack_sequences
from algorithm.rmappo.protocol import validate_checkpoint as validate_r
from algorithm.ea_mappo.factory import build_ea_mappo_trainer
from algorithm.ea_mappo.protocol import validate_checkpoint as validate_e

ROOT = Path(__file__).resolve().parents[1]
BUILD = {'mappo':build_mappo_trainer,'stea_mappo':build_stea_mappo_trainer,
         'rmappo':build_rmappo_trainer,'ea_mappo':build_ea_mappo_trainer}


def config(name):
    return yaml.safe_load((ROOT/f'configs/{name}_5v5.yaml').read_text())


def build(name, device='cpu', seed=27):
    cfg = config(name)
    if name == 'mappo':
        # Existing factory has no seed override; direct construction has identical kwargs.
        from algorithm.common.control_factory import trainer_kwargs
        return MAPPOTrainer(**trainer_kwargs(cfg,device,seed))
    return BUILD[name](cfg,device,seed=seed)


def observations(*shape):
    obs = torch.randn(*shape,65)
    obs[...,7:35].reshape(*shape,4,7)[...,-1] = 1
    obs[...,35:].reshape(*shape,5,6)[...,-1] = 1
    return obs


@pytest.mark.parametrize('seed',[0,3])
def test_four_critics_have_identical_class_keys_shapes_counts_and_weights(seed):
    trainers = [build(name,seed=seed) for name in BUILD]
    reference = trainers[0].critic.state_dict()
    for trainer in trainers:
        assert type(trainer.critic) is CentralizedMLPCritic
        assert sum(p.numel() for p in trainer.critic.parameters()) == 166145
        assert trainer.critic.value_network[0].in_features == 390
        assert trainer.critic.state_dict().keys() == reference.keys()
        for key,value in trainer.critic.state_dict().items():
            assert value.shape == reference[key].shape
            assert torch.equal(value,reference[key])


def test_four_actor_component_contracts_and_identical_temporal_core():
    m,r,e,s = (build(name).actor for name in ('mappo','rmappo','ea_mappo','stea_mappo'))
    assert not any(isinstance(mod,(nn.GRU,nn.LSTM)) for mod in m.modules())
    assert not any('attention' in key for key in m.state_dict())
    assert not any('attention' in key or 'entity' in key or key.startswith(('self_encoder','ally_encoder','enemy_encoder')) for key in r.state_dict())
    assert r.flat_encoder[0].in_features == 65 and r.flat_encoder[0].out_features == 128
    assert len(r.flat_encoder) == 2
    for actor in (r,s):
        assert actor.gru.input_size == actor.gru.hidden_size == 128
        assert actor.gru.num_layers == 1 and actor.gru.batch_first
        assert actor.actor_head[0].in_features == actor.actor_head[0].out_features == 128
        assert actor.mean.in_features == 128 and actor.mean.out_features == 3
    assert not any(isinstance(mod,(nn.GRU,nn.LSTM)) for mod in e.modules())
    assert not hasattr(e,'distribution_step') and not hasattr(e,'distribution_sequence')
    assert tuple(inspect.signature(e.forward).parameters) == ('observations',)
    assert hasattr(e,'ally_attention') and hasattr(e,'enemy_attention')
    assert hasattr(s,'ally_attention') and hasattr(s,'gru')
    assert not hasattr(build('ea_mappo'),'sequence_length')


@pytest.mark.parametrize('name',['rmappo','ea_mappo'])
def test_formal_config_gaussian_ppo_and_three_million_default(name):
    cfg = config(name); trainer=build(name); actor=trainer.actor
    t,i = cfg['training'],cfg['implementation']
    assert t['total_sampled_steps'] == 3000000
    assert (t['rollout_steps'],t['minibatch_size'],t['num_train_envs'],t['ppo_epochs']) == (256,512,16,10)
    assert isinstance(trainer.actor_optimizer,torch.optim.Adam)
    assert isinstance(trainer.critic_optimizer,torch.optim.Adam)
    assert trainer.actor_optimizer.param_groups[0]['lr'] == trainer.critic_optimizer.param_groups[0]['lr'] == 3e-4
    assert (trainer.gamma,trainer.gae_lambda,trainer.clip_ratio,trainer.value_loss_coefficient,
            trainer.entropy_coefficient,trainer.target_kl,trainer.max_grad_norm) == (.99,.95,.2,.5,.001,.015,.5)
    assert trainer.normalize_advantages and trainer.clip_value_loss
    assert actor.log_std_parameter.shape == (3,) and not hasattr(actor,'log_std')
    assert torch.equal(actor.log_std_parameter,torch.full((3,),-.5))
    assert (actor.log_std_min,actor.log_std_max) == (-5.,.5)
    torch.testing.assert_close(actor.mean.weight@actor.mean.weight.T,torch.eye(3)*.0001)
    assert torch.count_nonzero(actor.mean.bias) == 0
    with torch.no_grad(): actor.log_std_parameter.copy_(torch.tensor([-100.,0.,100.]))
    obs=observations(2,5)
    d=(actor.distribution_step(obs,torch.zeros(2,5,128),torch.ones(2,5),torch.zeros(2))[0]
       if name=='rmappo' else actor.distribution(obs))
    assert torch.equal(d.scale,torch.tensor([-5.,0.,.5]).exp().expand(2,5,3))


def test_rmappo_hidden_shapes_agent_independence_death_episode_and_history():
    actor=build('rmappo').actor
    obs=observations(2,5); alive=torch.ones(2,5); start=torch.zeros(2)
    hidden=torch.randn(2,5,128)
    d,h,_=actor.distribution_step(obs,hidden,alive,start)
    assert h.shape == (2,5,128) and d.mean.shape == (2,5,3)
    d0,_,_=actor.distribution_step(obs,torch.zeros_like(hidden),alive,start)
    assert not torch.allclose(d.mean,d0.mean)
    changed=hidden.clone(); changed[0,1] += 10
    d1,h1,_=actor.distribution_step(obs,changed,alive,start)
    assert torch.equal(h[0,0],h1[0,0]) and torch.equal(h[1],h1[1])
    alive[0,1]=0
    actions,dead=actor(obs,hidden,alive,start)
    assert torch.count_nonzero(actions[0,1]) == torch.count_nonzero(dead[0,1]) == 0
    reset=actor(obs,hidden,alive,torch.ones(2))
    zero=actor(obs,torch.zeros_like(hidden),alive,start)
    assert all(torch.equal(a,b) for a,b in zip(reset,zero))


def test_rmappo_sequence_history_future_and_boundary_gradient():
    actor=build('rmappo').actor
    obs=observations(1,6,5).requires_grad_()
    alive=torch.ones(1,6,5); starts=torch.zeros(1,6); hidden=torch.zeros(1,5,128)
    d,_,_=actor.distribution_sequence(obs,hidden,alive,starts)
    grad=torch.autograd.grad(d.mean[:,5].sum(),obs)[0]
    assert torch.count_nonzero(grad[:,0]) > 0
    changed=obs.detach().clone(); changed[:,4:] += 10
    alternate,_,_=actor.distribution_sequence(changed,hidden,alive,starts)
    assert torch.equal(d.mean[:,:4],alternate.mean[:,:4])
    starts[:,3]=1
    d,_,_=actor.distribution_sequence(obs,hidden,alive,starts)
    grad=torch.autograd.grad(d.mean[:,5].sum(),obs)[0]
    assert torch.count_nonzero(grad[:,:3]) == 0 and torch.count_nonzero(grad[:,3:]) > 0
    changed=obs.detach().clone(); changed[:,:3] += 100
    alternate,_,_=actor.distribution_sequence(changed,hidden+20,alive,starts)
    assert torch.equal(d.mean[:,3:],alternate.mean[:,3:])


def recurrent_rollout(trainer,t=40,e=2,saturated=False):
    rng=np.random.default_rng(71)
    obs=observations(t,e,5).numpy(); alive=np.ones((t,e,5),np.float32)
    alive[4:9,0,1]=0
    starts=np.zeros((t,e),np.float32); starts[0]=1; starts[11,0]=1; starts[17,1]=1
    hidden=np.zeros((e,5,128),np.float32)
    acts,raws,logs,states=[],[],[],[]
    for step in range(t):
        states.append(hidden.copy())
        a,r,l,hidden=trainer.act(obs[step],alive[step],hidden,starts[step])
        if saturated:
            # tanh(10) is already exactly saturated in float32. Extreme raw=25
            # amplifies CUDA matrix rounding in log N independently of tanh.
            r=np.full_like(r,10.)*alive[step,...,None]; a=np.tanh(r)
            with torch.no_grad():
                d=trainer.actor.distribution_step(trainer.tensor(obs[step]),trainer.tensor(states[-1]),
                    trainer.tensor(alive[step]),trainer.tensor(starts[step]))[0]
                l=(trainer.actor._squashed_log_prob(d,trainer.tensor(r),trainer.tensor(a))*trainer.tensor(alive[step])).cpu().numpy()
        acts.append(a);raws.append(r);logs.append(l)
        if step+1<t:
            hidden*=alive[step+1,...,None]; hidden[starts[step+1]>.5]=0
    dones=np.zeros_like(starts);dones[:-1]=starts[1:]
    nxt=np.concatenate((alive[1:],alive[-1:]),axis=0)
    return RecurrentRolloutBatch(obs,np.stack(acts),np.stack(raws),np.stack(logs),
        rng.normal(size=(t,e,5)).astype(np.float32),dones,alive,obs+.01,nxt,np.stack(states),starts)


@pytest.mark.parametrize('saturated',[False,True])
def test_rmappo_cuda_ratio_replay_uses_nonzero_chunk_hidden_and_rejects_corruption(saturated):
    assert torch.cuda.is_available(),'CUDA mandatory'
    trainer=build('rmappo','cuda'); rollout=recurrent_rollout(trainer,saturated=saturated)
    chunks=sequence_chunks(256,16,32)
    assert len(chunks)==128 and trainer.minibatch_size//32 == 16
    audit=trainer.audit_rollout_ratio(rollout)
    assert audit['pre_update_ratio_max_abs_error'] < 1e-4
    packed=trainer._packed({k:trainer.tensor(v) for k,v in vars(rollout).items()},sequence_chunks(40,2,32))
    assert torch.equal(packed['initial_hidden'][1],trainer.tensor(rollout.actor_hidden_states[32,0]))
    assert torch.count_nonzero(packed['initial_hidden'][1]) > 0
    bad=copy.deepcopy(rollout); bad.old_log_probs[2,0,0] += .1
    with pytest.raises(RuntimeError,match='ratio mismatch'): trainer.audit_rollout_ratio(bad)
    if not saturated:
        before=copy.deepcopy(trainer.actor.state_dict())
        metrics=trainer.update(rollout)
        assert all(np.isfinite(list(metrics.values())))
        assert metrics['sequence_chunks']==4 and metrics['padded_environment_transitions']==48
        assert metrics['configured_ppo_epochs']==10 and 1<=metrics['effective_ppo_epochs']<=10
        assert any(not torch.equal(before[k],v) for k,v in trainer.actor.state_dict().items())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in trainer.actor.parameters())


@pytest.mark.parametrize('side,start,end,slots,width', [('ally',7,35,4,7),('enemy',35,65,5,6)])
def test_ea_spatial_exactly_matches_stea_masks_empty_and_permutation(side,start,end,slots,width):
    e=build('ea_mappo').actor;s=build('stea_mappo').actor
    frontend=('self_encoder','ally_encoder','enemy_encoder','ally_attention','enemy_attention','spatial_fusion')
    for name in frontend: getattr(e,name).load_state_dict(getattr(s,name).state_dict())
    obs=observations(2,5);alive=torch.ones(2,5)
    entities=obs[...,start:end].reshape(2,5,slots,width);entities[...,0,-1]=0
    fused,diag=e.spatial(obs,alive); other,otherdiag=s.spatial(obs,alive)
    assert torch.equal(fused,other)
    assert all(torch.equal(diag[k],otherdiag[k]) for k in diag)
    assert torch.count_nonzero(diag[f'{side}_attention_weights'][...,0])==0
    perm=list(reversed(range(slots))); changed=obs.clone()
    changed[...,start:end]=entities[...,perm,:].reshape(2,5,slots*width)
    permuted,pdiag=e.spatial(changed,alive)
    torch.testing.assert_close(fused,permuted,atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(diag[f'{side}_attention_weights'][...,perm],pdiag[f'{side}_attention_weights'])
    changed=obs.clone(); changed[...,start:start+width-1]=1e6
    assert torch.equal(e.spatial(changed,alive)[0],fused)
    entities[...,-1]=0
    _,empty=e.spatial(obs,alive)
    assert torch.count_nonzero(empty[f'{side}_attention_weights'])==0
    assert torch.count_nonzero(empty[f'{side}_context'])==0
    assert torch.isfinite(empty[f'{side}_context']).all()


def stateless_rollout(trainer,t=8,e=2):
    obs=observations(t,e,5).numpy();alive=np.ones((t,e,5),np.float32)
    a,r,l=trainer.act(obs,alive,return_policy_data=True)
    return RolloutBatch(obs,a,r,l,np.ones((t,e,5),np.float32),np.zeros((t,e),np.float32),alive,obs+.01,alive.copy())


def test_ea_nonrecurrent_update_has_no_hidden_and_diagnostics_do_not_change_rng_or_weights():
    trainer=build('ea_mappo','cuda'); batch=stateless_rollout(trainer)
    assert not any('hidden' in key or 'episode_start' in key for key in vars(batch))
    clone=build('ea_mappo','cuda'); clone.actor.load_state_dict(trainer.actor.state_dict())
    clone.critic.load_state_dict(trainer.critic.state_dict())
    torch.manual_seed(41);torch.cuda.manual_seed_all(41);metrics=trainer.update(batch)
    after_rng=torch.cuda.get_rng_state()
    torch.manual_seed(41);torch.cuda.manual_seed_all(41);baseline=MAPPOTrainer.update(clone,batch)
    assert torch.equal(after_rng,torch.cuda.get_rng_state())
    for key,value in clone.actor.state_dict().items(): assert torch.equal(value,trainer.actor.state_dict()[key])
    for key,value in clone.critic.state_dict().items(): assert torch.equal(value,trainer.critic.state_dict()[key])
    assert all(np.isfinite(list(metrics.values())))
    for side in ('ally','enemy'):
        assert f'{side}_attention_entropy' in metrics and f'{side}_attention_top1_mean' in metrics


@pytest.mark.parametrize('name',['rmappo','ea_mappo'])
def test_deterministic_is_tanh_mean_and_saturation_safe_logprob(name):
    trainer=build(name); obs=observations(2,5);alive=torch.ones(2,5)
    if name=='rmappo':
        d=trainer.actor.distribution_step(obs,torch.zeros(2,5,128),alive,torch.zeros(2))[0]
        a=trainer.act(obs.numpy(),alive.numpy(),np.zeros((2,5,128),np.float32),np.zeros(2,np.float32),deterministic=True)[0]
    else:
        d=trainer.actor.distribution(obs)
        a=trainer.act(obs.numpy(),alive.numpy(),deterministic=True)
    assert np.array_equal(a,d.mean.tanh().detach().numpy())
    raw=torch.full_like(d.mean,50.)
    log=trainer.actor._squashed_log_prob(d,raw,raw.tanh())
    assert torch.isfinite(log).all()


def complete_state(name):
    trainer=build(name); cfg=config(name)
    env=yaml.safe_load((ROOT/'configs/combat_environment_v25.yaml').read_text())
    prefix='rmappo' if name=='rmappo' else 'ea_mappo'
    ac=sum(p.numel() for p in trainer.actor.parameters());cc=sum(p.numel() for p in trainer.critic.parameters())
    state=trainer.checkpoint_state(dict(algorithm=cfg['algorithm'],environment_version='2.5',
        observation_dim=65,action_dim=3,num_agents=5,training_seed=27,training_gamma=.99,
        training_num_envs=16,training_total_sampled_steps=3000000,training_smoke=False,
        effective_hidden_dim=256,critic_type='mlp',network_architecture=trainer.network_architecture,
        environment_config_sha256=config_sha256(env),algorithm_config_sha256=config_sha256(cfg),
        actor_parameter_count=ac,critic_parameter_count=cc,total_parameter_count=ac+cc,
        **{prefix+'_impl_version':1}))
    return trainer,state,env,cfg


@pytest.mark.parametrize('source',['rmappo','ea_mappo'])
def test_new_checkpoint_roundtrip_and_all_cross_algorithm_loads_rejected(source,tmp_path):
    trainer,state,env,cfg=complete_state(source)
    validator=validate_r if source=='rmappo' else validate_e
    validator(state,env,cfg)
    path=tmp_path/'checkpoint.pt';torch.save(state,path)
    build(source).load(path)
    for target in BUILD:
        if target==source:continue
        with pytest.raises(RuntimeError,match='checkpoint'):build(target).load(path)
    for validator_other,cfg_other in ((validate_mappo,config('mappo')),(validate_stea,config('stea_mappo')),
                                    (validate_e if source=='rmappo' else validate_r,config('ea_mappo' if source=='rmappo' else 'rmappo'))):
        with pytest.raises(RuntimeError):validator_other(state,env,cfg_other)
    for old in ('mappo','stea_mappo'):
        oldpath=tmp_path/(old+'.pt');build(old).save(oldpath)
        with pytest.raises(RuntimeError,match='checkpoint'):trainer.load(oldpath)


@pytest.mark.parametrize('name',['rmappo','ea_mappo'])
@pytest.mark.parametrize('field',['environment_version','observation_dim','action_dim','num_agents',
                                  'environment_config_sha256','algorithm_config_sha256','actor_parameter_count',
                                  'critic_parameter_count','network_architecture'])
def test_metadata_corruption_rejected(name,field):
    _,state,env,cfg=complete_state(name)
    state['extra'][field]='corrupted'
    with pytest.raises(RuntimeError):(validate_r if name=='rmappo' else validate_e)(state,env,cfg)


@pytest.mark.parametrize('name',['rmappo','ea_mappo'])
def test_cpu_runtime_and_wrong_environment_rejected(name,tmp_path):
    from algorithm.rmappo.runner import RMAPPOTrainingRunner
    from algorithm.ea_mappo.runner import EAMAPPOTrainingRunner
    cls=RMAPPOTrainingRunner if name=='rmappo' else EAMAPPOTrainingRunner
    env=yaml.safe_load((ROOT/'configs/combat_environment_v25.yaml').read_text())
    with pytest.raises(RuntimeError,match='CUDA'):cls(env,config(name),output_dir=tmp_path,device='cpu')
    env['environment_version']='2.4'
    with pytest.raises(ValueError,match='2.5'):cls(env,config(name),output_dir=tmp_path,device='cuda')


def test_rmappo_runner_resets_individual_death_and_whole_completed_environment():
    from types import SimpleNamespace
    from algorithm.rmappo.runner import RMAPPOTrainingRunner
    runner=object.__new__(RMAPPOTrainingRunner)
    runner.num_envs=2;runner.num_agents=5;runner.rollout_steps=1
    runner.observations=np.zeros((2,5,65),np.float32)
    runner.alive_masks=np.ones((2,5),np.float32)
    runner.actor_hidden_states=np.full((2,5,128),2.,np.float32)
    runner.episode_start_masks=np.zeros(2,np.float32)
    runner.trainer=SimpleNamespace(sampled_steps=0,vector_steps=0)
    runner.trainer.act=lambda obs,alive,before,start: (np.zeros((2,5,3),np.float32),
        np.zeros((2,5,3),np.float32),np.zeros((2,5),np.float32),np.full((2,5,128),7.,np.float32))
    next_alive=np.ones((2,5),np.float32);next_alive[0,1]=0
    result=SimpleNamespace(terminated=np.array([False,True]),truncated=np.zeros(2,bool),
        rewards=np.zeros((2,5),np.float32),transition_next_observations=runner.observations,
        next_alive_masks=next_alive,observations=runner.observations)
    runner.vector=SimpleNamespace(step_batch=lambda _:result,current_alive_masks=np.ones((2,5),np.float32))
    runner.vector.current_alive_masks[0,1]=0
    runner._completed=lambda _:[];runner._write_step_metrics=lambda *_:None
    batch=runner.collect_rollout()
    assert np.all(batch.actor_hidden_states[0]==2)
    assert np.count_nonzero(runner.actor_hidden_states[0,1])==0
    assert np.count_nonzero(runner.actor_hidden_states[1])==0
    assert np.all(runner.actor_hidden_states[0,0]==7)
    assert runner.episode_start_masks.tolist()==[0,1]
    nextbatch=runner.collect_rollout()
    assert np.count_nonzero(nextbatch.actor_hidden_states[0,0,1])==0
    assert np.count_nonzero(nextbatch.actor_hidden_states[0,1])==0
    assert nextbatch.episode_starts[0].tolist()==[0,1]


def test_rmappo_evaluator_resets_hidden_at_episode_start_and_after_death(monkeypatch):
    from types import SimpleNamespace
    from algorithm.rmappo import evaluation
    seen=[]
    class Episode:
        team_size=5
        def reset(self,seed):
            self.step_count=0;self.red_alive_mask=np.ones(5,np.float32)
            return np.ones((5,65),np.float32),{}
        def step(self,actions):
            self.step_count+=1;self.red_alive_mask[1]=0
            return np.ones((5,65),np.float32),np.zeros(5),self.step_count==2,False,{}
    def act(obs,alive,hidden,start,**kwargs):
        seen.append((hidden.copy(),float(start)))
        return np.zeros((5,3)),None,None,np.ones((5,128),np.float32),{}
    trainer=SimpleNamespace(device='cuda',actor=SimpleNamespace(gru_hidden_dim=128),act=act)
    monkeypatch.setattr(evaluation,'make_combat_environment',lambda _:Episode())
    monkeypatch.setattr(evaluation,'aggregate_combat_records',lambda records:{'count':len(records)})
    assert evaluation.evaluate(trainer,{},[65000000,65000001])['count']==2
    for index in (0,2):
        assert np.count_nonzero(seen[index][0])==0 and seen[index][1]==1
    for index in (1,3):
        assert np.count_nonzero(seen[index][0][1])==0 and np.all(seen[index][0][0]==1)
