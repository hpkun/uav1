"""Fixed-scale attention representation, strict ablation and baseline regressions."""
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Normal
import yaml
from algorithm.eaa_mappo.networks import EntityAttentionMAPPOActor, split_v338_entities
from algorithm.eaa_mappo.factory import build_eaa_mappo_trainer
from algorithm.eaa_mappo.protocol import validate_config, validate_checkpoint
from algorithm.mappo.networks import SharedMAPPOActor, CentralizedMLPCritic, CentralizedValueCritic
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.mappo.trainer import MAPPO_IMPL_VERSION, MAPPOTrainer, RolloutBatch

ROOT=Path(__file__).resolve().parents[1]


def configs(name='eaa_mappo'):
    return tuple(yaml.safe_load((ROOT/'configs'/p).read_text(encoding='utf-8')) for p in
        ['combat_environment_v338.yaml',name+'_4v4_v338.yaml'])


def test_parser_arbitrary_leading_dimensions_and_exact_boundaries():
    x=torch.arange(45.).expand(2,3,4,45)
    own,allies,enemies=split_v338_entities(x)
    assert own.shape==(2,3,4,7) and allies.shape==(2,3,4,3,6) and enemies.shape==(2,3,4,4,5)
    assert torch.equal(own[0,0,0],torch.arange(7.))
    assert torch.equal(allies[0,0,0].flatten(),torch.arange(7.,25.))
    assert torch.equal(enemies[0,0,0].flatten(),torch.arange(25.,45.))
    with pytest.raises(ValueError):split_v338_entities(torch.zeros(44))


def test_tokens_encoders_head_dimensions_and_no_extra_components():
    actor=EntityAttentionMAPPOActor()
    tokens,valid=actor.encode_entities(torch.randn(2,45))
    assert tokens.shape==(2,8,64) and valid.shape==(2,8)
    assert actor.attention_heads==4 and actor.head_dim==16
    for name,width in [('self_encoder',7),('ally_encoder',6),('enemy_encoder',5)]:
        encoder=getattr(actor,name);assert isinstance(encoder[0],nn.Linear) and isinstance(encoder[1],nn.ReLU)
        assert encoder[0].in_features==width and encoder[0].out_features==64
    for module in actor.modules():assert not isinstance(module,(nn.GRU,nn.LSTM,nn.Dropout,nn.LayerNorm,nn.Transformer))
    assert not any('position' in name or 'mask_generator' in name for name,_ in actor.named_parameters())
    assert [(m.in_features,m.out_features) for m in actor.backbone if isinstance(m,nn.Linear)]==[(128,256),(256,256)]


def test_scaled_dot_product_matches_independent_manual_formula():
    torch.manual_seed(7);actor=EntityAttentionMAPPOActor()
    tokens=torch.randn(2,8,64);valid=torch.ones(2,8,dtype=torch.bool);valid[:,3]=False
    output,weights=actor.entity_attention(tokens,valid)
    q,k,v=[layer(tokens).reshape(2,8,4,16).transpose(1,2) for layer in [actor.wq,actor.wk,actor.wv]]
    logits=q@k.transpose(-2,-1)/math.sqrt(16)
    expected=torch.softmax(logits.masked_fill(~valid[:,None,None,:],-torch.inf),-1)
    assert torch.allclose(weights,expected,atol=1e-7)
    assert torch.allclose(output,(expected@v).transpose(1,2).reshape(2,8,64),atol=1e-7)
    assert not torch.allclose(weights,torch.softmax((q@k.transpose(-2,-1)).masked_fill(~valid[:,None,None,:],-torch.inf),-1))


@pytest.mark.parametrize('live_allies,live_enemies',[(0,0),(0,2),(2,0),(3,4),(1,1)])
def test_dead_entities_exact_zero_and_self_only_safe(live_allies,live_enemies):
    actor=EntityAttentionMAPPOActor();obs=torch.zeros(5,45)
    own,allies,enemies=split_v338_entities(obs)
    own.copy_(torch.randn_like(own));allies[:,:live_allies]=1.;enemies[:,:live_enemies]=1.
    own[0]=0  # Even the all-zero dead-agent input must have a finite self-only fallback.
    # Nonzero biases make this a mask test, not a zero-input coincidence.
    with torch.no_grad():actor.ally_encoder[0].bias.fill_(3);actor.enemy_encoder[0].bias.fill_(5)
    fused,weights=actor.representation(obs,True)
    assert fused.shape==(5,128) and weights.shape==(5,4,8)
    assert torch.isfinite(fused).all() and torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(-1),torch.ones(5,4))
    assert torch.count_nonzero(weights[...,1+live_allies:4])==0
    assert torch.count_nonzero(weights[...,4+live_enemies:])==0
    if live_allies+live_enemies==0:assert torch.equal(weights[...,0],torch.ones(5,4))
    tokens,_=actor.encode_entities(obs);context,_=actor.entity_attention(tokens,actor.encode_entities(obs)[1])
    assert context[...,0,:].shape==(5,64)
    assert torch.equal(fused,torch.cat([tokens[...,0,:],context[...,0,:]],-1))


@pytest.mark.parametrize('token_permutation,dead_start',[
    ([0,2,1,3,4,5,6,7],19),([0,2,1,3,4,5,6,7],13),
    ([0,1,2,3,6,5,4,7],30),([0,1,2,3,6,5,4,7],25)])
def test_ally_enemy_permutations_preserve_self_representation_and_permute_weights(token_permutation,dead_start):
    actor=EntityAttentionMAPPOActor();obs=torch.randn(3,4,45)
    obs[...,dead_start:dead_start+(6 if dead_start<25 else 5)]=0
    own,allies,enemies=split_v338_entities(obs)
    if token_permutation[1]!=1:allies=allies[...,[1,0,2],:]
    else:enemies=enemies[...,[2,1,0,3],:]
    permuted=torch.cat([own,allies.flatten(-2),enemies.flatten(-2)],-1)
    z,w=actor.representation(obs,True);zp,wp=actor.representation(permuted,True)
    assert torch.allclose(z,zp,atol=2e-7,rtol=1e-6)
    assert torch.allclose(w[...,token_permutation],wp,atol=2e-7,rtol=1e-6)
    assert torch.allclose(actor.deterministic(obs),actor.deterministic(permuted),atol=1e-7)


def test_gaussian_and_latent_log_prob_exactly_reuse_baseline_math():
    actor=EntityAttentionMAPPOActor();obs=torch.randn(8,4,45)
    for method in ['sample','evaluate_actions','_squashed_log_prob','_initialize_policy_std','_policy_log_std','deterministic']:
        assert getattr(type(actor),method) is getattr(SharedMAPPOActor,method)
    distribution,attention=actor.distribution(obs,True)
    assert distribution.mean.shape==distribution.scale.shape==(8,4,3)
    assert torch.equal(distribution.scale,torch.full_like(distribution.scale,math.exp(-.5)))
    actions,raw,logp,_=actor.sample(obs);evaluated,entropy=actor.evaluate_actions(obs,actions,raw)
    assert torch.equal(logp,evaluated) and torch.isfinite(entropy).all()
    assert actor(obs,True)[1].shape==(8,4,4,8)
    assert actor.deterministic(obs).abs().max()<=1
    large=torch.full_like(raw,30.);bounded=torch.tanh(large)
    expected=(distribution.log_prob(large)-2*(math.log(2)-large-torch.nn.functional.softplus(-2*large))).sum(-1)
    assert torch.equal(actor._squashed_log_prob(distribution,large,bounded),expected)


@pytest.mark.parametrize('name,actor_cls,critic_cls',[
    ('mappo',SharedMAPPOActor,CentralizedMLPCritic),('mappo_ca',SharedMAPPOActor,CentralizedValueCritic),
    ('ea_mappo',EntityAttentionMAPPOActor,CentralizedMLPCritic),('eaa_mappo',EntityAttentionMAPPOActor,CentralizedValueCritic)])
def test_strict_ablation_construction_and_frozen_protocol(name,actor_cls,critic_cls):
    env,c=configs(name);_,base=configs('mappo')
    trainer=build_mappo_trainer(c,'cpu') if name=='mappo' else build_eaa_mappo_trainer(c,env,'cpu')
    assert type(trainer.actor) is actor_cls and type(trainer.critic) is critic_cls
    for section in ['training','implementation','runtime_logging','environment']:assert c[section]==base[section]
    assert MAPPO_IMPL_VERSION==2 and trainer.gamma==.99 and trainer.entropy_coefficient==.001
    obs=np.random.default_rng(7).normal(size=(2,4,45)).astype('float32');alive=np.ones((2,4),'float32');alive[:,2]=0
    actions=trainer.act(obs,alive,deterministic=True)
    assert np.count_nonzero(actions[:,2])==0 and np.isfinite(actions).all()


@pytest.mark.parametrize('masks',[[1,1,1,1],[1,0,1,0],[1,0,0,0],[0,0,0,0]])
def test_reused_critic_masks_and_no_other_agent_case(masks):
    critic=CentralizedValueCritic(45,256,2);obs=torch.randn(2,4,45);alive=torch.tensor([masks]*2,dtype=torch.float32)
    values,weights=critic(obs,alive,True)
    assert values.shape==(2,4) and weights.shape==(2,2,4,4)
    assert torch.isfinite(values).all() and torch.isfinite(weights).all()
    assert torch.count_nonzero(weights[...,alive[0]==0])==0
    assert torch.count_nonzero(values[:,alive[0]==0])==0
    valid_queries=alive.bool() & (alive.sum(-1,keepdim=True)>1)
    assert torch.allclose(weights.sum(-1).transpose(1,2)[valid_queries],torch.ones_like(weights.sum(-1).transpose(1,2)[valid_queries]))
    if sum(masks)<=1:assert torch.count_nonzero(weights)==0
    changed=obs.clone();changed[:,alive[0]==0]=1e3
    assert torch.equal(critic(obs,alive),critic(changed,alive))


@pytest.mark.parametrize('name',['mappo_ca','ea_mappo','eaa_mappo'])
def test_attention_diagnostics_normalized_and_do_not_change_rng(name):
    env,c=configs(name);trainer=build_eaa_mappo_trainer(c,env,'cpu')
    obs=np.random.default_rng(3).normal(size=(4,4,45)).astype('float32');alive=np.ones((4,4),'float32')
    rng=torch.get_rng_state().clone();metrics=trainer.attention_diagnostics(obs,alive)
    assert torch.equal(rng,torch.get_rng_state()) and np.isfinite(list(metrics.values())).all()
    if name!='mappo_ca':
        assert sum(metrics['actor_attention_'+k+'_mass'] for k in ['self','ally','enemy'])==pytest.approx(1,abs=2e-7)
        assert all('actor_attention_head_'+str(h)+'_entropy' in metrics for h in range(4))
    if name!='ea_mappo':assert 'critic_attention_entropy' in metrics and metrics['critic_attention_max_weight']<=1
    assert all(v==0 for k,v in trainer.attention_diagnostics(obs,np.zeros_like(alive)).items())


@pytest.mark.parametrize('section,key,value',[
    ('training','gamma',.98),('training','actor_learning_rate',1e-4),('training','entropy_coefficient',.01),
    ('network','actor_type','mlp'),('network','critic_type','mlp'),('network','entity_attention_heads',2),
    ('network','dropout',.1),('network','gru_hidden_dim',128),('implementation','policy_std_mode','state_dependent')])
def test_reject_non_ablation_config_changes(section,key,value):
    env,c=configs();c[section][key]=value
    with pytest.raises(ValueError):validate_config(env,c)


@pytest.mark.parametrize('key', ['algorithm','checkpoint_family','implementation_version','environment_config_sha256',
    'algorithm_config_sha256','network_architecture','ppo_hyperparameters','policy_protocol','parameter_counts'])
def test_checkpoint_rejects_modified_identity_contract(key):
    env,c=configs();trainer=build_eaa_mappo_trainer(c,env,'cpu');state=trainer.checkpoint_state()
    validate_checkpoint(state,env,c)
    state[key]='incorrect'
    with pytest.raises(RuntimeError):validate_checkpoint(state,env,c)


def test_baseline_and_legacy_ea_checkpoints_cannot_cross_into_v338_family():
    env,c=configs('ea_mappo');trainer=build_eaa_mappo_trainer(c,env,'cpu')
    _,base=configs('mappo');state=build_mappo_trainer(base,'cpu').checkpoint_state()
    with pytest.raises(RuntimeError):validate_checkpoint(state,env,c)
    new=trainer.checkpoint_state()
    for name in ['mappo_ca','eaa_mappo']:
        with pytest.raises(RuntimeError):validate_checkpoint(new,env,configs(name)[1])
    from algorithm.ea_mappo.trainer import EAMAPPO_IMPL_VERSION
    assert EAMAPPO_IMPL_VERSION==1 and new['checkpoint_family']=='eaa_mappo_v338'


def test_baseline_default_runner_constructor_is_unchanged(monkeypatch,tmp_path):
    import algorithm.mappo.runner as module
    env,c=configs('mappo')
    class Vector:
        def __init__(self,*args):self.current_alive_masks=np.ones((2,4));self.backend='test';self.num_workers=2
        def reset(self):return np.zeros((2,4,45))
    monkeypatch.setattr(module,'ParallelVectorEnv',Vector)
    runner=module.MAPPOTrainingRunner(env,c,2,512,'cpu',0,tmp_path)
    expected=build_mappo_trainer(c,'cpu')
    assert type(runner.trainer) is MAPPOTrainer
    for name in ['actor','critic']:
        assert all(torch.equal(v,getattr(expected,name).state_dict()[k]) for k,v in getattr(runner.trainer,name).state_dict().items())
    assert runner.trainer.policy_protocol()==expected.policy_protocol()
    assert all(key in runner.start_log_line() for key in ['actor_params=','critic_params=','total_params='])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('name',['mappo_ca','ea_mappo','eaa_mappo'])
def test_cuda_finite_gradients_and_checkpoint_roundtrip(name,tmp_path):
    env,c=configs(name);trainer=build_eaa_mappo_trainer(c,env,'cuda')
    obs=torch.randn(2,4,45,device='cuda');alive=torch.ones(2,4,device='cuda')
    actions,raw,logp,_=trainer.actor.sample(obs)
    lp,entropy=trainer.actor.evaluate_actions(obs,actions.detach(),raw.detach())
    (lp.mean()+entropy.mean()*.001).backward()
    names=['self_encoder','ally_encoder','enemy_encoder','wq','wk','wv','backbone'] if name!='mappo_ca' else ['backbone']
    for name_prefix in names:
        params=list(getattr(trainer.actor,name_prefix).parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        assert any(torch.count_nonzero(p.grad)>0 for p in params)
    value=trainer.critic(obs,alive);value.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainer.critic.parameters())
    before=trainer.act(obs.cpu().numpy(),alive.cpu().numpy(),deterministic=True)
    path=tmp_path/'model.pt';trainer.save(path)
    baseline=build_mappo_trainer(configs('mappo')[1],'cuda')
    with pytest.raises(RuntimeError,match='not a MAPPO checkpoint'):baseline.load(path)
    reloaded=build_eaa_mappo_trainer(c,env,'cuda');reloaded.load(path)
    assert np.array_equal(before,reloaded.act(obs.cpu().numpy(),alive.cpu().numpy(),deterministic=True))
