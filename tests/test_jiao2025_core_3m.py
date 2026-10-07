from copy import deepcopy
import numpy as np
import pytest
import torch
from tools.preflight_jiao2025_core_3m import CONFIGS,ENV,load_config,validate_pair,build_modular_mappo_trainer,validate_jiao2025_3m_config
from tools.smoke_jiao2025_core_3m import controlled_evaluation,assert_finite_update
from algorithm.modular_mappo.buffer import ModularRolloutBatch,episode_contiguous_chunks

def config(variant):return load_config(CONFIGS[variant])

@pytest.mark.parametrize('variant',CONFIGS)
def test_fixed_recipe(variant):
    cfg=config(variant);validate_jiao2025_3m_config(cfg,load_config(ENV))
    assert validate_pair(config('matched_plain'),config('core'),load_config(ENV))
    t=cfg['training']
    assert (t['seed'],t['total_sampled_steps'],t['rollout_steps'],t['num_train_envs'],t['minibatch_size'],t['ppo_epochs'])==(5301,3000000,256,24,512,10)
    assert (t['actor_learning_rate'],t['critic_learning_rate'],t['gamma'],t['gae_lambda'],t['clip_ratio'],t['entropy_coefficient'])==(.0005,.0005,.99,.95,.1,.01)
    assert cfg['implementation']['evaluation_seed_base']==44000000 and t['evaluation_episodes']==50
    assert cfg['development_protocol']['reserved_future_final_test']=={'seed_start':45000000,'seed_end':45000199,'executed':False}
    trainer=build_modular_mappo_trainer(cfg,'cpu')
    expected={'wave_context','recurrent_memory','popart'} if variant=='core' else set()
    assert set(trainer.module_protocol()['enabled_modules'])==expected
    assert not trainer.reward_adapter.enabled and not trainer.wave_balance.enabled
    assert not trainer.milestone_aware_retention_credit.enabled
    for step in (0,600000,900000,1500000,3000000):
        assert trainer.actor_lr_decay.apply(trainer.actor_optimizer,step,.0005)==.0005
        assert trainer.actor_optimizer.param_groups[0]['lr']==trainer.critic_optimizer.param_groups[0]['lr']==.0005

@pytest.mark.parametrize('variant',CONFIGS)
def test_architecture_and_context(variant):
    trainer=build_modular_mappo_trainer(config(variant),'cpu');core=variant=='core'
    assert trainer.actor.backbone[0].in_features==52+int(core)
    assert trainer.actor.backbone[0].out_features==trainer.actor.backbone[2].out_features==256
    assert trainer.actor.recurrent_hidden_dim==trainer.critic.recurrent_hidden_dim==(128 if core else 0)
    assert not trainer.recurrent.wave_boundary_reset
    assert not hasattr(trainer.actor,'phase_initial_hidden')
    assert trainer.popart.enabled==core
    if core:
        ctx=trainer.wave_context.encode_tensor(torch.tensor([1,2,3]),torch.tensor([3,3,3]))
        obs=torch.zeros(3,4,52)
        for network in (trainer.actor,trainer.critic):
            inputs=network._input(obs,ctx)
            assert inputs.shape==(3,4,53)
            assert torch.equal(inputs[...,-1],torch.tensor([[1.]*4,[2.]*4,[3.]*4]))
            assert all(p.requires_grad for p in network.gru.parameters())

def test_chunk_coverage_episode_only_not_wave():
    ep=np.ones((70,2),np.float32);ep[0]=0;ep[7,0]=0;ep[51,1]=0
    chunks=episode_contiguous_chunks(ep,32);coverage=np.zeros_like(ep)
    for env,start,stop in chunks:
        assert 0<stop-start<=32 and ep[start+1:stop,env].all()
        coverage[start:stop,env]+=1
    assert (coverage==1).all()
    # Wave switches at3/5 do not split the first episode chunk.
    assert (0,0,7) in chunks

def constructed_rollout(trainer):
    T,E,N=9,2,4;rng=np.random.default_rng(99001003)
    obs=rng.normal(size=(T,E,N,52)).astype(np.float32);alive=np.ones((T,E,N),np.float32)
    alive[3:,0,3]=0
    waves=np.repeat(np.asarray([1,1,1,2,2,3,3,3,3])[:,None],E,axis=1)
    ep=np.ones((T,E),np.float32);ep[0]=0;ep[6,1]=0
    ctx=trainer.wave_context.encode_numpy(waves,np.full((T,E),3))
    ah,ch=trainer.initial_hidden(E);saved_a=[];saved_c=[];actions=[];raw=[];logs=[]
    for t in range(T):
        trainer.recurrent.reset_for_episode(ah,ep[t]==0);trainer.recurrent.reset_for_episode(ch,ep[t]==0)
        ah=trainer.recurrent.apply_alive(ah,alive[t]);ch=trainer.recurrent.apply_alive(ch,alive[t])
        saved_a.append(ah.copy());saved_c.append(ch.copy())
        a,r,l,ah=trainer.act(obs[t],alive[t],False,True,ctx[t],ah,ep[t])
        _,ch=trainer.values_step(obs[t],alive[t],ctx[t],ch,ep[t])
        actions.append(a);raw.append(r);logs.append(l)
    rewards=rng.normal(size=(T,E,N)).astype(np.float32)
    dones=np.zeros((T,E),np.float32);dones[5,1]=1
    return ModularRolloutBatch(obs,np.asarray(actions),np.asarray(raw),np.asarray(logs),rewards,rewards.copy(),
        dones,alive,obs.copy(),alive.copy(),waves,np.full((T,E),3),ctx,ctx.copy(),np.asarray(saved_a),np.asarray(saved_c),ep)

def test_real_joint_update_coverage_gradients_and_detached_starts(monkeypatch):
    trainer=build_modular_mappo_trainer(config('core'),'cpu');batch=constructed_rollout(trainer)
    from algorithm.mappo.trainer import compute_gae
    tt=lambda value:torch.as_tensor(value,dtype=torch.float32)
    with torch.no_grad():
        values,next_values=trainer._value_rollout(batch,tt(batch.observations),tt(batch.next_observations))
        _,raw_target=compute_gae(tt(batch.rewards),values,next_values,tt(batch.dones),tt(batch.alive_masks),tt(batch.next_alive_masks),trainer.gamma,trainer.gae_lambda)
    covered=[];original=trainer._recurrent_minibatch
    actor_entry=[];critic_entry=[]
    def capture(r,group,*args):
        covered.extend(group)
        assert torch.isfinite(args[7]).all()
        assert torch.allclose(trainer.popart.denormalize_values(args[7]),raw_target,atol=1e-5)
        return original(r,group,*args)
    monkeypatch.setattr(trainer,'_recurrent_minibatch',capture)
    def hook(target):
        def record(module,args):target.append(args[1].grad_fn)
        return record
    ha=trainer.actor.gru.register_forward_pre_hook(hook(actor_entry));hc=trainer.critic.gru.register_forward_pre_hook(hook(critic_entry))
    metrics=trainer.update(batch);ha.remove();hc.remove()
    chunks=episode_contiguous_chunks(batch.episode_masks,32)
    assert len(covered)==len(chunks)*10
    for epoch in range(10):assert sorted(covered[epoch*len(chunks):(epoch+1)*len(chunks)])==sorted(chunks)
    assert metrics['ppo_epochs_executed']==10
    assert metrics['actor_gru_grad_norm']>0 and metrics['critic_gru_grad_norm']>0
    assert all(np.isfinite(v) for v in metrics.values())
    assert None in actor_entry and None in critic_entry
    assert not batch.actor_hidden_before_step[3:,0,3].any()
    assert not batch.critic_hidden_before_step[3:,0,3].any()
    assert np.array_equal(batch.rewards,batch.raw_environment_rewards)

def test_popart_recurrent_head_preserves_raw_output_and_resume(tmp_path):
    trainer=build_modular_mappo_trainer(config('core'),'cpu')
    latent=torch.randn(10,128);head=trainer.critic.output_layer
    before=trainer.popart.denormalize_values(head(latent)).detach()
    trainer.popart.update(torch.linspace(-25,40,100),head)
    assert torch.allclose(before,trainer.popart.denormalize_values(head(latent)).detach(),atol=1e-5)
    trainer.save(tmp_path/'core.pt')
    other=build_modular_mappo_trainer(config('core'),'cpu');other.load(tmp_path/'core.pt')
    for key,value in trainer.popart.state_dict().items():assert torch.equal(value,other.popart.state_dict()[key])
    assert np.isfinite(trainer.popart.normalize_targets(torch.tensor([1.,-1.])).detach().numpy()).all()

def test_plain_popart_bypass(monkeypatch):
    trainer=build_modular_mappo_trainer(config('matched_plain'),'cpu')
    from tools.smoke_marc_state_memory import state_memory_rollout
    def forbidden(*args,**kwargs):raise AssertionError('Plain must bypass PopArt')
    monkeypatch.setattr(trainer.popart,'update',forbidden)
    metrics=trainer.update(state_memory_rollout(trainer))
    assert metrics['ppo_epochs_executed']==10 and trainer.popart.count.item()==0
    assert_finite_update(metrics)

@pytest.mark.parametrize('variant',CONFIGS)
def test_canonical_evaluation_hidden_and_tanh(variant):
    trainer=build_modular_mappo_trainer(config(variant),'cpu')
    assert controlled_evaluation(trainer,99001004)['deterministic_tanh_mean']

@pytest.mark.parametrize('mutation',['lr','epochs','rollout','context','mode','popart','marc','balance','reward','bank','holdout','guard','decay','extra'])
def test_fail_closed(mutation):
    cfg=config('core');env=load_config(ENV)
    if mutation=='lr':cfg['training']['actor_learning_rate']=.0003
    elif mutation=='epochs':cfg['training']['ppo_epochs']=1
    elif mutation=='rollout':cfg['training']['rollout_steps']=150
    elif mutation=='context':cfg['modules']['wave_context']['encoding']='rich'
    elif mutation=='mode':cfg['modules']['recurrent_memory']['mode']='wave_segmented_actor_gru'
    elif mutation=='popart':cfg['modules']['popart']['enabled']=False
    elif mutation=='marc':cfg['modules']['milestone_aware_retention_credit']['enabled']=True
    elif mutation=='balance':cfg['modules']['wave_balancing']['enabled']=True
    elif mutation=='reward':env['reward']={'changed':True}
    elif mutation=='bank':cfg['implementation']['evaluation_seed_base']=45000000
    elif mutation=='holdout':cfg['development_protocol']['reserved_future_final_test']['executed']=True
    elif mutation=='guard':cfg['modules']['actor_kl_guard']['enabled']=True
    elif mutation=='decay':cfg['modules']['actor_lr_decay']['enabled']=True
    else:cfg['modules']['entity_attention']['enabled']=True
    with pytest.raises(ValueError):validate_jiao2025_3m_config(cfg,env)
