"""MARC-SM architecture, boundary lifecycle and cross-wave RL credit isolation."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.protocol import checkpoint_architecture
from algorithm.modular_mappo.buffer import wave_segmented_chunks
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modules import continuation_coefficients
from algorithm.modules.milestone_aware_retention_credit import compute_local_gae
from algorithm.mappo.trainer import compute_gae
from tools.preflight_marc_state_memory import CONFIG,CONTROL,ENV,validate_protocol,matched_critic
from tools.smoke_marc_state_memory import state_memory_rollout,controlled_evaluation

@pytest.fixture
def trainer():return build_modular_mappo_trainer(load_config(CONFIG),'cpu')

def test_exact_architecture(trainer):
    actor=trainer.actor;arch=checkpoint_architecture(trainer)
    assert actor.base_observation_dim==52 and actor.context_dim==3
    assert (actor.gru.input_size,actor.gru.hidden_size)==(55,128)
    assert (actor.backbone[0].in_features,actor.backbone[0].out_features)==(183,256)
    assert (actor.backbone[2].in_features,actor.backbone[2].out_features)==(256,256)
    assert isinstance(actor.backbone[1],torch.nn.ReLU) and isinstance(actor.backbone[3],torch.nn.ReLU)
    assert actor.mean.in_features==actor.log_std.in_features==256
    assert not hasattr(actor,'phase_initial_hidden')
    assert arch['actor_input_dim']==55 and arch['fused_policy_input_dim']==183
    assert arch['critic_input_dim']==52 and arch['critic_context_dim']==0 and arch['critic_gru_hidden_dim']==0

def test_protocol_and_preserved_common_parameters():
    assert validate_protocol(load_config(CONFIG),load_config(CONTROL),load_config(ENV))

@pytest.mark.parametrize('seed',[5301,5302])
def test_critic_and_rng_matched(seed):assert all(matched_critic('cpu',seed).values())

@pytest.mark.parametrize('wave',[1,2,3])
def test_one_hot_literal_actor_only(trainer,wave):
    alive=np.ones((2,4),np.float32);ctx=trainer.actor_context_numpy([wave,wave],alive,None)
    assert np.array_equal(ctx,np.broadcast_to(np.eye(3)[wave-1],(2,4,3)))
    assert trainer._ctx(torch.as_tensor(ctx),False) is None

def test_current_state_direct_concat_path(trainer):
    obs=torch.randn(2,4,52);alive=torch.ones(2,4);context=trainer.actor.wave_one_hot([1,3],alive)
    captured=[];hook=trainer.actor.backbone.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
    _,h=trainer.actor.distribution_step(obs,context,torch.zeros(2,4,128),torch.ones(2),alive)
    hook.remove();z=captured[0]
    assert z.shape==(2,4,183) and torch.equal(z[...,:55],torch.cat((obs,context),-1))
    assert torch.equal(z[...,55:],h)
    obs.requires_grad_();dist,_=trainer.actor.distribution_step(obs,context,torch.zeros_like(h),torch.ones(2),alive)
    dist.mean.sum().backward();assert obs.grad is not None and torch.isfinite(obs.grad).all() and obs.grad.norm()>0

def test_literal_zero_dead_and_carry_lifecycle(trainer):
    alive=np.asarray([[1,1,1,0]],np.float32);hidden=np.ones((1,4,128),np.float32)
    for wave in (1,2,3):
        assert not trainer.prepare_actor_hidden(hidden,alive,[wave],[True]).any()
        carry=trainer.prepare_actor_hidden(hidden,alive,[wave],[False])
        assert np.all(carry[0,:3]==1) and not carry[0,3].any()
    episode,_=trainer.initial_hidden(1);assert not episode.any()
    r=state_memory_rollout(trainer)
    assert np.all(r.actor_hidden_before_step[[0,3,6]]==0)
    assert np.any(r.actor_hidden_before_step[2]) and np.any(r.actor_hidden_before_step[5])
    assert np.all(r.actor_hidden_before_step[5:,0,3]==0)
    assert np.all(r.actions[5:,0,3]==0)

def test_chunk_exact_coverage_no_episode_or_wave_crossing():
    waves=np.ones((99,2),int);waves[35:70]=2;waves[70:]=3;waves[85:,1]=1
    flags=np.zeros_like(waves,bool);flags[[0,35,70]]=True;flags[85,1]=True
    ep=np.ones_like(waves,float);ep[0]=0;ep[85,1]=0;coverage=np.zeros_like(waves)
    for e,s,z in wave_segmented_chunks(waves,flags,ep,32):
        assert z-s<=32 and np.all(waves[s:z,e]==waves[s,e]) and not np.any(ep[s+1:z,e]==0)
        coverage[s:z,e]+=1
    assert np.all(coverage==1)

@pytest.mark.parametrize('boundary',[True,False])
def test_bptt_initial_is_zero_or_detached_saved(trainer,boundary):
    r=state_memory_rollout(trainer);t=3 if boundary else 4
    r.actor_hidden_before_step[t,0]=7 if boundary else .2
    cv=lambda x:torch.as_tensor(x,dtype=torch.float32)
    args=[cv(x) for x in (r.observations,r.actions,r.raw_actions,r.old_log_probs,r.alive_masks)]
    seen=[];original=trainer.actor.distribution_step
    def capture(obs,ctx,hidden,*a,**kw):seen.append(hidden);return original(obs,ctx,hidden,*a,**kw)
    trainer.actor.distribution_step=capture
    row=trainer._actor_recurrent_minibatch(r,[(0,t,6)],*args,torch.ones_like(args[-1]),torch.ones(r.wave_indices.shape),cv(r.contexts),cv)
    assert not seen[0].requires_grad
    assert torch.all(seen[0]==(0 if boundary else .2))
    assert row['actor_gru_grad_norm']>0 and np.isfinite(row['actor_gru_grad_norm'])

def test_memory_reset_does_not_reset_full_horizon_credit():
    rewards=torch.tensor([0.,0.,3.]).reshape(3,1,1).expand(3,1,4)
    value=torch.zeros_like(rewards);next_value=torch.ones_like(rewards)
    alive=torch.ones_like(rewards);dones=torch.zeros(3,1);transition=torch.tensor([[1.],[1.],[0.]])
    global_adv,returns=compute_gae(rewards,value,next_value,dones,alive,alive,.999,.95)
    local,local_returns=compute_local_gae(rewards,value,next_value,dones,alive,alive,transition,.999,.95)
    assert torch.allclose(local[0],torch.full_like(local[0],.999)) # TD bootstrap survives boundary
    assert torch.all(global_adv[0]>local[0]) # later-wave rewards still propagate
    cont=global_adv-local;eta=continuation_coefficients(torch.tensor([[1],[2],[3]])).unsqueeze(-1)
    assert torch.allclose(eta.flatten(),torch.tensor([1/3,1/2,1.]))
    actor_adv=local+eta*cont
    assert torch.all(actor_adv[0]>local[0]) and torch.all(returns[0]>local_returns[0])

def test_combined_marc_advantage_weight_target_and_ten_epochs(trainer,monkeypatch):
    r=state_memory_rollout(trainer);cv=lambda x:torch.as_tensor(x,dtype=torch.float32)
    with torch.no_grad():
        v,nv=trainer._value_rollout(r,cv(r.observations),cv(r.next_observations))
        args=[cv(x) for x in (r.rewards,v,nv,r.dones,r.alive_masks,r.next_alive_masks)]
        global_adv,returns=compute_gae(*args,trainer.gamma,trainer.gae_lambda)
        local,_=compute_local_gae(*args,cv(r.wave_transition_flags),trainer.gamma,trainer.gae_lambda)
        waves=torch.as_tensor(r.wave_indices);mask=cv(r.alive_masks)
        adv=local+trainer.milestone_aware_retention_credit.coefficients(waves).unsqueeze(-1)*(global_adv-local)
        live=adv[mask>.5];adv=((adv-live.mean())/live.std(unbiased=False).clamp_min(1e-8))*mask
        weights,_=trainer.milestone_aware_retention_credit.wave_weights(waves,mask)
    seen={};original=trainer._update_actor_recurrent_critic_flat
    def capture(*args):seen.update(adv=args[6],target=args[8],weights=args[11]);return original(*args)
    monkeypatch.setattr(trainer,'_update_actor_recurrent_critic_flat',capture)
    monkeypatch.setattr(trainer,'_update_flat_marc',lambda *a:pytest.fail('SM entered flat MARC'))
    metrics=trainer.update(r)
    assert torch.equal(seen['adv'],adv) and torch.equal(seen['target'],returns) and torch.equal(seen['weights'],weights)
    assert metrics['ppo_epochs_configured']==metrics['ppo_epochs_executed']==10
    assert metrics['actor_optimizer_steps_this_update']==metrics['actor_optimizer_steps_expected']
    assert metrics['critic_optimizer_steps_this_update']==metrics['critic_optimizer_steps_expected']
    assert metrics['actor_gru_grad_norm']>0 and all(np.isfinite(v) for v in metrics.values())

def test_critic_update_exact_control(trainer):
    control=build_modular_mappo_trainer(load_config(CONTROL),'cpu')
    control.update(state_memory_rollout(control));trainer.update(state_memory_rollout(trainer))
    assert all(torch.equal(v,trainer.critic.state_dict()[k]) for k,v in control.critic.state_dict().items())

def test_zero_retention_skips_state_only_elite(trainer,monkeypatch):
    m=trainer.milestone_aware_retention_credit;before=deepcopy(m.rng.bit_generator.state)
    monkeypatch.setattr(trainer.actor,'distribution',lambda *a:pytest.fail('state-only distillation called'))
    m.ingest_success_segments([{'invalid':'should not inspect'}],trainer.actor,trainer.device)
    loss,_=m.retention_loss(trainer.actor,trainer.device)
    assert loss.item()==0 and not m.retention_active and before==m.rng.bit_generator.state

@pytest.mark.parametrize('bad',['obs','context','hidden','sequence','retention','critic_gru','phase','mode','extra','credit','balance','epochs','width'])
def test_bad_protocol_rejected(bad):
    cfg=load_config(CONFIG);r=cfg['modules']['recurrent_memory'];m=cfg['modules']['milestone_aware_retention_credit']
    if bad=='obs':cfg['network']['observation_dim']=55
    elif bad=='context':r['wave_context_dim']=4
    elif bad=='hidden':r['hidden_dim']=256
    elif bad=='sequence':r['sequence_length']=64
    elif bad=='retention':m['deployment_distill_coefficient']=.05
    elif bad=='critic_gru':r['mode']='actor_critic_gru'
    elif bad=='phase':r['initial_hidden']='learnable_phase'
    elif bad=='mode':r['mode']='wave_segmented_actor_gru'
    elif bad=='extra':cfg['modules']['wave_context']['enabled']=True
    elif bad=='credit':m['continuation_alpha']=0
    elif bad=='balance':m['wave_balance_temperature']=1
    elif bad=='epochs':cfg['training']['ppo_epochs']=1
    else:cfg['network']['actor_hidden_layers']=[128,128]
    with pytest.raises((RuntimeError,ValueError)):build_modular_mappo_trainer(cfg,'cpu')

def test_invalid_one_hot_rollout_rejected(trainer):
    r=state_memory_rollout(trainer);r.contexts[2,0,0]=[0,1,0]
    with pytest.raises(RuntimeError,match='one-hot'):trainer.update(r)

def test_checkpoint_roundtrip(trainer,tmp_path):
    trainer.update(state_memory_rollout(trainer));path=tmp_path/'checkpoint.pt';trainer.save(path)
    loaded=build_modular_mappo_trainer(load_config(CONFIG),'cpu');loaded.load(path)
    assert all(torch.equal(v,loaded.actor.state_dict()[k]) for k,v in trainer.actor.state_dict().items())
    assert loaded.actor_optimizer.state_dict()['state'] and loaded.critic_optimizer.state_dict()['state']

def test_runner_off_by_one_carry_resume_and_eval(tmp_path,monkeypatch):
    runner=ModularMAPPOTrainingRunner(load_config(ENV),load_config(CONFIG),num_envs=1,device='cpu',seed=88722001,output_dir=tmp_path,smoke=True)
    try:
        original=runner.vector.step_batch;count=[0]
        def step(actions):
            result=original(actions);count[0]+=1
            for info in result.infos:
                info['wave_index']=1 if count[0]<2 else 2 if count[0]<4 else 3
                info['spawned_next_wave']=count[0] in (2,4)
            return result
        monkeypatch.setattr(runner.vector,'step_batch',step)
        first=runner.collect_rollout(3);second=runner.collect_rollout(3)
        assert first.wave_indices[:,0].tolist()==[1,1,2] and second.wave_indices[:,0].tolist()==[2,3,3]
        assert first.actor_recurrent_phase_reset_flags[:,0].tolist()==[True,False,True]
        assert second.actor_recurrent_phase_reset_flags[:,0].tolist()==[False,True,False]
        assert not first.actor_hidden_before_step[[0,2]].any() and not second.actor_hidden_before_step[1].any()
        assert first.actor_hidden_before_step[1].any() and second.actor_hidden_before_step[0].any()
        for rollout in (first,second):
            expected=runner.trainer.actor_context_numpy(rollout.wave_indices,rollout.alive_masks,None)
            assert np.array_equal(rollout.contexts,expected)
        assert runner.last_rollout_metrics['state_memory_one_hot_alive_count_wave_2']==4
        assert runner.last_rollout_metrics['state_memory_one_hot_alive_count_wave_3']==8
        runner.trainer.update(second);path=tmp_path/'latest.pt';runner.save_checkpoint(path)
        actor={k:v.detach().clone() for k,v in runner.trainer.actor.state_dict().items()};runner.resume(path)
        assert not runner.actor_hidden.any() and runner.actor_phase_reset_flags.all() and np.all(runner.wave==1)
        assert all(torch.equal(v,runner.trainer.actor.state_dict()[k]) for k,v in actor.items())
        trace=controlled_evaluation(runner.trainer)
        assert trace['actor_hidden_before_step_norm_trace'].shape==(6,4)
    finally:runner.vector.close()
