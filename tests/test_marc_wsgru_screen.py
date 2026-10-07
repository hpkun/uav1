"""Phase lifecycle, wave BPTT, MARC composition and matched protocol regressions."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from algorithm.modular_mappo.factory import build_modular_mappo_trainer
from algorithm.modular_mappo.buffer import wave_segmented_chunks,contiguous_chunks
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modular_mappo.evaluation import evaluate_modular_episode
from algorithm.modular_mappo.trainer import compute_local_gae
from algorithm.mappo.trainer import compute_gae
from algorithm.modules import tempered_wave_weights,continuation_coefficients
from algorithm.train_modular_mappo import load_config
from tools.preflight_marc_wsgru_screen import CONTROL,TREATMENT,ENV,validate_pair,matched_initialization
from tools.smoke_marc_wsgru_screen import phase_rollout

@pytest.fixture
def treatment():
    return build_modular_mappo_trainer(load_config(TREATMENT),'cpu')

def test_pair_protocol():
    assert validate_pair(load_config(CONTROL),load_config(TREATMENT),load_config(ENV))

@pytest.mark.parametrize('seed',[5301,5302])
def test_matched_initialization(seed):
    assert all(matched_initialization('cpu',seed).values())

def test_phase_parameter_topology(treatment):
    p=treatment.actor.phase_initial_hidden
    assert p.shape==(3,128) and torch.count_nonzero(p)==0 and p.requires_grad
    assert any(p is v for v in treatment.actor.trainable_policy_parameters())
    assert any(p is v for group in treatment.actor_optimizer.param_groups for v in group['params'])
    assert treatment.critic.recurrent_hidden_dim==0
    assert treatment.actor.backbone[0].in_features==52 and treatment.actor.backbone[0].out_features==256
    assert treatment.actor.mean.in_features==128

@pytest.mark.parametrize('wave',[1,2,3])
def test_live_phase_lifecycle(treatment,wave):
    with torch.no_grad():treatment.actor.phase_initial_hidden[wave-1].fill_(wave)
    hidden=np.full((1,4,128),-8,np.float32);alive=np.asarray([[1,1,1,0]],np.float32)
    reset=treatment.prepare_actor_hidden(hidden,alive,[wave],[True])
    assert np.all(reset[0,:3]==wave) and np.all(reset[0,3]==0)
    carried=treatment.prepare_actor_hidden(hidden,alive,[wave],[False])
    assert np.all(carried[0,:3]==-8) and np.all(carried[0,3]==0)
    # A rollout boundary by itself is no phase boundary.
    assert np.array_equal(carried,treatment.prepare_actor_hidden(carried,alive,[wave],[False]))

def test_old_actor_gru_not_phase_initialized():
    cfg=load_config(TREATMENT.parents[0]/'dev_actor_only_gru_history_3m.yaml')
    trainer=build_modular_mappo_trainer(cfg,'cpu')
    assert not trainer.recurrent.wave_segmented and not hasattr(trainer.actor,'phase_initial_hidden')
    hidden=np.ones((1,4,trainer.recurrent.hidden_dim),np.float32)
    assert trainer.prepare_actor_hidden(hidden,np.ones((1,4)),[2],[True]) is hidden
    trainer.recurrent.reset_for_episode(hidden,np.asarray([True]));assert not hidden.any()
    assert contiguous_chunks(6,1,32)==[(0,0,6)]

def test_missing_phase_provenance_fails(treatment):
    with pytest.raises((RuntimeError,ValueError)):
        treatment.act(np.zeros((1,4,52)),np.ones((1,4)))

def test_chunk_plan_exact_coverage():
    waves=np.ones((90,2),int);waves[20:60,0]=2;waves[60:,0]=3;waves[45:75,1]=2;waves[75:,1]=1
    flags=np.zeros_like(waves,bool);flags[0]=True;flags[[20,60],0]=True;flags[[45,75],1]=True
    episode=np.ones_like(waves,float);episode[0]=0;episode[75,1]=0
    counts=np.zeros_like(waves)
    for env,start,end in wave_segmented_chunks(waves,flags,episode,32):
        assert 0<end-start<=32 and len(set(waves[start:end,env]))==1
        assert not np.any(episode[start+1:end,env]==0)
        counts[start:end,env]+=1
    assert np.all(counts==1)

@pytest.mark.parametrize('invalid',['missing','spurious','episode','length'])
def test_chunk_provenance_rejected(invalid):
    waves=np.asarray([[1],[1],[2]]);flags=np.asarray([[1],[0],[1]]);ep=np.asarray([[0],[1],[1]])
    length=32
    if invalid=='missing':flags[2]=0
    elif invalid=='spurious':flags[1]=1
    elif invalid=='episode':flags[0]=0
    else:length=0
    with pytest.raises((RuntimeError,ValueError)):wave_segmented_chunks(waves,flags,ep,length)

@pytest.mark.parametrize('wave',[1,2,3])
def test_phase_start_chunk_real_ppo_gradient(treatment,wave):
    r=phase_rollout(treatment);t=3*(wave-1)
    convert=lambda x:torch.as_tensor(x,dtype=torch.float32)
    args=[convert(x) for x in (r.observations,r.actions,r.raw_actions,r.old_log_probs,r.alive_masks)]
    adv=torch.ones_like(args[-1]);weights=torch.ones(r.wave_indices.shape);ctx=convert(r.contexts)
    # Saved rollout copy is deliberately wrong: phase start must use current live e_w.
    r.actor_hidden_before_step[t]=99
    seen=[];original=treatment.actor.distribution_step
    def capture(obs,context,hidden,*a,**kw):
        seen.append(hidden.detach().clone());return original(obs,context,hidden,*a,**kw)
    treatment.actor.distribution_step=capture
    row=treatment._actor_recurrent_minibatch(r,[(0,t,t+3)],*args,adv,weights,ctx,convert)
    assert torch.count_nonzero(seen[0])==0
    assert np.isfinite(row[f'actor_phase_embedding_grad_norm_wave{wave}'])
    assert row[f'actor_phase_embedding_grad_norm_wave{wave}']>0
    grad=treatment.actor.phase_initial_hidden.grad
    assert all(torch.count_nonzero(grad[w-1])==0 for w in (1,2,3) if w!=wave)

def test_ordinary_chunk_uses_detached_stored_hidden(treatment):
    r=phase_rollout(treatment);r.actor_hidden_before_step[1,0]=.4
    cv=lambda x:torch.as_tensor(x,dtype=torch.float32)
    args=[cv(x) for x in (r.observations,r.actions,r.raw_actions,r.old_log_probs,r.alive_masks)]
    seen=[];original=treatment.actor.distribution_step
    def capture(obs,context,hidden,*a,**kw):seen.append(hidden);return original(obs,context,hidden,*a,**kw)
    treatment.actor.distribution_step=capture
    treatment._actor_recurrent_minibatch(r,[(0,1,3)],*args,torch.ones_like(args[-1]),torch.ones(r.wave_indices.shape),cv(r.contexts),cv)
    assert not seen[0].requires_grad and torch.all(seen[0]==.4)
    gradient=treatment.actor.phase_initial_hidden.grad
    assert gradient is None or torch.count_nonzero(gradient)==0

def test_marc_advantage_weights_targets_and_ten_epochs(treatment,monkeypatch):
    r=phase_rollout(treatment);cv=lambda x:torch.as_tensor(x,dtype=torch.float32)
    with torch.no_grad():
        values,next_values=treatment._value_rollout(r,cv(r.observations),cv(r.next_observations))
        args=[cv(x) for x in (r.rewards,values,next_values,r.dones,r.alive_masks,r.next_alive_masks)]
        global_adv,returns=compute_gae(*args,treatment.gamma,treatment.gae_lambda)
        local,_=compute_local_gae(*args,cv(r.wave_transition_flags),treatment.gamma,treatment.gae_lambda)
        eta=continuation_coefficients(torch.as_tensor(r.wave_indices),3,1.).unsqueeze(-1)
        adv=local+eta*(global_adv-local);mask=cv(r.alive_masks);live=adv[mask>.5]
        adv=((adv-live.mean())/live.std(unbiased=False).clamp_min(1e-8))*mask
        expected_weights,_=treatment.milestone_aware_retention_credit.wave_weights(torch.as_tensor(r.wave_indices),mask)
    original=treatment._update_actor_recurrent_critic_flat;seen={}
    def capture(*args,**kwargs):
        seen['adv']=args[6];seen['target']=args[8];seen['weights']=args[11]
        return original(*args,**kwargs)
    monkeypatch.setattr(treatment,'_update_actor_recurrent_critic_flat',capture)
    monkeypatch.setattr(treatment,'_update_flat_marc',lambda *a,**kw:pytest.fail('recurrent treatment entered flat MARC'))
    metrics=treatment.update(r)
    assert torch.equal(seen['adv'],adv) and torch.equal(seen['target'],returns) and torch.equal(seen['weights'],expected_weights)
    assert metrics['ppo_epochs_configured']==metrics['ppo_epochs_executed']==10
    assert metrics['actor_optimizer_steps_this_update']==metrics['actor_optimizer_steps_expected']
    assert metrics['critic_optimizer_steps_this_update']==metrics['critic_optimizer_steps_expected']
    assert all(np.isfinite(v) for v in metrics.values())

def test_zero_retention_never_state_only(treatment,monkeypatch):
    module=treatment.milestone_aware_retention_credit
    before=deepcopy(module.rng.bit_generator.state)
    monkeypatch.setattr(treatment.actor,'distribution',lambda *a:pytest.fail('state-only loss called'))
    module.ingest_success_segments([{'bad':'must not inspect'}],treatment.actor,treatment.device)
    loss,_=module.retention_loss(treatment.actor,treatment.device)
    assert loss.item()==0 and not module.retention_active and module.rng.bit_generator.state==before

@pytest.mark.parametrize('bad',['positive','old_mode','extra','formal_v2','critic_gru'])
def test_strict_experimental_isolation(bad):
    cfg=load_config(TREATMENT)
    if bad=='positive':cfg['modules']['milestone_aware_retention_credit']['deployment_distill_coefficient']=.05
    elif bad=='old_mode':cfg['modules']['recurrent_memory']['mode']='actor_gru'
    elif bad=='extra':cfg['modules']['popart']['enabled']=True
    elif bad=='formal_v2':cfg['development_method']='marc_mappo_v2'
    else:cfg['modules']['recurrent_memory']['mode']='actor_critic_gru'
    with pytest.raises((RuntimeError,ValueError)):build_modular_mappo_trainer(cfg,'cpu')

def test_phase_checkpoint_roundtrip(treatment,tmp_path):
    with torch.no_grad():treatment.actor.phase_initial_hidden.copy_(torch.arange(384).reshape(3,128))
    path=tmp_path/'checkpoint.pt';treatment.save(path)
    restored=build_modular_mappo_trainer(load_config(TREATMENT),'cpu');restored.load(path)
    assert torch.equal(restored.actor.phase_initial_hidden,treatment.actor.phase_initial_hidden)

def test_real_runner_transition_resume_and_eval_lifecycle(tmp_path,monkeypatch):
    cfg=load_config(TREATMENT);runner=ModularMAPPOTrainingRunner(load_config(ENV),cfg,num_envs=1,device='cpu',output_dir=tmp_path,smoke=True)
    try:
        with torch.no_grad():
            for w in (1,2,3):runner.trainer.actor.phase_initial_hidden[w-1].fill_(w/10)
        original=runner.vector.step_batch;step=[0]
        def controlled(actions):
            result=original(actions);step[0]+=1
            for info in result.infos:
                info['wave_index']=1 if step[0]<2 else 2 if step[0]<4 else 3
                info['spawned_next_wave']=step[0] in (2,4)
            return result
        monkeypatch.setattr(runner.vector,'step_batch',controlled)
        first=runner.collect_rollout(3);second=runner.collect_rollout(3)
        assert first.wave_indices[:,0].tolist()==[1,1,2]
        assert second.wave_indices[:,0].tolist()==[2,3,3]
        assert first.actor_recurrent_phase_reset_flags[:,0].tolist()==[True,False,True]
        assert second.actor_recurrent_phase_reset_flags[:,0].tolist()==[False,True,False]
        assert np.allclose(first.actor_hidden_before_step[0,0],.1)
        assert np.allclose(first.actor_hidden_before_step[2,0],.2)
        assert np.allclose(second.actor_hidden_before_step[1,0],.3)
        assert not np.allclose(second.actor_hidden_before_step[0,0],.2) # carry at rollout truncation
        path=tmp_path/'latest.pt';runner.save_checkpoint(path);runner.resume(path)
        assert runner.actor_phase_reset_flags.all() and np.allclose(runner.actor_hidden,.1)
        # Canonical evaluator with a 6-step controlled environment, not a formal bank.
        from algorithm.modular_mappo import evaluation
        from env.factory import make_combat_environment
        env=make_combat_environment(load_config(ENV));original_step=env.step;count=[0]
        def eval_step(actions):
            obs,reward,terminated,truncated,info=original_step(actions);count[0]+=1
            info['wave_index']=1 if count[0]<2 else 2 if count[0]<4 else 3
            info['spawned_next_wave']=count[0] in (2,4)
            return obs,reward,terminated,count[0]==6,info
        monkeypatch.setattr(env,'step',eval_step);monkeypatch.setattr(evaluation,'make_combat_environment',lambda c:env)
        trace=evaluate_modular_episode(runner.trainer,load_config(ENV),88721002,True)
        assert trace['wave_trace'].tolist()==[1,1,2,2,3,3]
        assert trace['actor_recurrent_phase_reset_trace'].tolist()==[True,False,True,False,True,False]
    finally:runner.vector.close()
