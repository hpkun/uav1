"""Tiny CUDA checkpoint/resume and canonical controlled six-step evaluation."""
from pathlib import Path
import json,sys,tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools.preflight_jiao2025_core_3m import CONFIGS,ENV,load_config
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modular_mappo.evaluation import evaluate_modular_episode
from env.factory import make_combat_environment

def assert_state_equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a.to(b.device),b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:assert_state_equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for left,right in zip(a,b):assert_state_equal(left,right)
    else:assert a==b

def assert_finite_update(metrics):
    # Disabled-module metadata may be None/string; PPO diagnostics must be numeric.
    required=('actor_loss','value_loss','entropy','approx_kl','actor_gru_grad_norm','critic_gru_grad_norm','ppo_epochs_executed')
    for key in required:
        assert isinstance(metrics[key],(int,float,np.integer,np.floating)) and np.isfinite(metrics[key])
    numeric={k:v for k,v in metrics.items() if isinstance(v,(int,float,np.integer,np.floating))}
    assert all(np.isfinite(value) for value in numeric.values())
    return {k:type(v).__name__ for k,v in metrics.items() if k not in numeric}

def controlled_evaluation(trainer,seed):
    """Real physics, controlled wave signals only; not a combat success result."""
    env=make_combat_environment(load_config(ENV));original_step=env.step;count=[0]
    original_act=trainer.act;original_values=trainer.values_step;calls=[];critic_calls=[]
    def step(actions):
        obs,reward,terminated,truncated,info=original_step(actions);count[0]+=1
        info['wave_index']=1 if count[0]<2 else 2 if count[0]<4 else 3
        info['spawned_next_wave']=count[0] in (2,4)
        return obs,reward,False,count[0]==6,info
    env.step=step
    def act(obs,alive,deterministic,return_data,ctx,hidden,episode,**kwargs):
        with torch.no_grad():
            conv=lambda x:None if x is None else torch.as_tensor(x,dtype=torch.float32,device=trainer.device)
            dist,_=trainer.actor.distribution_step(conv(obs),trainer._ctx(conv(ctx),True),conv(hidden),conv(episode),conv(alive))
            expected=torch.tanh(dist.loc).cpu().numpy()*alive[...,None]
        result=original_act(obs,alive,deterministic,return_data,ctx,hidden,episode,**kwargs)
        assert deterministic and not return_data
        assert np.allclose(result[0],expected,atol=1e-7)
        calls.append((ctx.copy(),None if hidden is None else hidden.copy(),result[1],episode.copy()))
        return result
    def values(obs,alive,ctx,hidden,episode,**kwargs):
        result=original_values(obs,alive,ctx,hidden,episode,**kwargs)
        critic_calls.append((None if hidden is None else hidden.copy(),result[1]));return result
    with patch('algorithm.modular_mappo.evaluation.make_combat_environment',return_value=env),patch.object(trainer,'act',side_effect=act),patch.object(trainer,'values_step',side_effect=values):
        trace=evaluate_modular_episode(trainer,load_config(ENV),seed,True)
    assert trace['wave_trace'].tolist()==[1,1,2,2,3,3]
    if trainer.recurrent.enabled:
        assert [float(row[0][0,0]) for row in calls]==[1,1,2,2,3,3]
        assert not calls[0][1].any() and not critic_calls[0][0].any()
        for t in range(1,6):
            assert np.array_equal(calls[t][1],calls[t-1][2])
            assert np.array_equal(critic_calls[t][0],critic_calls[t-1][1])
            assert calls[t][3].all()
    return {'canonical_evaluator':True,'deterministic_tanh_mean':True,'wave_trace':trace['wave_trace'].tolist(),
            'controlled_transition_signals':True,'episode_seed':seed,'episode_steps':6}

def smoke_variant(variant,device='cuda'):
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required')
    with tempfile.TemporaryDirectory(prefix='jiao3m_') as directory:
        runner=ModularMAPPOTrainingRunner(load_config(ENV),load_config(CONFIGS[variant]),num_envs=2,
            device=device,seed=99001001,output_dir=directory,smoke=True)
        try:
            for line in runner.startup_console_lines():print(line)
            # Exercise actual vector collection, with controlled wave-index signals.
            original=runner.vector.step_batch;count=[0]
            def step(actions):
                result=original(actions);count[0]+=1
                assert not (result.terminated|result.truncated).any()
                for info in result.infos:
                    info['wave_index']=1 if count[0]<2 else 2 if count[0]<4 else 3
                    info['spawned_next_wave']=count[0] in (2,4)
                return result
            with patch.object(runner.vector,'step_batch',side_effect=step):
                first=runner.collect_rollout(2)
                ah=None if runner.actor_hidden is None else runner.actor_hidden.copy()
                ch=None if runner.critic_hidden is None else runner.critic_hidden.copy()
                second=runner.collect_rollout(4)
            assert np.array_equal(first.rewards,first.raw_environment_rewards)
            assert np.array_equal(second.rewards,second.raw_environment_rewards)
            if variant=='core':
                assert first.contexts[:,0,0].tolist()==[1,1]
                assert second.contexts[:,0,0].tolist()==[2,2,3,3]
                assert np.array_equal(second.actor_hidden_before_step[0],ah) and ah.any()
                assert np.array_equal(second.critic_hidden_before_step[0],ch) and ch.any()
            metrics=runner.trainer.update(second)
            nonnumeric=assert_finite_update(metrics)
            assert metrics['ppo_epochs_executed']==10
            if variant=='core':
                assert metrics['actor_gru_grad_norm']>0 and metrics['critic_gru_grad_norm']>0
                assert runner.trainer.popart.count.item()>0
            before={name:{k:v.clone() for k,v in getattr(runner.trainer,name).state_dict().items()} for name in ('actor','critic','popart')}
            path=Path(directory)/'smoke.pt';runner.save_checkpoint(path)
            checkpoint=torch.load(path,map_location=runner.trainer.device,weights_only=False)
            for key in ('actor_optimizer','critic_optimizer','rng_state','sampled_steps','popart'):assert key in checkpoint
            runner.resume(path)
            assert_state_equal(checkpoint['actor_optimizer'],runner.trainer.actor_optimizer.state_dict())
            assert_state_equal(checkpoint['critic_optimizer'],runner.trainer.critic_optimizer.state_dict())
            assert runner.trainer.sampled_steps==checkpoint['sampled_steps']
            assert_state_equal(checkpoint['rng_state']['trainer_permutation_rng_state'],runner.trainer.rng.bit_generator.state)
            assert runner.trainer.rng_restore_metadata['rng_state_restored']
            for name in before:assert all(torch.equal(v,getattr(runner.trainer,name).state_dict()[k]) for k,v in before[name].items())
            if variant=='core':assert not runner.actor_hidden.any() and not runner.critic_hidden.any()
            evaluation=controlled_evaluation(runner.trainer,99001002)
            return dict(variant=variant,environment_sampled_steps=12,epochs=metrics['ppo_epochs_executed'],
                actor_optimizer_steps=metrics['actor_optimizer_steps_this_update'],critic_optimizer_steps=metrics['critic_optimizer_steps_this_update'],
                actor_GRU_grad=metrics['actor_gru_grad_norm'],critic_GRU_grad=metrics['critic_gru_grad_norm'],
                finite=True,nonnumeric_module_diagnostics=nonnumeric,checkpoint_resume='PASS',reward_unchanged=True,evaluation=evaluation)
        finally:runner.vector.close()

if __name__=='__main__':
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    print(json.dumps(dict(status='JIAO2025_3M_TINY_CUDA_SMOKE_PASS',cells=[smoke_variant(v) for v in CONFIGS],
        formal_training_started=False,development_44M_executed=False,future_45M_executed=False),indent=2))
