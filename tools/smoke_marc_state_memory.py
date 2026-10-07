"""Tiny CUDA MARC-SM rollout, three-phase synthetic update, resume and evaluation."""
from __future__ import annotations
import json,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.modular_mappo.evaluation import evaluate_modular_episode
from env.factory import make_combat_environment
from tools.preflight_marc_state_memory import CONFIG,ENV

def state_memory_rollout(trainer):
    T,E,N=9,2,4;rng=np.random.default_rng(88722001)
    obs=rng.normal(size=(T,E,N,52)).astype(np.float32)
    alive=np.ones((T,E,N),np.float32);alive[5:,0,3]=0
    waves=np.repeat(np.asarray([1,1,1,2,2,2,3,3,3])[:,None],E,axis=1)
    flags=np.zeros((T,E),bool);flags[[0,3,6]]=True
    episode=np.ones((T,E),np.float32);episode[0]=0
    contexts=trainer.actor_context_numpy(waves,alive,np.zeros((T,E,N,0),np.float32))
    raw=np.zeros((T,E,N,3),np.float32);actions=np.zeros_like(raw);old=np.zeros((T,E,N),np.float32)
    hidden,_=trainer.initial_hidden(E);saved=[]
    for t in range(T):
        kwargs={}
        if trainer.recurrent.wave_boundary_reset:
            hidden=trainer.prepare_actor_hidden(hidden,alive[t],waves[t],flags[t]);saved.append(hidden.copy())
            kwargs={'wave_indices':waves[t],'actor_phase_reset_flags':flags[t]}
        actions[t],raw[t],old[t],hidden=trainer.act(obs[t],alive[t],False,True,contexts[t],hidden,episode[t],**kwargs)
    rewards=rng.normal(size=(T,E,N)).astype(np.float32);dones=np.zeros((T,E),np.float32)
    transitions=np.zeros((T,E),bool);transitions[[2,5]]=True
    return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),dones,alive,obs.copy(),alive.copy(),waves,
        np.full((T,E),3),contexts,contexts.copy(),np.asarray(saved) if saved else None,None,episode,
        wave_transition_flags=transitions,actor_recurrent_phase_reset_flags=flags if saved else None)

def controlled_evaluation(trainer):
    """Six physics steps with controlled transition signals, not combat success evidence."""
    env=make_combat_environment(load_config(ENV));original=env.step;count=[0]
    def step(actions):
        obs,reward,terminated,truncated,info=original(actions);count[0]+=1
        info['wave_index']=1 if count[0]<2 else 2 if count[0]<4 else 3
        info['spawned_next_wave']=count[0] in (2,4)
        return obs,reward,terminated,count[0]==6,info
    env.step=step
    with patch('algorithm.modular_mappo.evaluation.make_combat_environment',return_value=env):
        trace=evaluate_modular_episode(trainer,load_config(ENV),88722002,True)
    assert trace['wave_trace'].tolist()==[1,1,2,2,3,3]
    assert trace['actor_recurrent_phase_reset_trace'].tolist()==[True,False,True,False,True,False]
    for t,w in enumerate(trace['wave_trace']):
        assert np.array_equal(trace['actor_wave_one_hot_trace'][t],np.broadcast_to(np.eye(3)[w-1],(4,3)))
        norms=trace['actor_hidden_before_step_norm_trace'][t]
        assert np.all(norms==0) if t in (0,2,4) else np.all(norms>0)
    return trace

def smoke():
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory')
    with tempfile.TemporaryDirectory(prefix='marc_sm_') as directory:
        cfg=load_config(CONFIG)
        runner=ModularMAPPOTrainingRunner(load_config(ENV),cfg,num_envs=2,device='cuda',seed=88722001,output_dir=directory,smoke=True)
        try:
            real=runner.collect_rollout(4);real_metrics=runner.trainer.update(real)
            batch=state_memory_rollout(runner.trainer);metrics=runner.trainer.update(batch)
            assert all(np.isfinite(v) for v in (*real_metrics.values(),*metrics.values()))
            assert metrics['actor_gru_grad_norm']>0 and metrics['ppo_epochs_executed']==10
            path=Path(directory)/'smoke.pt';runner.save_checkpoint(path)
            before={k:v.detach().clone() for k,v in runner.trainer.actor.state_dict().items()}
            runner.resume(path)
            assert all(torch.equal(v,runner.trainer.actor.state_dict()[k]) for k,v in before.items())
            assert not runner.actor_hidden.any() and runner.actor_phase_reset_flags.all()
            trace=controlled_evaluation(runner.trainer)
            return {'status':'MARC_SM_TINY_CUDA_SMOKE_PASS','device':torch.cuda.get_device_name(0),
                'actual_environment_sampled_steps':8,'constructed_wave_indices':[1,2,3],
                'controlled_evaluation_steps':6,'controlled_transition_signals':True,
                'pre_action_wave_trace':trace['wave_trace'].tolist(),
                'pre_action_reset_trace':trace['actor_recurrent_phase_reset_trace'].tolist(),
                'zero_boundary_hidden':'PASS','one_hot_context':'PASS','finite_update':True,
                'gru_grad_norm':metrics['actor_gru_grad_norm'],'ppo_epochs_executed':metrics['ppo_epochs_executed'],
                'actor_optimizer_steps':metrics['actor_optimizer_steps_this_update'],
                'critic_optimizer_steps':metrics['critic_optimizer_steps_this_update'],
                'sequence_chunks':metrics['sequence_chunks'],'recurrent_minibatches_per_epoch':metrics['recurrent_minibatches_per_epoch'],
                'checkpoint_resume':'PASS','formal_training_started':False,'formal_evaluation_started':False,'future_holdout_executed':False}
        finally:runner.vector.close()

if __name__=='__main__':print(json.dumps(smoke(),indent=2))
