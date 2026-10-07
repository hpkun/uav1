"""Tiny CUDA update/resume smoke; never calls train() or formal evaluation."""
from __future__ import annotations
import json,sys,tempfile
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.modular_mappo.buffer import ModularRolloutBatch
from algorithm.modular_mappo.runner import ModularMAPPOTrainingRunner
from algorithm.train_modular_mappo import load_config
from algorithm.modular_mappo.evaluation import evaluate_modular_episode
from env.factory import make_combat_environment
from tools.preflight_marc_wsgru_screen import CONTROL,TREATMENT,ENV

def phase_rollout(trainer):
    """Controlled three-phase data including both real pre-action reset positions."""
    T,E,N=9,2,4;rng=np.random.default_rng(88721001)
    obs=rng.normal(size=(T,E,N,52)).astype(np.float32)
    alive=np.ones((T,E,N),np.float32);alive[5:,0,3]=0
    waves=np.repeat(np.asarray([1,1,1,2,2,2,3,3,3])[:,None],E,axis=1)
    flags=np.zeros((T,E),bool);flags[[0,3,6]]=True
    episode=np.ones((T,E),np.float32);episode[0]=0
    raw=np.zeros((T,E,N,3),np.float32);actions=np.zeros_like(raw);old=np.zeros((T,E,N),np.float32)
    contexts=np.zeros((T,E,N,0),np.float32);saved=[]
    hidden,_=trainer.initial_hidden(E)
    for t in range(T):
        kwargs={}
        if trainer.recurrent.wave_segmented:
            hidden=trainer.prepare_actor_hidden(hidden,alive[t],waves[t],flags[t]);saved.append(hidden.copy())
            kwargs={'wave_indices':waves[t],'actor_phase_reset_flags':flags[t]}
        result=trainer.act(obs[t],alive[t],False,True,contexts[t],hidden,episode[t],**kwargs)
        actions[t],raw[t],old[t],hidden=result
    rewards=rng.normal(size=(T,E,N)).astype(np.float32)
    dones=np.zeros((T,E),np.float32);transitions=np.zeros((T,E),bool);transitions[[2,5]]=True
    return ModularRolloutBatch(obs,actions,raw,old,rewards,rewards.copy(),dones,alive,obs.copy(),alive.copy(),waves,
        np.full((T,E),3),contexts,contexts.copy(),np.asarray(saved) if saved else None,None,episode,
        wave_transition_flags=transitions,actor_recurrent_phase_reset_flags=flags if saved else None)

def tiny_phase_evaluation(trainer):
    """Six real physics steps with test-controlled transition signals, no formal bank."""
    env=make_combat_environment(load_config(ENV));original=env.step;counter=[0]
    def step(actions):
        obs,reward,terminated,truncated,info=original(actions);counter[0]+=1
        info['wave_index']=1 if counter[0]<2 else 2 if counter[0]<4 else 3
        info['spawned_next_wave']=counter[0] in (2,4)
        return obs,reward,terminated,counter[0]==6,info
    env.step=step
    with patch('algorithm.modular_mappo.evaluation.make_combat_environment',return_value=env):
        trace=evaluate_modular_episode(trainer,load_config(ENV),88721002,True)
    assert trace['wave_trace'].tolist()==[1,1,2,2,3,3]
    if trainer.recurrent.wave_segmented:
        assert trace['actor_recurrent_phase_reset_trace'].tolist()==[True,False,True,False,True,False]
    return {'steps':6,'synthetic_transition_signals':True,'wave_trace':[1,1,2,2,3,3],'lifecycle':'PASS'}

def smoke():
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory')
    report={'device':torch.cuda.get_device_name(0),'formal_training_executed':False,'formal_evaluation_executed':False,'holdout_executed':False}
    with tempfile.TemporaryDirectory(prefix='marc_wsgru_') as directory:
        for name,path in [('control',CONTROL),('treatment',TREATMENT)]:
            cfg=load_config(path);cfg['training']['seed']=88721001
            runner=ModularMAPPOTrainingRunner(load_config(ENV),cfg,num_envs=2,device='cuda',output_dir=Path(directory)/name,smoke=True)
            try:
                rollout=runner.collect_rollout(4);real_metrics=runner.trainer.update(rollout)
                constructed=phase_rollout(runner.trainer)
                metrics=runner.trainer.update(constructed)
                assert all(np.isfinite(v) for v in metrics.values())
                assert metrics['ppo_epochs_executed']==10
                state=runner.trainer.actor.state_dict()
                phase=state.get('phase_initial_hidden')
                phase=None if phase is None else phase.detach().clone()
                checkpoint=Path(directory)/name/'smoke.pt';runner.save_checkpoint(checkpoint)
                runner.resume(checkpoint)
                if phase is not None:
                    assert torch.equal(phase,runner.trainer.actor.phase_initial_hidden)
                    assert runner.actor_phase_reset_flags.all()
                    assert all(metrics[f'actor_phase_embedding_grad_norm_wave{w}']>0 for w in (1,2,3))
                    assert runner.trainer.critic.recurrent_hidden_dim==0
                report[name]={'enabled_modules':runner.trainer.module_protocol()['enabled_modules'],
                    'actual_environment_sampled_steps':8,'constructed_wave_indices':[1,2,3],
                    'ppo_epochs_executed':metrics['ppo_epochs_executed'],
                    'actor_steps':metrics['actor_optimizer_steps_this_update'],'critic_steps':metrics['critic_optimizer_steps_this_update'],
                    'finite_real_and_constructed_update':all(np.isfinite(v) for v in real_metrics.values()),
                    'checkpoint_resume':'PASS',
                    'tiny_canonical_evaluation':tiny_phase_evaluation(runner.trainer),
                    'phase_gradients':{str(w):metrics[f'actor_phase_embedding_grad_norm_wave{w}'] for w in (1,2,3)} if phase is not None else {}}
            finally:runner.vector.close()
    return report

if __name__=='__main__':print(json.dumps(smoke(),indent=2))
