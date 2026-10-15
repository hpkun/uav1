"""Observe the actual formal runner at short duration without schedule overrides."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse
import math
import torch
import yaml
from algorithm.madsac.runner import MADSACTrainingRunner
from tools.madsac_audit_reference import require_cuda,dump


def theoretical_counts(config,total=None):
    t=config['training']; m=int(t['num_train_envs']); total=int(total or t['total_sampled_steps'])
    if total%m: raise ValueError('total must be divisible by environments')
    g=int(t['gradient_steps_per_vector_step']); d=int(t['policy_delay'])
    first=max(1,math.ceil(int(t['learning_starts'])/m),math.ceil(int(t['minibatch_size'])/m))
    vectors=total//m; critics=max(0,vectors-first+1)*g; actors=critics//d
    return dict(num_train_envs=m,total_sampled_steps=total,vector_steps=vectors,
        learning_starts=t['learning_starts'],minibatch_size=t['minibatch_size'],
        gradient_steps_per_vector_step=g,policy_delay=d,first_update_vector_step=first,
        first_update_sampled_steps=first*m,critic_update_count=critics,actor_update_count=actors,
        critic1_optimizer_steps=critics,critic2_optimizer_steps=critics,combined_critic_optimizer_steps=2*critics,
        replay_size=min(total,t['replay_capacity']),
        steady_state=dict(critic_UTD=2*g/m,per_critic_UTD=g/m,critic_round_UTD=g/m,
            actor_UTD=g/(m*d),combined_critic_optimizer_UTD=2*g/m,
            critic_updates_per_vector_step=g,actor_updates_per_vector_step=g/d,
            critic_updates_per_1000_transitions=1000*g/m,actor_updates_per_1000_transitions=1000*g/(m*d),
            combined_critic_optimizer_steps_per_vector_step=2*g,
            combined_critic_optimizer_steps_per_1000_transitions=2000*g/m,
            transitions_per_critic_update=m/g,transitions_per_critic_optimizer_step=m/(2*g),
            transitions_per_actor_update=m*d/g),
        whole_run=dict(critic_UTD=2*critics/total,per_critic_UTD=critics/total,
            critic_round_UTD=critics/total,actor_UTD=actors/total,
            combined_critic_optimizer_UTD=2*critics/total),
        count_convention='critic_update_count = paired Q1/Q2 update rounds; each round performs TWO separate optimizer steps')


def run(config,env_config,output,total=24000):
    require_cuda(); torch.set_num_threads(1)
    c=yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    e=yaml.safe_load(Path(env_config).read_text(encoding='utf-8'))
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    formal={str(path):theoretical_counts(yaml.safe_load(path.read_text(encoding='utf-8')))
            for path in sorted((ROOT/'configs').glob('madsac_4v4_v33[567].yaml'))}
    expected=theoretical_counts(c,total)
    # smoke=False is essential: smoke=True changes learning_starts, batch and hidden width.
    runner=MADSACTrainingRunner(e,c,total_sampled_steps=total,device='cuda',output_dir=out/'schedule_smoke',smoke=False)
    actual=dict(critic1_optimizer_steps=0,critic2_optimizer_steps=0,actor_optimizer_steps=0,
                replay_sample_calls=0,replay_insert_calls=0,inserted_transitions=0)
    handles=[]
    for name in ('critic1','critic2','actor'):
        def hook(optimizer,args,kwargs,key=name+'_optimizer_steps'):
            actual[key]+=1
        handles.append(getattr(runner.trainer,name+'_optimizer').register_step_post_hook(hook))
    sample=runner.replay.sample; add=runner.replay.add_batch
    def observed_sample(*args,**kwargs):
        actual['replay_sample_calls']+=1
        return sample(*args,**kwargs)
    def observed_add(*args,**kwargs):
        actual['replay_insert_calls']+=1;actual['inserted_transitions']+=len(args[0])
        return add(*args,**kwargs)
    runner.replay.sample=observed_sample;runner.replay.add_batch=observed_add
    result=runner.run()
    for handle in handles: handle.remove()
    actual.update({k:result[k] for k in ('sampled_steps','vector_steps','replay_size','critic_update_count','actor_update_count')})
    checks={k:actual[k]==expected[k] for k in ('vector_steps','replay_size','critic_update_count','actor_update_count',
                                             'critic1_optimizer_steps','critic2_optimizer_steps')}
    checks.update(sampled_steps=actual['sampled_steps']==total,
        actor_optimizer=actual['actor_optimizer_steps']==expected['actor_update_count'],
        replay_sample_calls=actual['replay_sample_calls']==expected['critic_update_count'],
        replay_insert_calls=actual['replay_insert_calls']==expected['vector_steps'],
        inserted_transitions=actual['inserted_transitions']==total)
    payload=dict(formal_configs=formal,smoke_expected=expected,smoke_actual=actual,checks=checks,
        all_counts_match=all(checks.values()),smoke_flag=False,protocol=runner.startup_summary(),
        note='Only total_sampled_steps reduced; actual production run(), replay, optimizer and schedule retained.')
    dump(out/'schedule.json',payload)
    assert all(checks.values()),checks
    print(actual,flush=True)
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/madsac_4v4_v337.yaml')
    p.add_argument('--env-config',default='configs/combat_environment_v337.yaml')
    p.add_argument('--output-dir',default='outputs/madsac_algorithm_audit')
    p.add_argument('--total-sampled-steps',type=int,default=24000)
    a=p.parse_args();run(a.config,a.env_config,a.output_dir,a.total_sampled_steps)
