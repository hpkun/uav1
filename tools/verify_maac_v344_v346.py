"""Bounded CUDA integration audit of existing MAAC; never starts a 3M run."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.maac.runner import MAACTrainingRunner
from algorithm.maac.trainer import MAACTrainer
from algorithm.maac.protocol import validate_checkpoint,validate_checkpoint_for_resume
from algorithm.maac.metrics import evaluation_metrics
from algorithm.common.evaluator import evaluate
from env.config import load_config
from tools.record_combat_episode import load_recording_policy,record
from tools.combat_trace import read_trace
from tools.combat_replay_report import export_analysis


def dump(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def configs(v):
    return load_config(ROOT/f'configs/combat_environment_v{v}.yaml'),yaml.safe_load(
        (ROOT/f'configs/maac_4v4_v{v}.yaml').read_text())


def closed(runner):
    runner.vector.close()
    assert runner.vector._closed and all(not p.is_alive() for p in runner.vector._processes)


def formal_update(v,out):
    out.mkdir(parents=True,exist_ok=True)
    ec,ac=configs(v)
    # Explicit finite diagnostic budget; all formal algorithm parameters retained.
    runner=MAACTrainingRunner(ec,ac,out,total_sampled_steps=12000,seed=int(v),smoke=False)
    handles=[]
    try:
        t=runner.trainer
        assert runner.num_envs==12 and runner.batch_size==1024 and t.baseline_samples==16
        assert t.critic.hidden_dim==128 and t.critic.heads==4
        assert runner.replay.capacity==1000000 and runner.update_interval==100
        assert runner.observations.shape==(12,4,49)
        grads={name:dict(calls=0,max_abs=0.,finite=True) for name in ('actor','critic','attention')}
        def hook(name):
            def receive(g):
                assert torch.isfinite(g).all(),name+' gradient nonfinite'
                grads[name]['calls']+=1
                grads[name]['max_abs']=max(grads[name]['max_abs'],float(g.abs().max()))
            return receive
        for name in ('actor','critic'):
            for p in getattr(t,name).parameters():handles.append(p.register_hook(hook(name)))
        for p in t.critic.shared_parameters():handles.append(p.register_hook(hook('attention')))
        before={name:[p.clone() for p in getattr(t,name).parameters()]
                for name in ('actor','critic','target_actor','target_critic')}
        infos=[]
        for _ in range(1000):
            infos.extend(runner.collect_step())
            if t.sampled_steps>=1024 and len(runner.records)>=12:break
        assert len(runner.replay)==t.sampled_steps and len(runner.records)>=12
        batch=runner.replay.sample(1024);d=t.batch_tensors(batch)
        assert d['observations'].shape==(1024,4,49)
        all_samples=runner.replay.sample(len(runner.replay))
        assert all_samples.dones.sum()>=12 and np.any(all_samples.next_alive_masks==0)
        y=t.td_targets(d['rewards'],d['next_observations'],d['dones'],d['next_alive_masks'])
        stop=d['dones'][:,None].bool()|~d['next_alive_masks'].bool()
        torch.testing.assert_close(y[stop],d['rewards'][stop],rtol=0,atol=0)
        base=t.counterfactual_baseline(d['observations'],d['actions'],d['alive_masks'],0)
        assert torch.isfinite(base).all() and base.shape==(1024,) and not base.requires_grad
        runner.train_cycle()  # unchanged four critic + four shared-policy updates
        assert (t.critic_updates,t.actor_updates,t.policy_updates,t.target_updates)==(4,4,4,1)
        changes={}
        for name,old in before.items():
            changes[name]=float(sum((p-q).square().sum() for p,q in zip(getattr(t,name).parameters(),old)).sqrt().detach())
            assert changes[name]>0
        for values in grads.values():assert values['calls']>0 and values['max_abs']>0
        for optim in (t.actor_optimizer,t.critic_optimizer):
            assert optim.state
            for state in optim.state.values():
                assert all(torch.isfinite(x).all() for x in state.values() if isinstance(x,torch.Tensor))
        runner.write_metrics(infos)
        checkpoint=out/'diagnostic_bounded.pt';t.save(checkpoint)
        state=torch.load(checkpoint,map_location='cuda',weights_only=False)
        validate_checkpoint(state,ec,ac)
        for name in ('actor_optimizer','critic_optimizer'):assert state[name]['state']
        for name in before:
            assert all(torch.equal(x,state[name][k]) for k,x in getattr(t,name).state_dict().items())
        for name in ('sampled_steps','actor_updates','critic_updates','policy_updates','target_updates'):
            assert state[name]==getattr(t,name)
        reloaded,saved,provenance=load_recording_policy(checkpoint,ec,ac,'cuda')
        np.testing.assert_array_equal(t.act(runner.observations,runner.vector.current_alive_masks),
                                      reloaded.act(runner.observations,runner.vector.current_alive_masks))
        for name in before:
            for p,q in zip(getattr(t,name).parameters(),getattr(reloaded,name).parameters()):assert torch.equal(p,q)
        rejected=[]
        for other in ('341','344','346'):
            if other==v:continue
            try:validate_checkpoint(state,*configs(other))
            except RuntimeError:rejected.append(other)
            else:raise AssertionError('cross-version checkpoint accepted')
        try:validate_checkpoint_for_resume(state,ec,ac)
        except RuntimeError as e:assert 'exact resume unsupported' in str(e)
        else:raise AssertionError('exact resume unexpectedly accepted')
        # Run official runner evaluator with two diagnostic seeds, retaining config identity.
        runner.eval_seeds=list(range(10000000,10000002))
        evaluation=runner.evaluate();dump(out/'evaluation.json',evaluation)
        recording=out/'recording';metadata=record(reloaded,ec,20000000+int(v),recording,provenance)
        trace=read_trace(recording/'episode_trace.npz',metadata)
        export_analysis(recording/'episode_trace.npz',recording/'metadata.json',recording,{'csv','report'})
        assert metadata['event_semantics']=='continuous_dwell_neutralization'
        for side in ('red','blue'):
            assert metadata[side+'_fire_attempts']==metadata[side+'_weapon_hits']==0
            assert int(trace[side+'_dwell_completed_this_step'].sum())==metadata[side+'_dwell_completed']
            assert metadata[side+'_attack_kills']==metadata[side+'_dwell_neutralizations']
        reward_error=float(np.max(np.abs(trace['local_rewards']-trace['reward_components'].sum(axis=1))))
        assert reward_error<1e-10
        closed(runner)
        result=dict(version=v,device='cuda',cuda_name=torch.cuda.get_device_name(),
            formal_3m_training_started=False,purpose='bounded diagnostic; one unmodified MAAC replay cycle',
            sampled_steps=t.sampled_steps,completed_episodes=len(runner.records),replay_terminals=int(all_samples.dones.sum()),
            num_envs=12,hidden_dim=128,attention_heads=4,minibatch_size=1024,baseline_samples=16,
            critic_updates=t.critic_updates,actor_updates=t.actor_updates,target_updates=t.target_updates,
            gradients=grads,parameter_change_l2=changes,metrics=runner.last_metrics,
            checkpoint_training_smoke=False,checkpoint_reload_networks_and_actions_exact=True,
            saved_optimizer_states_and_counters_verified=True,exact_resume_supported=False,
            rejected_versions=rejected,workers_closed=True,all_worker_processes_exited=True,
            replay_recording_reward_error=reward_error,
            recording={k:metadata[k] for k in ('episode_steps','red_survivors','blue_survivors','draw',
                'red_dwell_completed','blue_dwell_completed','red_dwell_neutralizations','blue_dwell_neutralizations')})
        dump(out/'summary.json',result)
        return result
    finally:
        for h in handles:h.remove()
        closed(runner)


def historical_checkpoint(out):
    directory=ROOT/'outputs/maac_v341_seed1_3m';path=directory/'best_eval.pt'
    if not path.is_file():raise FileNotFoundError('historical v3.41 checkpoint audit requires '+str(path))
    ec=yaml.safe_load((directory/'env_config.yaml').read_text())
    ac=yaml.safe_load((directory/'algorithm_config.yaml').read_text())
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    trainer,state,_=load_recording_policy(path,ec,ac,'cuda')
    assert state['extra']['environment_version']=='3.41'
    result=evaluation_metrics(evaluate(trainer,ec,range(10000000,10000002)),ec)
    assert isinstance(result['fire_rate'],float)
    assert digest==hashlib.sha256(path.read_bytes()).hexdigest()
    for v in ('344','346'):
        try:validate_checkpoint(state,*configs(v))
        except RuntimeError:pass
        else:raise AssertionError('old checkpoint accepted for dwell')
    dump(out/'historical_v341_checkpoint.json',dict(checkpoint=str(path),sha256=digest,
        checkpoint_read_only=True,device='cuda',sampled_steps=state['sampled_steps'],
        historical_fire_rate_semantics_unchanged=True,new_versions_rejected=True,evaluation=result))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/maac_v344_v346_validation_20261022')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    torch.set_num_threads(1)
    results={}
    for v in ('344','346'):
        for name,fn in (('formal_update',formal_update),):
            print('[AUDIT]',v,name,flush=True)
            results[v+'_'+name]=fn(v,out/(v+'_'+name))
    historical_checkpoint(out)
    dump(out/'cuda_summary.json',results)
    print('[DONE] two formal-parameter bounded CUDA runs; no 3M training',flush=True)


if __name__=='__main__':main()
