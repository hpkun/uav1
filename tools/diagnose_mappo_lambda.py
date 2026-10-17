"""Small CUDA-only fixed-rollout GAE comparison and smoke artifact validation.

This never runs a training loop. Smoke training is explicitly invoked separately.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from algorithm.mappo.trainer import compute_gae,raw_gae_diagnostics
from algorithm.mappo.factory import build_mappo_trainer
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.common.checkpoint import validate_checkpoint_for_evaluation,validate_checkpoint_for_resume
from algorithm.common.evaluator import evaluate
from algorithm.train_mappo import ensure_fresh_output_directory

CONTROL='configs/mappo_4v4_v338.yaml'
TREATMENT='configs/mappo_4v4_v338_lambda996195.yaml'

def configs():
    load=lambda p:yaml.safe_load((ROOT/p).read_text(encoding='utf-8'))
    env,a,b=load('configs/combat_environment_v338.yaml'),load(CONTROL),load(TREATMENT)
    expected=copy.deepcopy(a);expected['training']['gae_lambda']=.996195
    if b!=expected:raise RuntimeError('Treatment must differ only in training.gae_lambda')
    return env,a,b

def tensor(value):return torch.as_tensor(value,dtype=torch.float32,device='cuda')
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def compare_fixed(rewards,values,next_values,dones,alive,next_alive):
    """Identical tensors and masks; only lambda changes. No gradient or optimizer."""
    masks=alive>.5;result={};arrays=[]
    with torch.no_grad():
        for label,lam in [('control',.95),('treatment',.996195)]:
            advantage,target=compute_gae(rewards,values,next_values,dones,alive,next_alive,.99,lam)
            result[label]=raw_gae_diagnostics(advantage,target,values,alive)
            arrays.append(advantage.cpu().numpy())
    live=masks.cpu().numpy();a,b=(x[live].astype(np.float64) for x in arrays)
    import pandas as pd
    # Ratios near zero are reported separately instead of hidden by a large epsilon.
    nonzero=abs(a)>1e-8
    result['comparison']=dict(alive_samples=int(live.sum()),pearson=float(np.corrcoef(a,b)[0,1]),
        spearman=float(np.corrcoef(pd.Series(a).rank(method='average'),pd.Series(b).rank(method='average'))[0,1]),
        sign_disagreement_fraction=float(np.mean(np.sign(a)!=np.sign(b))),
        abs_mean_ratio=float(abs(b).mean()/abs(a).mean()),
        mean_sample_abs_ratio_nonzero_control=float(np.mean(abs(b[nonzero])/abs(a[nonzero]))),
        median_sample_abs_ratio_nonzero_control=float(np.median(abs(b[nonzero])/abs(a[nonzero]))),
        zero_control_samples=int((~nonzero).sum()))
    return result,arrays

def time_to_event(event,dones,next_alive):
    """Distance to next same-agent event without crossing episode/death boundaries."""
    distance=np.full(event.shape,np.inf);remaining=np.full(event.shape[1:],np.inf)
    for step in range(len(event)-1,-1,-1):
        cut=(dones[step,:,None]>.5)|(next_alive[step]<.5)
        remaining=np.where(cut,np.inf,remaining+1)
        remaining=np.where(event[step],0,remaining)
        distance[step]=remaining
    return distance

def fixed(output,checkpoint):
    ensure_fresh_output_directory(output)
    env,cfg,_=configs()
    checkpoint=checkpoint.resolve()
    # This is a new frozen-policy data collection, not a resumed training run.
    runner=MAPPOTrainingRunner(env,cfg,num_envs=16,total_sampled_steps=6144,
        device='cuda',seed=1,output_dir=output/'collector',smoke=False)
    try:
        state=torch.load(checkpoint,map_location='cuda',weights_only=False)
        validate_checkpoint_for_evaluation(state,env,cfg);runner.trainer.load(checkpoint)
        runner.collect_rollout(128)  # Fixed warmup reaches engagement; no optimizer step.
        events=[];step_batch=runner.vector.step_batch
        def capture(actions):
            batch=step_batch(actions)
            death=(batch.alive_masks>.5)&(batch.next_alive_masks<.5)
            boundary=np.stack([np.asarray(i['r2_rewards'])<0 for i in batch.infos])
            own=death&~boundary
            r1=np.stack([i['r1_rewards'] for i in batch.infos])
            kill_credit=r1+10*own
            events.append(dict(kill=kill_credit>1e-5,own_death=own,boundary_death=boundary))
            return batch
        runner.vector.step_batch=capture
        batch=runner.collect_rollout(256)
        with torch.no_grad():
            shape=batch.rewards.shape
            values=runner.trainer.values(batch.observations.reshape(-1,4,45),batch.alive_masks.reshape(-1,4)).reshape(shape)
            next_values=runner.trainer.values(batch.next_observations.reshape(-1,4,45),batch.next_alive_masks.reshape(-1,4)).reshape(shape)
        inputs=dict(rewards=batch.rewards,values=values,next_values=next_values,dones=batch.dones,
            alive=batch.alive_masks,next_alive=batch.next_alive_masks)
        hashes={k:digest(v) for k,v in inputs.items()}
        result,advantages=compare_fixed(**{k:tensor(v) for k,v in inputs.items()})
        strata=[]
        for name in ['kill','own_death','boundary_death']:
            flags=np.stack([e[name] for e in events]);inputs['event_'+name]=flags
            distance=time_to_event(flags,batch.dones,batch.next_alive_masks)
            for lo,hi in [(0,10),(10,20),(20,50),(50,100),(100,200)]:
                valid=(distance>=lo)&(distance<hi)&(batch.alive_masks>.5)
                if not valid.any():continue
                a,b=(v[valid].astype(np.float64) for v in advantages)
                strata.append(dict(event=name,seconds_from=lo*.1,seconds_to=hi*.1,n=int(valid.sum()),
                    control_std=float(a.std()),treatment_std=float(b.std()),
                    control_abs_mean=float(abs(a).mean()),treatment_abs_mean=float(abs(b).mean()),
                    sign_disagreement=float(np.mean(np.sign(a)!=np.sign(b)))))
        assert hashes=={k:digest(inputs[k]) for k in hashes}
        result.update(input_sha256=hashes,checkpoint=str(checkpoint),checkpoint_steps=int(state['sampled_steps']),
            frozen_policy=True,optimizer_steps=0,warmup_sampled_steps=2048,fixed_sampled_steps=4096,
            rollout_shape=list(batch.rewards.shape),gamma=.99,lambdas=[.95,.996195],
            event_counts={k:int(inputs['event_'+k].sum()) for k in ['kill','own_death','boundary_death']},
            event_definition='kill=positive individual kill credit (R1 + 10*nonboundary death); own_death excludes boundary, includes ground; boundary includes ceiling',
            strata=strata,trace_weights={str(seconds):dict(control=(.99*.95)**round(seconds/.1),
                treatment=(.99*.996195)**round(seconds/.1)) for seconds in [1,2,5,10]},
            treatment_50_step_weight=(.99*.996195)**50)
        np.savez_compressed(output/'fixed_rollout.npz',**inputs,control_advantage=advantages[0],treatment_advantage=advantages[1])
        dump(output/'comparison.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='strata'},indent=2))
    finally:runner.vector.close()

def check_smoke(output,run):
    ensure_fresh_output_directory(output)
    env,_,_=configs();cfg=yaml.safe_load((run/'algorithm_config.yaml').read_text(encoding='utf-8'))
    # The evaluation factory defaults to seed=0; the CLI runner passes seed=1.
    # Use the same factory arguments and constructor with only that runtime seed.
    from unittest.mock import patch
    from algorithm.mappo.trainer import MAPPOTrainer
    with patch('algorithm.mappo.factory.MAPPOTrainer',
               side_effect=lambda *args,**kwargs: MAPPOTrainer(*args,**kwargs,seed=1)):
        initial=build_mappo_trainer(cfg,'cuda')
    actor={k:v.clone() for k,v in initial.actor.state_dict().items()}
    critic={k:v.clone() for k,v in initial.critic.state_dict().items()}
    state=torch.load(run/'latest.pt',map_location='cuda',weights_only=False)
    validate_checkpoint_for_evaluation(state,env,cfg);validate_checkpoint_for_resume(state,env,cfg)
    assert state['sampled_steps']==4096 and state['ppo_updates']==1
    assert state['extra']['training_seed']==1
    assert state['extra']['training_gae_lambda']==cfg['training']['gae_lambda']
    actor_changed=any(not torch.equal(actor[k],v) for k,v in state['actor'].items())
    critic_changed=any(not torch.equal(critic[k],v) for k,v in state['critic'].items())
    assert actor_changed and critic_changed
    initial=build_mappo_trainer(cfg,'cuda');initial.load(run/'latest.pt')
    from env.factory import make_combat_environment
    environment=make_combat_environment(env);observation,_=environment.reset(10000000)
    before=initial.act(observation,environment.red_alive_mask,deterministic=True)
    initial.save(output/'roundtrip.pt',state['extra'])
    restored=build_mappo_trainer(cfg,'cuda');restored.load(output/'roundtrip.pt')
    after=restored.act(observation,environment.red_alive_mask,deterministic=True)
    assert np.array_equal(before,after)
    metrics=[json.loads(line) for line in (run/'optimization_metrics.jsonl').read_text().splitlines()]
    assert all(np.isfinite(list(m.values())).all() for m in metrics)
    evaluation=evaluate(restored,env,[10000000])
    assert np.isfinite(list(evaluation.values())).all()
    result=dict(sampled_steps=state['sampled_steps'],ppo_updates=state['ppo_updates'],
        actor_changed=actor_changed,critic_changed=critic_changed,metrics_finite=True,
        actor_max_parameter_change=max(float((actor[k]-v).abs().max()) for k,v in state['actor'].items()),
        critic_max_parameter_change=max(float((critic[k]-v).abs().max()) for k,v in state['critic'].items()),
        roundtrip_actions_equal=True,gamma=state['extra']['training_gamma'],gae_lambda=state['extra']['training_gae_lambda'],
        actor_parameters=sum(v.numel() for v in state['actor'].values()),critic_parameters=sum(v.numel() for v in state['critic'].values()),
        evaluation_episodes=1,evaluation_seed=10000000,evaluation=evaluation,
        purpose='chain validation only; not formal performance evaluation')
    dump(output/'validation.json',result);print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['fixed','check-smoke'],required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/mappo_v338_seed1_3m/best_eval.pt');p.add_argument('--run-dir',type=Path)
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA mandatory; no CPU fallback')
    torch.set_num_threads(1)
    if args.mode=='fixed':fixed(args.output_dir,args.checkpoint)
    else:
        if args.run_dir is None:p.error('--run-dir required for check-smoke')
        check_smoke(args.output_dir,args.run_dir)

if __name__=='__main__':main()
