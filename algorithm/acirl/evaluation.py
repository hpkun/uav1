"""Stateful deterministic/stochastic evaluation, with isolated episode RNGs."""
import torch
import numpy as np
from env.factory import make_combat_environment
from algorithm.common.evaluator import aggregate_combat_records,episode_return_metrics
from .state import PolicySession
from .protocol import require_cuda,validate_checkpoint


def validate_seeds(seeds,config,holdout=False):
    seeds=tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('evaluation requires distinct fixed seeds')
    lower,upper=(70_000_000,2**31) if holdout else (10_000_000,70_000_000)
    if any(not lower<=s<upper for s in seeds):raise ValueError('evaluation seed block mismatch; use --holdout for >=70M')
    t=config['training']
    if any(t['seed']<=s<=t['seed']+t['total_sampled_steps']+t['num_train_envs'] for s in seeds):raise ValueError('training/evaluation seed overlap')
    return seeds


def evaluate(trainer,env_config,seeds,mode='deterministic',holdout=False):
    require_cuda(trainer.device)
    if mode not in ('deterministic','stochastic'):raise ValueError('invalid evaluation mode')
    seeds=validate_seeds(seeds,trainer.config,holdout);records=[];selections=[]
    for seed in seeds:
        env=make_combat_environment(env_config);obs,_=env.reset(seed)
        session=PolicySession(trainer.policy,trainer.device,seed,mode=='deterministic')
        returns=np.zeros(env.team_size);renew=forced=steps=eligible=0
        while True:
            actions=session.act(obs,env.red_alive_mask)
            row=session.last;renew+=int((row['gate']*row['gate_valid']).sum());forced+=int(row['forced'].sum())
            eligible+=int(row['gate_valid'].sum())
            selections.extend(row['intent'][row['intent_valid'].bool()].cpu().tolist())
            obs,reward,terminated,truncated,info=env.step(actions);returns+=reward;steps+=1
            if terminated or truncated:break
        team,agent=episode_return_metrics(returns)
        records.append(dict(episode_return=team,mean_agent_episode_return=agent,learned_renew_count=renew,forced_renew_count=forced,eligible_gate_count=eligible,**info))
    alerts=[]
    if selections and len(set(selections))==1:alerts.append('SINGLE_INTENTION_CLASS_IN_EVALUATION')
    eligible=sum(r['eligible_gate_count'] for r in records);renew=sum(r['learned_renew_count'] for r in records)
    if eligible and renew==0:alerts.append('NO_LEARNED_RENEW_IN_EVALUATION')
    if eligible and renew==eligible:alerts.append('ALL_ELIGIBLE_GATES_RENEW_IN_EVALUATION')
    return dict(algorithm='acirl',evaluation_mode=mode,holdout=holdout,evaluation_seeds=list(seeds),
        intention_selection_counts=np.bincount(selections,minlength=trainer.config['network']['intentions']).tolist(),
        average_episode_learned_renew_count=float(np.mean([r['learned_renew_count'] for r in records])),
        average_episode_forced_renew_count=float(np.mean([r['forced_renew_count'] for r in records])),
        average_episode_eligible_gate_decisions=float(np.mean([r['eligible_gate_count'] for r in records])),
        policy_diagnostic_alerts=alerts,
        diagnostic_scope='finite evaluation observation; deterministic argmax concentration is not proof of stochastic policy collapse',
        **aggregate_combat_records(records))


def load_checkpoint(path,env,config,device='cuda'):
    require_cuda(device)
    from .trainer import ACIRLTrainer
    state=torch.load(path,map_location='cpu',weights_only=False);validate_checkpoint(state,env,config)
    trainer=ACIRLTrainer(config,device);trainer.load(path,env)
    return trainer


def evaluate_checkpoint(path,env,config,device,seeds,mode='deterministic',holdout=False):
    trainer=load_checkpoint(path,env,config,device)
    return dict(checkpoint=str(path),checkpoint_sampled_steps=trainer.sampled_steps,
        **evaluate(trainer,env,seeds,mode,holdout))
