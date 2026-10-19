"""Explicit statistics scopes, collapse alerts and real parameter drift."""
import numpy as np
import torch


def statistics(values,mask=None):
    v=values.detach().flatten() if mask is None else values.detach()[mask.bool()]
    if not v.numel():return dict(mean=None,std=None,variance=None,min=None,max=None,count=0)
    return dict(mean=float(v.mean()),std=float(v.std(unbiased=False)),variance=float(v.var(unbiased=False)),min=float(v.min()),max=float(v.max()),count=v.numel())


def summarize(trainer,rollout,critic_rows,policy_rows,gate_adv,q,ratio_errors,epochs,stopped):
    d=rollout.data;n=trainer.config['network'];live=d['alive'].bool();gate=d['gate_valid'].bool();high=d['intent_valid'].bool()
    actions=d['actions'][live];intent=d['intent'][live];K=n['intentions'];alerts=[]
    counts=torch.bincount(intent[intent>=0],minlength=K);usage=(counts/counts.sum().clamp_min(1)).cpu().tolist()
    selections=d['intent'][high];selection_counts=torch.bincount(selections[selections>=0],minlength=K)
    if selections.numel()>=100 and (selection_counts>0).sum()==1:alerts.append('INTENTION_COLLAPSE: all >=100 selections in one class')
    learned=d['gate'][gate]
    if learned.numel()>=32 and (learned.sum()==0 or learned.sum()==learned.numel()):alerts.append('GATE_DEGENERACY: all learned decisions keep or all renew')
    mean=lambda key,rows:float(np.mean([row[key] for row in rows if key in row])) if any(key in r for r in rows) else None
    metrics=dict(statistics_scope='live executed actions; population std; independent event means for intention/gate; null for inactive heads',
        sampled_steps=trainer.sampled_steps,vector_steps=trainer.vector_steps,update_count=trainer.update_count,
        policy_optimizer_steps=trainer.policy_optimizer_steps,team_optimizer_steps=trainer.team_optimizer_steps,
        gate_optimizer_steps=trainer.gate_optimizer_steps,policy_rows=len(policy_rows),critic_rows=len(critic_rows),
        effective_epochs=epochs,kl_early_stop=stopped,behavior_logp_max_error=ratio_errors,
        valid_low_samples=int(live.sum()),valid_high_decisions=int(high.sum()),valid_gate_decisions=int(gate.sum()),
        original_environment_reward_sum=float(d['rewards'].sum()),original_team_scaled_reward_sum=float(d['raw_team_reward'].sum()),
        internal_training_reward_sum=float(d['training_reward'].sum()),switching_cost_sum=float(d['switching_cost'].sum()),
        intention_usage=usage,intention_selection_counts=selection_counts.cpu().tolist(),
        learned_renew_ratio=float(learned.mean()) if learned.numel() else None,
        learned_keep_ratio=float((1-learned).mean()) if learned.numel() else None,
        forced_renew_ratio=float(d['forced'][live].mean()) if live.any() else None,
        low_advantage=statistics(d['low_advantage'],d['alive']),high_advantage=statistics(d['high_advantage'],d['intent_valid']),
        gate_advantage=statistics(gate_adv,d['gate_valid']),parameter_drift_l2=trainer.parameter_drift())
    for key in ('team_value_loss','team_value_weighted_loss','gate_q_loss','gate_q_weighted_loss','gate_q_mse','team_grad_pre','team_grad_post','gate_critic_grad_pre','gate_critic_grad_post'):
        metrics[key]=mean(key,critic_rows)
    for key in policy_rows[0]:
        if key not in ('gradient_norms','inference_predicted_counts','inference_teacher_counts'):metrics[key]=mean(key,policy_rows)
    metrics['inference_predicted_counts']=np.sum([r['inference_predicted_counts'] for r in policy_rows],axis=0).tolist()
    metrics['inference_teacher_counts']=np.sum([r['inference_teacher_counts'] for r in policy_rows],axis=0).tolist()
    metrics['gradient_norms']={key:float(np.mean([r['gradient_norms'][key] for r in policy_rows])) for key in policy_rows[0]['gradient_norms']}
    for key,value in metrics['gradient_norms'].items():
        active=key not in ('intention_actor','gate_actor','inference') or (key=='intention_actor' and high.any()) or (key=='gate_actor' and gate.any()) or key=='inference'
        if active and value==0:alerts.append(f'ZERO_GRADIENT:{key}; inspect event/label counts')
    if q is not None:
        metrics.update(gate_q_keep=statistics(q[...,0],d['gate_valid']),gate_q_renew=statistics(q[...,1],d['gate_valid']),
            gate_q_delta=statistics(q[...,1]-q[...,0],d['gate_valid']),
            gate_q_scope='unexecuted branch is function-approximation counterfactual estimate, not an environment outcome',
            gate_supervised_keep_samples=int(((d['gate']==0)&gate).sum()),gate_supervised_renew_samples=int(((d['gate']==1)&gate).sum()))
    durations=[event['duration'] for event in rollout.events]
    metrics['macro_fragment_duration']={str(p):float(np.quantile(durations,p)) if durations else None for p in (0.,.25,.5,.75,.9,1.)}
    metrics['macro_fragment_duration']['mean']=float(np.mean(durations)) if durations else None
    metrics['macro_fragment_duration']['scope']='within-rollout fragments; rollout_bootstrap is censored, not a natural intention end'
    natural=[event['duration'] for event in rollout.intention_events]
    metrics['completed_intention_duration']={str(p):float(np.quantile(natural,p)) if natural else None for p in (.25,.5,.75,.9,1.)}
    metrics['completed_intention_duration']['mean']=float(np.mean(natural)) if natural else None
    metrics['completed_intention_count']=len(natural)
    metrics['action_mean']=actions.mean(0).cpu().tolist() if actions.numel() else None
    metrics['action_std']=actions.std(0,unbiased=False).cpu().tolist() if actions.numel() else None
    metrics['action_absolute_mean']=actions.abs().mean(0).cpu().tolist() if actions.numel() else None
    metrics['action_near_zero_ratio']=(actions.abs()<.05).float().mean(0).cpu().tolist() if actions.numel() else None
    metrics['action_saturation_ratio']=(actions.abs()>.95).float().mean(0).cpu().tolist() if actions.numel() else None
    continuing=live&~d['reset'].bool();change=(d['actions']-d['previous_action']).abs()[continuing]
    metrics['adjacent_action_absolute_change']=change.mean(0).cpu().tolist() if change.numel() else None
    metrics['actions_per_intention']=[dict(count=int((intent==k).sum()),mean=actions[intent==k].mean(0).cpu().tolist() if (intent==k).any() else None,
        std=actions[intent==k].std(0,unbiased=False).cpu().tolist() if (intent==k).any() else None) for k in range(K)]
    updates=(d['gate']*d['gate_valid']).flatten(0,1);centered=updates-updates.mean(0);std=centered.square().mean(0).sqrt()
    corr=(centered.T@centered)/len(updates)/std[:,None].clamp_min(1e-12)/std[None].clamp_min(1e-12)
    metrics['update_time_correlation']=[[float(corr[i,j]) if std[i]>0 and std[j]>0 else None for j in range(len(std))] for i in range(len(std))]
    metrics['alerts']=alerts
    return metrics
