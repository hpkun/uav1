"""On-policy recurrent PPO with ID-aware chunk reconstruction and strict audits."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .config import require_cuda, validate_config
from .memory import MemoryState
from .networks import EntityTemporalActor, CentralizedNavigationCritic, squashed_log_prob


def masked_mean(value, mask):
    return (value*mask).sum()/mask.sum().clamp_min(1)


def clipped_losses(log,old_log,advantage,value,old_value,returns,entropy,mask,clip,clip_value=True):
    """Only valid entries enter arithmetic, including before exp/square."""
    difference=torch.where(mask,log-old_log,0.)
    ratio=difference.exp()
    active_ratio=ratio[mask];active_advantage=advantage[mask]
    if not active_ratio.numel():raise ValueError('loss needs a valid sample')
    policy=-torch.minimum(active_ratio*active_advantage,active_ratio.clamp(1-clip,1+clip)*active_advantage).mean()
    prediction,old,target=value[mask],old_value[mask],returns[mask]
    error=(prediction-target).square()
    if clip_value:
        bounded=old+(prediction-old).clamp(-clip,clip)
        error=torch.maximum(error,(bounded-target).square())
    return policy,.5*error.mean(),entropy[mask].mean(),ratio,difference


def compute_gae(rewards, values, next_values, terminated, truncated, alive, next_alive, gamma, gae_lambda):
    """Time limits bootstrap true terminal frames, but never recurse across reset."""
    if rewards.shape != values.shape or rewards.shape != next_values.shape or alive.shape != values.shape or next_alive.shape != values.shape:
        raise ValueError('GAE agent tensor shape mismatch')
    if terminated.shape != rewards.shape[:2] or truncated.shape != terminated.shape or (terminated & truncated).any():
        raise ValueError('GAE termination/truncation shape or semantics mismatch')
    advantages=torch.zeros_like(rewards)
    future=torch.zeros_like(rewards[0])
    for step in reversed(range(len(rewards))):
        bootstrap=(~terminated[step]).unsqueeze(-1)*next_alive[step]
        continuation=(~(terminated[step]|truncated[step])).unsqueeze(-1)*next_alive[step]
        delta=rewards[step]+gamma*bootstrap*next_values[step]-values[step]
        future=(delta+gamma*gae_lambda*continuation*future)*alive[step]
        advantages[step]=future
    return advantages,(advantages+values)*alive


def sequence_chunks(time, envs, length):
    return [(env,start,min(time,start+length)) for env in range(envs) for start in range(0,time,length)]


def pack_tensor(tensor,chunks,length,pad=0):
    output=tensor.new_full((len(chunks),length,*tensor.shape[2:]),pad)
    for i,(env,start,end) in enumerate(chunks):output[i,:end-start]=tensor[start:end,env]
    return output


@dataclass
class EntityRollout:
    observations: dict
    next_observations: dict
    actions: np.ndarray
    raw_actions: np.ndarray
    old_log_probs: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    next_values: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_starts: np.ndarray
    memory: dict
    padding_mask: np.ndarray


def diagnostic_tensors(diagnostics, alive):
    result={}
    for side in ('self','peer','obstacle'):
        hidden=diagnostics[side+'_hidden']
        valid=alive if side=='self' else diagnostics[side+'_valid']
        result[side+'_hidden_norm']=masked_mean(hidden.norm(dim=-1),valid)
    gate=diagnostics['gate']
    expanded=alive.unsqueeze(-1).expand_as(gate)
    mean=masked_mean(gate,expanded)
    result.update(gate_mean=mean,gate_std=masked_mean((gate-mean).square(),expanded).sqrt(),
        gate_saturation_fraction=masked_mean(((gate<.05)|(gate>.95)).float(),expanded))
    for index in range(gate.shape[-1]):
        result[f'gate_dim_{index}_mean']=masked_mean(gate[...,index],alive)
        center=result[f'gate_dim_{index}_mean']
        result[f'gate_dim_{index}_std']=masked_mean((gate[...,index]-center).square(),alive).sqrt()
    for side in ('peer','obstacle'):
        weights=diagnostics[side+'_attention_weights']
        entropy=-(weights*weights.clamp_min(1e-12).log()).sum(-1).mean(-1)
        top=weights.max(-1).values.mean(-1) if weights.shape[-1] else torch.zeros_like(entropy)
        result[side+'_attention_entropy']=masked_mean(entropy,alive)
        result[side+'_attention_top1']=masked_mean(top,alive)
    for index in range(diagnostics['log_std'].shape[-1]):
        result[f'policy_sigma_{index}']=masked_mean(diagnostics['log_std'][...,index].exp(),alive)
    return result


class ETMMAPPOTrainer:
    def __init__(self,config,device='cuda'):
        validate_config(config);require_cuda(device)
        self.config=config;self.device=torch.device(device)
        n,t=config['network'],config['training']
        torch.set_num_threads(t['torch_threads'])
        torch.manual_seed(t['seed']);torch.cuda.manual_seed_all(t['seed'])
        self.rng=np.random.default_rng(t['seed'])
        self.actor=EntityTemporalActor(n).to(self.device)
        # Actor allocation differs by ablation. Isolate critic initialization so
        # a matched seed starts every variant from exactly the same critic.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(t['seed']+104729)
            self.critic=CentralizedNavigationCritic(n['self_dim'],n['critic_dim'],n['attention_heads']).to(self.device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=t['actor_learning_rate'])
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=t['critic_learning_rate'])
        self.sampled_steps=self.vector_steps=self.rollout_updates=0
        self.actor_optimizer_steps=self.critic_optimizer_steps=0

    def observation_tensors(self,observations):
        return {key:torch.as_tensor(value,device=self.device) for key,value in observations.items()}

    @torch.no_grad()
    def initial_state(self,observations):
        return self.actor.initial_state(self.observation_tensors(observations))

    @torch.no_grad()
    def values(self,observations):
        o=self.observation_tensors(observations)
        return self.critic(o['self_features'],o['agent_alive']).cpu().numpy()

    @torch.no_grad()
    def act(self,observations,state,episode_start,deterministic=False,diagnostics=False):
        o=self.observation_tensors(observations)
        dist,next_state,data=self.actor.step(o,state,torch.as_tensor(episode_start,dtype=torch.bool,device=self.device),
            reset_memory=self.config['training']['memory_reset_each_step'])
        raw=dist.mean if deterministic else dist.rsample()
        action=raw.tanh()*o['agent_alive'][...,None]
        log=squashed_log_prob(dist,raw)*o['agent_alive']
        result=(action.cpu().numpy(),(raw*o['agent_alive'][...,None]).cpu().numpy(),log.cpu().numpy(),next_state.detach())
        if diagnostics:
            return (*result,{key:float(value) for key,value in diagnostic_tensors(data,o['agent_alive']).items()})
        return result

    def tensor_rollout(self,rollout):
        result={key:torch.as_tensor(getattr(rollout,key),device=self.device) for key in (
            'actions','raw_actions','old_log_probs','rewards','values','next_values',
            'terminated','truncated','episode_starts','padding_mask')}
        result['observations']=self.observation_tensors(rollout.observations)
        result['memory']={key:torch.as_tensor(value,device=self.device) for key,value in rollout.memory.items()}
        if not result['padding_mask'].bool().all():
            raise ValueError('full rollout must contain only real transitions; padding is applied after GAE')
        return result

    def packed(self,tensors,chunks):
        length=self.config['training']['sequence_length']
        batch={key:pack_tensor(value,chunks,length,pad=True if key=='episode_starts' else 0)
            for key,value in tensors.items() if key not in ('observations','memory')}
        batch['observations']={key:pack_tensor(value,chunks,length,pad=-1 if key.endswith('_ids') else 0)
            for key,value in tensors['observations'].items()}
        batch['initial_memory']=MemoryState(**{key:torch.stack([value[start,env] for env,start,_ in chunks]).detach()
            for key,value in tensors['memory'].items()})
        batch['valid_steps']=torch.stack([torch.arange(length,device=self.device)<end-start for _,start,end in chunks])
        batch['live']=batch['observations']['agent_alive'] & batch['valid_steps'][...,None]
        return batch

    def evaluate_sequence(self,batch,entropy=True):
        state=batch['initial_memory']
        logs=[];entropies=[];rows=[]
        for step in range(self.config['training']['sequence_length']):
            o={key:value[:,step] for key,value in batch['observations'].items()}
            distribution,state,data=self.actor.step(o,state,batch['episode_starts'][:,step],
                reset_memory=self.config['training']['memory_reset_each_step'])
            raw=torch.where(o['agent_alive'][...,None],batch['raw_actions'][:,step],0.)
            log=squashed_log_prob(distribution,raw)
            if entropy:
                sampled=distribution.rsample()
                ent=-squashed_log_prob(distribution,sampled)
            else:ent=torch.zeros_like(log)
            logs.append(log);entropies.append(ent);rows.append(data)
        diagnostic={key:torch.stack([row[key] for row in rows],1) for key in rows[0]}
        return torch.stack(logs,1),torch.stack(entropies,1),diagnostic

    @torch.no_grad()
    def audit_rollout_ratio(self,rollout):
        t=self.tensor_rollout(rollout);time,envs=t['rewards'].shape[:2]
        length=self.config['training']['sequence_length']
        chunks=sequence_chunks(time,envs,length)
        per_batch=self.config['training']['minibatch_joint_transitions']//length
        ratios=[]
        for start in range(0,len(chunks),per_batch):
            batch=self.packed(t,chunks[start:start+per_batch])
            log,_,_=self.evaluate_sequence(batch,entropy=False)
            ratios.append((log-batch['old_log_probs']).exp()[batch['live']])
        live=torch.cat(ratios)
        error=float((live-1).abs().max()) if live.numel() else 0.
        if not torch.isfinite(live).all() or error>1e-4:
            raise RuntimeError(f'ETM pre-update ratio audit failed: max_abs_error={error}')
        return dict(pre_update_ratio_max_abs_error=error,
            pre_update_ratio_mean=float(live.mean()) if live.numel() else 1.)

    def update(self,rollout):
        audit=self.audit_rollout_ratio(rollout)
        tensors=self.tensor_rollout(rollout);o=tensors['observations']
        t=self.config['training'];time,envs=tensors['rewards'].shape[:2]
        live=o['agent_alive']
        following=torch.as_tensor(rollout.next_observations['agent_alive'],device=self.device)
        with torch.no_grad():
            advantage,returns=compute_gae(tensors['rewards'],tensors['values'],tensors['next_values'],
                tensors['terminated'],tensors['truncated'],live,following,t['gamma'],t['gae_lambda'])
            raw=advantage[live]
            raw_metrics=dict(advantage_mean=float(raw.mean()) if raw.numel() else 0.,
                advantage_std=float(raw.std(unbiased=False)) if raw.numel() else 0.,
                return_target_mean=float(returns[live].mean()) if raw.numel() else 0.,
                return_target_std=float(returns[live].std(unbiased=False)) if raw.numel() else 0.)
            if t['normalize_advantages'] and raw.numel():
                advantage=(advantage-raw.mean())/raw.std(unbiased=False).clamp_min(1e-8)*live
        tensors.update(advantages=advantage,returns=returns)
        length=t['sequence_length'];chunks=sequence_chunks(time,envs,length)
        per_batch=t['minibatch_joint_transitions']//length
        before_actor,before_critic=self.actor_optimizer_steps,self.critic_optimizer_steps
        rows=[];epochs=0;early=False
        for epoch in range(t['ppo_epochs']):
            epoch_rows=[]
            order=self.rng.permutation(len(chunks))
            for start in range(0,len(chunks),per_batch):
                batch=self.packed(tensors,[chunks[i] for i in order[start:start+per_batch]])
                mask=batch['live']
                if not mask.any():continue
                log,entropy,diagnostics=self.evaluate_sequence(batch)
                obs=batch['observations']
                value=self.critic(obs['self_features'],mask)
                policy_loss,value_loss,entropy_mean,ratio,difference=clipped_losses(log,batch['old_log_probs'],
                    batch['advantages'],value,batch['values'],batch['returns'],entropy,mask,t['clip_ratio'],t['clip_value_loss'])
                with torch.no_grad():
                    row={key:float(v) for key,v in diagnostic_tensors(diagnostics,mask).items()}
                    actual_ratio=ratio[mask]
                    row.update(policy_loss=float(policy_loss),actor_loss=float(policy_loss),value_loss=float(value_loss),
                        entropy=float(entropy_mean),approximate_kl=float(masked_mean((ratio-1)-difference,mask)),
                        clip_fraction=float(masked_mean((ratio-1).abs()>t['clip_ratio'],mask)),
                        ratio_mean=float(actual_ratio.mean()),ratio_min=float(actual_ratio.min()),ratio_max=float(actual_ratio.max()))
                    for index in range(batch['actions'].shape[-1]):
                        a=batch['actions'][...,index][mask]
                        row.update({f'action_mean_{index}':float(a.mean()),f'action_std_{index}':float(a.std(unbiased=False)),
                            f'action_saturation_{index}':float((a.abs()>.99).float().mean())})
                self.actor_optimizer.zero_grad(set_to_none=True)
                (policy_loss-t['entropy_coefficient']*entropy_mean).backward()
                actor_norm=nn.utils.clip_grad_norm_(self.actor.parameters(),t['max_grad_norm'])
                if not torch.isfinite(actor_norm):raise FloatingPointError('nonfinite actor gradient')
                self.actor_optimizer.step();self.actor_optimizer_steps+=1
                self.critic_optimizer.zero_grad(set_to_none=True)
                (t['value_loss_coefficient']*value_loss).backward()
                critic_norm=nn.utils.clip_grad_norm_(self.critic.parameters(),t['max_grad_norm'])
                if not torch.isfinite(critic_norm):raise FloatingPointError('nonfinite critic gradient')
                self.critic_optimizer.step();self.critic_optimizer_steps+=1
                row.update(actor_gradient_norm=float(actor_norm),critic_gradient_norm=float(critic_norm),
                    gradient_clipped_ratio=.5*(float(actor_norm>t['max_grad_norm'])+float(critic_norm>t['max_grad_norm'])))
                rows.append(row);epoch_rows.append(row)
            if epoch_rows:
                epochs+=1
                if t['target_kl'] is not None and np.mean([r['approximate_kl'] for r in epoch_rows])>t['target_kl']:
                    early=True;break
        self.rollout_updates+=1
        metrics={key:float(np.mean([r[key] for r in rows])) for key in rows[0]} if rows else {}
        metrics.update(audit,**raw_metrics,configured_ppo_epochs=t['ppo_epochs'],effective_ppo_epochs=epochs,
            kl_early_stop=int(early),actual_actor_optimizer_steps=self.actor_optimizer_steps-before_actor,
            actual_critic_optimizer_steps=self.critic_optimizer_steps-before_critic,
            valid_agent_samples=int(live.sum()),valid_joint_transitions=time*envs,
            padded_joint_transitions=len(chunks)*length-time*envs,sequence_chunks=len(chunks))
        if not np.isfinite(list(metrics.values())).all():raise FloatingPointError('nonfinite PPO metrics')
        return metrics

    def parameter_counts(self):
        result={name:sum(p.numel() for p in getattr(self,name).parameters()) for name in ('actor','critic')}
        result['total']=sum(result.values())
        return result
