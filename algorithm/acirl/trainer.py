"""Independent recurrent action/event PPO and return-supervised gate critic."""
from copy import deepcopy
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Bernoulli,Categorical
from algorithm.common.protocol import config_sha256
from .networks import ACIRLPolicy,CentralizedCritics,squashed_log_prob
from .state import PolicySession
from .protocol import require_cuda,architecture,validate_checkpoint,ACIRL_IMPL_VERSION
from .advantages import normalize,event_mean,ppo_component


def gradient_norm(module):
    gradients=[p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(gradients).norm()) if gradients else 0.


def clipped_step(loss,module,optimizer,maximum):
    if not torch.isfinite(loss):raise FloatingPointError('non-finite ACIRL loss')
    optimizer.zero_grad(set_to_none=True);loss.backward()
    pre=float(torch.nn.utils.clip_grad_norm_(module.parameters(),maximum,error_if_nonfinite=True))
    post=gradient_norm(module);optimizer.step()
    return pre,post


class ACIRLTrainer:
    def __init__(self,config,device='cuda',seed=None):
        require_cuda(device);self.device=torch.device(device);self.config=deepcopy(config)
        self.seed=config['training']['seed'] if seed is None else int(seed)
        self.config['training']['seed']=self.seed
        torch.manual_seed(self.seed)
        self.policy=ACIRLPolicy(self.config).to(self.device);self.critics=CentralizedCritics(self.config).to(self.device)
        t=config['training']
        self.policy_optimizer=torch.optim.Adam(self.policy.parameters(),lr=t['actor_learning_rate'])
        self.team_optimizer=torch.optim.Adam(self.critics.team.parameters(),lr=t['critic_learning_rate'])
        self.gate_optimizer=torch.optim.Adam(self.critics.gate.parameters(),lr=t['critic_learning_rate']) if self.critics.has_gate_q else None
        self.behavior_rng=torch.Generator(device=self.device).manual_seed(self.seed^0xB3A001)
        self.entropy_rng=torch.Generator(device=self.device).manual_seed(self.seed^0xE17001)
        self.permutation_rng=np.random.default_rng(self.seed^0xA017)
        self.sampled_steps=self.vector_steps=self.update_count=0
        self.policy_optimizer_steps=self.team_optimizer_steps=self.gate_optimizer_steps=0
        self.initial_parameters={name:torch.nn.utils.parameters_to_vector(module.parameters()).detach().clone()
            for name,module in self.modules_for_audit().items()}
        self._session=None

    def modules_for_audit(self):
        modules=dict(encoder=self.policy.encoder,gru=self.policy.gru,action_actor=self.policy.action_actor,team_critic=self.critics.team)
        for name in ('gate_actor','intention_actor','inference'):
            if hasattr(self.policy,name):modules[name]=getattr(self.policy,name)
        if self.critics.has_gate_q:modules['gate_critic']=self.critics.gate
        return modules

    def parameter_counts(self):
        return {**{name:sum(p.numel() for p in module.parameters()) for name,module in self.modules_for_audit().items()},
                'policy_total':sum(p.numel() for p in self.policy.parameters()),'critics_total':sum(p.numel() for p in self.critics.parameters())}

    @torch.no_grad()
    def parameter_drift(self):
        return {name:float((torch.nn.utils.parameters_to_vector(module.parameters())-self.initial_parameters[name]).norm())
                for name,module in self.modules_for_audit().items()}

    def tensor(self,value):return torch.as_tensor(value,dtype=torch.float32,device=self.device)

    @torch.no_grad()
    def _history(self,data,chunks):
        envs=torch.tensor([c[0] for c in chunks],device=self.device)
        starts=torch.tensor([c[1] for c in chunks],device=self.device)
        hidden=data['hidden_before'][0,envs].detach().clone()
        # Burn in the complete available prefix under the current parameters.
        # At the rollout boundary the recorded behavior memory is the TBPTT boundary.
        for step in range(int(starts.max())):
            new,*_=self.policy.memory(data['observations'][step,envs],hidden,data['previous_action'][step,envs],
                data['alive'][step,envs],data['reset'][step,envs])
            hidden=torch.where((step<starts)[:,None,None],new,hidden)
        return hidden.detach()

    def evaluate_sequences(self,data,chunks,entropy=True):
        L=self.config['training']['sequence_length'];T=len(data['observations'])
        starts=torch.tensor([c[1] for c in chunks],device=self.device)
        ends=torch.tensor([c[2] for c in chunks],device=self.device)
        envs=torch.tensor([c[0] for c in chunks],device=self.device)
        times=starts[None]+torch.arange(L,device=self.device)[:,None]
        valid=times<ends[None];indices=times.clamp_max(T-1)
        packed={key:value[indices,envs[None]] for key,value in data.items()}
        packed['alive']=packed['alive']*valid.unsqueeze(-1)
        for event_mask in ('gate_valid','intent_valid'):
            packed[event_mask]=packed[event_mask]*valid.unsqueeze(-1)
        packed['reset']=torch.where(valid.unsqueeze(-1),packed['reset'],torch.ones_like(packed['reset']))
        hidden=self._history(data,chunks);outputs={key:[] for key in ('action_logp','intent_logp','gate_logp','action_entropy','intent_entropy','gate_entropy','inference_logits','inference_valid')}
        for step in range(L):
            hidden,allies,ally_valid,_,_=self.policy.memory(packed['observations'][step],hidden,packed['previous_action'][step],packed['alive'][step],packed['reset'][step])
            context,inference=self.policy.context(hidden,allies,ally_valid)
            gate,intent,mean,log_std=self.policy.heads(hidden,context,packed['old_intent'][step],packed['age_before'][step],packed['intent'][step])
            logp=squashed_log_prob(mean,log_std,packed['raw_actions'][step])
            zero=logp*0
            action_entropy=zero
            if entropy:
                raw=mean+log_std.exp()*torch.randn(mean.shape,device=self.device,generator=self.entropy_rng)
                action_entropy=-squashed_log_prob(mean,log_std,raw)
            outputs['action_logp'].append(logp);outputs['action_entropy'].append(action_entropy)
            outputs['intent_logp'].append(Categorical(logits=intent).log_prob(packed['intent'][step].clamp_min(0)) if intent is not None else zero)
            outputs['intent_entropy'].append(Categorical(logits=intent).entropy() if intent is not None else zero)
            outputs['gate_logp'].append(Bernoulli(logits=gate).log_prob(packed['gate'][step]) if gate is not None else zero)
            outputs['gate_entropy'].append(Bernoulli(logits=gate).entropy() if gate is not None else zero)
            if inference is not None:
                outputs['inference_logits'].append(inference)
                outputs['inference_valid'].append(ally_valid&packed['alive'][step].bool().unsqueeze(-1))
        return packed,{key:torch.stack(values) for key,values in outputs.items() if values}

    @torch.no_grad()
    def audit_rollout_ratio(self,rollout):
        data=rollout.data;T,E,A=data['alive'].shape;L=self.config['training']['sequence_length'];errors={}
        for e in range(E):
            for start in range(0,T,L):
                p,out=self.evaluate_sequences(data,[(e,start,min(start+L,T))],entropy=False)
                for head,mask in (('action','alive'),('gate','gate_valid'),('intent','intent_valid')):
                    selected=p[mask].bool()
                    if selected.any():
                        difference=(out[head+'_logp']-p[head+'_old_logp'])[selected].abs()
                        errors[head]=max(errors.get(head,0.),float(difference.max()))
        return errors

    def _fit_critics(self,data):
        t=self.config['training'];w=self.config['loss_weights'];T,E=data['value'].shape
        flat={key:data[key].flatten(0,1).detach() for key in ('observations','alive','hidden','old_intent','age_before','value','returns','gate','gate_valid')}
        rows=[]
        for _ in range(t['ppo_epochs']):
            order=self.permutation_rng.permutation(T*E)
            for start in range(0,len(order),t['critic_minibatch_size']):
                ix=torch.as_tensor(order[start:start+t['critic_minibatch_size']],device=self.device)
                b={key:value[ix] for key,value in flat.items()}
                value,q=self.critics(b['observations'],b['alive'],b['hidden'],b['old_intent'],b['age_before'])
                error=(value-b['returns']).square()
                if t['value_clipping']:
                    clipped=b['value']+(value-b['value']).clamp(-t['clip_ratio'],t['clip_ratio'])
                    error=torch.maximum(error,(clipped-b['returns']).square())
                value_loss=.5*error.mean()
                vpre,vpost=clipped_step(w['team_value']*value_loss,self.critics.team,self.team_optimizer,t['max_grad_norm'])
                self.team_optimizer_steps+=1
                row=dict(team_value_loss=float(value_loss.detach()),team_value_weighted_loss=float(w['team_value']*value_loss.detach()),team_grad_pre=vpre,team_grad_post=vpost)
                if q is not None and b['gate_valid'].bool().any():
                    chosen=q.gather(-1,b['gate'].long().unsqueeze(-1)).squeeze(-1)
                    loss=.5*event_mean((chosen-b['returns'].unsqueeze(-1)).square(),b['gate_valid'])
                    pre,post=clipped_step(w['gate_q']*loss,self.critics.gate,self.gate_optimizer,t['max_grad_norm'])
                    self.gate_optimizer_steps+=1
                    row.update(gate_q_loss=float(loss.detach()),gate_q_weighted_loss=float(w['gate_q']*loss.detach()),gate_q_mse=float(2*loss.detach()),gate_critic_grad_pre=pre,gate_critic_grad_post=post)
                rows.append(row)
        with torch.no_grad():
            _,q=self.critics(data['observations'],data['alive'],data['hidden'],data['old_intent'],data['age_before'])
            if q is None:gate_adv=data['low_advantage'].clone()
            else:
                baseline=(1-data['gate_probability'])*q[...,0]+data['gate_probability']*q[...,1]
                gate_adv=q.gather(-1,data['gate'].long().unsqueeze(-1)).squeeze(-1)-baseline
        return rows,gate_adv.detach(),None if q is None else q.detach()

    def update(self,rollout):
        if rollout.behavior_version!=self.update_count:raise RuntimeError('stale/off-policy ACIRL rollout rejected')
        data=rollout.data;t=self.config['training'];w=self.config['loss_weights'];ent=self.config['entropy']
        ratio_errors=self.audit_rollout_ratio(rollout)
        if ratio_errors and max(ratio_errors.values())>2e-4:raise RuntimeError(f'behavior recurrent probability mismatch: {ratio_errors}')
        critic_rows,gate_adv,q=self._fit_critics(data)
        data['gate_advantage']=gate_adv
        for head,mask in (('low','alive'),('high','intent_valid'),('gate','gate_valid')):
            data[head+'_normalized']=normalize(data[head+'_advantage'],data[mask],t['normalize_advantages'])
        T,E,A=data['alive'].shape;L=t['sequence_length']
        chunks=[(e,start,min(start+L,T)) for e in range(E) for start in range(0,T,L)]
        per_batch=max(1,t['minibatch_size']//(L*A));policy_rows=[];stopped=False;epochs=0
        for _ in range(t['ppo_epochs']):
            order=self.permutation_rng.permutation(len(chunks));epoch_rows=[]
            for start in range(0,len(order),per_batch):
                selected=[chunks[ix] for ix in order[start:start+per_batch]]
                p,out=self.evaluate_sequences(data,selected)
                losses={};kls={};clips={};entropies={}
                for head,network,mask,adv in (('action','action','alive','low'),('intention','intent','intent_valid','high'),('gate','gate','gate_valid','gate')):
                    loss,kl,clip=ppo_component(out[network+'_logp'],p[network+'_old_logp'],p[adv+'_normalized'],p[mask]*p['alive'],t['clip_ratio'])
                    losses[head]=loss;kls[head]=float(kl);clips[head]=float(clip)
                    entropies[head]=event_mean(out[network+'_entropy'],p[mask]*p['alive'])
                inference_loss=out['action_logp'].sum()*0;inference_accuracy=0.;inference_count=0
                inferred_counts=[0]*self.config['network']['intentions'];teacher_counts=list(inferred_counts)
                if self.policy.has_inference:
                    peer_ids=torch.tensor([[j for j in range(A) if j!=i] for i in range(A)],device=self.device)
                    # Teacher labels enter only this supervised loss, never policy.context/heads.
                    labels=p['intent'][...,peer_ids]
                    valid=out['inference_valid']&(labels>=0)
                    logits=out['inference_logits']
                    inference_count=int(valid.sum())
                    if inference_count:
                        inference_loss=F.cross_entropy(logits[valid],labels[valid])
                        inference_accuracy=float((logits[valid].argmax(-1)==labels[valid]).float().mean())
                        inferred_counts=torch.bincount(logits[valid].argmax(-1),minlength=len(inferred_counts)).cpu().tolist()
                        teacher_counts=torch.bincount(labels[valid],minlength=len(teacher_counts)).cpu().tolist()
                losses['inference']=inference_loss
                total=sum(w[head]*loss for head,loss in losses.items())-sum(ent[head]*entropy for head,entropy in entropies.items())
                self.policy_optimizer.zero_grad(set_to_none=True)
                # Critics are separate optimizers and detached from the policy graph.
                self.team_optimizer.zero_grad(set_to_none=True)
                if self.gate_optimizer:self.gate_optimizer.zero_grad(set_to_none=True)
                if not torch.isfinite(total):raise FloatingPointError('non-finite ACIRL joint policy loss')
                total.backward()
                # Empty event sets must not advance Adam momentum in their heads.
                for name,mask in (('gate_actor','gate_valid'),('intention_actor','intent_valid')):
                    if hasattr(self.policy,name) and not (p[mask]*p['alive']).bool().any():
                        for parameter in getattr(self.policy,name).parameters():parameter.grad=None
                norms={name:gradient_norm(module) for name,module in self.modules_for_audit().items() if 'critic' not in name}
                pre=float(torch.nn.utils.clip_grad_norm_(self.policy.parameters(),t['max_grad_norm'],error_if_nonfinite=True))
                post=gradient_norm(self.policy);self.policy_optimizer.step();self.policy_optimizer_steps+=1
                row=dict(policy_loss=float(total.detach()),policy_grad_pre=pre,policy_grad_post=post,
                    gradient_norms=norms,inference_accuracy=inference_accuracy,inference_samples=inference_count,
                    inference_predicted_counts=inferred_counts,inference_teacher_counts=teacher_counts,
                    high_events=int((p['intent_valid']*p['alive']).sum()),gate_events=int((p['gate_valid']*p['alive']).sum()),
                    low_samples=int(p['alive'].sum()))
                for head,loss in losses.items():row[head+'_loss']=float(loss.detach());row[head+'_weighted_loss']=float(w[head]*loss.detach())
                for head,value in entropies.items():
                    row[head+'_entropy']=float(value.detach());row[head+'_weighted_entropy_bonus']=float(-ent[head]*value.detach())
                    row[head+'_approx_kl']=kls[head];row[head+'_clip_fraction']=clips[head]
                epoch_rows.append(row);policy_rows.append(row)
            epochs+=1
            # Each component has its own probability ratio and KL stopping rule.
            if any(np.average([r[head+'_approx_kl'] for r in epoch_rows],weights=[max(1,r[count]) for r in epoch_rows])>t['target_kl']
                   for head,count in (('action','low_samples'),('intention','high_events'),('gate','gate_events'))):
                stopped=True;break
        self.update_count+=1
        from .diagnostics import summarize
        return summarize(self,rollout,critic_rows,policy_rows,gate_adv,q,ratio_errors,epochs,stopped)

    def begin_episode(self,seed,deterministic=True):
        self._session=PolicySession(self.policy,self.device,seed,deterministic)

    def act(self,observations,alive_masks,deterministic=True):
        if self._session is None:self.begin_episode(self.seed,deterministic)
        if self._session.deterministic!=deterministic:raise ValueError('reset episode before changing evaluation mode')
        return self._session.act(observations,alive_masks)

    def recording_step_data(self):
        return {key:self._session.last[key][0].cpu().numpy().copy() for key in
            ('intent','old_intent','age','age_before','gate','gate_valid','forced','reason','intention_probability','gate_probability','reset')}

    def save(self,path,env,extra=None,runtime=None):
        common_extra=dict(environment_version=env['environment_version'],environment_config_sha256=config_sha256(env),
            algorithm_config_sha256=config_sha256(self.config),training_seed=self.seed,
            **{key:self.config['network'][key] for key in ('observation_dim','action_dim','num_agents')},**(extra or {}))
        state=dict(algorithm='acirl',implementation_version=ACIRL_IMPL_VERSION,
            environment_config=deepcopy(env),algorithm_config=deepcopy(self.config),
            environment_config_sha256=config_sha256(env),algorithm_config_sha256=config_sha256(self.config),
            network_architecture=architecture(self.config),ablation_mode=self.config['ablation']['mode'],
            parameter_counts=self.parameter_counts(),policy=self.policy.state_dict(),critics=self.critics.state_dict(),
            policy_optimizer=self.policy_optimizer.state_dict(),team_optimizer=self.team_optimizer.state_dict(),
            gate_optimizer=self.gate_optimizer.state_dict() if self.gate_optimizer else None,
            sampled_steps=self.sampled_steps,vector_steps=self.vector_steps,update_count=self.update_count,
            training_seed=self.seed,
            policy_optimizer_steps=self.policy_optimizer_steps,team_optimizer_steps=self.team_optimizer_steps,gate_optimizer_steps=self.gate_optimizer_steps,
            rng_states=dict(behavior=self.behavior_rng.get_state(),entropy=self.entropy_rng.get_state(),
                permutation=deepcopy(self.permutation_rng.bit_generator.state),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(),python=random.getstate()),
            initial_parameters={k:v.detach().cpu() for k,v in self.initial_parameters.items()},
            formal_exact_resume_supported=False,training_runtime=runtime,
            evaluation_history=(extra or {}).get('evaluation_history',[]),best_evaluation=(extra or {}).get('best_evaluation'),extra=deepcopy(common_extra))
        Path(path).parent.mkdir(parents=True,exist_ok=True);torch.save(state,path)

    def load(self,path,env):
        state=torch.load(path,map_location=self.device,weights_only=False);validate_checkpoint(state,env,self.config)
        if state['parameter_counts']!=self.parameter_counts():raise RuntimeError('ACIRL parameter count mismatch')
        for name in ('policy','critics'):
            expected=getattr(self,name).state_dict();saved=state[name]
            if set(expected)!=set(saved) or any(expected[k].shape!=saved[k].shape or not torch.isfinite(saved[k]).all() for k in expected):
                raise RuntimeError(f'ACIRL {name} tensor architecture/nonfinite mismatch')
        for name in ('policy','critics'):getattr(self,name).load_state_dict(state[name],strict=True)
        for name in ('policy_optimizer','team_optimizer','gate_optimizer'):
            optimizer=getattr(self,name)
            if optimizer:optimizer.load_state_dict(state[name])
        for name in ('sampled_steps','vector_steps','update_count','policy_optimizer_steps','team_optimizer_steps','gate_optimizer_steps'):
            setattr(self,name,state[name])
        self.initial_parameters={k:v.to(self.device) for k,v in state['initial_parameters'].items()}
        self.behavior_rng.set_state(state['rng_states']['behavior'].cpu());self.entropy_rng.set_state(state['rng_states']['entropy'].cpu())
        self.permutation_rng.bit_generator.state=deepcopy(state['rng_states']['permutation']);self._session=None
        return state

    def resume(self,*args,**kwargs):
        raise RuntimeError('exact resume unsupported; use --warm-start with fresh environment episodes and explicit provenance')
