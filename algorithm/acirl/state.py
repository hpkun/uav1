"""Per-environment/per-UAV runtime state; all stochastic draws use owned RNGs."""
from dataclasses import dataclass
import torch
from torch.distributions import Bernoulli, Categorical
from .networks import squashed_log_prob


@dataclass
class IntentionState:
    hidden:torch.Tensor
    previous_action:torch.Tensor
    intent:torch.Tensor
    age:torch.Tensor
    reset:torch.Tensor

    def clone(self):return IntentionState(*(v.detach().clone() for v in vars(self).values()))


def initial_state(policy,num_envs,device):
    n=policy.n;shape=(num_envs,n['num_agents'])
    return IntentionState(torch.zeros(*shape,n['gru_hidden'],device=device),torch.zeros(*shape,n['action_dim'],device=device),
        torch.full(shape,-1,device=device,dtype=torch.long),torch.zeros(shape,device=device),torch.ones(shape,device=device))


def gate_decision(logits,old_intent,age,alive,config,rng,deterministic=False):
    mode=config['ablation']['mode'];p=config['intention'];live=alive.bool()
    initial=(old_intent<0)&live
    maximum=(age>=p['max_duration'])&live
    if mode=='no_intent':return torch.zeros_like(initial),torch.zeros_like(initial),torch.zeros_like(initial),torch.zeros_like(age),torch.full_like(old_intent,5)
    if mode=='fixed_interval':
        renew=initial|((age>=p['fixed_interval'])&live)
        return renew,torch.zeros_like(initial),renew,torch.zeros_like(age),torch.where(initial,1,torch.where(renew,4,0))
    valid=live&~initial&~maximum&(age>=p['min_duration'])
    sampled=(logits>=0) if deterministic else (torch.rand(logits.shape,device=logits.device,generator=rng)<logits.sigmoid())
    renew=initial|maximum|(sampled&valid)
    forced=initial|maximum
    logp=Bernoulli(logits=logits).log_prob(renew.float())
    reason=torch.where(initial,1,torch.where(maximum,2,torch.where(sampled&valid,3,0)))
    return renew,valid,forced,torch.where(valid,logp,torch.zeros_like(logp)),reason


@torch.no_grad()
def decide(policy,observations,alive,state,rng,deterministic=False):
    old=state.clone();live=alive.bool()
    old.intent=torch.where(live,old.intent,torch.full_like(old.intent,-1))
    old.age=old.age*alive
    hidden,allies,valid,_,_=policy.memory(observations,old.hidden,old.previous_action,alive,old.reset)
    context,inference=policy.context(hidden,allies,valid)
    gate,intent_logits=policy.heads(hidden,context,old.intent,old.age)
    renew,gate_valid,forced,gate_logp,reason=gate_decision(gate,old.intent,old.age,alive,policy.config,rng,deterministic)
    selected=old.intent.clone();intent_logp=torch.zeros_like(alive)
    if policy.has_intent and renew.any():
        logits=intent_logits[renew]
        samples=logits.argmax(-1) if deterministic else torch.multinomial(logits.softmax(-1),1,generator=rng).squeeze(-1)
        selected[renew]=samples;intent_logp[renew]=Categorical(logits=logits).log_prob(samples)
    selected=torch.where(live,selected,torch.full_like(selected,-1))
    _,_,mean,log_std=policy.heads(hidden,context,old.intent,old.age,selected)
    raw=mean if deterministic else mean+log_std.exp()*torch.randn(mean.shape,device=mean.device,generator=rng)
    actions=raw.tanh()*alive.unsqueeze(-1)
    action_logp=squashed_log_prob(mean,log_std,raw)*alive
    age=torch.where(renew,torch.zeros_like(old.age),old.age)+alive
    next_state=IntentionState(hidden.detach(),actions.detach(),selected.detach(),age.detach(),torch.zeros_like(alive))
    data=dict(observations=observations,alive=alive,hidden_before=old.hidden,previous_action=old.previous_action,
        reset=old.reset,old_intent=old.intent,age_before=old.age,hidden=hidden,intent=selected,age=age,
        gate=renew.float(),gate_valid=gate_valid.float(),forced=forced.float(),reason=reason,
        gate_old_logp=gate_logp,intent_valid=(renew&live).float() if policy.has_intent else torch.zeros_like(alive),
        intent_old_logp=intent_logp,actions=actions,raw_actions=raw,action_old_logp=action_logp,
        gate_probability=gate.sigmoid() if gate is not None else torch.zeros_like(alive),
        intention_probability=intent_logits.softmax(-1).gather(-1,selected.clamp_min(0).unsqueeze(-1)).squeeze(-1) if policy.has_intent else torch.zeros_like(alive))
    return data,next_state


def finish_transition(state,next_alive,done):
    keep=next_alive*(1-done.unsqueeze(-1))
    state.hidden*=keep.unsqueeze(-1);state.previous_action*=keep.unsqueeze(-1)
    state.intent=torch.where(keep.bool(),state.intent,torch.full_like(state.intent,-1))
    state.age*=keep;state.reset=1-keep
    return state


class PolicySession:
    def __init__(self,policy,device,seed,deterministic=True):
        self.policy=policy;self.device=device;self.deterministic=deterministic
        self.rng=torch.Generator(device=device).manual_seed(int(seed)^0xAC17E)
        self.state=None;self.last=None

    def act(self,observations,alive):
        obs=torch.as_tensor(observations,dtype=torch.float32,device=self.device)
        mask=torch.as_tensor(alive,dtype=torch.float32,device=self.device)
        single=obs.ndim==2
        if single:obs=obs.unsqueeze(0);mask=mask.unsqueeze(0)
        if self.state is None:self.state=initial_state(self.policy,len(obs),self.device)
        if len(obs)!=len(self.state.intent):raise ValueError('reset session before changing environment batch size')
        self.last,self.state=decide(self.policy,obs,mask,self.state,self.rng,self.deterministic)
        actions=self.last['actions'].cpu().numpy()
        return actions[0] if single else actions
