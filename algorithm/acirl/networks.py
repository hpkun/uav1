"""Local-only recurrent hierarchical policy and separate centralized critics."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Bernoulli, Categorical, Normal


def mlp(inputs, widths, outputs):
    layers=[]
    for width in widths:
        layers.extend((nn.Linear(inputs,width),nn.ReLU()));inputs=width
    return nn.Sequential(*layers,nn.Linear(inputs,outputs))


def squashed_log_prob(mean,log_std,raw):
    jac=2*(math.log(2)-raw-F.softplus(-2*raw))
    return (Normal(mean,log_std.exp()).log_prob(raw)-jac).sum(-1)


class EntityEncoder(nn.Module):
    def __init__(self,n):
        super().__init__();self.n=dict(n)
        d=n['entity_embedding'];self.heads=n['attention_heads'];self.width=d//self.heads
        self.self_encoder=mlp(n['self_features'],[d],d)
        self.ally_encoder=mlp(n['ally_features'],[d],d)
        self.enemy_encoder=mlp(n['enemy_features'],[d],d)
        self.query=nn.Linear(d,d);self.key=nn.Linear(d,d);self.value=nn.Linear(d,d)
        self.fusion=mlp(3*d,[d],d)

    def pool(self,own,entities,valid):
        shape=own.shape[:-1];d=own.shape[-1];count=entities.shape[-2]
        q=self.query(own).reshape(*shape,self.heads,self.width)
        k=self.key(entities).reshape(*shape,count,self.heads,self.width).transpose(-3,-2)
        v=self.value(entities).reshape(*shape,count,self.heads,self.width).transpose(-3,-2)
        logits=(q.unsqueeze(-2)*k).sum(-1)/math.sqrt(self.width)
        mask=valid.unsqueeze(-2)
        weights=torch.softmax(logits.masked_fill(~mask,-1e9),-1)*mask
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        context=(weights.unsqueeze(-1)*v).sum(-2).reshape(*shape,d)
        return context,weights

    def forward(self,observations,alive):
        n=self.n;count=n['num_agents'];s=n['self_features'];a=n['ally_features'];e=n['enemy_features']
        obs=torch.where(alive.bool().unsqueeze(-1),observations,torch.zeros_like(observations))
        own=obs[...,:s]
        allies=obs[...,s:s+(count-1)*a].reshape(*obs.shape[:-1],count-1,a)
        enemies=obs[...,s+(count-1)*a:].reshape(*obs.shape[:-1],count,e)
        av=allies.abs().sum(-1)>0;ev=enemies.abs().sum(-1)>0
        # Zero slots are the only locally observable entity-validity signal.
        own=self.self_encoder(own)
        ae=self.ally_encoder(allies)*av.unsqueeze(-1)
        ee=self.enemy_encoder(enemies)*ev.unsqueeze(-1)
        ac,aw=self.pool(own,ae,av);ec,ew=self.pool(own,ee,ev)
        embedding=self.fusion(torch.cat((own,ac,ec),-1))*alive.unsqueeze(-1)
        return embedding,ae,av,aw,ew


class ACIRLPolicy(nn.Module):
    """All axes before the feature axis are independent local actor instances."""
    def __init__(self,config):
        super().__init__();self.config=config;self.n=config['network'];self.mode=config['ablation']['mode']
        n=self.n;d=n['entity_embedding'];h=n['gru_hidden'];k=n['intentions'];z=n['intention_embedding']
        self.has_intent=self.mode!='no_intent'
        self.has_gate=self.mode not in ('no_intent','fixed_interval')
        self.has_inference=self.mode not in ('no_intent','no_teammate_inference')
        self.encoder=EntityEncoder(n);self.gru=nn.GRUCell(d+n['action_dim'],h)
        if self.has_intent:self.intent_embedding=nn.Embedding(k,z)
        if self.has_inference:
            self.inference=mlp(h+d,[d],k);self.team_embedding=nn.Embedding(k,z)
        context=z if self.has_inference else 0
        if self.has_intent:self.intention_actor=mlp(h+context,n['actor_hidden'],k)
        if self.has_gate:self.gate_actor=mlp(h+z+1+context,n['actor_hidden'],1)
        self.action_actor=mlp(h+(z if self.has_intent else 0)+context,n['actor_hidden'],2*n['action_dim'])
        with torch.no_grad():
            head=self.action_actor[-1];head.weight.mul_(n['mean_head_gain']);head.bias.zero_()
            head.weight[n['action_dim']:].zero_();head.bias[n['action_dim']:].fill_(n['log_std_init'])

    def memory(self,obs,hidden,previous_action,alive,reset):
        keep=alive*(1-reset)
        hidden=torch.where(keep.bool().unsqueeze(-1),hidden,torch.zeros_like(hidden))
        previous_action=torch.where(keep.bool().unsqueeze(-1),previous_action,torch.zeros_like(previous_action))
        embedding,allies,valid,aw,ew=self.encoder(obs,alive)
        shape=hidden.shape
        hidden=self.gru(torch.cat((embedding,previous_action),-1).reshape(-1,embedding.shape[-1]+previous_action.shape[-1]),
                        hidden.reshape(-1,shape[-1])).reshape(shape)*alive.unsqueeze(-1)
        return hidden,allies,valid,aw,ew

    def context(self,hidden,allies,valid):
        if not self.has_inference:return hidden[...,:0],None
        own=hidden.unsqueeze(-2).expand(*allies.shape[:-1],hidden.shape[-1])
        logits=self.inference(torch.cat((own,allies),-1))
        estimates=logits.softmax(-1)@self.team_embedding.weight
        context=(estimates*valid.unsqueeze(-1)).sum(-2)/valid.sum(-1,keepdim=True).clamp_min(1)
        return context,logits

    def heads(self,hidden,context,old_intent,age,current_intent=None):
        base=torch.cat((hidden,context),-1)
        gate=None;intent=None
        if self.has_intent:
            intent=self.intention_actor(base)
            if self.has_gate:
                own=self.intent_embedding(old_intent.clamp_min(0))
                gate=self.gate_actor(torch.cat((base,own,age.unsqueeze(-1)/self.config['intention']['max_duration']),-1)).squeeze(-1)
        if current_intent is None:return gate,intent
        features=base
        if self.has_intent:features=torch.cat((base,self.intent_embedding(current_intent.clamp_min(0))),-1)
        mean,log_std=self.action_actor(features).chunk(2,-1)
        return gate,intent,mean,log_std.clamp(self.n['log_std_min'],self.n['log_std_max'])


class CentralizedCritics(nn.Module):
    def __init__(self,config):
        super().__init__();self.n=config['network'];self.mode=config['ablation']['mode']
        n=self.n;self.has_intent=self.mode!='no_intent'
        self.has_gate_q=self.mode not in ('no_intent','fixed_interval','no_gate_q')
        self.per_agent=n['observation_dim']+n['gru_hidden']+1+(n['intentions']+1 if self.has_intent else 0)
        self.joint_dim=n['num_agents']*self.per_agent
        self.team=mlp(self.joint_dim,n['critic_hidden'],1)
        if self.has_gate_q:self.gate=mlp(self.joint_dim+self.per_agent+n['num_agents'],n['critic_hidden'],2)

    def features(self,obs,alive,hidden,intent,age):
        mask=alive.bool().unsqueeze(-1)
        pieces=[torch.where(mask,obs,torch.zeros_like(obs)),torch.where(mask,hidden,torch.zeros_like(hidden)),alive.unsqueeze(-1)]
        if self.has_intent:
            z=F.one_hot(intent.clamp_min(0),self.n['intentions']).float()*((intent>=0)*alive).unsqueeze(-1)
            duration=torch.where(alive.bool(),age,torch.zeros_like(age))
            pieces.extend((z,duration.unsqueeze(-1)/self.n['max_duration_scale']))
        return torch.cat(pieces,-1)*alive.unsqueeze(-1)

    def forward(self,obs,alive,hidden,intent,age):
        local=self.features(obs,alive,hidden,intent,age);joint=local.flatten(-2)
        value=self.team(joint).squeeze(-1)
        q=None
        if self.has_gate_q:
            n=self.n['num_agents'];ids=torch.eye(n,device=obs.device).expand(*local.shape[:-2],n,n)
            q=self.gate(torch.cat((joint.unsqueeze(-2).expand(*local.shape[:-1],joint.shape[-1]),local,ids),-1))
        return value,q
