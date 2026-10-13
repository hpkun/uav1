"""Continuous MAAC: independent local policies and Eq.(5)-(6) attention Q."""
import math
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


class GaussianLocalActor(nn.Module):
    def __init__(self, observation_dim=56, action_dim=3, hidden_dim=128, log_std_min=-5., log_std_max=2.):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(observation_dim,hidden_dim),nn.LeakyReLU(.01),
            nn.Linear(hidden_dim,hidden_dim),nn.LeakyReLU(.01))
        self.mean = nn.Linear(hidden_dim,action_dim)
        self.log_std = nn.Linear(hidden_dim,action_dim)
        self.log_std_min,self.log_std_max=log_std_min,log_std_max

    def distribution(self, observations):
        embedding=self.backbone(observations)
        return Normal(self.mean(embedding),self.log_std(embedding).clamp(self.log_std_min,self.log_std_max).exp())

    @staticmethod
    def squashed_log_prob(distribution, raw_action):
        correction=2*(math.log(2.)-raw_action-F.softplus(-2*raw_action))
        return (distribution.log_prob(raw_action)-correction).sum(-1)

    def sample(self, observations, deterministic=False):
        distribution=self.distribution(observations)
        # Eq.(8) score-function sampling, deliberately NOT rsample().
        raw=distribution.mean if deterministic else distribution.sample()
        return raw.tanh(),self.squashed_log_prob(distribution,raw),raw

    def log_prob(self, observations, raw_action):
        return self.squashed_log_prob(self.distribution(observations),raw_action.detach())

    def forward(self, observations):
        return self.distribution(observations).mean.tanh()


class CentralizedAttentionCritic(nn.Module):
    def __init__(self, observation_dim=56, action_dim=3, num_agents=5, hidden_dim=128, attention_heads=4):
        super().__init__()
        if hidden_dim % attention_heads:raise ValueError('hidden_dim must divide into attention heads')
        self.observation_dim,self.action_dim,self.num_agents=observation_dim,action_dim,num_agents
        self.hidden_dim,self.heads=hidden_dim,attention_heads
        self.encoders=nn.ModuleList([nn.Sequential(nn.Linear(observation_dim+action_dim,hidden_dim),
            nn.LeakyReLU(.01)) for _ in range(num_agents)])
        self.query=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.key=nn.Linear(hidden_dim,hidden_dim,bias=False)
        self.value=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.LeakyReLU(.01))
        self.q_heads=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),nn.LeakyReLU(.01),
            nn.Linear(hidden_dim,1)) for _ in range(num_agents)])

    def shared_parameters(self):
        for module in (self.query,self.key,self.value):yield from module.parameters()

    def forward(self, observations, actions, alive_masks, return_attention=False):
        if (observations.ndim!=3 or observations.shape[1:]!=(self.num_agents,self.observation_dim)
                or actions.shape!=(*observations.shape[:2],self.action_dim)
                or alive_masks.shape!=observations.shape[:2]):
            raise ValueError('critic requires obs [B,5,56], actions [B,5,3], alive [B,5]')
        alive=alive_masks.bool()
        z=torch.cat((observations,actions),-1).masked_fill(~alive[...,None],0.)
        embedding=torch.stack([encoder(z[:,i]) for i,encoder in enumerate(self.encoders)],1)
        def heads(x):return x.reshape(x.shape[0],self.num_agents,self.heads,-1).transpose(1,2)
        query,key,value=heads(self.query(embedding)),heads(self.key(embedding)),heads(self.value(embedding))
        scores=(query@key.transpose(-1,-2))/math.sqrt(self.hidden_dim/self.heads)
        not_self=~torch.eye(self.num_agents,dtype=torch.bool,device=observations.device)
        allowed=alive[:,None,:,None]&alive[:,None,None,:]&not_self[None,None]
        scores=scores.masked_fill(~allowed,torch.finfo(scores.dtype).min)
        attention=scores.softmax(-1)*allowed
        context=(attention@value).transpose(1,2).reshape(observations.shape[0],self.num_agents,self.hidden_dim)
        q=torch.stack([head(torch.cat((embedding[:,i],context[:,i]),-1))
                       for i,head in enumerate(self.q_heads)],1)*alive[...,None]
        return (q,attention) if return_attention else q
