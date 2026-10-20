"""Local entity histories, masked cross-attention and centralized value learning."""
import math
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from .memory import MemoryState, align_hidden


def squashed_log_prob(distribution, raw):
    # Same saturation-safe identity verified in the existing PPO implementation;
    # kept here to avoid importing any task-specific runtime/environment.
    jacobian = 2 * (math.log(2.) - raw - F.softplus(-2 * raw))
    return (distribution.log_prob(raw) - jacobian).sum(-1)


class EntityEncoder(nn.Sequential):
    def __init__(self, input_dim, hidden):
        super().__init__(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())


class TemporalAttention(nn.Module):
    def __init__(self, dim=64, heads=2):
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError('attention dimension must divide into positive heads')
        self.dim, self.heads = dim, heads
        self.q, self.k, self.v = (nn.Linear(dim, dim) for _ in range(3))

    def forward(self, query, entities, valid):
        prefix, slots = query.shape[:-1], entities.shape[-2]
        if valid.shape != entities.shape[:-1] or valid.dtype != torch.bool:
            raise ValueError('attention validity shape/dtype mismatch')
        if slots == 0:
            return torch.zeros_like(query), query.new_zeros((*prefix, self.heads, 0))
        width = self.dim // self.heads
        q = self.q(query).reshape(*prefix, self.heads, width).unsqueeze(-2)
        k = self.k(entities).reshape(*prefix, slots, self.heads, width).transpose(-3, -2)
        v = self.v(entities).reshape(*prefix, slots, self.heads, width).transpose(-3, -2)
        scores = (q @ k.transpose(-2, -1)).squeeze(-2) / math.sqrt(width)
        mask = valid.unsqueeze(-2)
        # Empty rows use finite zero scores before softmax, then explicit masks.
        any_valid = mask.any(-1, keepdim=True)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        scores = torch.where(any_valid, scores, torch.zeros_like(scores))
        weights = scores.softmax(-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        context = (weights.unsqueeze(-2) @ v).squeeze(-2).reshape(*prefix, self.dim)
        return context, weights


class EntityTemporalActor(nn.Module):
    """B0/B1/B2/B3 differ only in the stated actor structural ablation."""
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.variant = config['variant']
        h, f = config['memory_dim'], config['fusion_dim']
        self.h, self.f = h, f
        self.self_encoder = EntityEncoder(config['self_dim'], h)
        self.peer_encoder = EntityEncoder(config['peer_dim'], h)
        self.obstacle_encoder = EntityEncoder(config['obstacle_dim'], h)
        if self.variant in ('B2', 'B3'):
            self.self_gru = nn.GRUCell(h, h)
            self.peer_gru = nn.GRUCell(h, h)
            self.obstacle_gru = nn.GRUCell(h, h)
        if self.variant != 'B0':
            self.peer_attention = TemporalAttention(h, config['attention_heads'])
            self.obstacle_attention = TemporalAttention(h, config['attention_heads'])
        if self.variant in ('B2','B3'):
            self.self_projection = nn.Linear(h, f)
            self.context_projection = nn.Linear(2*h, f)
            if self.variant == 'B3':
                self.gate = nn.Linear(2*f, f)
            self.fusion_norm = nn.LayerNorm(f)
        else:
            self.fusion = nn.Sequential(nn.Linear(3*h, f), nn.LayerNorm(f), nn.ReLU())
        if self.variant == 'B1':
            self.post_gru = nn.GRUCell(f, f)
        self.policy_hidden = nn.Sequential(nn.Linear(f, f), nn.ReLU())
        self.mean = nn.Linear(f, config['action_dim'])
        self.log_std = nn.Linear(f, config['action_dim'])
        nn.init.orthogonal_(self.mean.weight, gain=.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, config['log_std_init'])

    def initial_state(self, observations):
        own = observations['self_features']
        prefix = own.shape[:-1]
        peers, obstacles = observations['peer_ids'], observations['obstacle_ids']
        return MemoryState(own.new_zeros((*prefix, self.h)),
            own.new_zeros((*peers.shape, self.h)), own.new_zeros((*obstacles.shape, self.h)),
            torch.full_like(peers, -1), torch.full_like(obstacles, -1), own.new_zeros((*prefix, self.f)))

    def _validate(self, o, state, starts):
        prefix = o['self_features'].shape[:-1]
        if len(prefix) != 2 or o['self_features'].shape[-1] != self.config['self_dim']:
            raise ValueError('self features must be [env,agent,self_dim]')
        if o['agent_alive'].shape != prefix or o['agent_alive'].dtype != torch.bool:
            raise ValueError('agent_alive must be a bool [env,agent] mask')
        if starts.shape != prefix[:1] or starts.dtype != torch.bool:
            raise ValueError('episode_start must be bool [env]')
        for side, dimension in (('peer', self.config['peer_dim']), ('obstacle', self.config['obstacle_dim'])):
            features, ids, valid = (o[f'{side}_{key}'] for key in ('features','ids','valid'))
            if features.shape[:-1] != ids.shape or ids.shape != valid.shape or ids.shape[:-1] != prefix or features.shape[-1] != dimension:
                raise ValueError(f'{side} feature/id/mask shape mismatch')
        if state.self_hidden.shape != (*prefix, self.h) or state.fusion_hidden.shape != (*prefix,self.f):
            raise ValueError('self/fusion hidden shape mismatch')

    @staticmethod
    def _cell(cell, inputs, hidden):
        if inputs.shape[-2] == 0:
            return hidden
        return cell(inputs.reshape(-1, inputs.shape[-1]), hidden.reshape(-1, hidden.shape[-1])).reshape_as(hidden)

    def step(self, o, state, episode_start, reset_memory=False):
        self._validate(o, state, episode_start)
        # ID validation/alignment happens even in ablations, so input contracts
        # and information availability remain identical across B0--B3.
        live = o['agent_alive']
        keep = live & ~episode_start[:,None]
        if reset_memory:
            keep = torch.zeros_like(keep)
        own_previous = state.self_hidden * keep[...,None]
        old_peer_ids = torch.where(keep[...,None], state.peer_ids, -1)
        old_obstacle_ids = torch.where(keep[...,None], state.obstacle_ids, -1)
        peer_valid, obstacle_valid = o['peer_valid'], o['obstacle_valid']
        peer_previous = align_hidden(state.peer_hidden, old_peer_ids, o['peer_ids'], peer_valid)
        obstacle_previous = align_hidden(state.obstacle_hidden, old_obstacle_ids, o['obstacle_ids'], obstacle_valid)
        peer_valid = peer_valid & live[...,None]
        obstacle_valid = obstacle_valid & live[...,None]
        own = self.self_encoder(torch.where(live[...,None],o['self_features'],0)) * live[...,None]
        peers = self.peer_encoder(torch.where(peer_valid[...,None],o['peer_features'],0)) * peer_valid[...,None]
        obstacles = self.obstacle_encoder(torch.where(obstacle_valid[...,None],o['obstacle_features'],0)) * obstacle_valid[...,None]
        if self.variant in ('B2','B3'):
            own = self._cell(self.self_gru, own, own_previous) * live[...,None]
            peers = self._cell(self.peer_gru, peers, peer_previous) * peer_valid[...,None]
            obstacles = self._cell(self.obstacle_gru, obstacles, obstacle_previous) * obstacle_valid[...,None]
        if self.variant == 'B0':
            peer_context = peers.sum(-2) / peer_valid.sum(-1,keepdim=True).clamp_min(1)
            obstacle_context = obstacles.sum(-2) / obstacle_valid.sum(-1,keepdim=True).clamp_min(1)
            pw = own.new_zeros((*live.shape, self.config['attention_heads'], peers.shape[-2]))
            ow = own.new_zeros((*live.shape, self.config['attention_heads'], obstacles.shape[-2]))
        else:
            peer_context, pw = self.peer_attention(own, peers, peer_valid)
            obstacle_context, ow = self.obstacle_attention(own, obstacles, obstacle_valid)
        external = peer_valid.any(-1) | obstacle_valid.any(-1)
        if self.variant in ('B2','B3'):
            u = self.self_projection(own)
            # Mask *after* the linear transform, including its bias.
            v = self.context_projection(torch.cat((peer_context,obstacle_context),-1)) * external[...,None]
            gate = torch.sigmoid(self.gate(torch.cat((u,v),-1))) if self.variant=='B3' else torch.full_like(u,.5)
            fused = self.fusion_norm(gate*u+(1-gate)*v)
        else:
            v = torch.cat((peer_context,obstacle_context),-1)
            gate = own.new_zeros((*live.shape,self.f))
            fused = self.fusion(torch.cat((own,v),-1))
        fusion_hidden = torch.zeros_like(state.fusion_hidden)
        if self.variant == 'B1':
            fusion_hidden = self._cell(self.post_gru, fused, state.fusion_hidden*keep[...,None]) * live[...,None]
            fused = fusion_hidden
        features = self.policy_hidden(fused)
        distribution = Normal(self.mean(features),self.log_std(features).clamp(self.config['log_std_min'],self.config['log_std_max']).exp())
        next_state = MemoryState(own*live[...,None],peers*peer_valid[...,None],obstacles*obstacle_valid[...,None],
            torch.where(peer_valid,o['peer_ids'],-1),torch.where(obstacle_valid,o['obstacle_ids'],-1),fusion_hidden)
        diagnostics = dict(peer_attention_weights=pw,obstacle_attention_weights=ow,
            gate=gate,external_context=v,self_hidden=next_state.self_hidden,
            peer_hidden=next_state.peer_hidden,obstacle_hidden=next_state.obstacle_hidden,
            peer_valid=peer_valid,obstacle_valid=obstacle_valid,log_std=distribution.scale.log())
        return distribution,next_state,diagnostics


class CentralizedNavigationCritic(nn.Module):
    """Only this value function can aggregate across observing agents."""
    def __init__(self, state_dim, hidden=64, heads=2):
        super().__init__()
        self.encoder = EntityEncoder(state_dim,hidden)
        self.attention = TemporalAttention(hidden,heads)
        self.value = nn.Sequential(nn.Linear(2*hidden,128),nn.ReLU(),nn.Linear(128,1))

    def forward(self, states, alive):
        embedding = self.encoder(torch.where(alive[...,None],states,0)) * alive[...,None]
        count = states.shape[-2]
        entities = embedding.unsqueeze(-3).expand(*embedding.shape[:-2],count,count,embedding.shape[-1])
        valid = alive.unsqueeze(-2).expand(*alive.shape[:-1],count,count)
        context,_ = self.attention(embedding,entities,valid)
        return self.value(torch.cat((embedding,context),-1)).squeeze(-1)*alive
