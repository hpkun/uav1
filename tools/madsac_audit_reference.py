"""Read-only numerical reference; never installed in a production trainer."""
from pathlib import Path
import json
import math
import hashlib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('MADSAC audit requires CUDA; CPU fallback is forbidden')
    return 'cuda'


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def offline_forward(critic, observations, actions, alive, scaled=False):
    """Reconstruct CURRENT exactly; SCALED changes only the offline logits."""
    embedding = critic.embedding(torch.cat((observations, actions), -1))
    batch, agents, _ = embedding.shape
    def heads(x):
        return x.view(batch, agents, critic.attention_heads, critic.head_dim).transpose(1, 2)
    query, key = heads(critic.wq(embedding)), heads(critic.wk(embedding))
    value = heads(critic.value_activation(critic.wv(embedding)))
    raw_logits = query @ key.transpose(-2, -1)
    logits = raw_logits / math.sqrt(critic.head_dim) if scaled else raw_logits
    valid = (alive > .5)[:, None, None, :].expand_as(logits)
    weights = torch.softmax(logits.masked_fill(~valid, -1e9), -1) * valid
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    weights = weights * (alive > .5)[:, None, :, None]
    context = (weights @ value).transpose(1, 2).contiguous().view(batch, agents, critic.hidden_dim)
    q = critic.q_network(context).squeeze(-1) * alive
    return q, weights, logits, embedding


def describe(values):
    x = np.asarray(values, dtype=float)
    if not x.size:
        return {'count': 0, 'mean': None}
    return dict(count=int(x.size), mean=float(x.mean()), std=float(x.std()),
                min=float(x.min()), p50=float(np.quantile(x, .5)),
                p90=float(np.quantile(x, .9)), p95=float(np.quantile(x, .95)),
                p99=float(np.quantile(x, .99)), max=float(x.max()))


def attention_rows(logits, weights, alive, offset=0):
    """One record per replay item/head/live query; invalid keys never enter logits."""
    logits, weights, alive = (x.detach().cpu().numpy() for x in (logits, weights, alive))
    rows = []
    for b in range(len(alive)):
        keys = np.flatnonzero(alive[b] > .5)
        for h in range(weights.shape[1]):
            for i in keys:
                z, p = logits[b, h, i, keys].astype(float), weights[b, h, i, keys].astype(float)
                positive = p > 0
                entropy = float(-(p[positive] * np.log(p[positive])).sum())
                ordered = np.sort(p)
                peers = p[keys != i]
                rows.append(dict(item=b+offset, head=h, query=int(i), alive_keys=len(keys),
                    logit_mean=float(z.mean()), logit_std=float(z.std()), logit_abs_max=float(np.abs(z).max()),
                    attention_entropy=entropy,
                    normalized_attention_entropy=entropy/math.log(len(keys)) if len(keys)>1 else None,
                    max_attention_weight=float(ordered[-1]),
                    second_max_attention_weight=float(ordered[-2]) if len(keys)>1 else None,
                    effective_number_of_attended_agents=math.exp(entropy),
                    self_attention_weight=float(weights[b,h,i,i]),
                    max_peer_attention_weight=float(peers.max()) if peers.size else None))
    return rows


def summarize_rows(rows):
    metrics = [k for k in rows[0] if k not in ('item','head','query')] if rows else []
    result = {k: describe([r[k] for r in rows if r[k] is not None]) for k in metrics}
    def fractions(selected):
        return {str(t): sum(r['max_attention_weight'] >= t for r in selected)/len(selected)
                if selected else None for t in (.5,.75,.9,.95,.99)}
    result['saturation_fraction_ge'] = fractions(rows)
    result['nontrivial_saturation_fraction_ge'] = fractions([r for r in rows if r['alive_keys']>1])
    result['live_query_head_count'] = len(rows)
    result['single_alive_query_head_count'] = sum(r['alive_keys']==1 for r in rows)
    return result


def attention_audit(critic, data, output, label, batch_size=256):
    modes = {'unscaled': [], 'scaled': []}
    max_error = 0.
    global_logits = {k: [] for k in modes}
    with torch.no_grad():
        for start in range(0, len(data[0]), batch_size):
            obs, act, alive = [torch.as_tensor(x[start:start+batch_size],device='cuda',dtype=torch.float32) for x in data]
            q0,w0 = critic(obs,act,alive,True)
            for name in modes:
                q,w,z,_ = offline_forward(critic,obs,act,alive,name=='scaled')
                if name=='unscaled':
                    max_error = max(max_error,float((q-q0).abs().max()),float((w-w0).abs().max()))
                modes[name].extend(attention_rows(z,w,alive,start))
                mask = (alive>.5)[:,None,:,None] & (alive>.5)[:,None,None,:]
                global_logits[name].extend(z[mask.expand_as(z)].cpu().tolist())
    assert max_error < 1e-6, max_error
    stats = {name:summarize_rows(rows) for name,rows in modes.items()}
    for name,rows in modes.items():
        stats[name]['global_valid_logit_std'] = float(np.std(global_logits[name]))
        stats[name]['per_head_query'] = {f'{h}:{i}':summarize_rows([r for r in rows if r['head']==h and r['query']==i])
            for h in range(critic.attention_heads) for i in range(data[0].shape[1])}
    std0, std1 = (stats[k]['global_valid_logit_std'] for k in modes)
    result = dict(label=label, inputs=len(data[0]),hidden_dim=critic.hidden_dim,heads=critic.attention_heads,
        head_dim=critic.head_dim,theoretical_factor=math.sqrt(critic.head_dim),
        measured_logit_std_ratio=std0/std1 if std1 else None,
        current_reference_max_abs_error=max_error,**stats)
    dump(Path(output)/f'{label}_rows.json',modes)
    return result


def gradient_audit(critic, data):
    obs, act, alive = [torch.as_tensor(x[:256],device='cuda',dtype=torch.float32) for x in data]
    result={}
    for name in ('unscaled','scaled'):
        q,_,_,emb=offline_forward(critic,obs,act,alive,name=='scaled')
        groups={k:list(getattr(critic,k).parameters()) for k in ('wq','wk','wv','embedding')}
        parameters=[p for group in groups.values() for p in group]
        grads=torch.autograd.grad(q.sum()/alive.sum().clamp_min(1),parameters+[emb])
        norms={}; offset=0
        for group,ps in groups.items():
            norms['grad_norm_'+group]=float(torch.sqrt(sum(g.square().sum() for g in grads[offset:offset+len(ps)])))
            offset+=len(ps)
        norms['grad_norm_embedding_output']=float(grads[-1].norm())
        result[name]=norms
    result['scaled_over_unscaled']={k:result['scaled'][k]/v if v else None for k,v in result['unscaled'].items()}
    result['scalar']='sum alive Q / sum alive mask; identical inputs/weights; autograd.grad; no optimizer'
    return result
