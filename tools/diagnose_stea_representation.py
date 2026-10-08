"""Read-only STEA representation diagnostics on existing v2.5 run snapshots."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from algorithm.common.evaluator import aggregate_combat_records, episode_return_metrics
from algorithm.common.protocol import config_sha256
from algorithm.stea_mappo.networks import decompose_observations
from algorithm.stea_mappo.trainer import RecurrentRolloutBatch
from env.factory import make_combat_environment
from tools.evaluate_policy_modes import load_policy

DIMENSIONS = ('heading', 'pitch', 'speed')
CONCENTRATION = ('entropy', 'normalized_entropy', 'top1', 'uniform_top1', 'top1_excess', 'effective_entity_count')
LOG_FIELDS = ('sampled_steps', 'policy_sigma_heading', 'policy_sigma_pitch', 'policy_sigma_speed',
    'policy_log_std_heading', 'policy_log_std_pitch', 'policy_log_std_speed', 'effective_ppo_epochs',
    'kl_early_stop', 'last_epoch_mean_kl', 'approx_kl', 'clip_fraction', 'entropy',
    'ally_attention_entropy', 'enemy_attention_entropy', 'ally_attention_top1_mean',
    'enemy_attention_top1_mean', 'gru_hidden_norm_mean', 'pre_update_ratio_max_abs_error')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def distribution_summary(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(x):
        return dict(count=0, mean=None, median=None, p10=None, p90=None, min=None, max=None)
    if not np.isfinite(x).all():
        raise FloatingPointError('nonfinite diagnostic statistic')
    return dict(count=len(x), mean=float(x.mean()), median=float(np.median(x)),
        p10=float(np.quantile(x,.1)), p90=float(np.quantile(x,.9)), min=float(x.min()), max=float(x.max()))


def concentration_rows(weights, valid):
    """Actual per-head probabilities; caller excludes dead focal agents."""
    k = valid.sum(-1)
    h = -(weights * weights.clamp_min(1e-12).log()).sum(-1)
    top = weights.max(-1).values
    uniform = 1 / k.clamp_min(1).to(weights.dtype)
    h_norm = h / k.clamp_min(2).to(weights.dtype).log()[..., None]
    return torch.stack((h, h_norm, top, uniform[...,None].expand_as(h),
                        top-uniform[...,None], h.exp()), -1), k


def summarize_concentration(rows, counts):
    rows = np.asarray(rows, dtype=np.float64)  # live decisions, heads, metrics
    counts = np.asarray(counts, dtype=int)
    eligible = counts >= 2
    result = dict(live_decisions=len(counts), k_counts={str(k):int((counts==k).sum()) for k in np.unique(counts)},
        k0_excluded=int((counts==0).sum()), k1_excluded=int((counts==1).sum()),
        eligible_decisions=int(eligible.sum()),
        weighting='one live agent-decision; metrics averaged over heads AFTER computing head entropy/top1',
        decision_mean={}, per_head=[], by_valid_entity_count={})
    for index, name in enumerate(CONCENTRATION):
        result['decision_mean'][name] = distribution_summary(rows[eligible,:,index].mean(1))
    for head in range(rows.shape[1]):
        result['per_head'].append({name:distribution_summary(rows[eligible,head,index])
                                  for index,name in enumerate(CONCENTRATION)})
    for k in sorted(set(counts[eligible])):
        result['by_valid_entity_count'][str(k)] = {name:distribution_summary(rows[counts==k,:,index].mean(1))
                                                  for index,name in enumerate(CONCENTRATION)}
    hist, edges = np.histogram(np.clip(rows[eligible,:,1],0,1).reshape(-1), bins=np.linspace(0,1,11))
    result['normalized_entropy_head_histogram'] = dict(edges=edges.tolist(), counts=hist.tolist())
    result['numerically_one_hot_head_fraction_k_ge_2'] = (
        float((rows[eligible,:,2] >= 1-1e-6).mean()) if eligible.any() else None)
    return result


def uniform_context(attention, embeddings, valid):
    """Average the actual projected V vectors, preserving multi-head layout."""
    values = attention.wv(embeddings)
    return (values * valid[...,None]).sum(-2) / valid.sum(-1).clamp_min(1)[...,None]


@torch.no_grad()
def one_step_counterfactuals(actor, obs, hidden, alive, start, learned_action):
    """Same visited state and incoming memory; inputs and weights are untouched."""
    _, allies, enemies = decompose_observations(obs)
    own, ally, enemy = actor.encode_entities(obs)
    a_valid = (allies[...,-1]>.5) & (alive[...,None]>.5)
    e_valid = (enemies[...,-1]>.5) & (alive[...,None]>.5)
    a_context = uniform_context(actor.ally_attention, ally, a_valid)
    e_context = uniform_context(actor.enemy_attention, enemy, e_valid)
    spatial = actor.spatial_fusion(torch.cat((own,a_context,e_context),-1))
    reset = start
    while reset.ndim < alive.ndim:
        reset = reset.unsqueeze(-1)
    effective_hidden = hidden * alive[...,None] * (1-reset.to(hidden.dtype))[...,None]
    _, uniform_hidden = actor.gru(spatial.reshape(-1,1,spatial.shape[-1]),
                                  effective_hidden.reshape(1,-1,actor.gru_hidden_dim))
    uniform_hidden = uniform_hidden.reshape_as(hidden)*alive[...,None]
    uniform_action = actor.mean(actor.actor_head(uniform_hidden)).tanh()*alive[...,None]
    zero_dist, _, _ = actor.distribution_step(obs,torch.zeros_like(hidden),alive,start)
    zero_action = zero_dist.mean.tanh()*alive[...,None]
    return (learned_action-uniform_action).abs(), (learned_action-zero_action).abs()


def memory_input(hidden, memoryless):
    return torch.zeros_like(hidden) if memoryless else hidden


def ranked_slot(features, valid, chosen, column):
    """1-based stable ranks; ATA/HA use absolute normalized feature magnitudes."""
    values = features[...,column].abs()
    selected = values.gather(-1,chosen[...,None])
    slots = torch.arange(values.shape[-1],device=values.device)
    return 1 + ((valid & ((values < selected) | ((values == selected) & (slots < chosen[...,None]))))).sum(-1)


def scalar_info(info):
    return {key:value.item() if isinstance(value,np.generic) else value for key,value in info.items()
            if isinstance(value,(str,int,float,bool,type(None),np.generic))}


@torch.no_grad()
def rollout(trainer, env_config, seeds, *, representation=True, memoryless=False, progress=None):
    actor, device = trainer.actor, trainer.device
    records, attention_rows, k_rows = [], {'ally':[], 'enemy':[]}, {'ally':[], 'enemy':[]}
    action_rows, hidden_rows, semantic_rows = [], [], []
    audit_bank = {key:[] for key in ('observations','actions','raw_actions','old_log_probs','rewards','dones',
        'alive_masks','next_observations','next_alive_masks','actor_hidden_states','episode_starts')}
    for episode, seed in enumerate(seeds):
        env = make_combat_environment(env_config); observations,_ = env.reset(int(seed))
        hidden = torch.zeros(env.team_size,actor.gru_hidden_dim,device=device)
        returns = np.zeros(env.team_size); digest = hashlib.sha256(); start = torch.tensor(1.,device=device)
        while True:
            obs = torch.as_tensor(observations,dtype=torch.float32,device=device)
            alive = torch.as_tensor(env.red_alive_mask,dtype=torch.float32,device=device)
            before = memory_input(hidden,memoryless)
            dist, next_hidden, attention = actor.distribution_step(obs,before,alive,start)
            actions = dist.mean.tanh()*alive[...,None]; live = alive>.5
            if representation:
                _, allies, enemies = decompose_observations(obs)
                for side,features in (('ally',allies),('enemy',enemies)):
                    valid = (features[...,-1]>.5) & live[...,None]
                    rows,k = concentration_rows(attention[f'{side}_attention_weights'],valid)
                    attention_rows[side].append(rows[live].cpu().numpy()); k_rows[side].append(k[live].cpu().numpy())
                    if (attention[f'{side}_attention_weights'] * (~valid)[...,None,:]).count_nonzero():
                        raise RuntimeError('dead entity received attention')
                    if attention[f'{side}_context'][valid.sum(-1)==0].count_nonzero():
                        raise RuntimeError('empty context is nonzero')
                uniform_diff, zero_diff = one_step_counterfactuals(actor,obs,before,alive,start,actions)
                action_rows.append(torch.stack((uniform_diff[live],zero_diff[live]),1).cpu().numpy())
                effective = before * alive[:,None]*(1-start)
                hidden_rows.append(torch.stack((effective.norm(dim=-1)[live],next_hidden.norm(dim=-1)[live]),1).cpu().numpy())
                valid = (enemies[...,-1]>.5) & live[:,None]
                chosen = attention['enemy_attention_weights'].mean(-2).argmax(-1)
                ranks = torch.stack([ranked_slot(enemies,valid,chosen,col) for col in (0,3,4)],-1)
                eligible = live & valid.any(-1)
                rank_np = ranks[eligible].cpu().numpy();chosen_np = chosen.cpu().numpy()
                qualified = [env._in_fire_window(env.red[i],env.blue[int(chosen_np[i])])
                             for i in np.flatnonzero(eligible.cpu().numpy())]
                semantic_rows.append(np.column_stack((rank_np,qualified,valid.sum(-1)[eligible].cpu().numpy())))
            actions_np = actions.cpu().numpy();digest.update(actions_np.tobytes())
            next_obs,rewards,terminated,truncated,info = env.step(actions_np);returns += rewards
            done = terminated or truncated
            # A small state bank verifies replay with REAL checkpoint actions,
            # including a boundary between episode 0 and 1, without an update.
            if episode < 2 and not memoryless:
                log_prob = actor._squashed_log_prob(dist,dist.mean,actions)*alive
                row = dict(observations=observations.copy(),actions=actions_np,raw_actions=(dist.mean*alive[:,None]).cpu().numpy(),
                    old_log_probs=log_prob.cpu().numpy(),rewards=rewards,dones=float(done),alive_masks=alive.cpu().numpy(),
                    next_observations=next_obs.copy(),next_alive_masks=env.red_alive_mask.copy(),
                    actor_hidden_states=before.cpu().numpy(),episode_starts=float(start))
                for key,value in row.items():audit_bank[key].append(value)
            hidden = next_hidden * torch.as_tensor(env.red_alive_mask,device=device)[:,None]
            start = torch.tensor(0.,device=device);observations = next_obs
            if done:
                hidden.zero_();team,agent = episode_return_metrics(returns)
                records.append(dict(episode_return=team,mean_agent_episode_return=agent,**scalar_info(info),
                                    environment_seed=int(seed),executed_action_sha256=digest.hexdigest()))
                break
        if progress and ((episode+1)%10==0 or episode+1==len(seeds)):
            progress(f'{episode+1}/{len(seeds)} episodes done')
    result = dict(metrics=aggregate_combat_records(records),episodes=records,
                  execution='zero_hidden_each_step' if memoryless else 'normal_deterministic')
    if audit_bank['observations']:
        bank = {key:np.asarray(value,dtype=np.float32)[:,None] for key,value in audit_bank.items()}
        result['sequence_replay_audit'] = trainer.audit_rollout_ratio(RecurrentRolloutBatch(**bank))
        result['replay_state_bank_transitions'] = len(bank['observations'])
    if representation:
        differences = np.concatenate(action_rows);hidden = np.concatenate(hidden_rows)
        semantics = np.concatenate(semantic_rows)
        result.update(attention={side:summarize_concentration(np.concatenate(attention_rows[side]),np.concatenate(k_rows[side]))
                                for side in attention_rows},
            uniform_one_step_action_difference={dim:distribution_summary(differences[:,0,i]) for i,dim in enumerate(DIMENSIONS)},
            zero_hidden_one_step_action_difference={dim:distribution_summary(differences[:,1,i]) for i,dim in enumerate(DIMENSIONS)},
            gru_incoming_hidden_norm=distribution_summary(hidden[:,0]),gru_next_hidden_norm=distribution_summary(hidden[:,1]),
            enemy_semantics=dict(valid_decisions=len(semantics),
                top_attention_is_nearest_enemy_rate=float((semantics[:,0]==1).mean()) if len(semantics) else None,
                distance_rank=distribution_summary(semantics[:,0]),absolute_ata_rank=distribution_summary(semantics[:,1]),
                absolute_ha_rank=distribution_summary(semantics[:,2]),
                top_attention_is_weapon_qualified_rate=float(semantics[:,3].mean()) if len(semantics) else None,
                k_ge_2_nearest_enemy_rate=float((semantics[semantics[:,4]>=2,0]==1).mean()) if (semantics[:,4]>=2).any() else None,
                uniform_nearest_baseline=float((1/semantics[:,4]).mean()) if len(semantics) else None,
                slot_rule='argmax of head-mean weights; exact ties resolved by slot index',
                qualification_function='current MultiUAVCombatEnv._in_fire_window (no ATA/HA approximation)'),
            state_bank_definition='all normal-rollout live decisions; counterfactuals computed online at identical obs/incoming hidden',
            concentration_mask_and_empty_context_checks='PASS')
    return result


def checkpoint_diagnostic(run_dir, checkpoint, seed_base=65000000, episodes=50, device='cuda', memory_ablation=False):
    run_dir,checkpoint = Path(run_dir).resolve(),Path(checkpoint).resolve()
    env_path,config_path = run_dir/'env_config.yaml',run_dir/'algorithm_config.yaml'
    env = yaml.safe_load(env_path.read_text());config = yaml.safe_load(config_path.read_text())
    before = {str(p):sha256(p) for p in (checkpoint,env_path,config_path)}
    trainer,state = load_policy('stea-mappo',checkpoint,env,config,device)
    extra = state['extra'];training_end = extra['training_seed']+extra['training_total_sampled_steps']+extra['training_num_envs']
    seeds = list(range(seed_base,seed_base+episodes))
    if episodes < 1 or any(extra['training_seed'] <= seed <= training_end for seed in seeds):
        raise ValueError('invalid/overlapping diagnostic seeds')
    def progress(message):print(f"STEA seed={extra['training_seed']} steps={state['sampled_steps']}: {message}",flush=True)
    weights_before = {k:v.detach().clone() for k,v in trainer.actor.state_dict().items()}
    with torch.backends.cudnn.flags(benchmark=False,deterministic=True):
        execute = rollout_batched if len(seeds)>1 else rollout
        normal = execute(trainer,env,seeds,progress=progress)
        result = dict(checkpoint=str(checkpoint),checkpoint_sha256=before[str(checkpoint)],
            training_seed=int(extra['training_seed']),sampled_steps=int(state['sampled_steps']),algorithm=state['algorithm'],
            environment_version=extra['environment_version'],implementation_version=state['stea_mappo_impl_version'],
            env_config_sha256=config_sha256(env),algorithm_config_sha256=config_sha256(config),
            input_file_sha256=before,diagnostic_seed_base=seeds[0],diagnostic_seed_end=seeds[-1],episodes=episodes,device=str(device),
            seed_role='development diagnostics, not final formal holdout',normal=normal,
            log_std_parameter=trainer.actor.log_std_parameter.detach().cpu().tolist(),
            clamped_log_std=trainer.actor.log_std_parameter.detach().clamp(trainer.actor.log_std_min,trainer.actor.log_std_max).cpu().tolist(),
            sigma=trainer.actor.log_std_parameter.detach().clamp(trainer.actor.log_std_min,trainer.actor.log_std_max).exp().cpu().tolist())
        if memory_ablation:
            zero = execute(trainer,env,seeds,representation=False,memoryless=True,progress=progress)
            result['zero_hidden_each_step'] = zero
            result['memory_ablation_gap'] = {key:normal['metrics'][key]-zero['metrics'][key] for key in normal['metrics'] if key!='evaluation_episodes'}
            result['paired_win_difference'] = distribution_summary([float(a['red_success'])-float(b['red_success'])
                for a,b in zip(normal['episodes'],zero['episodes'])])
    assert all(torch.equal(v,weights_before[k]) for k,v in trainer.actor.state_dict().items()),'actor mutated'
    assert all(sha256(p)==digest for p,digest in before.items()),'input file mutated'
    result['input_files_and_actor_unchanged'] = True
    return result


@torch.no_grad()
def rollout_batched(trainer, env_config, seeds, *, representation=True, memoryless=False, progress=None):
    """Independent fixed-seed episodes, batched actor inference; no autoresets."""
    actor,device=trainer.actor,trainer.device
    envs=[make_combat_environment(env_config) for _ in seeds]
    observations=np.stack([env.reset(int(seed))[0] for env,seed in zip(envs,seeds)])
    n=len(seeds);agents=envs[0].team_size
    hidden=torch.zeros(n,agents,actor.gru_hidden_dim,device=device)
    starts=torch.ones(n,device=device);done=np.zeros(n,dtype=bool)
    returns=np.zeros((n,agents));records=[None]*n;digests=[hashlib.sha256() for _ in seeds]
    a_rows,k_rows={side:[] for side in ('ally','enemy')},{side:[] for side in ('ally','enemy')}
    action_rows,hidden_rows,semantic_rows=[],[],[]
    bank={key:[] for key in RecurrentRolloutBatch.__dataclass_fields__}
    completed=0
    while not done.all():
        obs=torch.as_tensor(observations,dtype=torch.float32,device=device)
        alive_np=np.stack([np.zeros(agents,np.float32) if done[i] else env.red_alive_mask for i,env in enumerate(envs)])
        alive=torch.as_tensor(alive_np,device=device);live=alive>.5
        before=memory_input(hidden,memoryless)
        dist,next_hidden,attention=actor.distribution_step(obs,before,alive,starts)
        actions=dist.mean.tanh()*alive[...,None]
        if representation:
            _,allies,enemies=decompose_observations(obs)
            for side,features in (('ally',allies),('enemy',enemies)):
                valid=(features[...,-1]>.5)&live[...,None]
                rows,k=concentration_rows(attention[f'{side}_attention_weights'],valid)
                a_rows[side].append(rows[live].cpu().numpy());k_rows[side].append(k[live].cpu().numpy())
                if (attention[f'{side}_attention_weights']*(~valid)[...,None,:]).count_nonzero():
                    raise RuntimeError('dead entity received attention')
                if attention[f'{side}_context'][valid.sum(-1)==0].count_nonzero():
                    raise RuntimeError('empty context is nonzero')
            uniform_diff,zero_diff=one_step_counterfactuals(actor,obs,before,alive,starts,actions)
            action_rows.append(torch.stack((uniform_diff[live],zero_diff[live]),1).cpu().numpy())
            effective=before*alive[...,None]*(1-starts)[:,None,None]
            hidden_rows.append(torch.stack((effective.norm(dim=-1)[live],next_hidden.norm(dim=-1)[live]),1).cpu().numpy())
            valid=(enemies[...,-1]>.5)&live[...,None]
            chosen=attention['enemy_attention_weights'].mean(-2).argmax(-1)
            ranks=torch.stack([ranked_slot(enemies,valid,chosen,col) for col in (0,3,4)],-1)
            eligible=live&valid.any(-1);selected=chosen.cpu().numpy()
            qualified=[envs[e]._in_fire_window(envs[e].red[i],envs[e].blue[int(selected[e,i])])
                       for e,i in np.argwhere(eligible.cpu().numpy())]
            semantic_rows.append(np.column_stack((ranks[eligible].cpu().numpy(),qualified,valid.sum(-1)[eligible].cpu().numpy())))
        actions_np=actions.cpu().numpy();before_done=done.copy()
        next_obs=np.zeros_like(observations);rewards=np.zeros((n,agents),np.float32)
        next_alive=np.zeros((n,agents),np.float32)
        for index,env in enumerate(envs):
            if done[index]:continue
            digests[index].update(actions_np[index].tobytes())
            next_obs[index],rewards[index],terminated,truncated,info=env.step(actions_np[index])
            returns[index]+=rewards[index];done[index]=terminated or truncated
            next_alive[index]=env.red_alive_mask if not done[index] else 0.
            if done[index]:
                team,agent=episode_return_metrics(returns[index])
                records[index]=dict(episode_return=team,mean_agent_episode_return=agent,**scalar_info(info),
                    environment_seed=int(seeds[index]),executed_action_sha256=digests[index].hexdigest())
                completed+=1
                if progress and (completed%10==0 or completed==n):progress(f'{completed}/{n} episodes done')
        if not memoryless and not before_done[:2].all():
            # [T,2,N,...], death/end/padding masks retained; no new rollouts.
            lp=actor._squashed_log_prob(dist,dist.mean,actions)*alive
            row=dict(observations=observations[:2].copy(),actions=actions_np[:2],raw_actions=(dist.mean*alive[...,None])[:2].cpu().numpy(),
                old_log_probs=lp[:2].cpu().numpy(),rewards=rewards[:2],dones=done[:2].astype(np.float32),
                alive_masks=alive_np[:2],next_observations=next_obs[:2].copy(),next_alive_masks=next_alive[:2],
                actor_hidden_states=before[:2].cpu().numpy(),episode_starts=starts[:2].cpu().numpy())
            for key,value in row.items():bank[key].append(value)
        hidden=next_hidden*torch.as_tensor(next_alive,device=device)[...,None]
        starts=torch.zeros(n,device=device);observations=next_obs
    result=dict(metrics=aggregate_combat_records(records),episodes=records,
        execution='zero_hidden_each_step' if memoryless else 'normal_deterministic',
        inference_batch_size=n,episode_order='fixed ascending seeds; inactive episodes masked, never autoreset')
    if bank['observations']:
        result['sequence_replay_audit']=trainer.audit_rollout_ratio(RecurrentRolloutBatch(**{
            key:np.asarray(value,np.float32) for key,value in bank.items()}))
        result['replay_state_bank_transitions']=int(sum(np.asarray(bank['alive_masks']).any(-1).sum(-1)))
    if representation:
        differences=np.concatenate(action_rows);hidden=np.concatenate(hidden_rows);semantics=np.concatenate(semantic_rows)
        result.update(attention={side:summarize_concentration(np.concatenate(a_rows[side]),np.concatenate(k_rows[side])) for side in a_rows},
            uniform_one_step_action_difference={dim:distribution_summary(differences[:,0,i]) for i,dim in enumerate(DIMENSIONS)},
            zero_hidden_one_step_action_difference={dim:distribution_summary(differences[:,1,i]) for i,dim in enumerate(DIMENSIONS)},
            gru_incoming_hidden_norm=distribution_summary(hidden[:,0]),gru_next_hidden_norm=distribution_summary(hidden[:,1]),
            enemy_semantics=dict(valid_decisions=len(semantics),top_attention_is_nearest_enemy_rate=float((semantics[:,0]==1).mean()),
                distance_rank=distribution_summary(semantics[:,0]),absolute_ata_rank=distribution_summary(semantics[:,1]),
                absolute_ha_rank=distribution_summary(semantics[:,2]),top_attention_is_weapon_qualified_rate=float(semantics[:,3].mean()),
                k_ge_2_nearest_enemy_rate=float((semantics[semantics[:,4]>=2,0]==1).mean()) if (semantics[:,4]>=2).any() else None,
                uniform_nearest_baseline=float((1/semantics[:,4]).mean()),slot_rule='argmax of head-mean weights; stable slot ties',
                qualification_function='current MultiUAVCombatEnv._in_fire_window'),
            state_bank_definition='all normal-rollout live decisions; counterfactuals use identical obs/incoming hidden',
            concentration_mask_and_empty_context_checks='PASS')
    return result


@torch.no_grad()
def mappo_final_rollout(trainer,env,seeds):
    records=[]
    for seed in seeds:
        episode=make_combat_environment(env);obs,_=episode.reset(seed);returns=np.zeros(episode.team_size);digest=hashlib.sha256()
        while True:
            actions=trainer.act(obs,episode.red_alive_mask,deterministic=True);digest.update(actions.tobytes())
            obs,rewards,terminated,truncated,info=episode.step(actions);returns+=rewards
            if terminated or truncated:
                team,agent=episode_return_metrics(returns)
                records.append(dict(episode_return=team,mean_agent_episode_return=agent,**scalar_info(info),
                    environment_seed=seed,executed_action_sha256=digest.hexdigest()));break
    return dict(metrics=aggregate_combat_records(records),episodes=records)


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def numeric_csv(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key,value in row.items():
            try:row[key]=float(value)
            except (TypeError,ValueError):pass
    return rows


def stability_summary(rows):
    result = {key:distribution_summary([row[key] for row in rows if row.get(key) is not None])
              for key in LOG_FIELDS if key!='sampled_steps'}
    result['updates'] = len(rows)
    result['kl_early_stop_count'] = sum(bool(row.get('kl_early_stop')) for row in rows)
    result['kl_early_stop_fraction'] = result['kl_early_stop_count']/len(rows) if rows else None
    return result


def weighted_training_summary(rows):
    keys = ('win_rate','average_return','average_red_loss','average_blue_loss','average_episode_length')
    weights = np.asarray([row['evaluation_episodes'] for row in rows])
    return dict(completed_episodes=int(weights.sum()),**{key:float(np.average([row[key] for row in rows],weights=weights))
        for key in keys}) if len(rows) else dict(completed_episodes=0)


def analyze_logs(run_dir, training_seed):
    run_dir = Path(run_dir)
    updates = list(read_jsonl(run_dir/'optimization_metrics.jsonl'))
    evaluations = numeric_csv(run_dir/'evaluation_history.csv')
    completed, recorded_updates = [], []
    for row in read_jsonl(run_dir/'training_metrics.jsonl'):
        if row.get('record_type')=='ppo_update':recorded_updates.append(row)
        elif row.get('evaluation_episodes',0)>0:completed.append(row)
    if len(recorded_updates)!=len(updates):raise RuntimeError('training/optimization update count differs')
    for left,right in zip(recorded_updates,updates):
        for key in LOG_FIELDS:
            if key in right and left.get(key)!=right[key]:raise RuntimeError(f'duplicate log differs: {key}')
    intervals = [(450000,600000),(800000,1000000),(1450000,1700000),(1800000,2000000)] if training_seed==1 else [
        (550000,750000),(800000,1000000),(1300000,1550000),(1750000,2000000)]
    windows = []
    for low,high in intervals:
        inside=lambda row:low<=row['sampled_steps']<=high
        local=[r for r in updates if inside(r)]
        windows.append(dict(step_start=low,step_end=high,optimization=stability_summary(local),
            stochastic_training=weighted_training_summary([r for r in completed if inside(r)]),
            periodic_evaluations=[r for r in evaluations if inside(r)]))
    # Align each 20-episode eval with the exact same-step optimizer update.
    aligned=[];by_step={row['sampled_steps']:row for row in updates}
    for row in evaluations:
        aligned.append({**row,**{key:by_step.get(row['sampled_steps'],{}).get(key) for key in LOG_FIELDS if key!='sampled_steps'}})
    return dict(stability=stability_summary(updates),windows=windows,
        training_completed_episode_count=sum(r['evaluation_episodes'] for r in completed),
        training_optimization_logs_consistent=True,training_metric_rows_with_completions=len(completed),
        optimization_timeseries=[{key:row.get(key) for key in LOG_FIELDS} for row in updates],
        periodic_evaluation_timeseries=aligned,
        training_completed_timeseries=[{key:row.get(key) for key in ('sampled_steps','evaluation_episodes','win_rate',
            'average_return','average_red_loss','average_blue_loss','average_episode_length')} for row in completed])


def save_json(path, result):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise FileExistsError(f'nonidentical output exists: {path}')
        return
    with path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)


def suite(output, seed_base, episodes, device):
    """Only existing runs; identify seeds from checkpoint metadata, never names."""
    output=Path(output).resolve()
    runs=[p for p in (ROOT/'outputs').glob('*5v5*mlpcritic') if p.is_dir() and (p/'checkpoint_2000000.pt').exists()]
    if not runs:raise FileNotFoundError('no completed MLP-critic runs found')
    protected={str(p):sha256(p) for run in runs for p in run.iterdir() if p.is_file()}
    protected.update({str(p):sha256(p) for prefix in ('algorithm','env','configs') for p in (ROOT/prefix).rglob('*')
                      if p.is_file() and p.suffix in ('.py','.yaml')})
    save_json(output/'input_hashes_before.json',protected)
    inventory=[];sigma_rows=[]
    for run in sorted(runs):
        state=torch.load(run/'checkpoint_2000000.pt',map_location='cpu',weights_only=False)
        algorithm=state['algorithm'];seed=int(state['extra']['training_seed'])
        entry=dict(run_dir=str(run),algorithm=algorithm,training_seed=seed,final_steps=state['sampled_steps'],
                   directory_seed_is_not_identity=True)
        inventory.append(entry)
        save_json(output/f'{algorithm.lower()}_seed{seed}_logs.json',analyze_logs(run,seed))
        cfg=yaml.safe_load((run/'algorithm_config.yaml').read_text());env=yaml.safe_load((run/'env_config.yaml').read_text())
        checkpoints=list(run.glob('checkpoint_*.pt'))
        steps={int(p.stem.split('_')[1]):p for p in checkpoints}
        for target in (500000,1000000,1500000,2000000):
            actual=min(steps,key=lambda s:abs(s-target));checkpoint=steps[actual]
            tr,st=load_policy('stea-mappo' if algorithm=='STEA-MAPPO' else 'mappo',checkpoint,env,cfg,device)
            parameter=tr.actor.log_std_parameter.detach();clamped=parameter.clamp(tr.actor.log_std_min,tr.actor.log_std_max)
            row=dict(algorithm=algorithm,training_seed=seed,sampled_steps=int(st['sampled_steps']),
                     checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                     log_std_parameter=parameter.cpu().tolist(),clamped_log_std=clamped.cpu().tolist(),sigma=clamped.exp().cpu().tolist())
            histories=st['extra'].get('evaluation_history',[])
            nearest=min(histories,key=lambda r:abs(r['sampled_steps']-st['sampled_steps'])) if histories else None
            row['nearest_periodic_evaluation']=nearest;sigma_rows.append(row)
            if algorithm=='STEA-MAPPO':
                path=output/f'seed{seed}_{actual:07d}.json'
                if path.exists():
                    prior=json.loads(path.read_text())
                    if (prior['checkpoint_sha256'],prior['diagnostic_seed_base'],prior['episodes']) != (row['checkpoint_sha256'],seed_base,episodes):
                        raise RuntimeError('cached diagnostic protocol mismatch')
                    continue
                result=checkpoint_diagnostic(run,checkpoint,seed_base,episodes,device,memory_ablation=target==2000000)
                save_json(path,result)
                print(json.dumps(dict(output=str(path),win=result['normal']['metrics']['win_rate'],
                                      return_=result['normal']['metrics']['average_return'])),flush=True)
        if algorithm=='MAPPO':
            # Same development seeds for final MAPPO; no stochastic rerun needed.
            tr,_=load_policy('mappo',run/'checkpoint_2000000.pt',env,cfg,device)
            with torch.backends.cudnn.flags(benchmark=False,deterministic=True):
                final=mappo_final_rollout(tr,env,list(range(seed_base,seed_base+episodes)))
            save_json(output/f'mappo_seed{seed}_final.json',dict(**entry,diagnostic_seed_base=seed_base,
                      diagnostic_seed_end=seed_base+episodes-1,diagnostic_episodes=episodes,**final))
    save_json(output/'inventory.json',dict(runs=inventory,missing_mappo_seeds=sorted({1,3}-{r['training_seed'] for r in inventory if r['algorithm']=='MAPPO'})))
    save_json(output/'sigma_checkpoints.json',sigma_rows)
    assert all(sha256(p)==digest for p,digest in protected.items()),'protected files changed'
    save_json(output/'input_hashes_after.json',protected)
    print('All existing source/config/run files remained byte-identical.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',action='store_true')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--seed-base',type=int,default=65000000)
    parser.add_argument('--episodes',type=int,default=50)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--memory-ablation',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from algorithm.stea_mappo.protocol import require_cuda
    require_cuda(args.device);torch.set_num_threads(1)
    # The suite output must be separate from all original runs.
    for p in (ROOT/'outputs').glob('*5v5*'):
        if args.output.resolve().is_relative_to(p.resolve()):raise ValueError('output must be outside original run directories')
    if args.suite:suite(args.output,args.seed_base,args.episodes,args.device)
    else:
        if not args.run_dir or not args.checkpoint:parser.error('--run-dir and --checkpoint required without --suite')
        if args.output.resolve().is_relative_to(args.run_dir.resolve()):raise ValueError('output must be outside original run')
        save_json(args.output,checkpoint_diagnostic(args.run_dir,args.checkpoint,args.seed_base,args.episodes,args.device,args.memory_ablation))


if __name__=='__main__':main()
