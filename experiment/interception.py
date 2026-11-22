"""Interception training and evaluation using the existing MAPPO/MADDPG learners."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from algorithm.mappo import MAPPO, RolloutBuffer
from algorithm.mappo_config import MAPPOConfig
from algorithm.maddpg import MADDPG
from algorithm.maddpg_config import MADDPGConfig
from algorithm.dual_critic import DualCritic, DualCriticConfig
from env.interception import InterceptionConfig, InterceptionEnv
from env.parallel import ParallelInterceptionEnvs
from .progress import ProgressLogger


def write_json(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2,default=_plain)+'\n',encoding='utf-8')


def _plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(env_path, algorithm_path=None):
    raw=yaml.safe_load(Path(env_path).read_text(encoding='utf-8'))
    if not isinstance(raw,dict) or set(raw)!={'schema','environment'} or raw['schema'] not in ('Interception/1','Interception/2'):
        raise ValueError('Expected supported interception environment configuration')
    config=InterceptionConfig(**raw['environment'])
    if InterceptionEnv(config).contract['schema']!=raw['schema']:
        raise ValueError('Configuration schema and observation layout disagree')
    if algorithm_path is None:return config,None,None
    learner=yaml.safe_load(Path(algorithm_path).read_text(encoding='utf-8'))
    if not isinstance(learner,dict) or set(learner)!={'algorithm','parameters','training'}:
        raise ValueError('Expected algorithm, parameters and training sections')
    name=learner['algorithm']
    if name not in ('mappo','maddpg','dual_critic'):
        raise ValueError('Unsupported interception algorithm')
    if name=='dual_critic' and InterceptionEnv(config).contract['version']!='interception_1':
        raise ValueError('Dual critic supports only Interception/1')
    parameters={'mappo':MAPPOConfig,'maddpg':MADDPGConfig,'dual_critic':DualCriticConfig}[name](**learner['parameters'])
    rollout=int(learner['training']['rollout_length'])
    if rollout<1:raise ValueError('rollout_length must be positive')
    return config,parameters,rollout


def make_policy(name,config,parameters,device,seed,environment_contract=None,environment_config=None):
    environment=InterceptionEnv(config)
    contract=environment_contract if environment_contract is not None else environment.contract
    snapshot=dict(environment=asdict(config) if environment_config is None else environment_config)
    if name=='mappo':
        return MAPPO(environment.observation_dim,environment.state_dim,5,parameters,device,seed,environment_config=snapshot,environment_contract=contract)
    if name=='maddpg':
        return MADDPG(parameters,device,seed,environment_config=snapshot,environment_contract=contract)
    if name=='dual_critic':
        if environment.contract['version']!='interception_1':
            raise ValueError('Dual critic supports only Interception/1')
        return DualCritic(parameters,device,seed,environment_config=snapshot,environment_contract=contract)
    raise ValueError(name)


def save_policy(policy,name,path,step,config):
    if name=='mappo':policy.save(path,global_step=step)
    else:policy.save(path,global_step=step,environment_config=dict(environment=asdict(config)))


def load_run(checkpoint,label='Best',device='cpu'):
    root=Path(checkpoint)
    if root.is_file():root=root.parent.parent if root.parent.name=='checkpoints' else root.parent
    settings=json.loads((root/'settings.json').read_text(encoding='utf-8'))
    if settings.get('attack_mode') not in ('direct_rush','split_attack'):
        raise ValueError('Checkpoint training attack mode is unsupported')
    saved_environment=settings['environment']
    restored_environment=dict(saved_environment)
    if 'observation_layout' not in restored_environment:
        restored_environment['observation_layout']='relative_with_base'
    if restored_environment['observation_layout']=='relative_assignment':
        restored_environment.setdefault('assignment_method','nearest_visible_legacy')
    restored_environment.setdefault('episode_protocol','first_breach')
    config=replace(InterceptionConfig(**restored_environment),
                   attack_mode=settings['attack_mode'])
    name=settings['algorithm']
    parameters={'mappo':MAPPOConfig,'maddpg':MADDPGConfig,'dual_critic':DualCriticConfig}[name](**settings['parameters'])
    model=(root/'checkpoints'/('best.pt' if label=='Best' else 'last.pt')) if Path(checkpoint).is_dir() else Path(checkpoint)
    index=json.loads((root/'checkpoints'/'index.json').read_text(encoding='utf-8'))
    expected=index['checkpoints'].get(label)
    if expected is None or expected['sha256']!=digest(model):raise ValueError('Checkpoint catalog digest mismatch')
    policy=make_policy(name,config,parameters,device,settings['seed'],
                       environment_contract=settings['environment_contract'],
                       environment_config=saved_environment)
    policy.load(model,load_optimizers=False)
    return config,policy,name,model,settings


def require_matching_attack_mode(settings,requested):
    trained=settings.get('attack_mode')
    if trained not in ('direct_rush','split_attack') or requested!=trained:
        raise ValueError(f'Checkpoint trained for {trained}; requested attack mode {requested}')


def evaluate_policy(config,policy,mode,episodes,seed_start):
    if mode!=config.attack_mode:
        raise ValueError('Evaluation mode differs from checkpoint training mode')
    rows=[]
    for offset in range(episodes):
        seed=seed_start+offset
        env=InterceptionEnv(replace(config,attack_mode=mode))
        obs,info=env.reset(seed=seed)
        reward_components={}
        while True:
            actions=policy.policy_actions(obs[None],info['active_masks'][None],deterministic=True)[0]
            obs,_,term,trunc,info=env.step(actions)
            for component,value in info['reward_components'].items():
                reward_components[component]=reward_components.get(component,0.0)+value
            if term or trunc:
                row=dict(info['episode_summary'])
                if row['seed']!=seed:
                    raise AssertionError('Evaluation episode seed mismatch')
                if not np.isclose(sum(reward_components.values()),row['team_return'],rtol=0,atol=1e-6):
                    raise AssertionError('Evaluation reward ledger mismatch')
                row['reward_components']=reward_components
                rows.append(row)
                break
        env.close()
    return rows


def summarize(rows):
    count=len(rows)
    if not count:raise ValueError('No evaluation episodes')
    outcomes={key:sum(row['outcome']==key for row in rows) for key in ('all_captured','penetrated','time_limit')}
    first=[row['first_capture_time_s'] for row in rows if row['first_capture_time_s'] is not None]
    component_names=sorted({name for row in rows for name in row.get('reward_components',{})})
    mean_reward_components={name:float(np.mean([row.get('reward_components',{}).get(name,0.0) for row in rows]))
                            for name in component_names}
    def interval(successes):
        z=1.96
        p=successes/count
        center=(p+z*z/(2*count))/(1+z*z/count)
        half=z*np.sqrt((p*(1-p)+z*z/(4*count))/count)/(1+z*z/count)
        return [float(max(0,center-half)),float(min(1,center+half))]
    return dict(episodes=count,counts=outcomes,all_capture_rate=outcomes['all_captured']/count,
                penetration_rate=outcomes['penetrated']/count,time_limit_rate=outcomes['time_limit']/count,
                all_capture_95pct_wilson=interval(outcomes['all_captured']),
                penetration_95pct_wilson=interval(outcomes['penetrated']),
                capture_efficiency=sum(row['captured_count'] for row in rows)/(5*count),
                mean_first_capture_time_s=float(np.mean(first)) if first else None,
                mean_duration_s=float(np.mean([row['duration_s'] for row in rows])),
                mean_defender_oob_count=float(np.mean([row['defender_oob_count'] for row in rows])),
                mean_penetrated_count=float(np.mean([row['penetrated_count'] for row in rows])),
                mean_reward_components=mean_reward_components,
                mean_team_return=float(np.mean([row['team_return'] for row in rows])))


class ReplayBuffer:
    """Joint transition replay for the existing MADDPG update loop."""
    def __init__(self,capacity,seed):
        self.capacity=capacity
        self.rows=[]
        self.cursor=0
        self.rng=np.random.default_rng(seed)

    def add(self,**row):
        copied={k:np.asarray(v,dtype=np.float32).copy() for k,v in row.items()}
        if len(self.rows)<self.capacity:self.rows.append(copied)
        else:self.rows[self.cursor]=copied
        self.cursor=(self.cursor+1)%self.capacity

    def __len__(self):return len(self.rows)

    def sample(self,size):
        chosen=self.rng.integers(0,len(self.rows),size=size)
        return {key:np.stack([self.rows[i][key] for i in chosen]) for key in self.rows[0]}


def evaluation_interval_for_budget(steps,num_envs):
    if steps<1 or num_envs<1 or steps%num_envs:
        raise ValueError('Invalid sampled-step budget')
    five_percent=(steps+19)//20
    return ((five_percent+num_envs-1)//num_envs)*num_envs


def train(config,parameters,rollout,name,*,steps,num_envs,seed,device,attack_mode,output,
          eval_interval=None,eval_episodes=10,log_every_steps=10000):
    if num_envs<1 or steps<1 or steps%num_envs or log_every_steps<1 or attack_mode not in ('direct_rush','split_attack'):
        raise ValueError('Invalid training budget or attack mode')
    if eval_interval is None:
        eval_interval=evaluation_interval_for_budget(steps,num_envs)
    if eval_interval<0 or eval_episodes<1:
        raise ValueError('Invalid evaluation schedule')
    config=replace(config,attack_mode=attack_mode)
    root=Path(output)
    environment=InterceptionEnv(config)
    if name=='dual_critic' and environment.contract['version']!='interception_1':
        raise ValueError('Dual critic supports only Interception/1')
    root.mkdir(parents=True,exist_ok=False)
    (root/'checkpoints').mkdir()
    settings=dict(schema='InterceptionRun/1',environment=asdict(config),environment_contract=InterceptionEnv(config).contract,
                  algorithm=name,parameters=asdict(parameters),rollout_length=rollout,seed=seed,
                  attack_mode=attack_mode,checkpoint_selection_mode=attack_mode,
                  total_sampled_steps=steps,num_envs=num_envs,
                  log_every_steps=log_every_steps,eval_interval=eval_interval,eval_episodes=eval_episodes,
                  learner_reward_interface=('team_scalar' if name=='mappo' else 'per_defender_vector' if name=='maddpg'
                                            else 'global_scalar_and_mean_local_scalar'))
    write_json(root/'settings.json',settings)
    policy=make_policy(name,config,parameters,device,seed)
    replay=ReplayBuffer(parameters.buffer_capacity,seed) if name!='mappo' else None
    episode_log=(root/'episodes.jsonl').open('x',encoding='utf-8')
    update_log=(root/'updates.jsonl').open('x',encoding='utf-8')
    evaluation_log=(root/'evaluations.jsonl').open('x',encoding='utf-8')
    progress_log=(root/'progress.jsonl').open('x',encoding='utf-8')
    progress=ProgressLogger(name,steps,log_every_steps,attack_mode=attack_mode)
    best_score=None
    best_entry=None
    next_eval=eval_interval if eval_interval else None
    sampled=0
    episodes=0
    reward_component_totals=[{} for _ in range(num_envs)]
    updates=0
    def record_evaluation(step):
        nonlocal best_score,best_entry
        summaries={attack_mode:summarize(evaluate_policy(config,policy,attack_mode,eval_episodes,600000))}
        evaluation_log.write(json.dumps(dict(step=step,modes=summaries),allow_nan=False)+'\n')
        evaluation_log.flush()
        print(ProgressLogger.evaluation(step,summaries),flush=True)
        own=summaries[attack_mode]
        score=(own['all_capture_rate'],own['capture_efficiency'],-own['penetration_rate'])
        if best_score is None or score>best_score:
            model=root/'checkpoints'/'best.pt'
            save_policy(policy,name,model,step,config)
            best_entry=dict(path='checkpoints/best.pt',sha256=digest(model),step=step,
                            selection='periodic_evaluation',selection_mode=attack_mode)
            best_score=score
    try:
        print(f'[train {name}] attack={attack_mode} workers={num_envs} device={device} '
              f'budget={steps:,} log_every={log_every_steps:,} eval_every={eval_interval:,} '
              f'eval_episodes={eval_episodes}',flush=True)
        with ParallelInterceptionEnvs(num_envs,config,seed=seed,attack_mode=attack_mode) as sampler:
            obs,states,masks,_=sampler.reset()
            while sampled<steps:
                horizon=min(rollout,(steps-sampled)//num_envs)
                buffer=(RolloutBuffer(horizon,num_envs,5,environment.observation_dim,environment.state_dim,2,action_transform=policy.action_transform,
                                      disk_radius=policy.disk_radius) if name=='mappo' else None)
                for _ in range(horizon):
                    if name=='mappo':
                        actions,logp,values,raw=policy.act_rollout(obs,states,masks)
                    else:
                        actions=policy.policy_actions(obs,masks,deterministic=False)
                    result=sampler.step(actions)
                    if name=='mappo':
                        buffer.insert(obs,states,actions,logp,values,result.rewards,result.dones,masks,raw_actions=raw)
                    else:
                        for i in range(num_envs):
                            item=dict(observations=obs[i],states=states[i],actions=actions[i],
                                      next_observations=result.transition_observations[i],next_states=result.transition_states[i],
                                      dones=result.dones[i],active_masks=masks[i],next_active_masks=result.transition_masks[i])
                            if name=='dual_critic':
                                item.update(global_rewards=result.infos[i]['global_reward'],
                                            local_rewards=float(np.mean(result.infos[i]['local_rewards'])))
                            else:item['rewards']=result.individual_rewards[i]
                            replay.add(**item)
                    for i,info in enumerate(result.infos):
                        totals=reward_component_totals[i]
                        for component,value in info['reward_components'].items():
                            totals[component]=totals.get(component,0.0)+value
                        if 'episode_summary' in info:
                            summary=info['episode_summary']
                            if not np.isclose(sum(totals.values()),summary['team_return'],rtol=0,atol=1e-6):
                                raise AssertionError('Episode reward ledger mismatch')
                            episodes+=1
                            episode_log.write(json.dumps(dict(worker=i,episode_id=episodes,**summary,
                                                             reward_components=totals),allow_nan=False)+'\n')
                            progress.episode(summary)
                            reward_component_totals[i]={}
                    obs,states,masks=result.observations,result.states,result.active_masks
                    sampled+=num_envs
                    if replay is not None and len(replay)>=parameters.warmup(config.episode_steps) and sampled%parameters.update_interval_steps==0:
                        for _update in range(parameters.updates_per_interval):
                            metrics=policy.update(replay.sample(parameters.batch_size));updates+=1
                            update_log.write(json.dumps(dict(step=sampled,update=updates,**metrics),allow_nan=False)+'\n')
                            progress.update(metrics)
                    if eval_interval and sampled>=next_eval and sampled<steps:
                        record_evaluation(sampled)
                        next_eval=(sampled//eval_interval+1)*eval_interval
                if buffer is not None:
                    buffer.compute_returns_and_advantages(policy.values(states),parameters.gamma,parameters.gae_lambda)
                    metrics=policy.update(buffer);updates+=1
                    update_log.write(json.dumps(dict(step=sampled,update=updates,**metrics),allow_nan=False)+'\n')
                    progress.update(metrics)
                if eval_interval and sampled==steps:
                    record_evaluation(sampled)
                if progress.ready(sampled,final=sampled==steps):
                    status=progress.snapshot(sampled,episodes,updates,
                                             replay_size=len(replay) if replay is not None else None,
                                             warmup=parameters.warmup(config.episode_steps) if replay is not None else None)
                    progress_log.write(json.dumps(status,allow_nan=False)+'\n')
                    progress_log.flush()
                    for line in ProgressLogger.lines(status):print(line,flush=True)
        last=root/'checkpoints'/'last.pt'
        save_policy(policy,name,last,sampled,config)
        if best_entry is None:
            best=root/'checkpoints'/'best.pt'
            save_policy(policy,name,best,sampled,config)
            best_entry=dict(path='checkpoints/best.pt',sha256=digest(best),step=sampled,
                            selection='final_unvalidated',selection_mode=attack_mode)
        write_json(root/'checkpoints'/'index.json',dict(schema='InterceptionCheckpoints/1',checkpoints=dict(
            Best=best_entry,Last=dict(path='checkpoints/last.pt',sha256=digest(last),step=sampled))))
        result=dict(sampled_steps=sampled,completed_episodes=episodes,
                    episodes_by_attack_mode={attack_mode:episodes},
                    updates=updates,best_score=best_score,best_selection=best_entry['selection'],
                    best_selection_mode=attack_mode)
        write_json(root/'summary.json',result)
        return result
    finally:
        episode_log.close();update_log.close();evaluation_log.close();progress_log.close()
