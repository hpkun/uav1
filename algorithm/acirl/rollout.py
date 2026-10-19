"""Aligned joint transitions and explicit macro events; no replay of PPO data."""
from dataclasses import dataclass
import torch
from .state import initial_state,decide,finish_transition
from .advantages import team_gae,macro_advantages


@dataclass
class ACIRLRollout:
    data:dict
    events:list
    episode_records:list
    behavior_version:int
    intention_events:list


class RolloutCollector:
    def __init__(self,trainer,vector):
        self.trainer=trainer;self.vector=vector
        self.obs=vector.reset();self.alive=vector.current_alive_masks.copy()
        self.state=initial_state(trainer.policy,vector.num_envs,trainer.device)
        shape=(vector.num_envs,trainer.config['network']['num_agents'])
        self.agent_returns=torch.zeros(shape,device=trainer.device)
        self.train_returns=torch.zeros(vector.num_envs,device=trainer.device)
        self.cost_returns=self.train_returns.clone();self.episode_ids=torch.zeros(vector.num_envs,dtype=torch.long,device=trainer.device)
        self.episode_renew=torch.zeros(shape,device=trainer.device)

    @torch.no_grad()
    def collect(self,steps):
        t=self.trainer;config=t.config;tr=config['training'];storage={};records=[];lifecycle=[]
        for _ in range(steps):
            obs=t.tensor(self.obs);alive=t.tensor(self.alive)
            row,new_state=decide(t.policy,obs,alive,self.state,t.behavior_rng)
            value,q=t.critics(obs,alive,row['hidden'],row['old_intent'],row['age_before'])
            actions=row['actions'].cpu().numpy()
            result=self.vector.step_batch(actions)
            executed=t.tensor(__import__('numpy').stack([info['executed_red_actions'] for info in result.infos]))
            if not torch.equal(executed,row['actions']):raise RuntimeError('ACIRL executed action differs from sampled bounded action')
            reward=t.tensor(result.rewards);done=t.tensor(result.terminated|result.truncated)
            next_alive=t.tensor(result.next_alive_masks);next_obs=t.tensor(result.transition_next_observations)
            next_hidden,*_=t.policy.memory(next_obs,new_state.hidden,new_state.previous_action,next_alive,torch.zeros_like(alive))
            next_value,_=t.critics(next_obs,next_alive,next_hidden,new_state.intent,new_state.age)
            original=reward.mean(-1) if tr['reward_reduction']=='mean' else reward.sum(-1)
            learned=row['gate']*row['gate_valid']
            cost=tr['switching_cost']*learned.mean(-1)
            train_reward=original-cost
            # Natural lifecycle events are distinct from rollout-censored PPO samples.
            lifecycle_data={key:row[key].cpu().numpy() for key in ('gate','old_intent','age_before','intent','age','reason')}
            episode_ids=self.episode_ids.cpu().tolist()
            for e in range(len(obs)):
                for i in range(alive.shape[-1]):
                    if lifecycle_data['gate'][e,i] and lifecycle_data['old_intent'][e,i]>=0:
                        duration=int(lifecycle_data['age_before'][e,i])
                        lifecycle.append(dict(environment=e,agent=i,episode_id=episode_ids[e],
                            intention=int(lifecycle_data['old_intent'][e,i]),start_vector_step=t.vector_steps-duration,
                            end_vector_step=t.vector_steps,duration=duration,end_reason=int(lifecycle_data['reason'][e,i]),trainable=False))
                    episode_done=bool(result.terminated[e] or result.truncated[e])
                    if self.alive[e,i] and (episode_done or not result.next_alive_masks[e,i]) and lifecycle_data['intent'][e,i]>=0:
                        duration=int(lifecycle_data['age'][e,i])
                        lifecycle.append(dict(environment=e,agent=i,episode_id=episode_ids[e],
                            intention=int(lifecycle_data['intent'][e,i]),start_vector_step=t.vector_steps+1-duration,
                            end_vector_step=t.vector_steps+1,duration=duration,end_reason='episode_end' if episode_done else 'agent_death',trainable=False))
            row.update(rewards=reward,raw_team_reward=original,training_reward=train_reward,switching_cost=cost,
                next_observations=next_obs,next_alive=next_alive,done=done,
                terminated=t.tensor(result.terminated),truncated=t.tensor(result.truncated),
                value=value,next_value=next_value,episode_id=self.episode_ids.clone(),
                sampled_step=torch.full((len(obs),),t.sampled_steps,device=t.device,dtype=torch.long))
            for key,v in row.items():storage.setdefault(key,[]).append(v.detach().clone())
            self.agent_returns+=reward;self.train_returns+=train_reward;self.cost_returns+=cost;self.episode_renew+=learned
            for env_index in torch.nonzero(done).flatten().tolist():
                returns=self.agent_returns[env_index]
                records.append(dict(episode_return=float(returns.sum()),mean_agent_episode_return=float(returns.mean()),
                    original_team_training_scale_return=float(returns.mean() if tr['reward_reduction']=='mean' else returns.sum()),
                    internal_training_return=float(self.train_returns[env_index]),switching_cost_total=float(self.cost_returns[env_index]),
                    learned_renew_per_agent=self.episode_renew[env_index].cpu().tolist(),**result.infos[env_index]))
                self.agent_returns[env_index].zero_();self.train_returns[env_index]=0;self.cost_returns[env_index]=0
                self.episode_renew[env_index].zero_();self.episode_ids[env_index]+=1
            self.state=finish_transition(new_state,next_alive,done)
            self.obs=result.observations;self.alive=self.vector.current_alive_masks.copy()
            t.sampled_steps+=self.vector.num_envs;t.vector_steps+=1
        data={key:torch.stack(rows) for key,rows in storage.items()}
        boundary=torch.zeros_like(data['done'])
        data['rollout_start']=boundary.clone();data['rollout_start'][0]=1
        data['rollout_end']=boundary.clone();data['rollout_end'][-1]=1
        low,returns=team_gae(data['training_reward'],data['value'],data['next_value'],data['done'],tr['gamma'],tr['gae_lambda'])
        high,events=macro_advantages(data['training_reward'],data['value'],data['next_value'],data['done'],
            data['alive'],data['next_alive'],data['intent_valid'],tr['gamma'],tr['lambda_high'])
        if config['ablation']['mode']=='no_temporal_critic':high=low.unsqueeze(-1).expand_as(high)*data['intent_valid']
        data.update(low_advantage=low.unsqueeze(-1).expand_as(data['alive']).clone(),high_advantage=high,returns=returns)
        for event in events:
            start=event['start'];e=event['environment'];i=event['agent']
            event.update(episode_id=int(data['episode_id'][start,e]),intention=int(data['intent'][start,e,i]),
                selection_probability=float(data['intention_probability'][start,e,i]),
                start_sampled_step=int(data['sampled_step'][start,e])+e,
                rollout_update=t.update_count,decision_reason=int(data['reason'][start,e,i]),
                trainable=True,fragment_semantics='same_behavior_rollout_only')
            event['advantage']=float(high[start,e,i])
            event['advantage_method']='step_gae_ablation' if config['ablation']['mode']=='no_temporal_critic' else 'smdp'
        return ACIRLRollout(data,events,records,t.update_count,lifecycle)

    def runtime_state(self):
        return dict(state={k:v.detach().cpu() for k,v in vars(self.state).items()},episode_ids=self.episode_ids.cpu(),
            agent_returns=self.agent_returns.cpu(),training_returns=self.train_returns.cpu(),switching_cost=self.cost_returns.cpu(),
            environment_state_included=False,continuation_semantics='warm_restart_new_episodes_only')
