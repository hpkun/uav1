"""Off-policy MAAC replay cycles, compact logs and deterministic evaluation."""
import json
from pathlib import Path
import numpy as np
import torch
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.evaluator import evaluate,episode_return_metrics,aggregate_combat_records
from algorithm.common.controller_diagnostics import controller_metric_values
from .trainer import MAACTrainer
from .replay import JointReplayBuffer
from .metrics import DWELL_VERSIONS,evaluation_metrics,empty_dwell_metrics
from env.v350_metrics import reward_outcome_identity


def steps_text(value):
    return f'{value/1e6:.1f}M' if value>=1000000 else f'{value/1000:g}k' if value>=1000 else str(value)


class MAACTrainingRunner:
    def __init__(self,env_config,config,output_dir,num_envs=None,total_sampled_steps=None,
                 seed=None,device='cuda',smoke=False):
        if device!='cuda' or not torch.cuda.is_available():raise RuntimeError('MAAC training requires CUDA; no CPU fallback')
        self.env_config,self.config,self.output_dir=env_config,config,Path(output_dir)
        t=config['training'];self.seed=t['seed'] if seed is None else seed
        self.num_envs=t['num_train_envs'] if num_envs is None else num_envs
        self.total_steps=(32 if smoke else t['total_sampled_steps']) if total_sampled_steps is None else total_sampled_steps
        if self.num_envs<=0 or self.total_steps<=0:raise ValueError('envs/steps must be positive')
        if self.total_steps%self.num_envs:raise ValueError('total steps must be divisible by num_envs')
        self.batch_size=min(16,t['minibatch_size']) if smoke else t['minibatch_size']
        self.learning_starts=self.batch_size if smoke else max(t['learning_starts'],self.batch_size)
        self.update_interval=4 if smoke else t['sampled_steps_per_update']
        base=config['implementation']['evaluation_seed_base']
        if self.seed+self.total_steps+self.num_envs>=base:raise ValueError('training/evaluation seed ranges may overlap')
        self.eval_seeds=list(range(base,base+(2 if smoke else t['evaluation_episodes'])))
        self.trainer=MAACTrainer(env_config,config,device,self.seed,smoke)
        self.replay=JointReplayBuffer(min(128,t['replay_capacity']) if smoke else t['replay_capacity'],self.seed,
            observation_dim=self.trainer.observation_dim,num_agents=self.trainer.num_agents,action_dim=self.trainer.action_dim)
        self.vector=ParallelVectorEnv(self.num_envs,env_config,self.seed,self.eval_seeds)
        self.records=[];self.agent_returns=np.zeros((self.num_envs,self.trainer.num_agents))
        self.last_metrics={};self.evaluation_history=[];self.best_key=None;self.update_cycles=0
        try:self.observations=self.vector.reset()
        except BaseException:self.vector.close();raise

    def collect_step(self):
        obs=self.observations.copy();alive=self.vector.current_alive_masks.copy()
        actions=self.trainer.act(obs,alive,deterministic=False)
        result=self.vector.step_batch(actions)
        self.replay.add_batch(obs,actions,result.rewards,result.transition_next_observations,
            result.terminated|result.truncated,result.alive_masks,result.next_alive_masks)
        self.agent_returns+=result.rewards
        for i,done in enumerate(result.terminated|result.truncated):
            if done:
                team,agent=episode_return_metrics(self.agent_returns[i])
                self.records.append(dict(episode_return=team,mean_agent_episode_return=agent,**result.infos[i]))
                self.agent_returns[i].fill(0)
        self.observations=result.observations
        self.trainer.sampled_steps+=self.num_envs
        return result.infos

    def train_cycle(self):
        t=self.config['training'];rows=[]
        for _ in range(t['critic_updates_per_cycle']):
            rows.append(self.trainer.update_critic(self.replay.sample(self.batch_size)))
        for _ in range(t['policy_updates_per_cycle']):
            rows.append(self.trainer.update_policies(self.replay.sample(self.batch_size)))
        self.trainer.update_targets();self.update_cycles+=1
        keys=set().union(*(r.keys() for r in rows))
        self.last_metrics={k:float(np.mean([r[k] for r in rows if k in r])) for k in keys}

    def recent(self):
        rows=self.records[-self.config['training']['recent_episode_window']:]
        if self.env_config['environment_version'] in ('3.48','3.49','3.50'):
            from env.v348_metrics import mechanism_metrics
            if not rows:
                return dict(mean_return=None, win_rate=None, fire_rate=None, kill_rate=None,
                            **reward_outcome_identity(self.env_config['environment_version']),
                            **mechanism_metrics(self.env_config['environment_version']))
            result=evaluation_metrics(aggregate_combat_records(rows),self.env_config)
            result['mean_return']=result['average_return']
            return result
        if self.env_config['environment_version'] in DWELL_VERSIONS:
            if not rows:return empty_dwell_metrics()
            result=evaluation_metrics(aggregate_combat_records(rows),self.env_config)
            result['mean_return']=result['average_return']
            return result
        if not rows:return dict(mean_return=None,win_rate=None,fire_rate=None,kill_rate=None)
        return dict(mean_return=float(np.mean([r['episode_return'] for r in rows])),
            win_rate=float(np.mean([r['red_success'] for r in rows])),
            fire_rate=float(np.mean([r['red_first_attempt_step'] is not None for r in rows])),
            kill_rate=float(np.mean([r['red_first_kill_step'] is not None for r in rows])))

    def write(self,name,row):
        with (self.output_dir/name).open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

    def write_metrics(self,infos):
        row=dict(record_type='maac_replay_cycle',sampled_steps=self.trainer.sampled_steps,
            episodes=len(self.records),replay_size=len(self.replay),update_cycles=self.update_cycles,
            **self.recent(),**self.last_metrics,
            **{k:float(v.mean()) for k,v in controller_metric_values(infos).items()})
        for side in ('red','blue'):
            for field in ('step_fire_attempts','step_weapon_hits','step_attack_kills'):
                key=f'{side}_{field}';row[key]=sum(i[key] for i in infos)
            if self.env_config['environment_version'] in DWELL_VERSIONS:
                row[f'{side}_step_dwell_completed']=sum(int(np.sum(i[f'{side}_dwell_completed_this_step'])) for i in infos)
        if self.env_config['environment_version'] in DWELL_VERSIONS:
            row['episode_metrics_scope']='equal-weight completed episodes in recent_episode_window'
            row['step_metrics_scope']='all transitions since previous metrics write'
        self.write('training_metrics.jsonl',row)
        self.write('optimization_metrics.jsonl',dict(sampled_steps=self.trainer.sampled_steps,**self.last_metrics))

    def evaluate(self):
        row=dict(sampled_steps=self.trainer.sampled_steps,evaluation_mode='deterministic',
            evaluation_seeds=self.eval_seeds,**evaluate(self.trainer,self.env_config,self.eval_seeds))
        row=evaluation_metrics(row,self.env_config)
        self.evaluation_history.append(row);self.write('evaluation_metrics.jsonl',row)
        key=(row['win_rate'],row['average_return'],-row['average_red_loss'])
        if self.best_key is None or key>self.best_key:
            self.best_key=key;self.trainer.save(self.output_dir/'best_eval.pt')
        event=(f"dwell_start={row['dwell_start_episode_rate']:.1%} | dwell_complete={row['dwell_complete_episode_rate']:.1%} | mutual_draw={row['mutual_annihilation_draw_rate']:.1%}"
               if self.env_config['environment_version'] in DWELL_VERSIONS else f"fire={row['fire_rate']:.1%}")
        print(f"[EVAL] {steps_text(row['sampled_steps'])} | deterministic | win={row['win_rate']:.1%} | {event} | kill={row['kill_rate']:.1%} | R={row['average_return']:.2f} | len={row['episode_length']:.0f}",flush=True)
        return row

    def run(self):
        t=self.config['training'];next_log=t['console_interval_sampled_steps']
        next_eval=t['evaluation_interval_sampled_steps'];next_save=t['checkpoint_interval_sampled_steps']
        print(f"[START] MAAC | env=v{self.env_config['environment_version']} | {self.trainer.num_agents}v{self.trainer.num_agents} | seed={self.seed} | cuda | envs={self.num_envs} | steps={steps_text(self.total_steps)}",flush=True)
        if self.env_config['environment_version'] in DWELL_VERSIONS:
            print('[EVENT] continuous_dwell_neutralization | fire_rate=N/A | dwell_complete counts attackers; neutralizations count unique victims',flush=True)
        pending=0;infos=[]
        try:
            while self.trainer.sampled_steps<self.total_steps:
                infos.extend(self.collect_step());pending+=self.num_envs
                if pending>=self.update_interval and len(self.replay)>=self.learning_starts:
                    self.train_cycle();self.write_metrics(infos);infos=[];pending=0
                if self.trainer.sampled_steps>=next_log:
                    self.log_train(self.recent())
                    next_log=(self.trainer.sampled_steps//t['console_interval_sampled_steps']+1)*t['console_interval_sampled_steps']
                if self.trainer.sampled_steps>=next_eval:
                    self.evaluate();next_eval=(self.trainer.sampled_steps//t['evaluation_interval_sampled_steps']+1)*t['evaluation_interval_sampled_steps']
                if self.trainer.sampled_steps>=next_save:
                    self.trainer.save(self.output_dir/f'checkpoint_{self.trainer.sampled_steps}.pt')
                    next_save=(self.trainer.sampled_steps//t['checkpoint_interval_sampled_steps']+1)*t['checkpoint_interval_sampled_steps']
            if infos:self.write_metrics(infos)
            self.log_train(self.recent())
            if not self.evaluation_history or self.evaluation_history[-1]['sampled_steps']!=self.trainer.sampled_steps:self.evaluate()
            for name in ('latest.pt','final.pt',f'checkpoint_{self.trainer.sampled_steps}.pt'):self.trainer.save(self.output_dir/name)
            return dict(**reward_outcome_identity(self.env_config['environment_version']),
                algorithm='MAAC',variant=self.config['implementation']['variant'],environment_version=self.env_config['environment_version'],
                sampled_steps=self.trainer.sampled_steps,completed_episodes=len(self.records),replay_size=len(self.replay),
                actor_updates=self.trainer.actor_updates,policy_updates=self.trainer.policy_updates,
                critic_updates=self.trainer.critic_updates,target_updates=self.trainer.target_updates,update_cycles=self.update_cycles,
                optimization_metrics=self.last_metrics,evaluation=self.evaluation_history[-1],
                formal_exact_resume_supported=False,replay_buffer_in_checkpoint=False)
        finally:self.vector.close()

    def log_train(self,recent):
        fmt=lambda x:'-' if x is None else f'{x:.3f}'
        event=(f"dwell_start={fmt(recent['dwell_start_episode_rate'])} | dwell_complete={fmt(recent['dwell_complete_episode_rate'])} | mutual_draw={fmt(recent['mutual_annihilation_draw_rate'])}"
               if self.env_config['environment_version'] in DWELL_VERSIONS else f"fire={fmt(recent['fire_rate'])}")
        print(f"[TRAIN] {steps_text(self.trainer.sampled_steps)}/{steps_text(self.total_steps)} | ep={len(self.records)} | win={fmt(recent['win_rate'])} | {event} | kill={fmt(recent['kill_rate'])} | R={fmt(recent['mean_return'])} | Qloss={fmt(self.last_metrics.get('critic_loss'))} | Aloss={fmt(self.last_metrics.get('actor_loss'))}",flush=True)
