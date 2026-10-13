"""Fresh on-policy parallel rollouts, compact logging and deterministic eval."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.evaluator import evaluate, aggregate_combat_records, episode_return_metrics
from algorithm.common.controller_diagnostics import controller_metric_values
from .trainer import MAACTrainer, RolloutBatch


def steps_text(value):
    return f'{value/1e6:.1f}M' if value>=1000000 else f'{value/1000:g}k' if value>=1000 else str(value)


class MAACTrainingRunner:
    def __init__(self, env_config, config, output_dir, num_envs=None, total_sampled_steps=None,
                 seed=None, device='cuda', smoke=False):
        if device!='cuda' or not torch.cuda.is_available():raise RuntimeError('MAAC training requires CUDA; no CPU fallback')
        self.env_config,self.config,self.output_dir=env_config,config,Path(output_dir)
        t=config['training'];self.seed=t['seed'] if seed is None else seed
        self.num_envs=num_envs or t['num_train_envs']
        self.total_steps=total_sampled_steps or (32 if smoke else t['total_sampled_steps'])
        if self.num_envs<=0 or self.total_steps<=0:raise ValueError('envs/steps must be positive')
        self.rollout_length=4 if smoke else t['rollout_length']
        base=config['implementation']['evaluation_seed_base']
        if self.seed+self.total_steps+self.num_envs>=base:raise ValueError('training/evaluation seed ranges may overlap')
        self.eval_seeds=list(range(base,base+(2 if smoke else t['evaluation_episodes'])))
        self.trainer=MAACTrainer(env_config,config,device,self.seed,smoke)
        self.vector=ParallelVectorEnv(self.num_envs,env_config,self.seed,self.eval_seeds)
        self.records=[];self.agent_returns=np.zeros((self.num_envs,5))
        self.last_metrics={};self.evaluation_history=[];self.best_key=None
        try:self.observations=self.vector.reset()
        except BaseException:self.vector.close();raise

    def collect(self, length):
        storage={k:[] for k in RolloutBatch.__dataclass_fields__};infos=[]
        for _ in range(length):
            obs=self.observations.copy();alive=self.vector.current_alive_masks.copy()
            actions=self.trainer.act(obs,alive)
            result=self.vector.step_batch(actions)
            values=dict(observations=obs,actions=actions,rewards=result.rewards,
                next_observations=result.transition_next_observations,
                dones=result.terminated|result.truncated,alive_masks=result.alive_masks,next_alive_masks=result.next_alive_masks)
            for k,v in values.items():storage[k].append(v)
            self.agent_returns+=result.rewards
            for i,done in enumerate(result.terminated|result.truncated):
                if done:
                    team,agent=episode_return_metrics(self.agent_returns[i])
                    self.records.append(dict(episode_return=team,mean_agent_episode_return=agent,**result.infos[i]))
                    self.agent_returns[i].fill(0)
            infos.extend(result.infos)
            self.observations=result.observations
            self.trainer.sampled_steps+=self.num_envs
        return RolloutBatch(**{k:np.stack(v).astype(np.float32) for k,v in storage.items()}),infos

    def recent(self):
        rows=self.records[-self.config['training']['recent_episode_window']:]
        if not rows:return dict(mean_return=None,win_rate=None,fire_rate=None,kill_rate=None)
        return dict(mean_return=float(np.mean([r['episode_return'] for r in rows])),
            win_rate=float(np.mean([r['red_success'] for r in rows])),
            fire_rate=float(np.mean([r['red_first_attempt_step'] is not None for r in rows])),
            kill_rate=float(np.mean([r['red_first_kill_step'] is not None for r in rows])))

    def write(self, name, row):
        with (self.output_dir/name).open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

    def evaluate(self):
        row=dict(sampled_steps=self.trainer.sampled_steps,evaluation_mode='deterministic',
            evaluation_seeds=self.eval_seeds,**evaluate(self.trainer,self.env_config,self.eval_seeds))
        row.update(fire_rate=row['red_attempt_episode_rate'],kill_rate=row['red_kill_episode_rate'])
        self.evaluation_history.append(row);self.write('evaluation_metrics.jsonl',row)
        key=(row['win_rate'],row['average_return'],-row['average_red_loss'])
        if self.best_key is None or key>self.best_key:
            self.best_key=key;self.trainer.save(self.output_dir/'best_eval.pt')
        print(f"[EVAL] {steps_text(row['sampled_steps'])} | deterministic | win={row['win_rate']:.1%} | fire={row['fire_rate']:.1%} | kill={row['kill_rate']:.1%} | R={row['average_return']:.2f} | len={row['episode_length']:.0f}",flush=True)
        return row

    def run(self):
        t=self.config['training'];next_log=t['console_interval_sampled_steps']
        next_eval=t['evaluation_interval_sampled_steps'];next_save=t['checkpoint_interval_sampled_steps']
        print(f"[START] MAAC | env=v3.29 | 5v5 | seed={self.seed} | cuda | envs={self.num_envs} | steps={steps_text(self.total_steps)}",flush=True)
        try:
            while self.trainer.sampled_steps<self.total_steps:
                length=min(self.rollout_length,math.ceil((self.total_steps-self.trainer.sampled_steps)/self.num_envs))
                batch,infos=self.collect(length)
                self.last_metrics=self.trainer.update(batch)
                recent=self.recent()
                row=dict(record_type='rollout_summary',sampled_steps=self.trainer.sampled_steps,
                    episodes=len(self.records),rollout_transitions=len(infos),mean_step_reward=float(batch.rewards.mean()),
                    **recent,**self.last_metrics,
                    **{k:float(v.mean()) for k,v in controller_metric_values(infos).items()})
                for side in ('red','blue'):
                    for field in ('step_fire_attempts','step_weapon_hits','step_attack_kills'):
                        key=f'{side}_{field}';row[key]=sum(i[key] for i in infos)
                self.write('training_metrics.jsonl',row)
                self.write('optimization_metrics.jsonl',dict(sampled_steps=self.trainer.sampled_steps,**self.last_metrics))
                if self.trainer.sampled_steps>=next_log:
                    self.log_train(recent)
                    next_log=(self.trainer.sampled_steps//t['console_interval_sampled_steps']+1)*t['console_interval_sampled_steps']
                if self.trainer.sampled_steps>=next_eval:
                    self.evaluate();next_eval=(self.trainer.sampled_steps//t['evaluation_interval_sampled_steps']+1)*t['evaluation_interval_sampled_steps']
                if self.trainer.sampled_steps>=next_save:
                    self.trainer.save(self.output_dir/f'checkpoint_{self.trainer.sampled_steps}.pt')
                    next_save=(self.trainer.sampled_steps//t['checkpoint_interval_sampled_steps']+1)*t['checkpoint_interval_sampled_steps']
            self.log_train(self.recent())
            if not self.evaluation_history or self.evaluation_history[-1]['sampled_steps']!=self.trainer.sampled_steps:self.evaluate()
            for name in ('latest.pt','final.pt',f'checkpoint_{self.trainer.sampled_steps}.pt'):self.trainer.save(self.output_dir/name)
            return dict(algorithm='MAAC',variant=self.config['implementation']['variant'],environment_version='3.29',
                sampled_steps=self.trainer.sampled_steps,completed_episodes=len(self.records),
                actor_updates=self.trainer.actor_updates,critic_updates=self.trainer.critic_updates,
                optimization_metrics=self.last_metrics,evaluation=self.evaluation_history[-1],
                rollout_reuse=False,formal_exact_resume_supported=False)
        finally:self.vector.close()

    def log_train(self,recent):
        fmt=lambda x:'-' if x is None else f'{x:.3f}'
        print(f"[TRAIN] {steps_text(self.trainer.sampled_steps)}/{steps_text(self.total_steps)} | ep={len(self.records)} | win={fmt(recent['win_rate'])} | fire={fmt(recent['fire_rate'])} | kill={fmt(recent['kill_rate'])} | R={fmt(recent['mean_return'])} | Qloss={fmt(self.last_metrics.get('critic_loss'))} | Aloss={fmt(self.last_metrics.get('actor_loss'))}",flush=True)
