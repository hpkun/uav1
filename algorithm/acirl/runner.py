"""ACIRL parallel collection, independent updates, evaluation and provenance."""
from copy import deepcopy
from pathlib import Path
import csv
import json
import numpy as np
import torch
import yaml
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.evaluator import aggregate_combat_records
from algorithm.common.checkpoint import evaluation_selection_key
from algorithm.common.protocol import config_sha256
from .protocol import validate_config,require_cuda,architecture
from .trainer import ACIRLTrainer
from .rollout import RolloutCollector
from .evaluation import evaluate


def json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


class ACIRLTrainingRunner:
    def __init__(self,env,config,output_dir,device=None,seed=None,num_envs=None,total_sampled_steps=None,smoke=False,warm_start=None):
        self.env=deepcopy(env);self.requested=deepcopy(config);self.config=deepcopy(config);self.smoke=smoke
        t=self.config['training']
        if smoke:t.update(num_train_envs=2,total_sampled_steps=128,rollout_steps=32,ppo_epochs=2,
            minibatch_size=128,critic_minibatch_size=64,evaluation_episodes=1,evaluation_interval=128,checkpoint_interval=128)
        for key,value in (('device',device),('seed',seed),('num_train_envs',num_envs),('total_sampled_steps',total_sampled_steps)):
            if value is not None:t[key]=value
        validate_config(self.env,self.config);require_cuda(t['device'])
        self.output=Path(output_dir)
        if self.output.exists() and any(self.output.iterdir()):raise FileExistsError('ACIRL requires a fresh nonempty-free output directory')
        self.output.mkdir(parents=True,exist_ok=True)
        self.trainer=ACIRLTrainer(self.config,t['device']);self.history=[];self.best=None;self.completed=[]
        self.warm_start=str(warm_start) if warm_start else None
        if warm_start:
            state=self.trainer.load(warm_start,self.env);self.history=state['evaluation_history'];self.best=state['best_evaluation']
            if self.trainer.sampled_steps>=t['total_sampled_steps']:raise ValueError('warm-start checkpoint already reached configured budget')
        base=t['seed']+(self.trainer.sampled_steps+t['num_train_envs'] if warm_start else 0)
        self.vector=ParallelVectorEnv(t['num_train_envs'],self.env,base,range(t['evaluation_seed_base'],t['evaluation_seed_base']+t['evaluation_episodes']))
        try:
            self.collector=RolloutCollector(self.trainer,self.vector)
            for filename,value in (('env_config.yaml',self.env),('algorithm_config.yaml',self.config),('requested_algorithm_config.yaml',self.requested)):
                (self.output/filename).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')
            self.write_json('run_config.json',dict(algorithm='acirl',device=str(self.trainer.device),smoke=smoke,
                requested_algorithm_config_sha256=config_sha256(self.requested),algorithm_config_sha256=config_sha256(self.config),
                environment_config_sha256=config_sha256(self.env),effective_training=t,network_architecture=architecture(self.config),
                parameter_counts=self.trainer.parameter_counts(),worker_pids=self.vector.worker_pids,worker_metadata=self.vector.worker_metadata,
                sampled_steps_unit='joint_environment_transitions_not_agent_transitions',warm_start_source=self.warm_start,
                warm_start_semantics='weights_optimizers_counters_algorithm_RNG; fresh episodes; pending macros discarded',
                environment_seed_base=base,formal_exact_resume_supported=False))
        except BaseException:self.vector.close();raise

    def write_json(self,name,value):
        (self.output/name).write_text(json.dumps(value,indent=2,allow_nan=False,default=json_default),encoding='utf-8')

    def append(self,name,value):
        with (self.output/name).open('a',encoding='utf-8') as f:f.write(json.dumps(value,allow_nan=False,default=json_default)+'\n')

    def save(self,name):
        self.trainer.save(self.output/name,self.env,dict(evaluation_history=self.history,best_evaluation=self.best,
            warm_start_source=self.warm_start,smoke=self.smoke),self.collector.runtime_state())

    def evaluate(self):
        t=self.config['training'];row=dict(sampled_steps=self.trainer.sampled_steps,
            **evaluate(self.trainer,self.env,range(t['evaluation_seed_base'],t['evaluation_seed_base']+t['evaluation_episodes']),t['evaluation_mode']))
        self.history.append(row)
        # Selection always uses original environment return and the configured evaluation mode.
        if self.best is None or evaluation_selection_key(row)>evaluation_selection_key(self.best):self.best=row.copy();self.save('best_eval.pt')
        with (self.output/'evaluation_history.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(self.history)
        print(f"[EVAL] {self.trainer.sampled_steps} | {t['evaluation_mode']} | win={row['win_rate']:.1%} | original_return={row['average_return']:.3f} | alerts={row['policy_diagnostic_alerts']}",flush=True)

    def run(self):
        t=self.config['training'];last={}
        next_eval=(self.trainer.sampled_steps//t['evaluation_interval']+1)*t['evaluation_interval']
        next_checkpoint=(self.trainer.sampled_steps//t['checkpoint_interval']+1)*t['checkpoint_interval']
        print(f"[START] ACIRL {self.config['ablation']['mode']} | env={self.env['environment_version']} | CUDA | envs={t['num_train_envs']} | transitions={t['total_sampled_steps']}",flush=True)
        try:
            while self.trainer.sampled_steps<t['total_sampled_steps']:
                steps=min(t['rollout_steps'],(t['total_sampled_steps']-self.trainer.sampled_steps)//t['num_train_envs'])
                rollout=self.collector.collect(steps);last=self.trainer.update(rollout)
                self.completed.extend(rollout.episode_records)
                for record in rollout.episode_records:
                    self.append('episode_metrics.jsonl',{key:record[key] for key in ('episode_return','mean_agent_episode_return',
                        'original_team_training_scale_return','internal_training_return','switching_cost_total','learned_renew_per_agent',
                        'episode_length','red_losses','blue_losses')})
                self.append('optimization_metrics.jsonl',last)
                for event in rollout.events:self.append('macro_decisions.jsonl',event)
                for event in getattr(rollout,'intention_events',[]):self.append('intention_events.jsonl',event)
                combat=aggregate_combat_records(rollout.episode_records) if rollout.episode_records else {}
                self.append('training_metrics.jsonl',dict(sampled_steps=self.trainer.sampled_steps,vector_steps=self.trainer.vector_steps,
                    update_count=self.trainer.update_count,completed_episodes_total=len(self.completed),completed_episodes=len(rollout.episode_records),
                    original_environment_reward_sum=last['original_environment_reward_sum'],
                    original_team_scaled_reward_sum=last['original_team_scaled_reward_sum'],internal_training_reward_sum=last['internal_training_reward_sum'],
                    switching_cost_sum=last['switching_cost_sum'],
                    **combat,mean_learned_renew_count_per_completed_episode=float(np.mean([sum(r['learned_renew_per_agent']) for r in rollout.episode_records])) if rollout.episode_records else None))
                print(f"[TRAIN] {self.trainer.sampled_steps} | updates={self.trainer.update_count} | low/high/gate={last['valid_low_samples']}/{last['valid_high_decisions']}/{last['valid_gate_decisions']} | loss={last['policy_loss']:.4g} | alerts={last['alerts']}",flush=True)
                if self.trainer.sampled_steps>=next_eval:
                    self.evaluate()
                    while next_eval<=self.trainer.sampled_steps:next_eval+=t['evaluation_interval']
                if self.trainer.sampled_steps>=next_checkpoint:
                    self.save(f'checkpoint_{self.trainer.sampled_steps}.pt');self.save('latest.pt')
                    while next_checkpoint<=self.trainer.sampled_steps:next_checkpoint+=t['checkpoint_interval']
            if not self.history or self.history[-1]['sampled_steps']!=self.trainer.sampled_steps:self.evaluate()
            self.save('latest.pt');self.save('final.pt')
            summary=dict(algorithm='acirl',status='completed',ablation=self.config['ablation']['mode'],sampled_steps=self.trainer.sampled_steps,
                vector_steps=self.trainer.vector_steps,update_count=self.trainer.update_count,parameter_counts=self.trainer.parameter_counts(),
                parameter_drift_l2=self.trainer.parameter_drift(),final_optimization_metrics=last,evaluation_history=self.history,best_evaluation=self.best,
                completed_training_episodes=len(self.completed),formal_exact_resume_supported=False,warm_start_source=self.warm_start)
            self.write_json('run_summary.json',summary);return summary
        finally:self.vector.close()
