"""Independent simulator sampling and file/log lifecycle for navigation PPO."""
from dataclasses import fields
import json
from pathlib import Path
import time
import csv
import numpy as np
import torch
import yaml
from .config import validate_config, fingerprint
from .navigation import NavigationVectorEnv
from .memory import MemoryState
from .trainer import ETMMAPPOTrainer,EntityRollout
from .evaluation import evaluate
from .checkpoint import save_checkpoint


class ETMTrainingRunner:
    def __init__(self,config,output_dir):
        validate_config(config)
        self.config=config;self.output=Path(output_dir)
        if self.output.exists() and any(self.output.iterdir()):raise FileExistsError('experiment output must be fresh')
        n=config['network']
        if (n['self_dim'],n['peer_dim'],n['obstacle_dim'],n['action_dim'])!=(10,4,5,2):
            raise ValueError('the supplied navigation simulator emits 10/4/5 features and 2 actions')
        self.output.mkdir(parents=True,exist_ok=True)
        self.trainer=ETMMAPPOTrainer(config,config['training']['device'])
        t=config['training']
        self.vector=NavigationVectorEnv(t['num_envs'],config['environment'],t['seed'])
        self.observations=self.vector.current
        self.state=self.trainer.initial_state(self.observations)
        self.episode_start=np.ones(t['num_envs'],dtype=bool)
        self.completed=[];self.evaluation_history=[];self.best=None;self.last_metrics={}
        self.started=time.perf_counter()
        (self.output/'algorithm_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
        (self.output/'env_config.yaml').write_text(yaml.safe_dump(config['environment'],sort_keys=False),encoding='utf-8')
        self.run_config=dict(algorithm='ETM-MAPPO',variant=n['variant'],config_sha256=fingerprint(config),
            environment_sha256=fingerprint(config['environment']),seed=t['seed'],device=t['device'],
            gpu=torch.cuda.get_device_name(),parameter_counts=self.trainer.parameter_counts(),
            sample_unit='joint_environment_transitions',bit_exact_training_resume=False)
        (self.output/'run_config.json').write_text(json.dumps(self.run_config,indent=2),encoding='utf-8')

    def collect_rollout(self,steps):
        storage={key:[] for key in EntityRollout.__dataclass_fields__}
        for _ in range(steps):
            observations={key:value.copy() for key,value in self.observations.items()}
            before=self.state.numpy()
            starts=self.episode_start.copy()
            values=self.trainer.values(observations)
            actions,raw,log,state=self.trainer.act(observations,self.state,starts)
            transition,rewards,terminated,truncated,infos=self.vector.step(actions)
            executed=np.stack([info['executed_actions'] for info in infos])
            if not np.array_equal(actions,executed):raise RuntimeError('policy/executed action mismatch')
            next_values=self.trainer.values(transition)
            row=dict(observations=observations,next_observations=transition,actions=executed,
                raw_actions=raw,old_log_probs=log,rewards=rewards,values=values,next_values=next_values,
                terminated=terminated,truncated=truncated,episode_starts=starts,memory=before,
                padding_mask=np.ones(len(actions),dtype=bool))
            for key,value in row.items():storage[key].append(value)
            done=terminated|truncated
            for field in fields(state):
                value=getattr(state,field.name)
                mask=torch.as_tensor(done,device=value.device).reshape(len(done),*([1]*(value.ndim-1)))
                setattr(state,field.name,torch.where(mask,-1 if field.name.endswith('_ids') else 0,value))
            self.state=state.detach();self.episode_start=done
            self.observations=self.vector.current
            self.trainer.sampled_steps+=len(actions);self.trainer.vector_steps+=1
            for i in np.flatnonzero(done):
                record={key:value for key,value in infos[i].items() if key!='executed_actions'}
                self.completed.append(record)
                self._jsonl('episode_metrics.jsonl',dict(sampled_steps=self.trainer.sampled_steps,**record))
        result={}
        for key,rows in storage.items():
            result[key]={name:np.stack([r[name] for r in rows]) for name in rows[0]} if isinstance(rows[0],dict) else np.stack(rows)
        return EntityRollout(**result)

    def _jsonl(self,name,row):
        with (self.output/name).open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')

    def save(self,name):
        save_checkpoint(self.trainer,self.output/name,dict(best_evaluation=self.best,evaluation_history=self.evaluation_history,
            training_seed_provenance=self.vector.used_seeds,episode_indices=self.vector.episode_indices.tolist()))

    def evaluation(self,stage):
        t=self.config['training']
        seeds=range(t['evaluation_seed_base'],t['evaluation_seed_base']+t['evaluation_episodes'])
        row=dict(sampled_steps=self.trainer.sampled_steps,stage=stage,**evaluate(self.trainer,seeds))
        self.evaluation_history.append(row)
        # CSV stays scalar/flat; complete provenance remains in JSON/checkpoints.
        scalars=[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in self.evaluation_history]
        with (self.output/'evaluation_history.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(scalars[0]));writer.writeheader();writer.writerows(scalars)
        self._jsonl('evaluation_history.jsonl',row)
        key=lambda r:(r['task_completion_rate'],-r['collision_rate'],r['episode_return'])
        if stage!='initial' and (self.best is None or key(row)>key(self.best)):
            self.best=dict(row);self.save('best_eval.pt')
        return row

    def run(self):
        t=self.config['training']
        initial=self.evaluation('initial')
        next_eval=t['evaluation_interval'];next_checkpoint=t['checkpoint_interval']
        try:
            while self.trainer.sampled_steps<t['total_sampled_steps']:
                steps=min(t['rollout_steps'],(t['total_sampled_steps']-self.trainer.sampled_steps)//t['num_envs'])
                rollout=self.collect_rollout(steps)
                self.last_metrics=self.trainer.update(rollout)
                record=dict(sampled_steps=self.trainer.sampled_steps,rollout_update=self.trainer.rollout_updates,
                    elapsed_seconds=time.perf_counter()-self.started,**self.last_metrics)
                self._jsonl('optimization_metrics.jsonl',record)
                self._jsonl('training_metrics.jsonl',dict(record,completed_episodes=len(self.completed),
                    recent_episode_return=float(np.mean([r['episode_return'] for r in self.completed[-50:]])) if self.completed else None))
                print(f"[PPO] {self.config['network']['variant']} steps={self.trainer.sampled_steps} ratio_error={self.last_metrics['pre_update_ratio_max_abs_error']:.2g} actor_steps={self.trainer.actor_optimizer_steps}",flush=True)
                if self.trainer.sampled_steps>=next_eval:
                    self.evaluation('development')
                    while next_eval<=self.trainer.sampled_steps:next_eval+=t['evaluation_interval']
                if self.trainer.sampled_steps>=next_checkpoint:
                    self.save(f'checkpoint_{self.trainer.sampled_steps}.pt')
                    while next_checkpoint<=self.trainer.sampled_steps:next_checkpoint+=t['checkpoint_interval']
                self.save('latest.pt')
            if self.evaluation_history[-1]['sampled_steps']!=self.trainer.sampled_steps:self.evaluation('development')
            final=self.evaluation_history[-1]
            generalization=evaluate(self.trainer,range(t['evaluation_seed_base']+1000,t['evaluation_seed_base']+1000+t['evaluation_episodes']),scene='random')
            self.save('latest.pt');self.save('final.pt')
            summary=dict(self.run_config,sampled_steps=self.trainer.sampled_steps,vector_steps=self.trainer.vector_steps,
                rollout_updates=self.trainer.rollout_updates,actor_optimizer_steps=self.trainer.actor_optimizer_steps,
                critic_optimizer_steps=self.trainer.critic_optimizer_steps,initial_evaluation=initial,final_evaluation=final,
                random_layout_evaluation=generalization,return_change=final['episode_return']-initial['episode_return'],
                elapsed_seconds=time.perf_counter()-self.started,last_optimization=self.last_metrics,
                completed_episodes=len(self.completed),bit_exact_training_resume=False)
            (self.output/'run_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
            return summary
        except BaseException as exc:
            self._jsonl('exceptions.jsonl',dict(sampled_steps=self.trainer.sampled_steps,error=repr(exc)))
            raise
