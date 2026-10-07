"""Independent recurrent rollout lifecycle; reuse baseline reporting and scheduling."""
from pathlib import Path
import numpy as np
import torch
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.protocol import config_sha256
from env.config import ENVIRONMENT_VERSION
from .factory import build_stea_mappo_trainer, validate_config
from .protocol import require_cuda, validate_checkpoint
from .trainer import RecurrentRolloutBatch, STEA_MAPPO_IMPL_VERSION
from .evaluation import evaluate


class STEAMAPPOTrainingRunner(MAPPOTrainingRunner):
    def __init__(self,env_config,algorithm_config,num_envs=None,total_sampled_steps=None,
                 device=None,seed=None,output_dir=None,smoke=False):
        validate_config(algorithm_config)
        self.env_config,self.algorithm_config = env_config,algorithm_config
        self.smoke = bool(smoke)
        t,n,i = (algorithm_config[key] for key in ("training","network","implementation"))
        self.device = str(t["device"] if device is None else device)
        require_cuda(self.device)
        self.observation_dim,self.action_dim,self.num_agents = 52,3,4
        self.num_envs = int(t["num_train_envs"] if num_envs is None else num_envs)
        self.total_sampled_steps = int(t["total_sampled_steps"] if total_sampled_steps is None else total_sampled_steps)
        self.seed = int(t["seed"] if seed is None else seed)
        if self.num_envs <= 0 or self.total_sampled_steps <= 0 or self.seed < 0:
            raise ValueError("environment count and target must be positive; seed must be nonnegative")
        if output_dir is None:
            raise ValueError("output_dir must be the final run directory")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.trainer = build_stea_mappo_trainer(algorithm_config,self.device,seed=self.seed,smoke=smoke)
        self.effective_hidden_dim = self.trainer.network_architecture["critic_hidden_dim"]
        self.rollout_steps = int(n["recurrent_sequence_length"] if smoke else t["rollout_steps"])
        base = int(i["evaluation_seed_base"])
        self.evaluation_seeds = list(range(base,base+(2 if smoke else int(t["evaluation_episodes"]))))
        if self.seed+self.total_sampled_steps+self.num_envs >= base or not self.evaluation_seeds:
            raise ValueError("invalid or overlapping evaluation seed protocol")
        self.console_interval = int(32 if smoke else algorithm_config["runtime_logging"]["console_interval_sampled_steps"])
        self.recent_episode_window = int(algorithm_config["runtime_logging"]["recent_episode_window"])
        self.evaluation_interval = self.total_sampled_steps if smoke else int(t["evaluation_interval_sampled_steps"])
        self.checkpoint_interval = self.total_sampled_steps if smoke else int(i["checkpoint_interval_sampled_steps"])
        if min(self.console_interval,self.recent_episode_window,self.evaluation_interval,self.checkpoint_interval) <= 0:
            raise ValueError("reporting intervals must be positive")
        self.next_console_log,self.next_evaluation,self.next_checkpoint = self.console_interval,self.evaluation_interval,self.checkpoint_interval
        self.completed_records,self.evaluation_history = [],[]
        self.last_metrics,self.best_evaluation = {},None
        self.agent_episode_returns = np.zeros((self.num_envs,4),dtype=float)
        self.vector = ParallelVectorEnv(self.num_envs,env_config,self.seed,self.evaluation_seeds)
        try:
            self.observations = self.vector.reset()
        except BaseException:
            self.vector.close()
            raise
        self.alive_masks = self.vector.current_alive_masks.copy()
        self.actor_hidden_states = np.zeros((self.num_envs,4,int(n["gru_hidden_dim"])),dtype=np.float32)
        self.episode_start_masks = np.ones(self.num_envs,dtype=np.float32)

    def startup_summary(self):
        result = super().startup_summary()
        result.update(algorithm="STEA-MAPPO",stea_mappo_impl_version=STEA_MAPPO_IMPL_VERSION,
                      network_architecture=dict(self.trainer.network_architecture))
        return result

    def start_log_line(self):
        return super().start_log_line().replace("algorithm=MAPPO","algorithm=STEA-MAPPO") + f" | sequence={self.trainer.sequence_length} | gru={self.trainer.actor.gru_hidden_dim}"

    @staticmethod
    def done_log_line(summary):
        return MAPPOTrainingRunner.done_log_line(summary).replace("algorithm=MAPPO","algorithm=STEA-MAPPO")

    def collect_rollout(self,vector_steps=None):
        storage = {key:[] for key in RecurrentRolloutBatch.__dataclass_fields__}
        for _ in range(self.rollout_steps if vector_steps is None else int(vector_steps)):
            obs,alive = self.observations.copy(),self.alive_masks.copy()
            before,start = self.actor_hidden_states.copy(),self.episode_start_masks.copy()
            actions,raw,log_prob,hidden = self.trainer.act(obs,alive,before,start)
            result = self.vector.step_batch(actions)
            done = (result.terminated|result.truncated).astype(np.float32)
            row = dict(observations=obs,actions=actions,raw_actions=raw,old_log_probs=log_prob,
                rewards=result.rewards,dones=done,alive_masks=alive,
                next_observations=result.transition_next_observations,next_alive_masks=result.next_alive_masks,
                actor_hidden_states=before,episode_starts=start)
            for key,value in row.items():
                storage[key].append(value)
            # Reset immediately after death, before the next observation/action.
            hidden *= result.next_alive_masks[...,None]
            hidden[done>.5] = 0.
            self.actor_hidden_states = hidden
            self.episode_start_masks = done
            rows = self._completed(result)
            self.observations,self.alive_masks = result.observations,self.vector.current_alive_masks.copy()
            self.trainer.sampled_steps += self.num_envs
            self.trainer.vector_steps += 1
            self._write_step_metrics(result,rows)
        return RecurrentRolloutBatch(**{key:np.stack(values).astype(np.float32) for key,values in storage.items()})

    def _evaluation_record(self):
        return {"sampled_steps":self.trainer.sampled_steps,"evaluation_seed_base":self.evaluation_seeds[0],
            "evaluation_seed_end":self.evaluation_seeds[-1],**evaluate(self.trainer,self.env_config,self.evaluation_seeds)}

    def save_checkpoint(self,path):
        counts = {f"{name}_parameter_count":sum(p.numel() for p in module.parameters())
                  for name,module in (("actor",self.trainer.actor),("critic",self.trainer.critic))}
        counts["total_parameter_count"] = counts["actor_parameter_count"]+counts["critic_parameter_count"]
        self.trainer.save(path,{"algorithm":"STEA-MAPPO","stea_mappo_impl_version":STEA_MAPPO_IMPL_VERSION,
            "environment_version":ENVIRONMENT_VERSION,"observation_dim":52,"action_dim":3,"num_agents":4,
            "training_seed":self.seed,"training_gamma":self.trainer.gamma,"training_num_envs":self.num_envs,
            "training_total_sampled_steps":self.total_sampled_steps,"training_smoke":self.smoke,
            "effective_hidden_dim":self.effective_hidden_dim,"critic_type":"attention",
            "network_architecture":dict(self.trainer.network_architecture),**counts,
            "environment_config_sha256":config_sha256(self.env_config),
            "algorithm_config_sha256":config_sha256(self.algorithm_config),
            "episode_indices":self.vector.episode_indices.tolist(),
            "evaluation_history":self.evaluation_history,"best_evaluation":self.best_evaluation})

    def resume(self,path):
        state = torch.load(path,map_location="cpu",weights_only=False)
        extra = validate_checkpoint(state,self.env_config,self.algorithm_config)
        if (extra["training_seed"],extra["training_num_envs"],extra["training_smoke"]) != (self.seed,self.num_envs,self.smoke):
            raise RuntimeError("resume training seed/environment count/smoke protocol mismatch")
        previous = np.asarray(extra["episode_indices"],dtype=np.int64)
        if previous.shape != (self.num_envs,):
            raise RuntimeError("resume episode index shape mismatch")
        self.trainer.load(path)
        # As in baseline, resume starts fresh episodes with new scenario seeds.
        self.vector.episode_indices = previous+1
        self.observations = self.vector.reset()
        self.alive_masks = self.vector.current_alive_masks.copy()
        self.actor_hidden_states.fill(0.)
        self.episode_start_masks.fill(1.)
        self.agent_episode_returns.fill(0.)
        self.evaluation_history = list(extra["evaluation_history"])
        self.best_evaluation = extra["best_evaluation"]
        for target,interval in (("next_console_log",self.console_interval),("next_evaluation",self.evaluation_interval),("next_checkpoint",self.checkpoint_interval)):
            setattr(self,target,(self.trainer.sampled_steps//interval+1)*interval)
