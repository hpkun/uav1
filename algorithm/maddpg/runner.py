"""Independent MADDPG vector rollout, OU exploration and one-step replay training."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import numpy as np
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.evaluator import episode_return_metrics, aggregate_combat_records
from algorithm.common.checkpoint import evaluation_selection_key
from algorithm.common.protocol import config_sha256
from .factory import build_maddpg_trainer
from .protocol import validate_config, require_cuda
from .replay_buffer import JointReplayBuffer
from .noise import OUNoise, exploration_scale
from .evaluation import evaluate, validate_seeds
from .trainer import MADDPG_IMPL_VERSION


class MADDPGTrainingRunner:
    def __init__(self, env_config, algorithm_config, num_envs=None, total_sampled_steps=None,
                 device=None, seed=None, output_dir=None, smoke=False):
        self.env_config, self.algorithm_config = deepcopy(env_config), deepcopy(algorithm_config)
        self.requested_algorithm_config = deepcopy(algorithm_config)
        self.smoke = bool(smoke)
        t, i, n = (self.algorithm_config[key] for key in ('training', 'implementation', 'network'))
        if smoke:
            t.update(total_sampled_steps=1024, learning_starts=128, minibatch_size=64,
                     replay_capacity=4096, evaluation_episodes=2, evaluation_interval_sampled_steps=1024)
            i['checkpoint_interval_sampled_steps'] = 1024
            self.algorithm_config['runtime_logging']['console_interval_sampled_steps'] = 256
        validate_config(self.env_config, self.algorithm_config)
        self.num_envs = int(t['num_train_envs'] if num_envs is None else num_envs)
        self.total_sampled_steps = int(t['total_sampled_steps'] if total_sampled_steps is None else total_sampled_steps)
        self.seed = int(t['seed'] if seed is None else seed)
        self.device = str(t['device'] if device is None else device)
        require_cuda(self.device)
        if min(self.num_envs, self.total_sampled_steps) <= 0 or self.seed < 0 or self.total_sampled_steps % self.num_envs:
            raise ValueError('positive target must be divisible by num_envs; seed must be nonnegative')
        self.output_dir = Path(output_dir)
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError(f'fresh MADDPG output directory is non-empty: {self.output_dir}')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trainer = build_maddpg_trainer(self.algorithm_config, self.device, self.seed)
        self.replay = JointReplayBuffer(t['replay_capacity'], n['num_agents'], n['observation_dim'], 3, self.seed ^ 0x31A5)
        self.noise = OUNoise(self.num_envs, n['num_agents'], 3, self.seed ^ 0x0A17,
                             i['ou_theta'], i['ou_sigma'], i['ou_mu'])
        self.evaluation_seeds = validate_seeds(range(i['evaluation_seed_base'], i['evaluation_seed_base']+t['evaluation_episodes']),
            {'training_seed': self.seed, 'training_total_sampled_steps': self.total_sampled_steps, 'training_num_envs': self.num_envs})
        self.evaluation_history, self.completed_records, self.last_metrics = [], [], {}
        self.best_evaluation = None
        self.agent_returns = np.zeros((self.num_envs, n['num_agents']), np.float64)
        self.vector = ParallelVectorEnv(self.num_envs, self.env_config, self.seed, self.evaluation_seeds)
        try:
            self.observations = self.vector.reset()
            self.alive = self.vector.current_alive_masks.copy()
            self.write_json('run_config.json', self.startup_summary())
            import yaml
            for name, config in (('env_config.yaml', self.env_config), ('algorithm_config.yaml', self.algorithm_config),
                                 ('requested_algorithm_config.yaml', self.requested_algorithm_config)):
                (self.output_dir/name).write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        except BaseException:
            self.vector.close()
            raise

    def write_json(self, name, value):
        (self.output_dir/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')

    def startup_summary(self):
        return {'algorithm': 'maddpg', 'implementation_version': MADDPG_IMPL_VERSION,
            'mode': 'smoke' if self.smoke else 'formal', 'device': self.device, 'seed': self.seed,
            'num_envs': self.num_envs, 'total_sampled_steps': self.total_sampled_steps,
            'environment_backend': self.vector.backend, 'worker_pids': self.vector.worker_pids,
            'network_architecture': self.trainer.network_architecture, 'parameter_counts': self.trainer.parameter_counts(),
            'environment_config_sha256': config_sha256(self.env_config),
            'algorithm_config_sha256': config_sha256(self.algorithm_config),
            'requested_algorithm_config_sha256': config_sha256(self.requested_algorithm_config),
            'effective_training': deepcopy(self.algorithm_config['training']),
            'sampled_steps_unit': 'environment_transitions_not_agent_transitions',
            'formal_exact_resume_supported': False}

    def behavior_actions(self, observations, alive):
        i = self.algorithm_config['implementation']
        scale = exploration_scale(self.trainer.sampled_steps, i['exploration_scale_start'],
                                  i['exploration_scale_end'], i['exploration_decay_sampled_steps'])
        mean = self.trainer.act(observations, alive, deterministic=True)
        return (np.clip(mean+scale*self.noise.sample(), -1, 1)*alive[..., None]).astype(np.float32)

    def store_vector_step(self, observations, actions, result):
        executed = np.stack([info.get('executed_red_actions', actions[e]) for e, info in enumerate(result.infos)]).astype(np.float32)
        if not np.array_equal(executed, actions):
            raise RuntimeError('executed_red_actions differs from clipped behavior action')
        self.replay.add_batch(observations, executed, result.rewards, result.transition_next_observations,
            result.terminated | result.truncated, result.alive_masks, result.next_alive_masks)

    def checkpoint_extra(self):
        t = self.algorithm_config['training']
        return {'environment_version': self.env_config['environment_version'],
            'environment_config': deepcopy(self.env_config), 'algorithm_config': deepcopy(self.algorithm_config),
            'environment_config_sha256': config_sha256(self.env_config), 'algorithm_config_sha256': config_sha256(self.algorithm_config),
            'requested_algorithm_config_sha256': config_sha256(self.requested_algorithm_config),
            **{key: getattr(self.trainer, key) for key in ('observation_dim', 'action_dim', 'num_agents', 'gamma', 'tau')},
            'training_seed': self.seed, 'training_num_envs': self.num_envs,
            'training_total_sampled_steps': self.total_sampled_steps, 'training_smoke': self.smoke,
            'network_architecture': self.trainer.network_architecture, 'parameter_counts': self.trainer.parameter_counts(),
            'ou_noise_state': self.noise.state_dict(), 'replay_sampling_rng_state': deepcopy(self.replay.rng.bit_generator.state),
            'episode_indices': self.vector.episode_indices.tolist(), 'evaluation_history': deepcopy(self.evaluation_history),
            'best_evaluation': deepcopy(self.best_evaluation), 'learning_starts': t['learning_starts'],
            'replay_capacity': self.replay.capacity, 'minibatch_size': t['minibatch_size'],
            'gradient_steps_per_vector_step': t['gradient_steps_per_vector_step'],
            'formal_exact_resume_supported': False, 'replay_buffer_in_checkpoint': False}

    def save_checkpoint(self, path):
        self.trainer.save(path, self.checkpoint_extra())

    def evaluate(self):
        row = {'sampled_steps': self.trainer.sampled_steps, **evaluate(self.trainer, self.env_config, self.evaluation_seeds)}
        self.evaluation_history.append(row)
        with (self.output_dir/'evaluation_history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerows(self.evaluation_history)
        if self.best_evaluation is None or evaluation_selection_key(row) > evaluation_selection_key(self.best_evaluation):
            self.best_evaluation = row.copy(); self.save_checkpoint(self.output_dir/'best_eval.pt')
        print(f"[EVAL] steps={self.trainer.sampled_steps} win={row['win_rate']:.2f} return={row['average_return']:.2f}", flush=True)

    def run(self):
        t, i, logging = (self.algorithm_config[key] for key in ('training', 'implementation', 'runtime_logging'))
        next_log = logging['console_interval_sampled_steps']
        next_eval, next_checkpoint = t['evaluation_interval_sampled_steps'], i['checkpoint_interval_sampled_steps']
        print(f'[START] MADDPG {json.dumps(self.startup_summary())}', flush=True)
        try:
            while self.trainer.sampled_steps < self.total_sampled_steps:
                obs, alive = self.observations.copy(), self.alive.copy()
                actions = self.behavior_actions(obs, alive)
                result = self.vector.step_batch(actions)
                self.store_vector_step(obs, actions, result)
                done = result.terminated | result.truncated
                self.agent_returns += result.rewards
                completed = []
                for e in np.flatnonzero(done):
                    team, agent = episode_return_metrics(self.agent_returns[e])
                    completed.append({'episode_return': team, 'mean_agent_episode_return': agent, **result.infos[e]})
                    self.agent_returns[e].fill(0.)
                self.completed_records.extend(completed)
                self.noise.reset(np.flatnonzero(done))
                self.observations, self.alive = result.observations, self.vector.current_alive_masks.copy()
                self.trainer.sampled_steps += self.num_envs
                self.trainer.vector_steps += 1
                if self.trainer.sampled_steps >= t['learning_starts'] and len(self.replay) >= t['minibatch_size']:
                    for _ in range(t['gradient_steps_per_vector_step']):
                        self.last_metrics = self.trainer.update(self.replay.sample(t['minibatch_size']))
                        with (self.output_dir/'optimization_metrics.jsonl').open('a') as stream:
                            stream.write(json.dumps({'sampled_steps': self.trainer.sampled_steps, **self.last_metrics}, allow_nan=False)+'\n')
                row = {'sampled_steps': self.trainer.sampled_steps, 'vector_steps': self.trainer.vector_steps,
                    'replay_size': len(self.replay), 'mean_step_reward': float(result.rewards.mean()),
                    'completed_episodes': len(completed), **(aggregate_combat_records(completed) if completed else {})}
                with (self.output_dir/'training_metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                if self.trainer.sampled_steps >= next_log:
                    recent = self.completed_records[-logging['recent_episode_window']:]
                    print(f"[TRAIN] steps={self.trainer.sampled_steps}/{self.total_sampled_steps} eps={len(self.completed_records)} "
                          f"return={np.mean([r['episode_return'] for r in recent]) if recent else None} "
                          f"actor_updates={self.trainer.actor_update_count} critic_updates={self.trainer.critic_update_count}", flush=True)
                    while next_log <= self.trainer.sampled_steps:
                        next_log += logging['console_interval_sampled_steps']
                if self.trainer.sampled_steps >= next_eval:
                    self.evaluate()
                    while next_eval <= self.trainer.sampled_steps:
                        next_eval += t['evaluation_interval_sampled_steps']
                if self.trainer.sampled_steps >= next_checkpoint:
                    self.save_checkpoint(self.output_dir/f'checkpoint_{self.trainer.sampled_steps}.pt')
                    while next_checkpoint <= self.trainer.sampled_steps:
                        next_checkpoint += i['checkpoint_interval_sampled_steps']
            if not self.evaluation_history or self.evaluation_history[-1]['sampled_steps'] != self.trainer.sampled_steps:
                self.evaluate()
            for name in (f'checkpoint_{self.trainer.sampled_steps}.pt', 'latest.pt', 'final.pt'):
                self.save_checkpoint(self.output_dir/name)
            summary = {'algorithm': 'maddpg', 'sampled_steps': self.trainer.sampled_steps,
                'vector_steps': self.trainer.vector_steps, 'replay_size': len(self.replay),
                'actor_update_count': self.trainer.actor_update_count, 'critic_update_count': self.trainer.critic_update_count,
                'actor_updates_per_agent': self.trainer.actor_updates_per_agent, 'critic_updates_per_agent': self.trainer.critic_updates_per_agent,
                'final_optimization_metrics': self.last_metrics, 'latest_evaluation': self.evaluation_history[-1],
                'formal_exact_resume_supported': False, 'protocol': self.startup_summary()}
            self.write_json('run_summary.json', summary)
            return summary
        finally:
            self.vector.close()
