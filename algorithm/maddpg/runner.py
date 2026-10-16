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
from .protocol import validate_config, require_cuda, schedule_contract
from .replay_buffer import JointReplayBuffer
from .noise import OUNoise, exploration_scale
from .evaluation import evaluate, validate_seeds
from .trainer import MADDPG_IMPL_VERSION
from .schedule import SampledTransitionSchedule
from .logging import format_steps, number, combat_fields


class MADDPGTrainingRunner:
    def __init__(self, env_config, algorithm_config, num_envs=None, total_sampled_steps=None,
                 device=None, seed=None, output_dir=None, smoke=False):
        self.env_config, self.algorithm_config = deepcopy(env_config), deepcopy(algorithm_config)
        self.requested_algorithm_config = deepcopy(algorithm_config)
        self.smoke = bool(smoke)
        t, i, n = (self.algorithm_config[key] for key in ('training', 'implementation', 'network'))
        if smoke:
            t.update(total_sampled_steps=8192, learning_starts=128, minibatch_size=64,
                     replay_capacity=4096, evaluation_episodes=2, evaluation_interval_sampled_steps=8192)
            i['checkpoint_interval_sampled_steps'] = 8192
            self.algorithm_config['runtime_logging']['console_interval_sampled_steps'] = 256
        validate_config(self.env_config, self.algorithm_config)
        schedule_contract(self.algorithm_config)
        self.schedule = SampledTransitionSchedule(t['learning_starts'], t['sampled_steps_per_update'])
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
            'environment_version': self.env_config['environment_version'],
            'num_envs': self.num_envs, 'total_sampled_steps': self.total_sampled_steps,
            'environment_backend': self.vector.backend, 'worker_pids': self.vector.worker_pids,
            'network_architecture': self.trainer.network_architecture, 'parameter_counts': self.trainer.parameter_counts(),
            'environment_config_sha256': config_sha256(self.env_config),
            'algorithm_config_sha256': config_sha256(self.algorithm_config),
            'requested_algorithm_config_sha256': config_sha256(self.requested_algorithm_config),
            'effective_training': deepcopy(self.algorithm_config['training']),
            'sampled_steps_unit': 'environment_transitions_not_agent_transitions',
            **schedule_contract(self.algorithm_config)}

    @property
    def transitions_since_update(self):
        return self.schedule.credit

    def current_exploration_scale(self):
        i = self.algorithm_config['implementation']
        return exploration_scale(self.trainer.sampled_steps, i['exploration_scale_start'],
                                 i['exploration_scale_end'], i['exploration_decay_sampled_steps'])

    def startup_log_line(self):
        t, i, n = (self.algorithm_config[k] for k in ('training','implementation','network'))
        return (f"[START] MADDPG-v{MADDPG_IMPL_VERSION} | env=v{self.env_config['environment_version']} | "
            f"{n['num_agents']}v{n['num_agents']} | seed={self.seed} | {self.device} | envs={self.num_envs} | "
            f"steps={format_steps(self.total_sampled_steps)} | actor={n['observation_dim']}-256-256-3 | "
            f"critic={n['num_agents']*(n['observation_dim']+3)}-256-256-1x{n['num_agents']} | "
            f"batch={t['minibatch_size']} | replay={format_steps(t['replay_capacity'])} | "
            f"warmup={format_steps(t['learning_starts'])} | update_every={t['sampled_steps_per_update']} | "
            f"OU={i['ou_theta']:.2f}/{i['ou_sigma']:.2f} | eval=deterministic")

    def training_log_line(self):
        rows = self.completed_records[-self.algorithm_config['runtime_logging']['recent_episode_window']:]
        recent = aggregate_combat_records(rows) if rows else {}
        return (f"[TRAIN] {format_steps(self.trainer.sampled_steps)}/{format_steps(self.total_sampled_steps)} "
            f"({self.trainer.sampled_steps/self.total_sampled_steps:.1%}) | ep={len(self.completed_records)} | "
            f"{combat_fields(recent)} | Qloss={number(self.last_metrics.get('critic_loss'),3)} | "
            f"Aloss={number(self.last_metrics.get('actor_loss'),3)} | OU={self.current_exploration_scale():.2f}")

    @staticmethod
    def evaluation_log_line(row, best):
        return f"[EVAL{'*' if best else ''}] {format_steps(row['sampled_steps'])} | deterministic | {combat_fields(row)}"

    def write_metrics(self, filename, row):
        with (self.output_dir/filename).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')

    def update_from_credit(self):
        t = self.algorithm_config['training']
        while self.schedule.credit >= t['sampled_steps_per_update']:
            if len(self.replay) < t['minibatch_size']:
                raise RuntimeError('eligible update credit without sufficient replay')
            self.last_metrics = self.trainer.update(self.replay.sample(t['minibatch_size']))
            self.schedule.consume()
            self.write_metrics('optimization_metrics.jsonl', {
                'sampled_steps': self.trainer.sampled_steps, 'vector_steps': self.trainer.vector_steps,
                'replay_size': len(self.replay), **self.last_metrics,
                'exploration_scale': self.current_exploration_scale(),
                'transitions_since_update': self.transitions_since_update,
                'actor_updates_per_agent': list(self.trainer.actor_updates_per_agent),
                'critic_updates_per_agent': list(self.trainer.critic_updates_per_agent)})

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
            'transitions_since_update': self.transitions_since_update,
            **schedule_contract(self.algorithm_config)}

    def save_checkpoint(self, path):
        self.trainer.save(path, self.checkpoint_extra())

    def evaluate(self):
        row = {'sampled_steps': self.trainer.sampled_steps, 'environment_version': self.env_config['environment_version'],
            'evaluation_mode': 'deterministic', 'evaluation_seed_base': self.evaluation_seeds[0],
            'evaluation_seed_end': self.evaluation_seeds[-1], **evaluate(self.trainer, self.env_config, self.evaluation_seeds)}
        self.evaluation_history.append(row)
        with (self.output_dir/'evaluation_history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row)); writer.writeheader(); writer.writerows(self.evaluation_history)
        best = self.best_evaluation is None or evaluation_selection_key(row) > evaluation_selection_key(self.best_evaluation)
        if best:
            self.best_evaluation = row.copy(); self.save_checkpoint(self.output_dir/'best_eval.pt')
        print(self.evaluation_log_line(row, best), flush=True)

    def run(self):
        t, i, logging = (self.algorithm_config[key] for key in ('training', 'implementation', 'runtime_logging'))
        next_log = logging['console_interval_sampled_steps']
        next_eval, next_checkpoint = t['evaluation_interval_sampled_steps'], i['checkpoint_interval_sampled_steps']
        print(self.startup_log_line(), flush=True)
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
                previous_steps = self.trainer.sampled_steps
                self.trainer.sampled_steps += self.num_envs
                self.trainer.vector_steps += 1
                self.schedule.add(previous_steps, self.trainer.sampled_steps)
                self.update_from_credit()
                combat = aggregate_combat_records(completed) if completed else {}
                row = {'sampled_steps': self.trainer.sampled_steps, 'vector_steps': self.trainer.vector_steps,
                    'replay_size': len(self.replay), 'mean_step_reward': float(result.rewards.mean()),
                    'completed_episodes': len(completed), 'completed_episodes_this_step': len(completed),
                    'completed_episodes_total': len(self.completed_records),
                    **{key: None for key in ('win_rate','red_win_rate','average_return','average_red_loss','average_blue_loss',
                        'red_fire_window_episode_rate','red_attempt_episode_rate','red_hit_episode_rate','red_kill_episode_rate',
                        'timeout_rate','draw_rate','episode_length','red_boundary_exits','blue_boundary_exits',
                        'red_ground_losses','blue_ground_losses','red_ceiling_losses','blue_ceiling_losses')}, **combat,
                    'team_episode_return': combat.get('average_return'), 'red_loss': combat.get('average_red_loss'),
                    'blue_loss': combat.get('average_blue_loss'), 'fire_rate': combat.get('red_attempt_episode_rate'),
                    'hit_rate': combat.get('red_hit_episode_rate'), 'kill_rate': combat.get('red_kill_episode_rate'),
                    'exploration_scale': self.current_exploration_scale(), 'transitions_since_update': self.transitions_since_update,
                    'actor_update_count': self.trainer.actor_update_count, 'critic_update_count': self.trainer.critic_update_count}
                for name in ('r1','r2','r3','r4'):
                    if all(f'{name}_rewards' in info for info in result.infos):
                        row[f'mean_{name}_reward'] = float(np.stack([info[f'{name}_rewards'] for info in result.infos]).mean())
                self.write_metrics('training_metrics.jsonl', row)
                if self.trainer.sampled_steps >= next_log:
                    print(self.training_log_line(), flush=True)
                    while next_log <= self.trainer.sampled_steps:
                        next_log += logging['console_interval_sampled_steps']
                if self.trainer.sampled_steps >= next_eval:
                    self.evaluate()
                    while next_eval <= self.trainer.sampled_steps:
                        next_eval += t['evaluation_interval_sampled_steps']
                if self.trainer.sampled_steps >= next_checkpoint:
                    self.save_checkpoint(self.output_dir/f'checkpoint_{self.trainer.sampled_steps}.pt')
                    print(f'[CKPT] {format_steps(self.trainer.sampled_steps)} | saved', flush=True)
                    while next_checkpoint <= self.trainer.sampled_steps:
                        next_checkpoint += i['checkpoint_interval_sampled_steps']
            if not self.evaluation_history or self.evaluation_history[-1]['sampled_steps'] != self.trainer.sampled_steps:
                self.evaluate()
            for name in (f'checkpoint_{self.trainer.sampled_steps}.pt', 'latest.pt', 'final.pt'):
                self.save_checkpoint(self.output_dir/name)
            summary = {'algorithm': 'maddpg', 'implementation_version': MADDPG_IMPL_VERSION, 'status': 'completed',
                'sampled_steps': self.trainer.sampled_steps,
                'vector_steps': self.trainer.vector_steps, 'replay_size': len(self.replay),
                'actor_update_count': self.trainer.actor_update_count, 'critic_update_count': self.trainer.critic_update_count,
                'actor_updates_per_agent': self.trainer.actor_updates_per_agent, 'critic_updates_per_agent': self.trainer.critic_updates_per_agent,
                'final_optimization_metrics': self.last_metrics, 'latest_evaluation': self.evaluation_history[-1],
                'transitions_since_update': self.transitions_since_update,
                'best_sampled_steps': self.best_evaluation['sampled_steps'], 'best_evaluation': self.best_evaluation,
                'formal_exact_resume_supported': False, 'replay_buffer_in_checkpoint': False, 'protocol': self.startup_summary()}
            self.write_json('run_summary.json', summary)
            print(f"[DONE] MADDPG-v{MADDPG_IMPL_VERSION} | steps={format_steps(self.trainer.sampled_steps)} | "
                f"best_step={format_steps(summary['best_sampled_steps'])} | best_win={number(self.best_evaluation['win_rate'],percent=True)} | "
                f"final_win={number(self.evaluation_history[-1]['win_rate'],percent=True)} | output={self.output_dir}", flush=True)
            return summary
        finally:
            self.vector.close()
