"""PS-IPPO sampling; baseline reporting, evaluation selection and scheduling reused."""
from pathlib import Path
from typing import Any
import json
import numpy as np
from algorithm.mappo.runner import MAPPOTrainingRunner
from algorithm.common.vector_env import ParallelVectorEnv
from algorithm.common.critic_protocol import layer_width
from algorithm.common.protocol import config_sha256
from env.config import environment_dimensions
from .trainer import RolloutBatch
from .factory import build_ippo_trainer
from .protocol import validate_algorithm_config, require_cuda

class IPPOTrainingRunner(MAPPOTrainingRunner):
    def __init__(self, env_config: dict, algorithm_config: dict,
                 num_envs: int | None = None,
                 total_sampled_steps: int | None = None,
                 device: str | None = None, seed: int | None = None,
                 output_dir: str | Path | None = None,
                 smoke: bool = False) -> None:
        validate_algorithm_config(algorithm_config, env_config)
        require_cuda(device or algorithm_config['training']['device'])
        self.env_config, self.algorithm_config = env_config, algorithm_config
        self.smoke = bool(smoke)
        training, network = algorithm_config["training"], algorithm_config["network"]
        implementation = algorithm_config["implementation"]
        configured = (int(network["observation_dim"]), int(network["action_dim"]),
                      int(network["num_agents"]))
        expected = environment_dimensions(env_config)
        if configured != expected:
            raise ValueError(f"network/environment dimension mismatch: configured obs/action/agents={configured}, environment={expected}")
        if 'environment' in algorithm_config or str(env_config['environment_version']) in ('3.38', '3.39','3.40', '3.41', '3.42', '3.43'):
            expected_identity = {
                'environment_version': str(env_config['environment_version']),
                'team_size': expected[2],
            }
            actual_identity = algorithm_config.get('environment')
            if actual_identity != expected_identity:
                raise RuntimeError(
                    'IPPO algorithm/environment identity mismatch: '
                    f'expected={expected_identity!r}, actual={actual_identity!r}'
                )
        self.observation_dim, self.action_dim, self.num_agents = configured
        self.num_envs = int(training["num_train_envs"] if num_envs is None else num_envs)
        self.total_sampled_steps = int(training["total_sampled_steps"] if total_sampled_steps is None else total_sampled_steps)
        self.device = str(device or training["device"])
        self.seed = int(training["seed"] if seed is None else seed)
        if output_dir is None:
            raise ValueError("output_dir must be the final run directory")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_steps = int(training["rollout_steps"])
        actor_width = layer_width(network, "actor_hidden_layers")
        critic_width = layer_width(network, "critic_hidden_layers")
        hidden_dim = actor_width
        self.effective_hidden_dim = hidden_dim
        # The factory seed is explicitly overridden for independent training runs.
        self.trainer = build_ippo_trainer(algorithm_config, self.device, seed=self.seed)
        evaluation_base = int(implementation["evaluation_seed_base"])
        self.evaluation_seeds = list(range(
            evaluation_base, evaluation_base + int(training["evaluation_episodes"])
        ))
        if self.seed + self.total_sampled_steps + self.num_envs >= evaluation_base:
            raise ValueError("configured training seed range can overlap evaluation seeds")
        if self.num_envs <= 0 or self.total_sampled_steps <= 0 or self.total_sampled_steps % self.num_envs:
            raise ValueError('sample budget must be positive and divisible by num_envs')
        if self.seed < 0 or self.seed + self.total_sampled_steps + self.num_envs >= 10_000_000:
            raise ValueError('training seed range must stay below development and recording ranges')
        self.vector = ParallelVectorEnv(self.num_envs, env_config, self.seed,
                                        self.evaluation_seeds)
        self.observations = self.vector.reset()
        self.alive_masks = self.vector.current_alive_masks.copy()
        self.agent_episode_returns = np.zeros((self.num_envs, self.num_agents), dtype=float)
        self.completed_records: list[dict[str, Any]] = []
        self.last_metrics: dict[str, float] = {}
        self.evaluation_history: list[dict[str, float]] = []
        self.best_evaluation: dict[str, float] | None = None
        logging = algorithm_config["runtime_logging"]
        self.console_interval = int(logging["console_interval_sampled_steps"])
        self.recent_episode_window = int(logging["recent_episode_window"])
        self.evaluation_interval = int(training["evaluation_interval_sampled_steps"])
        self.checkpoint_interval = int(implementation["checkpoint_interval_sampled_steps"])
        if min(self.rollout_steps, self.console_interval, self.recent_episode_window,
               self.evaluation_interval, self.checkpoint_interval) <= 0:
            raise ValueError("IPPO rollout and reporting intervals must be positive")
        self.next_console_log = self.console_interval
        self.next_evaluation = self.evaluation_interval
        self.next_checkpoint = self.checkpoint_interval

    def collect_rollout(self, vector_steps: int | None = None) -> RolloutBatch:
        self._begin_rollout_metrics()
        storage = {key: [] for key in ("observations", "actions", "raw_actions", "old_log_probs",
                   "rewards", "dones", "alive_masks", "next_observations",
                   "next_alive_masks")}
        for _ in range(int(vector_steps or self.rollout_steps)):
            observations, alive = self.observations.copy(), self.alive_masks.copy()
            actions, raw_actions, log_probs = self.trainer.act(
                observations, alive, return_policy_data=True
            )
            result = self.vector.step_batch(actions)
            executed = np.stack([info['executed_red_actions'] for info in result.infos]).astype(np.float32)
            if not np.array_equal(actions, executed):
                raise RuntimeError('sampled/executed actions differ; PPO latent action no longer valid')
            storage["observations"].append(observations)
            storage["actions"].append(executed)
            storage["raw_actions"].append(raw_actions)
            storage["old_log_probs"].append(log_probs)
            storage["rewards"].append(result.rewards)
            storage["dones"].append((result.terminated | result.truncated).astype(np.float32))
            storage["alive_masks"].append(result.alive_masks)
            storage["next_observations"].append(result.transition_next_observations)
            storage["next_alive_masks"].append(result.next_alive_masks)
            rows = self._completed(result)
            self.observations, self.alive_masks = result.observations, self.vector.current_alive_masks.copy()
            self.trainer.sampled_steps += self.num_envs; self.trainer.vector_steps += 1
            self._accumulate_rollout_metrics(result, rows)
        batch = RolloutBatch(**{key: np.stack(value).astype(np.float32)
                               for key, value in storage.items()})
        from .diagnostics import rollout_action_diagnostics
        diagnostics = rollout_action_diagnostics(batch,self.env_config['environment_version'])
        bank_path = self.output_dir / 'diagnostic_observations.npz'
        if not bank_path.exists():
            from .diagnostics import save_observation_bank
            save_observation_bank(bank_path, batch.observations[:min(len(batch.observations),16)],
                dict(split='training',seed_provenance=sorted(self.vector.used_training_seeds),
                     environment_version=self.env_config['environment_version'],
                     environment_config_sha256=config_sha256(self.env_config),
                     source='first rollout observations before optimization'))
        with (self.output_dir / 'action_diagnostics.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(sampled_steps=self.trainer.sampled_steps, **diagnostics))+'\n')
        return batch

    def startup_summary(self):
        return dict(super().startup_summary(), algorithm='ippo', parameter_sharing=True,
                    critic_scope='local_observation', bit_exact_training_resume=False)

    def start_log_line(self):
        return super().start_log_line().replace('algorithm=MAPPO', 'algorithm=ippo')

    @staticmethod
    def done_log_line(summary):
        return MAPPOTrainingRunner.done_log_line(summary).replace('algorithm=MAPPO','algorithm=ippo')

    def save_checkpoint(self, path):
        # The baseline method supplies complete run counters and selection provenance;
        # IPPOTrainer replaces its architecture and implementation identity.
        self.trainer.evaluation_seed_base = self.evaluation_seeds[0]
        super().save_checkpoint(path)

    def resume(self, path):
        raise RuntimeError('IPPO supports deterministic evaluation reload only; exact training resume unsupported')

    def run(self):
        summary = super().run()
        if not self.evaluation_history or self.evaluation_history[-1]['sampled_steps'] != self.trainer.sampled_steps:
            row = self._evaluation_record()
            self.evaluation_history.append(row)
            self._write_evaluation()
            self._consider_best_evaluation(row)
        self.save_checkpoint(self.output_dir / 'latest.pt')
        self.save_checkpoint(self.output_dir / 'final.pt')
        # A complete diagnostic flight uses a third, disjoint seed range. It
        # never feeds best-checkpoint selection or changes training parameters.
        from tools.record_combat_episode import record
        record(self.trainer, self.env_config, 20_000_000 + self.seed,
            self.output_dir / 'diagnostic_flight', dict(algorithm='ippo',
                source='final_checkpoint_diagnostic', evaluation_split='recording',
                checkpoint_sampled_steps=self.trainer.sampled_steps))
        return self.summary()
