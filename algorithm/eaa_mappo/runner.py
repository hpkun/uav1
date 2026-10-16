"""Reuse baseline rollout, evaluator, update scheduling and checkpoint selection."""
from algorithm.mappo.runner import MAPPOTrainingRunner
from .factory import build_eaa_mappo_trainer
from .protocol import require_cuda, validate_config, contract


class EAAMAPPOTrainingRunner(MAPPOTrainingRunner):
    def __init__(self, env_config, algorithm_config, num_envs=None, total_sampled_steps=None,
                 device=None, seed=None, output_dir=None, smoke=False):
        validate_config(env_config, algorithm_config)
        require_cuda(device or algorithm_config['training']['device'])
        t = algorithm_config['training']
        count = t['num_train_envs'] if num_envs is None else num_envs
        budget = t['total_sampled_steps'] if total_sampled_steps is None else total_sampled_steps
        if count <= 0 or budget <= 0 or budget % count or (seed is not None and seed < 0):
            raise ValueError('positive budget divisible by num_envs and nonnegative seed required')
        constructor = lambda *args, **kwargs: build_eaa_mappo_trainer(
            algorithm_config, env_config, kwargs['device'] if 'device' in kwargs else (device or algorithm_config['training']['device']),
            algorithm_config['training']['seed'] if seed is None else seed)
        # New smoke keeps formal networks and PPO parameters; only CLI budget/env count shrink.
        super().__init__(env_config,algorithm_config,num_envs,total_sampled_steps,device,seed,
                         output_dir,False,trainer_factory=constructor)
        self.smoke = bool(smoke)
        if smoke:
            self.evaluation_seeds = self.evaluation_seeds[:1]
            self.console_interval = self.evaluation_interval = self.checkpoint_interval = self.total_sampled_steps
            self.next_console_log = self.next_evaluation = self.next_checkpoint = self.total_sampled_steps

    def startup_summary(self):
        result = super().startup_summary()
        result.update(algorithm=self.trainer.variant, **contract(self.env_config,self.algorithm_config))
        return result

    def start_log_line(self):
        return super().start_log_line().replace('algorithm=MAPPO',f'algorithm={self.trainer.variant}',1)

    def train_log_line(self):
        line = super().train_log_line()
        if 'actor_attention_self_mass' in self.last_metrics:
            line += ' | Attn S/A/E=' + '/'.join(f"{self.last_metrics['actor_attention_'+n+'_mass']:.2f}" for n in ('self','ally','enemy'))
        return line

    def _finalize_rollout_metrics(self):
        result = super()._finalize_rollout_metrics()
        result.update({k:v for k,v in self.last_metrics.items() if '_attention_' in k})
        return result

    def resume(self, path):
        raise RuntimeError('EAA v1 supports checkpoint evaluation/load; training resume is not implemented')
