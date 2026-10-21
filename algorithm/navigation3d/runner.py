"""Collect the standard RolloutBatch; never replace the original PPO/GAE math."""
import numpy as np
from algorithm.mappo.trainer import RolloutBatch
from .vector_env import NavigationVectorEnv
from .trainer import NavigationTrainer


class NavigationRunner:
    def __init__(self, env_config, algorithm_config, num_envs=2, seed=0, device='cuda'):
        self.trainer = NavigationTrainer(env_config, algorithm_config, device, seed)
        self.vector = NavigationVectorEnv(num_envs, env_config, seed)
        self.observations = self.vector.reset()
        self.rollout_steps = algorithm_config['training']['rollout_steps']
        self.completed = []

    def collect_rollout(self, vector_steps=None):
        storage = {k: [] for k in RolloutBatch.__dataclass_fields__}
        for _ in range(self.rollout_steps if vector_steps is None else vector_steps):
            obs, mask = self.observations.copy(), self.vector.current_alive_masks.copy()
            actions, raw, logp = self.trainer.act(obs, mask, return_policy_data=True)
            step = self.vector.step_batch(actions)
            executed = np.stack([row['executed_actions'] for row in step.infos]).astype(np.float32)
            if not np.array_equal(executed, actions):
                raise RuntimeError('navigation sampled/executed action mismatch')
            values = dict(observations=obs, actions=actions, raw_actions=raw, old_log_probs=logp,
                rewards=step.rewards, dones=step.terminated | step.truncated,
                alive_masks=step.alive_masks, next_observations=step.transition_next_observations,
                next_alive_masks=step.next_alive_masks)
            for key, value in values.items():
                storage[key].append(value)
            for done, info in zip(step.terminated | step.truncated, step.infos):
                if done:
                    self.completed.append(info)
            self.observations = step.observations
            self.trainer.sampled_steps += self.vector.num_envs
            self.trainer.vector_steps += 1
        return RolloutBatch(**{k: np.array(v, dtype=np.float32) for k, v in storage.items()})

    def close(self):
        self.vector.close()
