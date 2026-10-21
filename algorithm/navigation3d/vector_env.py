"""Long-lived spawn workers with explicit terminal frames and separate resets."""
import multiprocessing as mp
import os
import traceback
import numpy as np
from algorithm.common.vector_env import ParallelVectorEnv, VectorStep
from env.navigation3d import Navigation3DEnv


def navigation_worker(connection, config):
    try:
        env = Navigation3DEnv(config)
        connection.send(('ready', dict(pid=os.getpid(), environment_id=env.environment_id)))
        while True:
            command, payload = connection.recv()
            if command == 'reset':
                obs, _ = env.reset(payload)
                connection.send(('ok', (obs, env.valid_mask.astype(np.float32))))
            elif command == 'step':
                connection.send(('ok', env.step(payload)))
            elif command == 'close':
                connection.send(('ok', None))
                break
            else:
                raise RuntimeError(f'unknown navigation worker command {command}')
    except EOFError:
        pass
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


class NavigationVectorEnv(ParallelVectorEnv):
    """Reuse seed scheduling, transport validation and shutdown from the baseline."""
    def __init__(self, num_envs, config, base_seed=0, forbidden_seeds=()):
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError('positive navigation worker count required')
        env = Navigation3DEnv(config)
        self.num_envs, self.base_seed = num_envs, int(base_seed)
        self.forbidden_seeds = set(map(int, forbidden_seeds))
        self.used_training_seeds = set()
        self.episode_indices = np.zeros(num_envs, dtype=np.int64)
        self.last_reset_seeds = np.zeros(num_envs, dtype=np.int64)
        self._closed, self._connections, self._processes = False, [], []
        self.observation_dim, self.action_dim, self.team_size = env.observation_dim, 3, env.num_agents
        self.current_alive_masks = np.zeros((num_envs, self.team_size), dtype=np.float32)
        self.current_observations = np.zeros((num_envs, self.team_size, self.observation_dim), dtype=np.float32)
        context = mp.get_context('spawn')
        try:
            for index in range(num_envs):
                parent, child = context.Pipe()
                process = context.Process(target=navigation_worker, args=(child, config), name=f'navigation3d-{index}', daemon=True)
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
            self.worker_metadata = [self._receive(c, 'startup') for c in self._connections]
            self.worker_pids = [v['pid'] for v in self.worker_metadata]
        except BaseException:
            self.close()
            raise

    def step_batch(self, actions, auto_reset=True):
        self._ensure_open()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.team_size, 3) or not np.isfinite(actions).all():
            raise ValueError('invalid batched navigation actions')
        before = self.current_alive_masks.copy()
        for connection, action in zip(self._connections, actions):
            connection.send(('step', action))
        rows = [self._receive(c, 'step') for c in self._connections]
        terminal_obs = self._validate_observations(np.stack([r[0] for r in rows]), 'step')
        rewards = np.stack([r[1] for r in rows])
        terminated, truncated = (np.array([r[k] for r in rows], bool) for k in (2, 3))
        infos = [r[4] for r in rows]
        terminal_masks = np.stack([r['valid_mask'] for r in infos]).astype(np.float32)
        current, masks = terminal_obs.copy(), terminal_masks.copy()
        if auto_reset:
            indices = np.flatnonzero(terminated | truncated).tolist()
            for index in indices:
                self.episode_indices[index] += 1
            for index, (obs, mask) in zip(indices, self._reset_many(indices)):
                current[index], masks[index] = obs, mask
        self.current_observations, self.current_alive_masks = current, masks
        return VectorStep(current.copy(), terminal_obs, rewards, terminated, truncated, infos, before, terminal_masks)
