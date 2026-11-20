"""Multiprocess interception sampler with preserved terminal transitions."""
from dataclasses import dataclass, replace
import multiprocessing as mp
import traceback

import numpy as np

from .interception import InterceptionConfig, InterceptionEnv


@dataclass
class VectorStep:
    observations: np.ndarray
    states: np.ndarray
    active_masks: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    dones: np.ndarray
    transition_observations: np.ndarray
    transition_states: np.ndarray
    transition_masks: np.ndarray
    infos: list


def _mode(requested,seed):
    return ('direct_rush' if seed%2==0 else 'split_attack') if requested=='mixed' else requested


def _worker(pipe,config,attack_mode):
    env=None
    try:
        pipe.send(('ready',None))
        base_seed=None
        episode=0
        while True:
            command,payload=pipe.recv()
            if command=='close':
                break
            if command=='reset':
                base_seed=int(payload)
                episode=0
                env=InterceptionEnv(replace(config,attack_mode=_mode(attack_mode,base_seed)))
                obs,info=env.reset(seed=base_seed)
                pipe.send(('result',(obs,info['central_state'],info['active_masks'],info)))
            elif command=='step':
                obs,_,term,trunc,info=env.step(payload)
                final=(obs,info['central_state'],info['active_masks'])
                reward=info['team_reward']
                if term or trunc:
                    episode+=1
                    seed=base_seed+1_000_003*episode
                    env.close()
                    env=InterceptionEnv(replace(config,attack_mode=_mode(attack_mode,seed)))
                    obs,reset=env.reset(seed=seed)
                    current=(obs,reset['central_state'],reset['active_masks'])
                else:
                    current=final
                pipe.send(('result',(current,final,reward,term,trunc,info)))
            else:
                raise ValueError('Unknown sampler command')
    except EOFError:
        pass
    except BaseException:
        try:pipe.send(('error',traceback.format_exc()))
        except (EOFError,BrokenPipeError):pass
    finally:
        if env is not None:env.close()
        pipe.close()


class ParallelInterceptionEnvs:
    def __init__(self,num_envs,config:InterceptionConfig,*,seed=0,attack_mode='mixed'):
        if num_envs<1 or attack_mode not in ('mixed','direct_rush','split_attack'):
            raise ValueError('Invalid sampler configuration')
        self.num_envs=num_envs
        self.seed=seed
        self.connections=[]
        self.processes=[]
        context=mp.get_context('spawn')
        try:
            for _ in range(num_envs):
                parent,child=context.Pipe()
                process=context.Process(target=_worker,args=(child,config,attack_mode),daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            for i in range(num_envs):self._receive(i,'ready')
        except BaseException:
            self.close()
            raise

    def _receive(self,index,expected='result'):
        connection=self.connections[index]
        if not connection.poll(60):
            raise RuntimeError(f'Interception worker {index} timed out')
        try:tag,data=connection.recv()
        except (EOFError,OSError) as error:
            raise RuntimeError(f'Interception worker {index} disconnected') from error
        if tag!=expected:
            raise RuntimeError(f'Interception worker {index}: {data}')
        return data

    def reset(self):
        for index,connection in enumerate(self.connections):
            connection.send(('reset',self.seed+index))
        rows=[self._receive(i) for i in range(self.num_envs)]
        return *(np.stack([row[k] for row in rows]) for k in range(3)),[row[3] for row in rows]

    def step(self,actions):
        actions=np.asarray(actions)
        if actions.shape!=(self.num_envs,5,2):
            raise ValueError('Expected [worker,defender,xy] actions')
        for connection,action in zip(self.connections,actions):
            connection.send(('step',action))
        rows=[self._receive(i) for i in range(self.num_envs)]
        current=[np.stack([row[0][k] for row in rows]) for k in range(3)]
        final=[np.stack([row[1][k] for row in rows]) for k in range(3)]
        term=np.asarray([row[3] for row in rows],bool)
        trunc=np.asarray([row[4] for row in rows],bool)
        return VectorStep(*current,np.asarray([row[2] for row in rows],np.float32),term,trunc,
                          (term|trunc).astype(np.float32),*final,[row[5] for row in rows])

    def close(self):
        for connection,process in zip(self.connections,self.processes):
            if process.is_alive():
                try:connection.send(('close',None))
                except (EOFError,BrokenPipeError,OSError):pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for connection in self.connections:connection.close()

    def __enter__(self):return self

    def __exit__(self,*_):self.close()
