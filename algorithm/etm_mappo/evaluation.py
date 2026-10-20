"""Deterministic navigation evaluation, with explicit scene and seed provenance."""
import copy
import numpy as np
from .config import fingerprint, require_cuda
from .navigation import CooperativeNavigationEnv, stack_observations


def evaluate(trainer,seeds,scene=None):
    require_cuda(trainer.device)
    seeds=list(map(int,seeds))
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('evaluation seeds must be nonempty and unique')
    training=trainer.config['training']
    if min(seeds)<training['evaluation_seed_base']:
        raise ValueError('evaluation seed must be in the reserved evaluation range')
    environment=copy.deepcopy(trainer.config['environment'])
    if scene is not None:
        if scene not in ('simple','random'):raise ValueError('invalid evaluation scene')
        environment['scene']=scene
    rows=[]
    for seed in seeds:
        env=CooperativeNavigationEnv(environment)
        observation=stack_observations([env.reset(seed)])
        state=trainer.initial_state(observation);start=np.ones(1,dtype=bool)
        while True:
            action,_,_,state=trainer.act(observation,state,start,deterministic=True)
            following,reward,terminated,truncated,info=env.step(action[0])
            if terminated or truncated:
                rows.append(dict(info,terminated=terminated,truncated=truncated));break
            observation=stack_observations([following]);start.fill(False)
    mean=lambda key:float(np.mean([r[key] for r in rows]))
    return dict(episode_return=mean('episode_return'),task_completion_rate=mean('success'),
        collision_rate=mean('collision_episode'),timeout_rate=mean('truncated'),
        mean_episode_length=mean('episode_length'),goal_error=mean('goal_error'),formation_error=mean('formation_error'),
        **{f'reward_{key}':float(np.mean([r['component_totals'][key] for r in rows])) for key in environment['reward']},
        evaluation_seeds=seeds,evaluation_episodes=len(seeds),scene=environment['scene'],
        evaluation_environment_sha256=fingerprint(environment),deterministic_policy=True)

