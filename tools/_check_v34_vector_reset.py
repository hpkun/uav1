"""Bounded acceptance of actual spawn-worker automatic episode reset."""
import json
from pathlib import Path
import numpy as np
from algorithm.common.vector_env import ParallelVectorEnv

def main():
    with_env=ParallelVectorEnv(1,'configs/combat_environment_v34.yaml',base_seed=39000000)
    try:
        with_env.reset();last=None
        for _ in range(1000):
            result=with_env.step_batch(np.zeros((1,5,3)))
            last=result.infos[0]
            if result.terminated[0] or result.truncated[0]:break
        else:raise AssertionError('episode did not finish')
        assert sum(last[f'blue_{k}_agent_steps'] for k in ('search','pursuit','evade','guard'))>5
        nxt=with_env.step_batch(np.zeros((1,5,3))).infos[0]
        assert nxt['episode_length']==1
        assert sum(nxt[f'blue_{k}_agent_steps'] for k in ('search','pursuit','evade','guard'))==5
        assert nxt['blue_evade_episode']==('EVADE' in nxt['blue_policy_modes'])
        Path('outputs/v34_validation/vector_reset.json').write_text(json.dumps(dict(
            first_episode_evaded=last['blue_evade_episode'],first_episode_steps=last['episode_length'],
            next_episode_steps=nxt['episode_length'],next_episode_modes=nxt['blue_policy_modes'],
            counters_reset=True,evade_flag_reset=True),indent=2))
    finally:with_env.close()

if __name__=='__main__':main()
