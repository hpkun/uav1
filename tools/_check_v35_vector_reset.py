"""Compare post-auto-reset spawn worker with a fresh same-seed environment."""
import json
from pathlib import Path
import numpy as np
from algorithm.common.vector_env import ParallelVectorEnv
from env.combat_env import MultiUAVCombatEnv
from tools._capture_v33_reference import canonical

def main():
    vector=ParallelVectorEnv(1,'configs/combat_environment_v35.yaml',base_seed=40000000)
    try:
        vector.reset()
        for _ in range(1000):
            result=vector.step_batch(np.zeros((1,5,3)));last=result.infos[0]
            if result.terminated[0] or result.truncated[0]:break
        else:raise AssertionError('no episode end')
        seed=int(vector.last_reset_seeds[0])
        fresh=MultiUAVCombatEnv('configs/combat_environment_v35.yaml');fresh.reset(seed)
        observation,reward,_,_,info=fresh.step(np.zeros((5,3)))
        nxt=vector.step_batch(np.zeros((1,5,3)))
        np.testing.assert_array_equal(nxt.observations[0],observation)
        np.testing.assert_array_equal(nxt.rewards[0],reward)
        assert canonical(nxt.infos[0])==canonical(info)
        assert info['episode_length']==1 and sum(info[f'blue_{k}_agent_steps'] for k in ('detected_pursuit','detected_escape','guerrilla','centripetal'))==5
        Path('outputs/v35_validation/vector_reset.json').write_text(json.dumps(dict(
            first_episode_steps=last['episode_length'],next_episode_seed=seed,
            counters_flags_returns_and_blue_rng_reset=True,
            next_episode_first_step_exactly_matches_fresh_environment=True),indent=2))
    finally:vector.close()

if __name__=='__main__':main()
