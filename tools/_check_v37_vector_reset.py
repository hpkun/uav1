"""Ensure new tactical diagnostics and RNG reset in actual spawn workers."""
import json
from pathlib import Path
import numpy as np
from algorithm.common.vector_env import ParallelVectorEnv
from env.combat_env import MultiUAVCombatEnv
from tools._capture_v33_reference import canonical

def main():
    vector=ParallelVectorEnv(1,'configs/combat_environment_v37.yaml',base_seed=42000000)
    try:
        vector.reset()
        for _ in range(1000):
            result=vector.step_batch(np.zeros((1,5,3)))
            if result.terminated[0] or result.truncated[0]:break
        else:raise AssertionError('episode did not finish')
        seed=int(vector.last_reset_seeds[0]);fresh=MultiUAVCombatEnv('configs/combat_environment_v37.yaml');fresh.reset(seed)
        obs,reward,_,_,info=fresh.step(np.zeros((5,3)))
        nxt=vector.step_batch(np.zeros((1,5,3)))
        np.testing.assert_array_equal(nxt.observations[0],obs);np.testing.assert_array_equal(nxt.rewards[0],reward)
        assert canonical(nxt.infos[0])==canonical(info)
        Path('outputs/v37_validation/vector_reset.json').write_text(json.dumps(dict(
            seed=seed,first_step_identical_to_fresh_environment=True,new_tactical_metrics_reset=True),indent=2))
    finally:vector.close()

if __name__=='__main__':main()
