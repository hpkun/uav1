"""56D multiprocessing batches and full reset agreement with a fresh env."""
import json
from pathlib import Path
import numpy as np
from algorithm.common.vector_env import ParallelVectorEnv
from env.combat_env import MultiUAVCombatEnv
from tools._capture_v33_reference import canonical
from tools.audit_combat_v39 import check_decomposition


def main():
    vector = ParallelVectorEnv(1, 'configs/combat_environment_v312.yaml', base_seed=45000000)
    try:
        obs = vector.reset()
        assert obs.shape == (1,5,56) and vector.observation_dim == 56
        accumulated = np.zeros(5)
        for _ in range(1000):
            result = vector.step_batch(np.zeros((1,5,3)))
            assert result.observations.shape == result.transition_next_observations.shape == (1,5,56)
            accumulated += result.rewards[0]
            check_decomposition(result.rewards[0],result.infos[0],accumulated)
            if result.terminated[0] or result.truncated[0]:
                break
        else:
            raise AssertionError('episode did not finish')
        seed = int(vector.last_reset_seeds[0])
        fresh = MultiUAVCombatEnv('configs/combat_environment_v312.yaml')
        fresh.reset(seed)
        assert fresh.fixed_policy.last_targets == [] and (fresh.red_ammo == 6).all()
        obs, reward, _, _, info = fresh.step(np.zeros((5,3)))
        nxt = vector.step_batch(np.zeros((1,5,3)))
        # Vector env retains its historical float32 network-input convention.
        np.testing.assert_array_equal(nxt.observations[0], obs.astype(np.float32))
        np.testing.assert_array_equal(nxt.rewards[0],reward)
        assert canonical(nxt.infos[0]) == canonical(info)
        check_decomposition(nxt.rewards[0],nxt.infos[0],nxt.rewards[0])
        Path('outputs/v312_validation/vector_reset.json').write_text(json.dumps(dict(seed=seed,
            observation_shape=[1,5,56], observation_dim=56,
            first_step_identical_to_fresh_environment=True, ammo_and_blue_diagnostics_reset=True),indent=2))
    finally:
        vector.close()


if __name__ == '__main__':
    main()
