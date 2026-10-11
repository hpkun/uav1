"""Check v3.9 reward diagnostics reset in an actual spawn worker."""
import json
from pathlib import Path
import numpy as np
from algorithm.common.vector_env import ParallelVectorEnv
from env.combat_env import MultiUAVCombatEnv
from tools._capture_v33_reference import canonical
from tools.audit_combat_v39 import check_decomposition


def main():
    vector = ParallelVectorEnv(1, 'configs/combat_environment_v39.yaml', base_seed=44000000)
    try:
        vector.reset()
        total = np.zeros(5)
        for _ in range(1000):
            result = vector.step_batch(np.zeros((1, 5, 3)))
            total += result.rewards[0]
            check_decomposition(result.rewards[0], result.infos[0], total)
            if result.terminated[0] or result.truncated[0]:
                break
        else:
            raise AssertionError('episode did not finish')
        seed = int(vector.last_reset_seeds[0])
        fresh = MultiUAVCombatEnv('configs/combat_environment_v39.yaml')
        fresh.reset(seed)
        obs, reward, _, _, info = fresh.step(np.zeros((5, 3)))
        nxt = vector.step_batch(np.zeros((1, 5, 3)))
        np.testing.assert_array_equal(nxt.observations[0], obs)
        np.testing.assert_array_equal(nxt.rewards[0], reward)
        assert canonical(nxt.infos[0]) == canonical(info)
        check_decomposition(nxt.rewards[0], nxt.infos[0], nxt.rewards[0])
        Path('outputs/v39_validation/vector_reset.json').write_text(json.dumps(dict(
            seed=seed, first_step_identical_to_fresh_environment=True,
            paper_safety_extended_metrics_reset=True, decomposition_passed=True), indent=2))
    finally:
        vector.close()


if __name__ == '__main__':
    main()
