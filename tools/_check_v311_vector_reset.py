"""Real multiprocessing worker auto-reset with the new initialization/fire state."""
import json
from pathlib import Path
import numpy as np
from algorithm.common.vector_env import ParallelVectorEnv
from env.combat_env import MultiUAVCombatEnv
from tools._capture_v33_reference import canonical
from tools.audit_combat_v39 import check_decomposition


def main():
    vector = ParallelVectorEnv(1, 'configs/combat_environment_v311.yaml', base_seed=45000000)
    try:
        vector.reset()
        accumulated = np.zeros(5)
        for _ in range(1000):
            result = vector.step_batch(np.zeros((1, 5, 3)))
            accumulated += result.rewards[0]
            check_decomposition(result.rewards[0], result.infos[0], accumulated)
            if result.terminated[0] or result.truncated[0]:
                break
        else:
            raise AssertionError('episode did not finish')
        seed = int(vector.last_reset_seeds[0])
        fresh = MultiUAVCombatEnv('configs/combat_environment_v311.yaml')
        fresh.reset(seed)
        assert fresh.fixed_policy.last_targets == []
        assert fresh.fixed_policy.pursuit_agent_steps == fresh.fixed_policy.target_switches == 0
        assert fresh.red_fire_states is fresh.blue_fire_states is None
        assert (fresh.red_ammo == 6).all() and (fresh.blue_ammo == 6).all()
        obs, reward, _, _, info = fresh.step(np.zeros((5, 3)))
        nxt = vector.step_batch(np.zeros((1, 5, 3)))
        np.testing.assert_array_equal(nxt.observations[0], obs)
        np.testing.assert_array_equal(nxt.rewards[0], reward)
        assert canonical(nxt.infos[0]) == canonical(info)
        assert nxt.infos[0]['blue_paper_nearest_agent_steps'] == 5
        assert nxt.infos[0]['blue_target_switches'] == 0
        assert (nxt.infos[0]['red_ammo'] == 6).all() and (nxt.infos[0]['blue_ammo'] == 6).all()
        check_decomposition(nxt.rewards[0], nxt.infos[0], nxt.rewards[0])
        Path('outputs/v311_validation/vector_reset.json').write_text(json.dumps(dict(
            seed=seed, first_step_identical_to_fresh_environment=True, ammo_reset=True,
            blue_targets_and_counters_reset=True, phi_zero=True, paper_safety_extended_reset=True), indent=2))
    finally:
        vector.close()


if __name__ == '__main__':
    main()
