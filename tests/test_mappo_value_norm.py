import numpy as np
import torch

from algorithm.mappo import RunningValueNorm
from algorithm.mappo_config import MAPPOConfig
from env.interception import InterceptionConfig, InterceptionEnv
from experiment.interception import make_policy


def test_mappo_seed_controls_weights_and_first_stochastic_action():
    config = InterceptionConfig()
    env = InterceptionEnv(config)
    observations, info = env.reset(seed=7)
    args = (observations[None], info['central_state'][None], info['active_masks'][None])
    first = make_policy('mappo', config, MAPPOConfig(hidden_size=16), 'cpu', 7)
    first_weights = {key: value.clone() for key, value in first.actor.state_dict().items()}
    first_actions = first.act_rollout(*args)[0]
    second = make_policy('mappo', config, MAPPOConfig(hidden_size=16), 'cpu', 7)
    for key, value in second.actor.state_dict().items():
        torch.testing.assert_close(value, first_weights[key], rtol=0, atol=0)
    np.testing.assert_array_equal(second.act_rollout(*args)[0], first_actions)


def test_value_norm_roundtrip_and_running_statistics():
    normalizer = RunningValueNorm()
    normalizer.update([-100.0, -50.0])
    normalizer.update([0.0, 50.0])
    assert normalizer.count == 4
    assert normalizer.mean == -25.0
    assert normalizer.std == np.std([-100.0, -50.0, 0.0, 50.0])
    values = torch.tensor([-70.0, 5.0])
    torch.testing.assert_close(normalizer.denormalize(normalizer.normalize(values)), values)
    restored = RunningValueNorm()
    restored.load_state_dict(normalizer.state_dict())
    assert restored.state_dict() == normalizer.state_dict()


def test_new_and_historical_mappo_checkpoint_value_scale(tmp_path):
    config = InterceptionConfig()
    parameters = MAPPOConfig(hidden_size=16)
    policy = make_policy('mappo', config, parameters, 'cpu', 7)
    states = np.zeros((2, policy.critic.network[0].in_features), np.float32)
    policy.value_normalizer.update([-100.0, 0.0, 50.0])
    expected = policy.values(states)
    path = tmp_path / 'new.pt'
    policy.save(path, global_step=5)
    loaded = make_policy('mappo', config, parameters, 'cpu', 8)
    loaded.load(path, load_optimizers=True)
    assert loaded.use_value_norm
    assert loaded.value_normalizer.state_dict() == policy.value_normalizer.state_dict()
    np.testing.assert_array_equal(loaded.values(states), expected)

    # Pre-ValueNorm checkpoints still use their original raw critic output.
    policy.use_value_norm = False
    historical_values = policy.values(states)
    old_path = tmp_path / 'old.pt'
    policy.save(old_path, global_step=5)
    payload = torch.load(old_path, map_location='cpu', weights_only=False)
    payload['extra'].pop('value_norm_state')
    payload['extra'].pop('cuda_rng_states')
    torch.save(payload, old_path)
    old = make_policy('mappo', config, parameters, 'cpu', 8)
    old.load(old_path, load_optimizers=True)
    assert not old.use_value_norm
    np.testing.assert_array_equal(old.values(states), historical_values)
