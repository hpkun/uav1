"""Real CUDA IPPO checkpoint rejects missing or corrupt v3.47 dimensions."""
from copy import deepcopy
from pathlib import Path
import pytest
import torch
import yaml
from algorithm.ippo.runner import IPPOTrainingRunner
from algorithm.ippo.protocol import validate_checkpoint
from env.config import load_config


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    assert torch.cuda.is_available(), 'CUDA mandatory; no CPU fallback'
    torch.set_num_threads(1)
    ec = load_config('configs/combat_environment_v347.yaml')
    ac = yaml.safe_load(Path('configs/ippo_4v4_v347.yaml').read_text())
    out = tmp_path_factory.mktemp('v347_checkpoint')
    runner = IPPOTrainingRunner(ec, ac, num_envs=1, total_sampled_steps=1,
                                output_dir=out, device='cuda')
    try:
        runner.save_checkpoint(out / 'identity.pt')
        state = torch.load(out / 'identity.pt', map_location='cuda', weights_only=False)
        validate_checkpoint(state, ec, ac)
        return state, ec, ac
    finally:
        runner.vector.close()


@pytest.mark.parametrize('key', ['observation_dim', 'action_dim', 'num_agents'])
@pytest.mark.parametrize('value', [None, 49, '45', True])
def test_ippo_dimension_metadata_fail_closed(checkpoint, key, value):
    state, ec, ac = checkpoint
    bad = deepcopy(state)
    if value is None:
        bad['extra'].pop(key)
    else:
        bad['extra'][key] = value
    with pytest.raises(RuntimeError, match=key):
        validate_checkpoint(bad, ec, ac)
